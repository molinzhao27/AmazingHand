#!/usr/bin/env python3
"""Drive the powered wrist (2 x STS3215, IDs 21/22) together with the 5-digit hand (10 x SCS0009, IDs 1-10).

NOT TESTED ON HARDWARE: the wrist was verified in simulation only (Demo/AHSimulation/wrist). Start with the
torque limit low (TORQUE_LIMIT) and the hand clear of anything, and check the direction step below first.

Both servo families sit on one TTL bus but use different register layouts (SCS0009: big-endian, 1024
steps over 300 deg; STS3215: little-endian, 4096 steps per turn), and a serial port can only be opened
once, so this uses rustypot's mixed-model `Bus` (rustypot >= 1.12, `pip install -U rustypot`) with raw
register values and the same radian conversions as rustypot's Scs0009PyController / Sts3215PyController.
With a second USB bus adapter for the wrist (--wrist-port), the wrist uses its own Sts3215PyController.

The wrist kinematics (wrist angles <-> servo angles) come from cad/five_finger/wrist/wrist_design.py.

    python wrist_hw_demo.py --port /dev/ttyACM0 --check-direction     # first run: see README
    python wrist_hw_demo.py --port /dev/ttyACM0                       # demo
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rustypot

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "wrist"))
import wrist_design as wd  # noqa: E402

HAND_IDS = list(range(1, 11))                 # right hand: index 1/2, middle 3/4, ring 5/6, thumb 7/8, pinky 9/10
WRIST_IDS = [21, 22]                          # motor1 = ulnar (pinky-side) servo, motor2 = radial (index-side)
MIDDLE_POS = [3, 0, -5, -8, -2, 5, -12, 0, 0, 0]   # deg, your hand calibration (AmazingHand_Hand_FingerMiddlePos.py)
WRIST_ZERO = [0.0, 0.0]                       # deg, servo angle with the horn arm square to the rod at the neutral wrist
WRIST_INVERT = [False, False]                 # set True if --check-direction shows the rod going the wrong way
TORQUE_LIMIT = 0.5                            # fraction of the STS3215 torque for the first runs (register max_torque_limit / torque_limit)
SCS_RANGE = math.radians(300.0)


def scs_to_raw(rad):
    return int(1024.0 * rad / SCS_RANGE + 511.0)


def sts_to_raw(rad):
    return int(4096.0 * (math.pi + rad) / (2 * math.pi))


def sts_from_raw(raw):
    return 2 * math.pi * raw / 4096.0 - math.pi


class Robot:
    def __init__(self, port, wrist_port=None, baudrate=1_000_000):
        if wrist_port:      # two adapters: the stock per-model controllers
            self.hand = rustypot.Scs0009PyController(serial_port=port, baudrate=baudrate, timeout=0.05)
            self.wrist = rustypot.Sts3215PyController(serial_port=wrist_port, baudrate=baudrate, timeout=0.05)
            self.bus = None
        else:               # one adapter, one port: mixed-model bus
            motors = {i: rustypot.Scs0009PyController.definition() for i in HAND_IDS}
            motors.update({i: rustypot.Sts3215PyController.definition() for i in WRIST_IDS})
            self.bus = rustypot.Bus(port, baudrate, 0.05, motors)

    def torque(self, on):
        limit = [int(1000 * TORQUE_LIMIT)] * len(WRIST_IDS)        # STS3215 torque_limit: 0..1000 = 0..100 %
        if self.bus:
            if on:
                self.bus.sync_write_register(WRIST_IDS, "torque_limit", limit)
            self.bus.set_torque(HAND_IDS + WRIST_IDS, on)
        else:
            if on:
                self.wrist.sync_write_torque_limit(WRIST_IDS, limit)
            self.hand.sync_write_torque_enable(HAND_IDS, [1 if on else 0] * len(HAND_IDS))
            self.wrist.sync_write_torque_enable(WRIST_IDS, [on] * len(WRIST_IDS))

    def wrist_goal(self, theta):
        """Servo angles (rad, wrist_design convention) -> goal positions, with zero offsets and inversion."""
        cmd = [(-1 if inv else 1) * (t + math.radians(z)) for t, z, inv in zip(theta, WRIST_ZERO, WRIST_INVERT)]
        if self.bus:
            self.bus.sync_write_register(WRIST_IDS, "goal_position", [sts_to_raw(c) for c in cmd])
        else:
            self.wrist.sync_write_goal_position(WRIST_IDS, cmd)

    def wrist_present(self):
        if self.bus:
            raw = self.bus.sync_read_register(WRIST_IDS, "present_position")
            cmd = [sts_from_raw(r if r < 32768 else -(r - 32768)) for r in raw]      # sign-magnitude
        else:
            cmd = list(self.wrist.sync_read_present_position(WRIST_IDS))
        return np.array([(-1 if inv else 1) * c - math.radians(z) for c, z, inv in zip(cmd, WRIST_ZERO, WRIST_INVERT)])

    def finger(self, k, a1_deg, a2_deg):
        ids = [2 * k + 1, 2 * k + 2]
        pos = [math.radians(MIDDLE_POS[2 * k] + a1_deg), math.radians(MIDDLE_POS[2 * k + 1] + a2_deg)]
        if self.bus:
            self.bus.sync_write_register(ids, "goal_position", [scs_to_raw(p) for p in pos])
        else:
            self.hand.sync_write_goal_position(ids, pos)


def move_wrist(robot, flex_deg, dev_deg, hold=0.0, outer_loop=True):
    """Go to (flexion, deviation) in deg; flexion > 0 palm-ward, deviation > 0 radial (towards the thumb).
    The outer loop re-reads the servo angles and nudges the goals so the P-controlled servos end where the
    IK wants them under gravity load (in simulation: 1.5-3.6 deg of sag removed, to < 0.15 deg)."""
    phi, psi = wd.clamp_wrist(math.radians(flex_deg), math.radians(dev_deg))   # coupled operating ROM
    target = wd.servo_ik(phi, psi)
    goal = target.copy()
    robot.wrist_goal(goal)
    t_end = time.time() + hold
    while outer_loop and time.time() < t_end:
        time.sleep(0.05)
        err = target - robot.wrist_present()
        goal = np.clip(goal + 0.1 * err, -wd.SERVO_LIMIT, wd.SERVO_LIMIT)
        robot.wrist_goal(goal)


def check_direction(robot):
    """Move each wrist servo alone by +10 deg from the neutral pose and ask what the rod did."""
    robot.torque(True)
    move_wrist(robot, 0, 0, hold=1.0, outer_loop=False)
    for k, name in enumerate(("motor1 (ID 21, ulnar / pinky side)", "motor2 (ID 22, radial / index side)")):
        th = wd.servo_ik(0.0, 0.0)
        th[k] = math.radians(10)
        robot.wrist_goal(th)
        time.sleep(1.0)
        expect = "UP" if wd.KAPPA[k + 1] > 0 else "DOWN"
        print(f"{name}: its push rod should have moved {expect} (towards the hand). If not, set WRIST_INVERT[{k}] = True.")
        move_wrist(robot, 0, 0, hold=1.0, outer_loop=False)


def demo(robot):
    robot.torque(True)
    for k in range(5):
        robot.finger(k, -35, 35)                       # OpenHand
    move_wrist(robot, 0, 0, hold=1.0)
    for _ in range(2):                                  # wave: radial / ulnar
        move_wrist(robot, 0, 20, hold=0.6)
        move_wrist(robot, 0, -20, hold=0.6)
    for _ in range(2):                                  # nod: palm-ward / extension
        move_wrist(robot, 45, 0, hold=0.8)
        move_wrist(robot, -45, 0, hold=0.8)
    for a in np.linspace(0, 2 * np.pi, 25):             # circumduction (inside the coupled ROM)
        move_wrist(robot, 40 * np.sin(a), 20 * np.cos(a), hold=0.15)
    move_wrist(robot, 0, 0, hold=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--wrist-port", default=None, help="second bus adapter for the wrist (optional)")
    ap.add_argument("--check-direction", action="store_true")
    args = ap.parse_args()
    robot = Robot(args.port, args.wrist_port)
    try:
        check_direction(robot) if args.check_direction else demo(robot)
    finally:
        robot.torque(False)


if __name__ == "__main__":
    main()
