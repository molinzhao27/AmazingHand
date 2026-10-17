#!/usr/bin/env python3
"""The powered wrist (2 x STS3215, IDs 21/22) with the final 5-digit hand: a separate entry point next to
AmazingHand_Demo_5F.py, which drives the hand here (same IDs, calibration, clamps and torque limits).

Every wrist command goes through wrist_design.clamp_wrist() (the coupled operating ROM: flexion +-60 deg, deviation
+-25 deg tapering to +-15 at full flexion; outside it the push rods reach their servos' gear ends) and the closed-form IK
of cad/five_finger/wrist/wrist_design.py. Moves are streamed along straight lines in (flexion, deviation), which stay in
the (convex) ROM, and every frame is clamped. Right before every bus write, check_wrist() refuses a servo goal that is not
the IK of a ROM pose (give or take the outer loop's sag correction, at most SAG_MAX).

NOT TESTED ON HARDWARE: the wrist is verified in simulation only (Demo/AHSimulation/wrist). Read
cad/five_finger/wrist/README.md ("Control code changes") first, run --check-direction, set WRIST_ZERO / WRIST_INVERT.

    python AmazingHand_Wrist_5F.py --dry-run
    python AmazingHand_Wrist_5F.py --self-test
    python AmazingHand_Wrist_5F.py --port /dev/ttyACM0 --check-direction            # first run
    python AmazingHand_Wrist_5F.py --port /dev/ttyACM0                              # hand and wrist on one adapter
    python AmazingHand_Wrist_5F.py --port /dev/ttyACM0 --wrist-port /dev/ttyACM1    # the wrist on a 2nd adapter
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import AmazingHand_Demo_5F as ah

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "wrist"))
import wrist_design as wd  # noqa: E402

WRIST_IDS = [21, 22]           # motor1 = ulnar (pinky-side) servo, motor2 = radial (index-side)
WRIST_ZERO = [0.0, 0.0]        # deg, servo angle with the horn arm square to the rod at the neutral wrist (your calibration)
WRIST_INVERT = [False, False]  # True for a servo whose rod --check-direction shows going the wrong way
WRIST_TORQUE_LIMIT = 500       # STS3215 torque_limit (RAM, 0..1000 = 0..100 %) for the first runs
WristSpeed = 1.0               # rad/s of the wrist angles
SAG_MAX = 6.0                  # deg the outer loop may move a goal off the IK (sag in the sim: 2.2-4.0 deg)
SCS_RANGE = math.radians(300.0)


def scs_to_raw(rad):
    return int(1024.0 * rad / SCS_RANGE + 511.0)


def sts_to_raw(rad):
    return int(4096.0 * (math.pi + rad) / (2 * math.pi))


def sts_from_raw(raw):
    return 2 * math.pi * raw / 4096.0 - math.pi


def check_wrist(phi, psi, theta):
    """Violations of a wrist goal: (phi, psi) wrist angles and theta the servo angles (rad, wrist_design convention)."""
    if not all(math.isfinite(float(v)) for v in (phi, psi, *theta)):
        return [f"non-finite wrist goal ({phi}, {psi}) {list(theta)}"]
    bad = []
    if abs(phi) > wd.FLEX_MAX + 1e-9 or abs(psi) > wd.rom_dev_limit(phi) + 1e-9:
        bad.append(f"wrist ({math.degrees(phi):.2f}, {math.degrees(psi):.2f}) deg outside the operating ROM")
    ik = wd.servo_ik(phi, psi)
    if not np.all(np.isfinite(ik)):
        bad.append("no IK")
    elif np.any(np.abs(np.asarray(theta) - ik) > math.radians(SAG_MAX) + 1e-9):
        bad.append(f"servo goal {np.degrees(theta).round(2)} more than {SAG_MAX:g} deg off the IK {np.degrees(ik).round(2)}")
    if np.any(np.abs(theta) > wd.SERVO_LIMIT + 1e-9):
        bad.append(f"servo goal {np.degrees(theta).round(2)} beyond +-{math.degrees(wd.SERVO_LIMIT):g} deg")
    return bad


class MixedBus:
    """Hand and wrist on one adapter: rustypot's mixed-model Bus (rustypot >= 1.12), raw register values with the
    conversions of Scs0009PyController / Sts3215PyController (as Demo/AHSimulation/wrist/control/wrist_hw_demo.py)."""
    realtime = True

    def __init__(self, port, baudrate=1_000_000):
        import rustypot
        motors = {i: rustypot.Scs0009PyController.definition() for i in ah.ALL_IDS}
        motors.update({i: rustypot.Sts3215PyController.definition() for i in WRIST_IDS})
        self.bus = rustypot.Bus(port, baudrate, 0.05, motors)

    def write_positions(self, ids, rad):
        self.bus.sync_write_register(list(ids), "goal_position", [scs_to_raw(r) for r in rad])

    def write_speeds(self, ids, rad_s):
        self.bus.sync_write_register(list(ids), "goal_speed", [int(v / (SCS_RANGE / 1024)) for v in rad_s])

    def torque(self, ids, on):
        failed = self.bus.set_torque(list(ids), on)
        if failed:
            raise RuntimeError(f"torque {'on' if on else 'off'} failed: {failed}")

    def read_position(self, i):
        return (self.bus.read_register(i, "present_position") - 511) * SCS_RANGE / 1024

    def read_torque_limit(self, i):
        return int(self.bus.read_register(i, "max_torque_limit"))

    def write_torque_limit(self, i, raw):
        self.bus.write_register(i, "lock", 0)
        self.bus.write_register(i, "max_torque_limit", int(raw))
        self.bus.write_register(i, "lock", 1)

    def wrist_torque_limit(self, raw):
        self.bus.sync_write_register(WRIST_IDS, "torque_limit", [int(raw)] * len(WRIST_IDS))

    def wrist_torque(self, on):
        self.torque(WRIST_IDS, on)

    def wrist_write(self, rad):
        self.bus.sync_write_register(WRIST_IDS, "goal_position", [sts_to_raw(r) for r in rad])

    def wrist_read(self):
        return [sts_from_raw(r) for r in self.bus.sync_read_register(WRIST_IDS, "present_position")]


class TwoPortBus(ah.ScsBus):
    """The hand on one adapter (ah.ScsBus), the wrist on a second one."""

    def __init__(self, port, wrist_port, baudrate=1_000_000):
        super().__init__(port, baudrate)
        from rustypot import Sts3215PyController
        self.w = Sts3215PyController(serial_port=wrist_port, baudrate=baudrate, timeout=0.05)

    def wrist_torque_limit(self, raw):
        self.w.sync_write_torque_limit(WRIST_IDS, [int(raw)] * len(WRIST_IDS))

    def wrist_torque(self, on):
        self.w.sync_write_torque_enable(WRIST_IDS, [bool(on)] * len(WRIST_IDS))

    def wrist_write(self, rad):
        self.w.sync_write_goal_position(WRIST_IDS, [float(r) for r in rad])

    def wrist_read(self):
        return [float(r) for r in self.w.sync_read_present_position(WRIST_IDS)]


class DryBus(ah.DryBus):
    """No hardware: the hand's DryBus plus the wrist (servo angles kept, so the outer loop sees no error)."""

    def __init__(self, verbose=False, wrist=(0.0, 0.0)):
        super().__init__(verbose=verbose)
        self.wrist = list(wrist)

    def wrist_torque_limit(self, raw):
        print(f"  torque_limit {WRIST_IDS} <- {raw}")

    def wrist_torque(self, on):
        print(f"  torque {'on' if on else 'off'} {WRIST_IDS}")

    def wrist_write(self, rad):
        self.wrist = list(rad)
        if self.verbose:
            print("    wrist goal " + " ".join(f"{i}:{math.degrees(r):.1f}" for i, r in zip(WRIST_IDS, rad)))

    def wrist_read(self):
        return list(self.wrist)


class Wrist:
    """Wrist commands in deg: flexion > 0 palm-ward, deviation > 0 radial (towards the thumb)."""

    def __init__(self, bus, log=print):
        self.bus, self.log, self.q, self.theta = bus, log, None, None

    def _servo(self, theta):
        return [(-1 if inv else 1) * (t + math.radians(z)) for t, z, inv in zip(theta, WRIST_ZERO, WRIST_INVERT)]

    def _write(self, phi, psi, theta):
        bad = check_wrist(phi, psi, theta)
        if bad:
            raise ah.LimitError("refused: " + "; ".join(bad))
        self.bus.wrist_write(self._servo(theta))
        self.q, self.theta = (phi, psi), np.array(theta, float)

    def present(self):
        """Servo angles (wrist_design convention) and the wrist pose they give."""
        th = np.array([(-1 if inv else 1) * s - math.radians(z)
                       for s, z, inv in zip(self.bus.wrist_read(), WRIST_ZERO, WRIST_INVERT)])
        return th, wd.servo_fk(th)

    def start(self):
        self.bus.wrist_torque_limit(WRIST_TORQUE_LIMIT)
        _, q = self.present()
        if not np.all(np.isfinite(q)):
            q = np.zeros(2)
        phi, psi = wd.clamp_wrist(*q)
        self._write(phi, psi, wd.servo_ik(phi, psi))
        self.bus.wrist_torque(True)
        self.move(0, 0, WristSpeed / 2, hold=0.5)

    def move(self, flex_deg, dev_deg, speed=WristSpeed, hold=0.0):
        """Straight streamed move to (flexion, deviation), clamped into the ROM; then hold with the outer loop, which
        re-reads the servos and nudges the goals so the P-controlled servos end where the IK wants them under load."""
        goal = np.array(wd.clamp_wrist(math.radians(flex_deg), math.radians(dev_deg)))
        q0 = np.array(self.q)
        n = max(1, int(math.ceil(np.abs(goal - q0).max() / speed * ah.STREAM_HZ)))
        t0 = time.monotonic()
        for k in range(1, n + 1):
            phi, psi = wd.clamp_wrist(*(q0 + (goal - q0) * k / n))
            self._write(phi, psi, wd.servo_ik(phi, psi))
            if self.bus.realtime:
                time.sleep(max(0.0, t0 + k / ah.STREAM_HZ - time.monotonic()))
        clamped = "" if np.allclose(np.degrees(goal), (flex_deg, dev_deg)) else f" (asked {flex_deg:g}, {dev_deg:g})"
        self.log(f"wrist ({math.degrees(goal[0]):+.1f}, {math.degrees(goal[1]):+.1f}) deg{clamped}: servo goals "
                 f"{np.degrees(self.theta).round(1).tolist()} deg, {n} frames")
        if not self.bus.realtime:
            return
        target, t_end = self.theta.copy(), time.monotonic() + hold
        while time.monotonic() < t_end:
            time.sleep(0.05)
            err = target - self.present()[0]
            theta = np.clip(self.theta + 0.1 * err, target - math.radians(SAG_MAX), target + math.radians(SAG_MAX))
            self._write(*self.q, np.clip(theta, -wd.SERVO_LIMIT, wd.SERVO_LIMIT))

    def stop(self):
        self.move(0, 0, WristSpeed / 2)
        self.bus.wrist_torque(False)


def check_direction(hand, wrist):
    """Move each wrist servo alone by +10 deg from the neutral pose and say what its rod should have done."""
    for k, name in enumerate(("motor1 (ID 21, ulnar / pinky side)", "motor2 (ID 22, radial / index side)")):
        th = wd.servo_ik(0.0, 0.0)
        th[k] += math.radians(10)
        wrist._write(*wd.servo_fk(th), th)
        hand.hold(1.0)
        expect = "UP" if wd.KAPPA[k + 1] > 0 else "DOWN"
        print(f"{name}: its push rod should have moved {expect} (towards the hand). If not, set WRIST_INVERT[{k}] = True.")
        wrist.move(0, 0, WristSpeed / 2, hold=1.0)


def demo(hand, wrist):
    ah.Pose(hand, "OpenHand")
    wrist.move(0, 0, hold=1.0)
    for _ in range(2):  # wave: radial / ulnar
        wrist.move(0, 20, hold=0.6)
        wrist.move(0, -20, hold=0.6)
    for _ in range(2):  # nod: palm-ward / extension
        wrist.move(45, 0, hold=0.8)
        wrist.move(-45, 0, hold=0.8)
    for a in np.linspace(0, 2 * np.pi, 25):  # circumduction, inside the coupled ROM
        wrist.move(40 * np.sin(a), 20 * np.cos(a), hold=0.15)
    wrist.move(30, 0, hold=0.5)
    ah.Pose(hand, "Peace")
    hand.hold(1.0)
    ah.Pose(hand, "OpenHand")
    wrist.move(0, 0, hold=1.0)


def self_test(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    fails = []

    def expect(ok, msg):
        if not ok:
            fails.append(msg)
            if len(fails) <= 20:
                print("  FAIL", msg)

    for phi, psi in np.radians(rng.uniform(-100, 100, (n, 2))):
        p, s = wd.clamp_wrist(phi, psi)
        expect(abs(p) <= wd.FLEX_MAX + 1e-12 and abs(s) <= wd.rom_dev_limit(p) + 1e-12, f"clamp_wrist -> ({p}, {s})")
        expect(not check_wrist(p, s, wd.servo_ik(p, s)), f"({p}, {s}): {check_wrist(p, s, wd.servo_ik(p, s))}")
        expect(bool(check_wrist(phi, psi, np.zeros(2))) or (p, s) == (phi, psi), "an out-of-ROM goal passed")
    expect(bool(check_wrist(float("nan"), 0.0, [0.0, 0.0])), "NaN accepted")
    print(f"clamp_wrist: {n} random commands in +-100 deg, every one lands in the ROM with an IK inside +-75 deg")

    class CheckBus(DryBus):
        def __init__(self):
            super().__init__(wrist=Wrist(None)._servo(wd.servo_ik(*np.radians([64.0, 28.0]))))  # resting past the ROM
            self.frames = 0

        def wrist_write(self, rad):
            th = np.array([(-1 if inv else 1) * r - math.radians(z) for r, z, inv in zip(rad, WRIST_ZERO, WRIST_INVERT)])
            q = wd.servo_fk(th)
            expect(not check_wrist(*q, th), f"bus got wrist servo goals {np.degrees(th)} -> {np.degrees(q)}")
            self.frames += 1
            super().wrist_write(rad)

        def wrist_torque_limit(self, raw):
            pass

        def wrist_torque(self, on):
            expect(self.frames > 0 or not on, "wrist torque on before a first goal")

        def write_speeds(self, ids, rad_s):
            pass

        def torque(self, ids, on):
            pass

        def write_torque_limit(self, i, raw):
            self.limit[i] = int(raw)

    bus = CheckBus()
    hand, wrist = ah.Hand(bus, log=lambda *a: None), Wrist(bus, log=lambda *a: None)
    hand.start(ah.TORQUE_LIMITS["pa-cf"])
    wrist.start()
    demo(hand, wrist)
    for flex, dev in rng.uniform(-90, 90, (200, 2)):
        wrist.move(flex, dev, speed=rng.uniform(0.2, 3))
    wrist.stop()
    hand.stop()
    print(f"wrist demo + 200 random moves: {bus.frames} wrist bus writes, every one the IK of a pose inside the ROM")
    print("SELF-TEST " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="bus adapter of the hand (and of the wrist without --wrist-port)")
    ap.add_argument("--wrist-port", help="a second bus adapter for the wrist")
    ap.add_argument("--dry-run", action="store_true", help="no hardware: print the commands")
    ap.add_argument("--self-test", action="store_true", help="sweep random wrist commands through the clamps and exit")
    ap.add_argument("--check-direction", action="store_true", help="move each wrist servo +10 deg alone, then exit")
    ap.add_argument("--cores", choices=sorted(ah.TORQUE_LIMITS), default="pa-cf", help="as AmazingHand_Demo_5F.py")
    ap.add_argument("-v", "--verbose", action="store_true", help="dry run: print every frame")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.dry_run and not args.port:
        ap.error("give --port, or --dry-run")
    if args.dry_run:
        bus = DryBus(verbose=args.verbose)
    elif args.wrist_port:
        bus = TwoPortBus(args.port, args.wrist_port)
    else:
        bus = MixedBus(args.port)
    hand, wrist = ah.Hand(bus), Wrist(bus)
    try:
        hand.start(ah.TORQUE_LIMITS[args.cores])
        wrist.start()
        check_direction(hand, wrist) if args.check_direction else demo(hand, wrist)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        for part in (wrist, hand):
            if (part.q if part is wrist else part.goal) is not None:
                try:
                    part.stop()
                except Exception as e:  # the bus may be what failed: at least try to let go
                    print(f"stop: {e}")
                    (bus.wrist_torque(False) if part is wrist else bus.torque(ah.ALL_IDS, False))


if __name__ == "__main__":
    main()
