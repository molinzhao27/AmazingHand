#!/usr/bin/env python3
"""AmazingHand demo for the final 5-digit right hand (AH_Right_5F_3S_Wrist, FIVE_FINGER.md): 3-phalanx index, middle,
ring and pinky (cad/five_finger/ultimate_finger/README.md), the slim-mount thumb, 10 SCS0009 servos. Same idea as
AmazingHand_Demo.py (rustypot, MiddlePos, one function per pose), plus the limits this hand needs before it is powered:

* IDs: index 1/2, middle 3/4, ring 5/6, thumb 7/8, pinky 9/10; motor1 is the odd ID (FIVE_FINGER.md, "Servo IDs").
* Thumb (7/8): every command is clamped to the slim mount's envelope (THUMB_ENVELOPE, thumb_command_ok and clamp_thumb,
  ported unchanged from cad/five_finger/layout.py). Outside it the thumb runs into the plate, its own unit or the palm
  shell; it keeps 22 % of the +-90 deg command square. Home is (0, 22.5): (0, 0) is inside the palm shell.
* 3-phalanx fingers, with f = (m1 - m2)/2 and a = (m1 + m2)/2 (> 0 towards the index), in degrees:
  straight at f = -56; f >= -60 (m1 - m2 >= -120: P1 meets the finger frame from f ~ -65); |a| <= 90 - |f| (both motors
  within +-90); |a| <= 30 towards a neighbour (the index and the pinky may spread outward up to 90 - |f|); neighbours
  converge by <= 15 (a of the pinky-side finger - a of the index-side finger), rising linearly to 30 as the more curled
  of the two goes from f = 60 to f = 90 (converge_limit; Demo/AHSimulation/ultimate_finger/reports/neighbour_sweep.txt).
* max_torque_limit (EEPROM addr 16) = 800 on IDs 1-6 and 9-10 (--cores: lower for PETG or PLA cores, ultimate_finger
  README section 5), read back before the torque goes on. The thumb keeps its own.
* Moves are streamed at 50 Hz along a straight line in command space, so the fingers' limits (a convex set) hold all the
  way; every frame is clamped, so the thumb follows the edge of its (non-convex) envelope instead of cutting a corner.
  Right before every bus write, check_hand() refuses any frame outside the limits.

Poses: Demo/AHSimulation/ultimate_finger/data/hand_poses.json (sim-checked on AH_Right_5F_3S), copied into POSES.
Right hand only. NOT TESTED ON HARDWARE.

    python AmazingHand_Demo_5F.py --dry-run                  # no hardware: prints the commands
    python AmazingHand_Demo_5F.py --self-test                # random commands through the clamps (run it in the repo)
    python AmazingHand_Demo_5F.py --port COM11               # the demo, looping (Linux: --port /dev/ttyACM0);
                                                             # first runs: add --speed-scale 0.3 --once
    python AmazingHand_Demo_5F.py --port COM11 --setup-only  # only write and check the torque limits (e.g. for AHControl)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

#Speed (rad/s, as in AmazingHand_Demo.py)
MaxSpeed = 7
CloseSpeed = 3
HomeSpeed = 1.5

#Fingers middle poses, by servo ID 1..10
MiddlePos = [3, 0, -5, -8, -2, 5, -12, 0, 0, 0] # replace values by your calibration results (pinky: ID_1 = 9, ID_2 = 10 in AmazingHand_Hand_FingerMiddlePos.py)

IDS = dict(index=(1, 2), middle=(3, 4), ring=(5, 6), thumb=(7, 8), pinky=(9, 10))  # (motor1, motor2)
FINGERS = ["index", "middle", "ring", "pinky"]  # index side -> pinky side
DIGITS = FINGERS + ["thumb"]
ALL_IDS = [i for d in DIGITS for i in IDS[d]]
FINGER_IDS = [i for d in FINGERS for i in IDS[d]]

# max_torque_limit (0..1000 = 0..100 % of stall) that keeps every finger part safe for contact up to f = +45
# (ultimate_finger README section 5): Link, L2, P1, P2 in PA-CF -> 800 (recommended); all PETG -> 460; all PLA -> 580.
TORQUE_LIMITS = {"pa-cf": 800, "petg": 460, "pla": 580}

# 3-phalanx finger limits (deg)
F_STRAIGHT = -56.0
F_MIN = -60.0
F_MAX = 90.0
M_MAX = 90.0
A_INNER = 30.0
CONVERGE_MAX = 15.0
CONVERGE_CURLED = 30.0                  # when the more curled finger of the pair is at f >= CURL_FULL
CURL_FROM, CURL_FULL = 60.0, 90.0       # the limit rises linearly from CONVERGE_MAX to CONVERGE_CURLED between these
TOL = 1e-6

STREAM_HZ = 50
THUMB_PATH_STEP = 0.25  # deg, sampling of the thumb's clamped path

# Thumb command envelope of the slim mount, right hand (m1, m2) in deg: m2_min(m1) <= m2 <= m2_max(m1), linear between
# the tabulated m1, none outside them. Ported from cad/five_finger/layout.py (THUMB_ENVELOPE["slim"]); --self-test checks
# that the two stay identical. Measured with one 7.5 deg grid step to spare on m2_max; m2_min and the lowest m1 keep the
# thumb >= 1.2 mm off the closed palm shell and grip pad.
THUMB_ENVELOPE = dict(m1=np.arange(-67.5, 90.1, 7.5),
                      m2_min=np.array([30, 30, 30, 30, 30, 30, 30, 30, 22.5, 22.5, 22.5, 22.5, 22.5, 30, 30, 37.5, 45, 52.5, 60,
                                       67.5, 75, 82.5], float),
                      m2_max=np.array([30, 37.5, 45, 52.5, 67.5, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                                       90], float))


def thumb_command_ok(m1, m2, side="right"):
    """Whether the thumb command (m1, m2) (deg) is inside the envelope (left hand: mirrored, (-m2, -m1))."""
    env = THUMB_ENVELOPE
    if side == "left":
        m1, m2 = -m2, -m1
    lo = np.interp(m1, env["m1"], env["m2_min"])
    hi = np.interp(m1, env["m1"], env["m2_max"])
    return bool(lo - 1e-9 <= m2 <= hi + 1e-9 and env["m1"][0] - 1e-9 <= m1 <= env["m1"][-1] + 1e-9)


def clamp_thumb(m1, m2, side="right"):
    """The nearest command inside the envelope, keeping m1 (deg)."""
    env = THUMB_ENVELOPE
    if side == "left":
        m1, m2 = -m2, -m1
    m1 = float(np.clip(m1, env["m1"][0], env["m1"][-1]))
    m2 = float(np.clip(m2, np.interp(m1, env["m1"], env["m2_min"]), np.interp(m1, env["m1"], env["m2_max"])))
    return (-m2, -m1) if side == "left" else (m1, m2)


THUMB_HOME = clamp_thumb(0.0, 0.0)  # (0, 22.5)
HOME = dict(index=(0.0, 0.0), middle=(0.0, 0.0), ring=(0.0, 0.0), pinky=(0.0, 0.0), thumb=THUMB_HOME)

# Right-hand (m1, m2) per digit, deg, from Demo/AHSimulation/ultimate_finger/data/hand_poses.json (--self-test compares)
POSES = {
    "OpenHand": dict(index=(-53.1, 53.1), middle=(-53.1, 53.1), ring=(-53.1, 53.1), pinky=(-53.1, 53.1), thumb=(-52.5, 45.0)),
    "CloseHand": dict(index=(90.0, -90.0), middle=(90.0, -90.0), ring=(90.0, -90.0), pinky=(90.0, -90.0), thumb=(-19.5, 45.0)),
    "SpreadHand": dict(index=(-30.0, 90.0), middle=(-49.7, 49.7), ring=(-85.0, 35.0), pinky=(-90.0, 30.0), thumb=(-36.0, 71.0)),
    "ClenchHand": dict(index=(-52.4, 42.4), middle=(-48.1, 58.1), ring=(-38.1, 68.1), pinky=(-45.1, 61.1), thumb=(-50.0, 37.0)),
    "Index_Pointing": dict(index=(-58.5, 58.5), middle=(90.0, -90.0), ring=(90.0, -90.0), pinky=(90.0, -90.0), thumb=(-1.0, 23.5)),
    "Perfect": dict(index=(35.0, -55.0), middle=(-13.0, 13.0), ring=(-35.9, 35.9), pinky=(-35.9, 35.9), thumb=(-6.0, 67.5)),
    "Victory": dict(index=(-33.5, 83.5), middle=(-83.5, 33.5), ring=(90.0, -90.0), pinky=(90.0, -90.0), thumb=(48.0, 41.0)),
    "Pinched": dict(index=(0.0, 0.0), middle=(90.0, -90.0), ring=(90.0, -90.0), pinky=(90.0, -90.0), thumb=(-7.5, 42.0)),
    "Fuck": dict(index=(90.0, -90.0), middle=(-56.0, 56.0), ring=(90.0, -90.0), pinky=(90.0, -90.0), thumb=(-19.5, 45.0)),
    "Peace": dict(index=(-26.0, 86.0), middle=(-86.0, 26.0), ring=(90.0, -90.0), pinky=(90.0, -90.0), thumb=(48.0, 41.0)),
    "MiddleFinger": dict(index=(90.0, -90.0), middle=(-56.0, 56.0), ring=(90.0, -90.0), pinky=(90.0, -90.0), thumb=(-19.5, 45.0)),
}


class LimitError(RuntimeError):
    pass


def finger_fa(m1, m2):
    return (m1 - m2) / 2.0, (m1 + m2) / 2.0


def finger_m(f, a):
    return a + f, a - f


def abduction_range(finger, f):
    """(lo, hi) of a finger's abduction at flexion f (deg): within 90 - |f|, and within A_INNER towards a neighbour."""
    room = max(0.0, M_MAX - abs(f))
    lo, hi = -min(room, A_INNER), min(room, A_INNER)
    if finger == "index":
        hi = room  # spreading outward, towards the thumb
    if finger == "pinky":
        lo = -room
    return lo, hi


def converge_limit(f_left, f_right):
    """How far a neighbour pair may converge (a_right - a_left, deg): CONVERGE_MAX, rising linearly to CONVERGE_CURLED
    as the more curled of the two goes from f = CURL_FROM to CURL_FULL. Swept on AH_Right_5F_3S over both fingers'
    whole range (Demo/AHSimulation/ultimate_finger/reports/neighbour_sweep.txt): the pose pairs this allows beyond 15
    keep >= 3 mm, more than inside the 15 deg limit (1.3 mm); with both fingers at f <= 0, 25-40 deg of convergence
    makes them touch."""
    s = min(max((max(f_left, f_right) - CURL_FROM) / (CURL_FULL - CURL_FROM), 0.0), 1.0)
    return CONVERGE_MAX + (CONVERGE_CURLED - CONVERGE_MAX) * s


def clamp_hand(cmd):
    """A whole-hand command {digit: (m1, m2)} (deg) moved inside every limit. A command already inside comes back
    unchanged. Fingers: f clipped to [F_MIN, F_MAX], then a to abduction_range(); a neighbour pair converging by more
    than converge_limit() is pulled apart by turning back whichever of the two is swung towards the other, in proportion
    to how far. That only ever moves an abduction towards 0 and never changes f (so a pair's limit stays put), so it
    never breaks the per-finger limits, and a fix can only push a violation further out along the row (a finger's a
    keeps its sign), so a few passes settle it."""
    for d in DIGITS:
        if not all(math.isfinite(float(v)) for v in cmd[d]):
            raise ValueError(f"{d}: non-finite command {cmd[d]}")
    eps = 1e-9  # what is inside by this much is left alone, so (m1, m2) -> (f, a) -> (m1, m2) rounding never moves it
    fa, moved = {}, {}
    for k in FINGERS:
        f, a = finger_fa(float(cmd[k][0]), float(cmd[k][1]))
        f2 = min(max(f, F_MIN), F_MAX)
        lo, hi = abduction_range(k, f2)
        a2 = min(max(a, lo), hi)
        fa[k], moved[k] = [f2, a2], abs(f2 - f) > eps or abs(a2 - a) > eps
    for _ in range(2 * len(FINGERS)):
        changed = False
        for left, right in zip(FINGERS, FINGERS[1:]):
            a0, a1 = fa[left][1], fa[right][1]
            lim = converge_limit(fa[left][0], fa[right][0])
            if a1 - a0 <= lim + eps:
                continue
            p, n = max(0.0, a1), max(0.0, -a0)       # how far each is swung towards the other
            rest = min(0.0, a1) - max(0.0, a0)        # <= 0
            k = (lim - rest) / (p + n)                # 0 <= k < 1
            if a0 < 0:
                fa[left][1], moved[left] = -n * k, True
            if a1 > 0:
                fa[right][1], moved[right] = p * k, True
            changed = True
        if not changed:
            break
    out = {}
    for k in FINGERS:
        if moved[k]:
            m1, m2 = finger_m(*fa[k])
            out[k] = (min(max(m1, -M_MAX), M_MAX), min(max(m2, -M_MAX), M_MAX))
        else:
            out[k] = (float(cmd[k][0]), float(cmd[k][1]))
    out["thumb"] = clamp_thumb(float(cmd["thumb"][0]), float(cmd["thumb"][1]))
    return out


def check_hand(cmd):
    """Every limit of this hand, checked on a whole-hand command {digit: (m1, m2)} (deg). Returns the violations."""
    bad = []
    for d in DIGITS:
        m1, m2 = cmd[d]
        if not (math.isfinite(m1) and math.isfinite(m2)):
            bad.append(f"{d}: non-finite ({m1}, {m2})")
            return bad
        if abs(m1) > M_MAX + TOL or abs(m2) > M_MAX + TOL:
            bad.append(f"{d}: ({m1:.2f}, {m2:.2f}) beyond +-{M_MAX:g}")
    f, a = {}, {}
    for k in FINGERS:
        f[k], a[k] = finger_fa(*cmd[k])
        lo, hi = abduction_range(k, f[k])
        if f[k] < F_MIN - TOL or f[k] > F_MAX + TOL:
            bad.append(f"{k}: flexion f = {f[k]:.2f} outside [{F_MIN:g}, {F_MAX:g}]")
        if abs(a[k]) > M_MAX - abs(f[k]) + TOL or not lo - TOL <= a[k] <= hi + TOL:
            bad.append(f"{k}: abduction a = {a[k]:.2f} outside [{lo:.2f}, {hi:.2f}] at f = {f[k]:.2f}")
    for left, right in zip(FINGERS, FINGERS[1:]):
        lim = converge_limit(f[left], f[right])
        if a[right] - a[left] > lim + TOL:
            bad.append(f"{left}/{right}: converging by {a[right] - a[left]:.2f} > {lim:.2f}")
    if not thumb_command_ok(*cmd["thumb"]):
        bad.append(f"thumb: ({cmd['thumb'][0]:.2f}, {cmd['thumb'][1]:.2f}) outside the slim-mount envelope")
    return bad


def thumb_path(t0, t1):
    """The thumb's way from t0 to t1: the straight line, clamped into the envelope point by point, as a polyline
    (xs, ys, cumulative max-norm length); max-norm, because each servo moves at its own speed."""
    d = max(abs(t1[0] - t0[0]), abs(t1[1] - t0[1]))
    n = max(1, int(math.ceil(d / THUMB_PATH_STEP)))
    pts = [tuple(t0)]
    for k in range(n + 1):
        p = clamp_thumb(t0[0] + (t1[0] - t0[0]) * k / n, t0[1] + (t1[1] - t0[1]) * k / n)
        if max(abs(p[0] - pts[-1][0]), abs(p[1] - pts[-1][1])) > 1e-9:
            pts.append(p)
    xs, ys = np.array(pts).T
    seg = np.maximum(np.abs(np.diff(xs)), np.abs(np.diff(ys)))
    return xs, ys, np.concatenate([[0.0], np.cumsum(seg)])


def trajectory(start, goal, speed):
    """Frames (whole-hand commands) from start to goal at speed (rad/s), STREAM_HZ apart, every one clamped. Where the
    clamps move a finger faster than the straight line does (a curled neighbour uncurling shrinks converge_limit under
    it), the move gets more frames, so no motor steps more than speed / STREAM_HZ between frames."""
    v = math.degrees(speed)
    xs, ys, cum = thumb_path(start["thumb"], goal["thumb"])
    d_fingers = max(abs(goal[k][j] - start[k][j]) for k in FINGERS for j in (0, 1))
    n = n0 = max(1, int(math.ceil(max(d_fingers, cum[-1]) / v * STREAM_HZ)))
    ref = clamp_hand(start)  # start may be outside the limits (where the hand is at power-on): its first frame jumps
    while True:
        frames = []
        for i in range(1, n + 1):
            s = i / n
            cmd = {k: tuple(start[k][j] + (goal[k][j] - start[k][j]) * s for j in (0, 1)) for k in FINGERS}
            cmd["thumb"] = (float(np.interp(s * cum[-1], cum, xs)), float(np.interp(s * cum[-1], cum, ys)))
            frames.append(clamp_hand(cmd))
        frames[-1] = goal
        step = max(max(abs(b[d][j] - a[d][j]) for d in DIGITS for j in (0, 1)) for a, b in zip([ref] + frames, frames))
        if step <= v / STREAM_HZ * (1 + 1e-9) or n >= 4 * n0:
            return frames
        n = min(4 * n0, max(n + 1, int(math.ceil(n * step / (v / STREAM_HZ)))))


def fmt(cmd, middle_pos=MiddlePos):
    lines = []
    for d in DIGITS:
        m1, m2 = cmd[d]
        i1, i2 = IDS[d]
        fa = "f {:+6.1f}  a {:+6.1f}".format(*finger_fa(m1, m2)) if d in FINGERS else " " * 18
        lines.append(f"    {d:6s} ({m1:6.1f}, {m2:6.1f})  {fa}   servo goal ID{i1:<2d} {middle_pos[i1 - 1] + m1:6.1f}"
                     f"  ID{i2:<2d} {middle_pos[i2 - 1] + m2:6.1f} deg")
    return "\n".join(lines)


def _scalar(x):
    return float(np.atleast_1d(np.asarray(x, float))[0])


class ScsBus:
    """The hand's 10 SCS0009 on one rustypot controller (radians, as Scs0009PyController)."""
    realtime = True

    def __init__(self, port, baudrate=1_000_000):
        from rustypot import Scs0009PyController
        self.c = Scs0009PyController(serial_port=port, baudrate=baudrate, timeout=0.5)

    def write_positions(self, ids, rad):
        self.c.sync_write_goal_position(list(ids), [float(r) for r in rad])

    def write_speeds(self, ids, rad_s):
        self.c.sync_write_goal_speed(list(ids), [float(v) for v in rad_s])

    def torque(self, ids, on):
        self.c.sync_write_torque_enable(list(ids), [1 if on else 0] * len(ids))

    def read_position(self, i):
        return _scalar(self.c.read_present_position(i))

    def read_torque_limit(self, i):
        return int(_scalar(self.c.read_raw_max_torque_limit(i)))

    def write_torque_limit(self, i, raw):
        self.c.write_lock(i, False)  # open the EEPROM lock so the value is kept at power-off
        self.c.write_raw_max_torque_limit(i, int(raw))
        self.c.write_lock(i, True)


class DryBus:
    """No hardware: prints what would go on the bus (every frame with --verbose) and keeps the register values."""
    realtime = False

    def __init__(self, middle_pos=MiddlePos, verbose=False, start=HOME):
        self.verbose = verbose
        self.pos = {i: math.radians(middle_pos[i - 1] + start[d][j]) for d in DIGITS for j, i in enumerate(IDS[d])}
        self.limit = {i: 1000 for i in ALL_IDS}
        self.writes = 0

    def write_positions(self, ids, rad):
        self.writes += 1
        self.pos.update(zip(ids, rad))
        if self.verbose:
            print("    goal " + " ".join(f"{i}:{math.degrees(r):.1f}" for i, r in zip(ids, rad)))

    def write_speeds(self, ids, rad_s):
        print(f"  goal_speed {list(ids)} <- {[round(v, 2) for v in rad_s]} rad/s")

    def torque(self, ids, on):
        print(f"  torque {'on' if on else 'off'} {list(ids)}")

    def read_position(self, i):
        return self.pos[i]

    def read_torque_limit(self, i):
        return self.limit[i]

    def write_torque_limit(self, i, raw):
        print(f"  max_torque_limit ID {i} <- {raw}")
        self.limit[i] = int(raw)


class Hand:
    """The 5-digit hand, in right-hand motor commands (m1, m2) (deg, as Move_Index & co. in AmazingHand_Demo.py). Every
    frame it writes has passed check_hand()."""

    def __init__(self, bus, middle_pos=MiddlePos, speed_scale=1.0, log=print):
        self.bus, self.middle, self.scale, self.log = bus, list(middle_pos), speed_scale, log
        self.goal = None

    def set_torque_limits(self, raw):
        for i in FINGER_IDS:
            if self.bus.read_torque_limit(i) != raw:
                self.bus.write_torque_limit(i, raw)
            got = self.bus.read_torque_limit(i)
            if got != raw:
                raise RuntimeError(f"ID {i}: max_torque_limit reads {got} after writing {raw}")
        self.log(f"max_torque_limit = {raw} on IDs {FINGER_IDS} (checked)")

    def present(self):
        cmd = {}
        for d in DIGITS:
            try:
                rad = [self.bus.read_position(i) for i in IDS[d]]
            except Exception as e:
                raise RuntimeError(f"no answer from {d} (IDs {IDS[d]}): check its IDs (FIVE_FINGER.md, Servo IDs) and "
                                   f"cabling ({e})")
            cmd[d] = tuple(math.degrees(r) - self.middle[i - 1] for r, i in zip(rad, IDS[d]))
        return cmd

    def start(self, torque_limit):
        """Torque limits, then torque on at the first frame of a slow move from where the hand is to HOME."""
        self.set_torque_limits(torque_limit)
        here = self.present()
        self.bus.write_speeds(ALL_IDS, [HomeSpeed] * len(ALL_IDS))  # the servos' own cap while homing
        frames = trajectory(here, clamp_hand(HOME), HomeSpeed)
        self.log(f"start: from where the hand is to home, {len(frames) / STREAM_HZ:.2f} s, {len(frames)} frames")
        self._write(frames[0])
        self.bus.torque(ALL_IDS, True)
        self._play(frames[1:])
        self.bus.write_speeds(ALL_IDS, [MaxSpeed] * len(ALL_IDS))

    def _write(self, cmd):
        bad = check_hand(cmd)
        if bad:
            raise LimitError("refused: " + "; ".join(bad))
        rad = [math.radians(self.middle[i - 1] + m) for d in DIGITS for i, m in zip(IDS[d], cmd[d])]
        self.bus.write_positions(ALL_IDS, rad)
        self.goal = cmd

    def _play(self, frames):
        t0 = time.monotonic()
        for k, cmd in enumerate(frames):
            self._write(cmd)
            if self.bus.realtime:
                time.sleep(max(0.0, t0 + (k + 1) / STREAM_HZ - time.monotonic()))

    def move(self, name, speed=MaxSpeed, **targets):
        """Move the given digits (the others hold) to (m1, m2), clamped, in one straight streamed move."""
        unknown = set(targets) - set(DIGITS)
        if unknown:
            raise ValueError(f"unknown digits {sorted(unknown)}; use {DIGITS}")
        want = dict(self.goal)
        want.update({d: tuple(map(float, v)) for d, v in targets.items()})
        goal = clamp_hand(want)
        frames = trajectory(self.goal, goal, speed * self.scale)
        self.log(f"{name}: {len(frames) / STREAM_HZ:.2f} s, {len(frames)} frames")
        for d in DIGITS:
            if max(abs(goal[d][0] - want[d][0]), abs(goal[d][1] - want[d][1])) > 0.05:
                self.log(f"    clamped {d}: ({want[d][0]:.1f}, {want[d][1]:.1f}) -> ({goal[d][0]:.1f}, {goal[d][1]:.1f})")
        self._play(frames)
        if not self.bus.realtime:
            self.log(fmt(goal, self.middle))

    def hold(self, seconds):
        if self.bus.realtime:
            time.sleep(seconds)

    def stop(self):
        self.move("home", HomeSpeed, **HOME)
        self.bus.torque(ALL_IDS, False)


def Pose(hand, name, speed=MaxSpeed):
    hand.move(name, speed, **POSES[name])


def OpenHand_Progressive(hand):
    for d in ["index", "middle", "ring", "pinky", "thumb"]:
        hand.move(f"OpenHand_Progressive {d}", MaxSpeed - 2, **{d: POSES["OpenHand"][d]})


def Nonono(hand):
    Pose(hand, "Index_Pointing")
    f = finger_fa(*POSES["Index_Pointing"]["index"])[0]
    for i in range(3):
        hand.hold(0.2)
        hand.move("Nonono", MaxSpeed, index=finger_m(f, 15))
        hand.hold(0.2)
        hand.move("Nonono", MaxSpeed, index=finger_m(f, -15))
    Pose(hand, "Index_Pointing")
    hand.hold(0.4)


def Scissors(hand):
    Pose(hand, "Victory")
    f = finger_fa(*POSES["Victory"]["index"])[0]
    for i in range(3):
        hand.hold(0.2)
        hand.move("Scissors", MaxSpeed, index=finger_m(f, 0), middle=finger_m(f, 0))
        hand.hold(0.2)
        hand.move("Scissors", MaxSpeed, index=POSES["Victory"]["index"], middle=POSES["Victory"]["middle"])


def demo(hand):
    Pose(hand, "OpenHand")
    hand.hold(0.5)
    Pose(hand, "CloseHand", CloseSpeed)
    hand.hold(3)
    OpenHand_Progressive(hand)
    hand.hold(0.5)
    Pose(hand, "SpreadHand")
    hand.hold(0.6)
    Pose(hand, "ClenchHand")
    hand.hold(0.6)
    Pose(hand, "OpenHand")
    hand.hold(0.2)
    Nonono(hand)
    hand.hold(0.5)
    Pose(hand, "OpenHand")
    hand.hold(0.3)
    Pose(hand, "Perfect")
    hand.hold(0.8)
    Pose(hand, "OpenHand")
    hand.hold(0.4)
    Scissors(hand)
    hand.hold(0.5)
    Pose(hand, "OpenHand")
    hand.hold(0.4)
    Pose(hand, "Pinched")
    hand.hold(1)
    Pose(hand, "Fuck")
    hand.hold(0.8)
    Pose(hand, "OpenHand")
    hand.hold(0.4)
    Pose(hand, "Peace")
    hand.hold(1)


# ----------------------------------------------------------------------------- self-test
def self_test(n=20000, seed=0):
    """Random commands through the clamps; the thumb is checked against cad/five_finger/layout.py itself."""
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "cad" / "five_finger"))
    import layout  # noqa: E402
    rng = np.random.default_rng(seed)
    fails = []

    def expect(ok, msg):
        if not ok:
            fails.append(msg)
            if len(fails) <= 20:
                print("  FAIL", msg)

    # 1. the port is the original
    env = layout.THUMB_ENVELOPE["slim"]
    expect(layout.THUMB_MOUNT == "slim", f"layout.THUMB_MOUNT is {layout.THUMB_MOUNT!r}, the demo assumes 'slim'")
    for key in ("m1", "m2_min", "m2_max"):
        expect(np.array_equal(env[key], THUMB_ENVELOPE[key]), f"THUMB_ENVELOPE[{key}] differs from layout.py")
    expect(THUMB_HOME == layout.clamp_thumb(0, 0) == (0.0, 22.5), f"thumb home {THUMB_HOME}")
    for side in ("right", "left"):
        for m1, m2 in rng.uniform(-150, 150, (n, 2)):
            expect(thumb_command_ok(m1, m2, side) == layout.thumb_command_ok(m1, m2, side=side),
                   f"thumb_command_ok({m1:.3f}, {m2:.3f}, {side}) differs from layout")
            c = clamp_thumb(m1, m2, side)
            expect(c == layout.clamp_thumb(m1, m2, side=side) and layout.thumb_command_ok(*c, side=side),
                   f"clamp_thumb({m1:.3f}, {m2:.3f}, {side}) = {c}")
    print(f"thumb port: envelope identical, {2 * n} random commands agree with layout.py, every clamp inside")

    # 2. random whole-hand commands: wild, in range, and near the limits
    def rand_cmd(kind):
        if kind == 0:
            v = rng.uniform(-150, 150, (5, 2))
        elif kind == 1:
            v = rng.uniform(-90, 90, (5, 2))
        else:  # f, a near their limits, the thumb near its envelope's edge
            v = np.zeros((5, 2))
            for j in range(4):
                f = rng.choice([F_MIN, F_STRAIGHT, 0.0, CURL_FROM, (CURL_FROM + CURL_FULL) / 2, F_MAX,
                                rng.uniform(-70, 95)]) + rng.normal(0, 2)
                a = rng.choice([-1, 1]) * rng.choice([A_INNER, M_MAX - abs(f), rng.uniform(0, 40)]) + rng.normal(0, 2)
                v[j] = finger_m(f, a)
            m1 = rng.uniform(-75, 95)
            v[4] = (m1, rng.choice([np.interp(m1, env["m1"], env["m2_min"]), np.interp(m1, env["m1"], env["m2_max"])])
                    + rng.normal(0, 2))
        return {d: (float(v[j, 0]), float(v[j, 1])) for j, d in enumerate(DIGITS)}

    for fl, fr, want in ((F_STRAIGHT, F_STRAIGHT, CONVERGE_MAX), (CURL_FROM, F_STRAIGHT, CONVERGE_MAX),
                         (F_STRAIGHT, (CURL_FROM + CURL_FULL) / 2, (CONVERGE_MAX + CONVERGE_CURLED) / 2),
                         (F_MAX, F_MIN, CONVERGE_CURLED), (F_MAX, F_MAX, CONVERGE_CURLED)):
        expect(abs(converge_limit(fl, fr) - want) < 1e-12 and converge_limit(fr, fl) == converge_limit(fl, fr),
               f"converge_limit({fl}, {fr}) = {converge_limit(fl, fr)}, want {want}")
    changed = 0
    for i in range(n):
        cmd = rand_cmd(i % 3)
        out = clamp_hand(cmd)
        bad = check_hand(out)
        expect(not bad, f"clamp_hand({cmd}) -> {out}: {bad}")
        expect(layout.thumb_command_ok(*out["thumb"]), f"thumb {out['thumb']} rejected by layout.thumb_command_ok")
        expect(clamp_hand(out) == out, f"clamp_hand not idempotent at {out}")
        if not check_hand(cmd):
            expect(out == cmd, f"a command inside the limits was changed: {cmd} -> {out}")
        changed += out != cmd
    inside = 0
    while inside < n // 4:  # commands inside every limit pass through unchanged
        cmd = {d: (float(x), float(y)) for d, (x, y) in zip(DIGITS, rng.uniform(-90, 90, (5, 2)))}
        cmd["thumb"] = (float(rng.uniform(-67.5, 90)), float(rng.uniform(22.5, 90)))
        if check_hand(cmd):
            continue
        inside += 1
        expect(clamp_hand(cmd) == cmd, f"a command inside the limits was changed: {cmd}")
    for bad_value in (float("nan"), float("inf"), -float("inf")):
        cmd = dict(HOME)
        cmd["ring"] = (bad_value, 0.0)
        try:
            clamp_hand(cmd)
            expect(False, f"clamp_hand accepted {bad_value}")
        except ValueError:
            pass
        expect(bool(check_hand(cmd)), f"check_hand accepted {bad_value}")
    print(f"clamp_hand: {n} random commands ({changed} needed clamping) all land inside every limit and on a fixed "
          f"point; {inside} commands inside the limits pass unchanged; NaN/inf refused")

    # 3. streamed moves between random commands: every frame inside, no frame jumps
    worst = 0.0
    for i in range(400):
        a, b = clamp_hand(rand_cmd(i % 3)), clamp_hand(rand_cmd((i + 1) % 3))
        speed = rng.uniform(0.5, MaxSpeed)
        frames = trajectory(a, b, speed)
        prev = a
        for fr in frames:
            expect(not check_hand(fr) and layout.thumb_command_ok(*fr["thumb"]), f"frame outside the limits: {fr}")
            step = max(abs(fr[d][j] - prev[d][j]) for d in DIGITS for j in (0, 1))
            worst = max(worst, step / (math.degrees(speed) / STREAM_HZ))
            prev = fr
        expect(frames[-1] == b, "a move does not end at its goal")
    expect(worst <= 1.05, f"a frame steps {worst:.2f} x speed / rate")
    print(f"trajectory: 400 random moves, every frame inside the limits; largest step {worst:.3f} x speed / {STREAM_HZ} Hz")

    # 4. the poses are hand_poses.json's, and every one is inside the limits, so the clamps leave it as it is (Victory
    # and Peace converge the straight middle onto the fully curled ring by 25 / 30 deg: allowed by converge_limit)
    data = json.loads((repo / "Demo/AHSimulation/ultimate_finger/data/hand_poses.json").read_text())
    expect(set(data) == set(POSES), f"poses differ: {sorted(set(data) ^ set(POSES))}")
    for name, pose in data.items():
        expect({d: tuple(v) for d, v in pose.items()} == POSES.get(name), f"pose {name} differs from hand_poses.json")
        out = clamp_hand(POSES[name])
        for d in DIGITS:
            if out[d] != POSES[name][d]:
                print(f"  note: {name} {d} {POSES[name][d]} -> ({out[d][0]:.1f}, {out[d][1]:.1f}) by the clamps "
                      f"({'; '.join(check_hand(POSES[name]))})")
        expect(out == POSES[name], f"pose {name} is changed by the clamps")
    print(f"poses: all {len(POSES)} of hand_poses.json are inside the limits and pass the clamps unchanged")

    # 5. the whole demo as it would run, on a bus that checks every write itself
    class CheckBus(DryBus):
        def __init__(self):
            super().__init__(start=dict(index=(-20, 70), middle=(0, 0), ring=(0, 0), pinky=(0, 0), thumb=(0, 0)))
            self.torque_on, self.frames, self.limit_writes = False, 0, []

        def write_positions(self, ids, rad):
            expect(list(ids) == ALL_IDS, f"IDs {ids}")
            cmd = {d: tuple(math.degrees(rad[ALL_IDS.index(i)]) - MiddlePos[i - 1] for i in IDS[d]) for d in DIGITS}
            cmd = {d: tuple(round(x, 9) for x in v) for d, v in cmd.items()}
            expect(not check_hand(cmd) and layout.thumb_command_ok(*cmd["thumb"]), f"bus got {cmd}: {check_hand(cmd)}")
            self.frames += 1
            super().write_positions(ids, rad)

        def write_speeds(self, ids, rad_s):
            pass

        def torque(self, ids, on):
            expect(self.frames > 0 or not on, "torque on before a first goal")
            self.torque_on = on

        def write_torque_limit(self, i, raw):
            self.limit_writes.append(i)
            self.limit[i] = int(raw)

    bus = CheckBus()
    hand = Hand(bus, log=lambda *a: None)
    hand.start(TORQUE_LIMITS["pa-cf"])
    expect(sorted(bus.limit_writes) == [1, 2, 3, 4, 5, 6, 9, 10], f"torque limit written to {bus.limit_writes}")
    demo(hand)
    hand.stop()
    expect(not bus.torque_on, "torque left on")
    print(f"demo run: {bus.frames} bus writes, every one inside the limits; torque limit on IDs {sorted(bus.limit_writes)}")

    print("SELF-TEST " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port of the bus adapter (COM11, /dev/ttyACM0, ...)")
    ap.add_argument("--dry-run", action="store_true", help="no hardware: print the commands")
    ap.add_argument("--self-test", action="store_true", help="sweep random commands through the clamps and exit")
    ap.add_argument("--setup-only", action="store_true", help="write and check the torque limits, then exit")
    ap.add_argument("--cores", choices=sorted(TORQUE_LIMITS), default="pa-cf",
                    help="what Link, L2, P1 and P2 are printed in; sets max_torque_limit (pa-cf 800, petg 460, pla 580)")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="scales every speed (e.g. 0.5 for the first runs)")
    ap.add_argument("--once", action="store_true", help="run the demo once instead of looping")
    ap.add_argument("-v", "--verbose", action="store_true", help="dry run: print every frame")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.dry_run and not args.port:
        ap.error("give --port, or --dry-run")
    if not 0 < args.speed_scale <= 1.5:
        ap.error("--speed-scale must be in (0, 1.5]")
    bus = DryBus(verbose=args.verbose) if args.dry_run else ScsBus(args.port)
    hand = Hand(bus, speed_scale=args.speed_scale)
    limit = TORQUE_LIMITS[args.cores]
    print(f"cores: {args.cores} -> max_torque_limit {limit} (FIVE_FINGER.md; ultimate_finger README section 5)")
    if args.setup_only:
        hand.set_torque_limits(limit)
        return
    try:
        hand.start(limit)
        while True:
            demo(hand)
            if args.dry_run or args.once:
                break
    except KeyboardInterrupt:
        print("stopping")
    finally:
        if hand.goal is not None:
            try:
                hand.stop()
            except Exception as e:  # the bus may be what failed: at least try to let go
                print(f"stop: {e}")
                bus.torque(ALL_IDS, False)


if __name__ == '__main__':
    main()
