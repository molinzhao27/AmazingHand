"""Compute how far the final hand's thumb may close before it touches a finger (THUMB_CLOSE_MAX in teleop.py).

The sim does not collide the digits with each other, so teleop.py keeps the thumb out of the fingers itself: for
each index curl (0..1 in eighths) and middle curl (in quarters), with the ring and pinky curled and the index and
middle spread towards the thumb as far as teleop.retarget lets them at that curl, the thumb walks from open
towards its rest pose (teleop.thumb_rest) in steps of 0.05 until it comes within --margin of a finger.
Distances are measured on the hand-only model AH_Right_5F_3S with the surface sampling of
Demo/AHSimulation/five_finger/hand_poses.py.

--check then settles random teleop commands (curls, abductions) and reports the closest pairs of digits, with the
penetration depth of any that touch.

Needs more than requirements.txt:  pip install scipy trimesh rtree
Usage:  python thumb_limits.py [--margin 0.5] [--check 200]
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "AHSimulation" / "five_finger"))
import teleop as T  # noqa: E402

STEP = 0.05
_pm = None


def pose_model():
    global _pm
    if _pm is None:
        from hand_poses import PoseModel
        _pm = PoseModel("AH_Right_5F_3S")
    return _pm


def finger_flex(s):
    return T.FINAL_FLEX_OPEN + s * (T.FINAL_FLEX_CLOSED - T.FINAL_FLEX_OPEN)


def close_max(args):
    si, sm, margin = args
    pm = pose_model()
    curls = {"index": si, "middle": sm, "ring": 1.0, "pinky": 1.0}
    # index and middle spread towards the thumb as far as their curl allows
    robot = T.ROBOTS["final"]
    reach = {d: robot.abd_max * max(0.0, 1 - curls[d] / robot.abd_fade_end) for d in ("index", "middle")}
    cmd = {d: T.to_motors(finger_flex(curls[d]), reach.get(d, 0.0)) for d in T.FINGERS}
    rest = T.thumb_rest(curls)
    best = 0.0
    for s in np.arange(STEP, 1 + 1e-9, STEP):
        cmd["thumb"] = T.clamp_thumb(*T.lerp(T.THUMB_OPEN, rest, s))
        pm.settle(cmd, steps=600)
        if min(pm.clearance("thumb", d)[0] for d in T.FINGERS) < margin:
            break
        best = float(s)
    return best


def check(args):
    """Settle one random teleop command; returns (closest pairs, commands)."""
    from hand_poses import check_pose
    seed, T.THUMB_CLOSE_MAX = args
    digits = T.ROBOTS["final"].digits
    rng = np.random.default_rng(seed)
    curls = {d: float(rng.choice([0.0, 1.0, rng.uniform()])) for d in digits}
    # human abduction relative to the calibrated neutral, as retarget() sees it (deg, + towards the thumb)
    angles, calib = {}, T.Calibration()
    for d in digits:
        lo, hi = calib.curl_open[d], calib.curl_fist[d]
        spread = rng.normal(0, 8) + {"index": 6, "middle": 0, "ring": -6, "pinky": -10, "thumb": 0}[d]
        angles[d] = (lo + curls[d] * (hi - lo), calib.abd_neutral[d] + spread)
    motors, _ = T.retarget(angles, calib, T.ROBOTS["final"])
    names = {v: k for k, v in T.ROBOT_FINGER.items()}
    cmd = {names[k]: v for k, v in motors.items()}
    pairs = check_pose(pose_model(), cmd)
    return [(a, b, dist, depth) for a, b, dist, depth, _ in pairs if dist < 1e-3], cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=0.5, help="clearance to keep from the fingers (mm)")
    ap.add_argument("--check", type=int, default=0, metavar="N", help="also check N random teleop commands")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    shape = T.THUMB_CLOSE_MAX.shape
    grid = [(si, sm, args.margin * 1e-3) for si in np.linspace(0, 1, shape[0]) for sm in np.linspace(0, 1, shape[1])]
    with Pool(args.jobs) as pool:
        table = np.array(pool.map(close_max, grid)).reshape(shape)
        print("THUMB_CLOSE_MAX = np.array([")
        for row in table:
            print("    [" + ", ".join(f"{v:.2f}" for v in row) + "],")
        print("])")
        if args.check:
            touching = 0
            for pairs, cmd in pool.map(check, [(seed, table) for seed in range(args.check)]):
                for a, b, dist, depth in pairs:
                    touching += dist < 0.225e-3
                    print(f"{a}/{b} {dist * 1e3:.2f} mm, penetration {depth * 1e3:.2f} mm: "
                          + ", ".join(f"{d} ({m1:+.0f}, {m2:+.0f})" for d, (m1, m2) in cmd.items()))
            print(f"{args.check} random commands: {touching} touching pairs")


if __name__ == "__main__":
    main()
