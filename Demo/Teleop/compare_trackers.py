"""Run several hand trackers on the same recorded clip and compare them.

Record a clip of hard motions first (fist, wide spread, pinches, back of hand to camera):
    python teleop.py --record hard.mp4
Then:
    python compare_trackers.py hard.mp4 --trackers mediapipe wilor --out compare.mp4

Writes a side-by-side video (one panel per tracker, with keypoints and per-digit curl bars) and
prints, per tracker:
  detected   share of frames with a hand, split by palm-facing and back-facing frames
  curl p5-p95  range of curl each digit actually reaches, in degrees (wider = less compressed)
  jitter     median frame-to-frame curl change in degrees while tracking (lower = steadier)
  ms/frame   average processing time
"""

import argparse
import time

import cv2
import numpy as np

from teleop import hand_angles
from trackers import CONNECTIONS, make_tracker, palm_facing_camera

DIGITS = ["thumb", "index", "middle", "ring", "pinky"]
COLORS = [(80, 80, 255), (80, 200, 255), (80, 255, 120), (255, 200, 80), (255, 80, 200)]


def draw(panel, hand, name, curls):
    for a, b in CONNECTIONS:
        cv2.line(panel, tuple(hand.xy[a].astype(int)), tuple(hand.xy[b].astype(int)), (255, 255, 255), 2)
    for k, pt in enumerate(hand.xy):
        cv2.circle(panel, tuple(pt.astype(int)), 4, COLORS[(k - 1) // 4 if k else 0], -1)
    for i, d in enumerate(DIGITS):
        v = min(curls[d] / 250.0, 1.0)
        y = 40 + 18 * i
        cv2.rectangle(panel, (10, y - 10), (10 + int(160 * v), y + 4), COLORS[i], -1)
        cv2.putText(panel, f"{d} {curls[d]:.0f}", (176, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--trackers", nargs="+", default=["mediapipe", "wilor"])
    ap.add_argument("--out", default="compare.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    trackers = [make_tracker(n) for n in args.trackers]
    stats = {t.name: {"frames": 0, "palm": [0, 0], "back": [0, 0], "curls": [], "ms": 0.0} for t in trackers}
    prev = {t.name: None for t in trackers}
    jitter = {t.name: [] for t in trackers}

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        frame = cv2.flip(frame, 1)
        panels = []
        # A face-on reference for palm/back split: whichever tracker sees the hand decides first.
        orientation = None
        for t in trackers:
            panel = frame.copy()
            t0 = time.monotonic()
            hands = t.detect(frame)
            s = stats[t.name]
            s["ms"] += (time.monotonic() - t0) * 1000
            s["frames"] += 1
            hand = hands[0] if hands else None
            if hand is not None and orientation is None:
                orientation = "palm" if palm_facing_camera(hand) else "back"
            if hand is not None:
                ang = hand_angles(hand.world.copy())
                curls = {d: ang[d][0] for d in DIGITS}
                s["curls"].append([curls[d] for d in DIGITS])
                if prev[t.name] is not None:
                    jitter[t.name].append(np.abs(np.array(s["curls"][-1]) - prev[t.name]).mean())
                prev[t.name] = np.array(s["curls"][-1])
                draw(panel, hand, t.name, curls)
            else:
                prev[t.name] = None
            cv2.putText(panel, t.name + ("" if hand else "  (no hand)"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append((panel, hand is not None))
        for t, (_, found) in zip(trackers, panels):
            if orientation:
                stats[t.name][orientation][0] += found
                stats[t.name][orientation][1] += 1
        view = np.hstack([p for p, _ in panels])
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (view.shape[1], view.shape[0]))
        writer.write(view)
        n += 1
    if writer:
        writer.release()
    for t in trackers:
        t.close()

    print(f"\n{n} frames -> {args.out}\n")
    for name, s in stats.items():
        c = np.array(s["curls"]) if s["curls"] else np.zeros((0, 5))
        det = len(c) / max(s["frames"], 1)
        pb = "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in (("palm", s["palm"]), ("back", s["back"])))
        print(f"{name:10s} detected {det*100:5.1f}%  ({pb})   {s['ms']/max(s['frames'],1):6.1f} ms/frame   "
              f"jitter {np.median(jitter[name]) if jitter[name] else float('nan'):.1f} deg")
        if len(c):
            lo, hi = np.percentile(c, 5, axis=0), np.percentile(c, 95, axis=0)
            print("           curl p5-p95: " + "  ".join(f"{d} {a:.0f}-{b:.0f}" for d, a, b in zip(DIGITS, lo, hi)))


if __name__ == "__main__":
    main()
