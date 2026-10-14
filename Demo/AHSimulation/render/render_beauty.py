#!/usr/bin/env python3
"""Studio beauty shots of the hand: pose in MuJoCo (export_pose.py), render in Blender Cycles (blender_render.py).

Usage (project venv):
  python render_beauty.py                      # all shots, final quality -> five_finger/renders/beauty/
  python render_beauty.py --preview            # fast low-sample previews (beauty/preview/)
  python render_beauty.py --only hero,peace    # a subset
  python render_beauty.py --model AH_Right_5F_3S_Wrist --blender /path/to/Blender

Each shot is a pose (a named pose from hand_poses.json, with optional per-digit overrides), a wrist pose and
the render options of blender_render.py (camera direction, framing, lighting overrides, ...). See README.md.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AHSIM = HERE.parent
OUT_DIR = AHSIM / "five_finger" / "renders" / "beauty"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

# A gently fanned open hand with a relaxed thumb (OpenHand's thumb, -35/35, points straight out sideways).
OPEN_FAN = {"index": [-46, 60], "middle": [-53, 53], "ring": [-60, 46], "pinky": [-66, 40], "thumb": [-12, 12]}
# A relaxed hand: the fingers curl progressively from index to pinky.
CASCADE = {"index": [-45, 45], "middle": [-30, 30], "ring": [-15, 15], "pinky": [0, 0], "thumb": [0, -10]}
FINGERS_ONLY = dict(frame="fingers", frame_classes="white_plastic,black_tpu", frame_digits="index,middle,ring,pinky")

# name: pose (hand_poses.json), digit overrides {digit: [m1, m2] deg}, wrist (flexion > 0 palm-ward, deviation > 0
# radial) deg, blender_render.py options (keys without the leading "--", "_" for "-"; lists for multi-value options).
# Camera az/el in MuJoCo's convention: az 180 looks at the palm, az 0 at the back, az 270 from the thumb side.
SHOTS = {
    # 3/4 palm view, open hand, wrist slightly extended; forearm leaves the frame at the bottom
    "hero": dict(
        pose="OpenHand", digits=OPEN_FAN, wrist=(-15, 0),
        render=dict(az=145, el=-8, frame="hand", fill=0.72, offset=[0.0, 0.06])),
    "peace": dict(
        pose="Peace", digits={}, wrist=(-15, 5),
        render=dict(az=160, el=-6, frame="hand", fill=0.72, offset=[0.0, 0.05])),
    # forearm horizontal (model turned 90 deg about X, then pronated 45 deg about the forearm), resting on the sweep
    "pointing": dict(
        pose="Index_Pointing", digits={}, wrist=(-10, 10),
        render=dict(az=195, el=-18, frame="hand", fill=0.62, offset=[0.04, 0.04], rotate=[90, 45, 0], floor_gap=0.0)),
    # OK / Perfect from the thumb side, so the index-thumb ring reads
    "ok": dict(
        pose="Perfect", digits={}, wrist=(-25, -10),
        render=dict(az=285, el=-5, frame="hand", fill=0.72, offset=[0.0, 0.05])),
    "back": dict(
        pose="OpenHand", digits=CASCADE, wrist=(10, 0),
        render=dict(az=345, el=-15, frame="hand", fill=0.72, offset=[0.0, 0.05])),
    # close-up of the four fingers from the pinky side, shallow depth of field; thumb tucked out of the way
    "fingers": dict(
        pose="OpenHand", digits={**CASCADE, "thumb": [90, -90]}, wrist=(10, 0),
        render=dict(az=118, el=-8, fill=0.75, fstop=4, lens=100, **FINGERS_ONLY)),
}


def to_args(opts):
    out = []
    for k, v in opts.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            out += [flag] if v else []
        elif isinstance(v, (list, tuple)):
            out += [flag] + [str(x) for x in v]
        elif isinstance(v, dict):
            out += [flag, json.dumps(v)]
        else:
            out += [flag, str(v)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated shot names (default: all): " + ", ".join(SHOTS))
    ap.add_argument("--preview", action="store_true", help="fast, small, few samples; writes to <out>/preview/")
    ap.add_argument("--model", default="AH_Right_5F_3S_Wrist")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--work", type=Path, help="keep the GLB/JSON exports here (default: a temp dir, deleted)")
    ap.add_argument("--blender", default=shutil.which("blender") or BLENDER)
    ap.add_argument("--extra", default="", help="extra blender_render.py arguments for every shot (one string)")
    ap.add_argument("--sheet", action="store_true", help="also write a contact sheet of the shots")
    args = ap.parse_args()

    names = args.only.split(",") if args.only else list(SHOTS)
    unknown = [n for n in names if n not in SHOTS]
    if unknown:
        ap.error(f"unknown shots {unknown}; have {list(SHOTS)}")
    out_dir = args.out / "preview" if args.preview else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    work = args.work or Path(tempfile.mkdtemp(prefix="beauty_"))
    work.mkdir(parents=True, exist_ok=True)
    times, written = [], []
    try:
        for name in names:
            s = SHOTS[name]
            glb = work / f"{name}.glb"
            t0 = time.time()
            cmd = [sys.executable, str(HERE / "export_pose.py"), "--model", args.model, "--pose", s["pose"],
                   "--wrist", str(s["wrist"][0]), str(s["wrist"][1]), "--out", str(glb)]
            for dig, (m1, m2) in s.get("digits", {}).items():
                cmd += ["--digit", f"{dig}={m1},{m2}"]
            subprocess.run(cmd, check=True)
            t1 = time.time()
            img = out_dir / f"{name}.jpg"
            cmd = [args.blender, "-b", "--factory-startup", "-P", str(HERE / "blender_render.py"), "--",
                   "--glb", str(glb), "--out", str(img)] + to_args(s.get("render", {}))
            if args.preview:
                cmd.append("--preview")
            if args.extra:
                cmd += args.extra.split()
            r = subprocess.run(cmd, capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.startswith("[blender_render]") or "Error" in line or "Traceback" in line:
                    print("   ", line)
            if r.returncode != 0 or not img.exists():
                print(r.stdout[-3000:], r.stderr[-3000:])
                raise SystemExit(f"blender failed on shot {name}")
            t2 = time.time()
            times.append((name, t1 - t0, t2 - t1))
            written.append(img)
            print(f"{name}: pose {t1 - t0:.1f} s, render {t2 - t1:.1f} s -> {img}", flush=True)
    finally:
        if args.work is None:
            shutil.rmtree(work, ignore_errors=True)
    print("\nshot          pose (s)  blender (s)")
    for name, tp, tr in times:
        print(f"{name:<12} {tp:8.1f} {tr:12.1f}")
    print(f"total {sum(t[1] + t[2] for t in times):.0f} s")
    if args.sheet and written:
        from PIL import Image
        ims = [Image.open(p) for p in written]
        w, h = ims[0].size
        tw, th = 640, 640 * h // w
        cols = min(3, len(ims))
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (tw * cols, th * rows), (40, 40, 40))
        for k, im in enumerate(ims):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), ((k % cols) * tw, (k // cols) * th))
        sheet.save(out_dir / "contact_sheet.jpg", quality=90)
        print("wrote", out_dir / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
