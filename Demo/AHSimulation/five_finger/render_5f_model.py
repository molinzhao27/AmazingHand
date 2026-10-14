#!/usr/bin/env python3
"""Offscreen renders of the 5-digit AmazingHand model (and the stock model for comparison).

Writes JPEG contact sheets to five_finger/renders/: one sheet per pose with six camera angles,
plus a stock-vs-5-finger comparison.

Usage: python render_5f_model.py [--model AH_Right_5F|AH_Left_5F] [--shells] [--out DIR] [--size 480]
       --shells adds the 5-finger palm/back shells and grip pads (cad/five_finger/stl)
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_shells import shells  # noqa: E402
from verify_5f_model import DEMO_POSES_5F, SIM_DIR, Hand, pose_ctrl  # noqa: E402

VIEWS = {  # name: (azimuth, elevation, distance)
    "palm": (180, -12, 0.34),
    "back": (0, -12, 0.34),
    "pinky side": (90, -10, 0.34),
    "index side": (270, -10, 0.34),
    "top": (180, -80, 0.34),
    "3/4 palm": (140, -28, 0.36),
}
LOOKAT = np.array([0.012, -0.03, 0.085])

POSES = {
    "keyframe zero": None,
    **{k: DEMO_POSES_5F[k] for k in ("OpenHand", "CloseHand", "SpreadHand", "ClenchHand", "Perfect")},
    # worst case of the interference sweep: ring and pinky fully abducted towards each other
    "ring+pinky abducted together (contact)": dict(ring=(-90, -90), pinky=(90, 90)),
    "ring+pinky fully flexed": dict(ring=(90, -90), pinky=(90, -90)),
}


def settle(h, pose):
    h.reset()
    if pose is None:
        return
    target = pose_ctrl(h, pose)
    start = h.d.ctrl.copy()
    for k in range(600):
        h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / 300)
        mujoco.mj_step(h.m, h.d)


def shoot(h, renderer, view, size):
    az, el, dist = VIEWS[view]
    if h.side == "left":  # mirror the camera too (the left hand is the right one mirrored in y)
        az = -az % 360
    cam = mujoco.MjvCamera()
    cam.lookat[:] = LOOKAT * ([1, -1, 1] if h.side == "left" else 1)
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # hide group 3 (closing sites live there too)
    renderer.update_scene(h.d, cam, opt)
    return renderer.render()


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(im)


SHELL_RGBA = {"top_shell": [0.95, 0.95, 0.96, 1], "palm_shell": [0.92, 0.93, 0.94, 1]}
PAD_RGBA = [0.07, 0.07, 0.08, 1]


def make_hand(model, size, with_shells=False):
    extra = shells("5f", "L" if "Left" in model else "R") if with_shells else None
    h = Hand(SIM_DIR / model / "mjcf", extra)
    for name in extra or ():
        g = h.part_geom[name]
        h.m.geom_matid[g] = -1
        h.m.geom_rgba[g] = SHELL_RGBA.get(name, PAD_RGBA)
    h.m.vis.global_.offwidth = max(h.m.vis.global_.offwidth, size * 2)
    h.m.vis.global_.offheight = max(h.m.vis.global_.offheight, size * 2)
    floor = mujoco.mj_name2id(h.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    h.m.geom_rgba[floor, 3] = 0  # hide the floor
    for b in range(h.m.nbody):  # hide the mocap target spheres
        if h.m.body_mocapid[b] >= 0:
            for g in range(h.m.body_geomadr[b], h.m.body_geomadr[b] + h.m.body_geomnum[b]):
                h.m.geom_rgba[g, 3] = 0
    return h, mujoco.Renderer(h.m, size, int(size * 4 / 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "renders")
    ap.add_argument("--size", type=int, default=480, help="height of one view in pixels")
    ap.add_argument("--model", default="AH_Right_5F", choices=["AH_Right_5F", "AH_Left_5F"])
    ap.add_argument("--shells", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = ("" if args.model == "AH_Right_5F" else "left_") + ("shells_" if args.shells else "")

    h, r = make_hand(args.model, args.size, args.shells)
    poses = {k: v for k, v in POSES.items() if k in ("OpenHand", "CloseHand", "SpreadHand", "Perfect")} if args.shells else POSES
    for i, (name, pose) in enumerate(poses.items()):
        settle(h, pose)
        tiles = [label(shoot(h, r, v, args.size), f"{name} - {v}") for v in VIEWS]
        sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
        fname = args.out / f"{prefix}5f_{i:02d}_{name.split(' (')[0].replace(' ', '_').replace('+', '_')}.jpg"
        Image.fromarray(sheet).save(fname, quality=88)
        print("wrote", fname.relative_to(HERE))

    if args.shells:
        return
    stock, rs = make_hand(args.model.replace("_5F", ""), args.size)
    rows = []
    for pose_name in ("OpenHand", "CloseHand"):
        row = []
        for hand, rend, tag in ((stock, rs, "stock 3F+thumb"), (h, r, "5-digit")):
            settle(hand, DEMO_POSES_5F[pose_name])
            for v in ("palm", "back"):
                row.append(label(shoot(hand, rend, v, args.size), f"{tag} - {pose_name} - {v}"))
        rows.append(np.concatenate(row, 1))
    fname = args.out / f"{prefix}compare_stock_vs_5f.jpg"
    Image.fromarray(np.concatenate(rows, 0)).save(fname, quality=88)
    print("wrote", fname.relative_to(HERE))


if __name__ == "__main__":
    main()
