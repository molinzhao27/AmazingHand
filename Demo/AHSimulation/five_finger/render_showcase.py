#!/usr/bin/env python3
"""Showcase renders of the full build: forearm + powered wrist + 5-digit hand with the new palm/back
shells and grip pads (default model AH_Right_5F_Slim_Wrist: slim fingers).

Usage: python render_showcase.py [--model AH_Right_5F_Slim_Wrist] [--out renders/showcase.jpg]
       python render_showcase.py --model AH_Right_5F_3S_Wrist --poses ../ultimate_finger/data/hand_poses.json \
           --shots gestures --out renders/showcase_3s.jpg      (3-phalanx fingers: their own pose commands)
"""

import json

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent / "cad" / "five_finger" / "wrist"))
import wrist_design as wd  # noqa: E402
from verify_5f_model import DEMO_POSES_5F, SIM_DIR  # noqa: E402

DIGITS = {"index": "finger1", "middle": "finger2", "ring": "finger3", "thumb": "finger4", "pinky": "finger5"}
# (title, hand pose, wrist flexion deg (> 0 palm-ward), wrist deviation deg (> 0 radial), camera az, el)
SHOTS = [
    ("OpenHand, wrist neutral", "OpenHand", 0, 0, 150, -12),
    ("Perfect, wrist extended 25 / ulnar 10", "Perfect", -25, -10, 165, -8),
    ("CloseHand, wrist flexed 40", "CloseHand", 40, 0, 120, -15),
    ("SpreadHand, wrist radial 15", "SpreadHand", 0, 15, 180, -10),
    ("Index_Pointing, wrist flexed 20 / radial 10", "Index_Pointing", 20, 10, 210, -12),
    ("Victory, wrist extended 40", "Victory", -40, 0, 135, -20),
]
# the gestures (peace sign, middle finger, OK, pinch) for a pose file that has them
GESTURE_SHOTS = [
    ("OpenHand, wrist neutral", "OpenHand", 0, 0, 150, -12),
    ("Peace, wrist extended 20", "Peace", -20, 0, 165, -10),
    ("MiddleFinger, wrist neutral", "MiddleFinger", 0, 0, 200, -8),
    ("Perfect (OK), wrist extended 25 / ulnar 10", "Perfect", -25, -10, 165, -8),
    ("Pinched, wrist flexed 20", "Pinched", 20, 0, 130, -15),
    ("CloseHand, wrist flexed 40", "CloseHand", 40, 0, 120, -15),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="AH_Right_5F_Slim_Wrist")
    ap.add_argument("--out", type=Path, default=HERE / "renders" / "showcase.jpg")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--poses", type=Path, help="JSON {pose: {digit: [m1, m2] deg}} overriding the demo poses")
    ap.add_argument("--shots", choices=["demo", "gestures"], default="demo")
    args = ap.parse_args()
    poses = dict(DEMO_POSES_5F)
    if args.poses:
        poses.update(json.loads(args.poses.read_text()))

    m = mujoco.MjModel.from_xml_path(str(SIM_DIR / args.model / "mjcf" / "scene_shells.xml"))
    m.vis.global_.offwidth, m.vis.global_.offheight = 2400, 2400
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    m.geom_rgba[floor, 3] = 0
    for b in range(m.nbody):
        if m.body_mocapid[b] >= 0:
            m.geom_rgba[m.body_geomadr[b]:m.body_geomadr[b] + m.body_geomnum[b], 3] = 0
    d = mujoco.MjData(m)
    act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    r = mujoco.Renderer(m, args.size, int(args.size * 0.8))
    tiles = []
    for title, pose, phi, psi, az, el in (SHOTS if args.shots == "demo" else GESTURE_SHOTS):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        target = np.zeros(m.nu)
        for dig, (a1, a2) in poses[pose].items():
            target[[act[f"{DIGITS[dig]}_motor1"], act[f"{DIGITS[dig]}_motor2"]]] = np.radians([a1, a2])
        target[[act["wrist_motor1"], act["wrist_motor2"]]] = wd.servo_ik(*wd.clamp_wrist(np.radians(phi), np.radians(psi)))
        for k in range(1500):
            d.ctrl[:] = target * min(1.0, (k + 1) / 600)
            mujoco.mj_step(m, d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.01, -0.005, 0.02]
        cam.azimuth, cam.elevation, cam.distance = az, el, 0.62
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        r.update_scene(d, cam, opt)
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text((8, 6), title, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(args.out, quality=88)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
