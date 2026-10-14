#!/usr/bin/env python3
"""Offscreen renders (MuJoCo) of the ultimate finger and the AH_Right_5F_3S hand.

renders/finger_compare_side.jpg   stock 2-phalanx | 3-phalanx concept | ultimate, side view at 4 poses
renders/finger_closeup.jpg        the ultimate finger, 3/4 views, straight and flexed
renders/hand_<pose>.jpg           palm, back, 3/4 palm, 3/4 back, side, top for each pose of data/hand_poses.json
renders/hand_<pose>_shells.jpg    with --shells: the same pose with the palm/back shells of scene_shells.xml (palm, back, 3/4 palm)
renders/hand_compare_OpenHand.jpg stock AH_Right_5F | AH_Right_5F_Slim | AH_Right_5F_3S, open hand, palm and back

Usage: python render.py [--size 480] [--poses Peace,MiddleFinger,...] [--only fingers|hands|compare] [--shells]
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "five_finger"))
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "three_segment"))
import verify_5f_model as V  # noqa: E402
from finger_common import Finger, flexion_ctrl, stock_roles, three_seg_roles  # noqa: E402

RENDER_DIR = HERE / "renders"
HAND_VIEWS = {"palm": (180, -12, 0.36), "back": (0, -12, 0.36), "3/4 palm": (140, -28, 0.37), "3/4 back": (35, -25, 0.37),
              "index side": (270, -8, 0.36), "top": (180, -80, 0.36)}
HAND_LOOKAT = np.array([0.012, -0.03, 0.09])
FINGER_VIEWS = {"side": (270, -5, 0.25), "3/4 palm": (215, -25, 0.26), "3/4 back": (330, -20, 0.26)}


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([4, 3, 10 + 6 * len(text), 17], fill=(20, 20, 20))
    d.text((7, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def hide_helpers(m):
    for b in range(m.nbody):
        if m.body_mocapid[b] >= 0:
            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
                m.geom_rgba[g, 3] = 0
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fl >= 0:
        m.geom_rgba[fl, 3] = 0


def shoot(m, d, r, cam_spec, lookat):
    az, el, dist = cam_spec
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    r.update_scene(d, cam, opt)
    return r.render()


def settle(h, pose):
    h.reset()
    if pose is None:
        return
    target = V.pose_ctrl(h, pose)
    start = h.d.ctrl.copy()
    for k in range(900):
        h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / 450)
        mujoco.mj_step(h.m, h.d)


def make_hand(model, size):
    h = V.Hand(V.SIM_DIR / model / "mjcf")
    h.m.vis.global_.offwidth, h.m.vis.global_.offheight = size * 2, size * 2
    hide_helpers(h.m)
    return h, mujoco.Renderer(h.m, size, int(size * 4 / 3))


def hands(size, names, shells=False):
    poses = json.loads((HERE / "data" / "hand_poses.json").read_text())
    h, r = make_hand("AH_Right_5F_3S", size)
    if shells:
        ms = mujoco.MjModel.from_xml_path(str(V.SIM_DIR / "AH_Right_5F_3S" / "mjcf" / "scene_shells.xml"))
        assert ms.nq == h.m.nq
        ms.vis.global_.offwidth, ms.vis.global_.offheight = size * 2, size * 2
        hide_helpers(ms)
        ds, rs = mujoco.MjData(ms), mujoco.Renderer(ms, size, int(size * 4 / 3))
    for name in names:
        settle(h, {k: tuple(v) for k, v in poses[name].items()})
        tiles = [label(shoot(h.m, h.d, r, v, HAND_LOOKAT), f"{name} - {k}") for k, v in HAND_VIEWS.items()]
        img = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
        Image.fromarray(img).save(RENDER_DIR / f"hand_{name}.jpg", quality=88)
        print("wrote", RENDER_DIR / f"hand_{name}.jpg")
        if not shells:
            continue
        ds.qpos[:] = h.d.qpos
        ds.mocap_pos[:], ds.mocap_quat[:] = h.d.mocap_pos, h.d.mocap_quat
        mujoco.mj_forward(ms, ds)
        tiles = [label(shoot(ms, ds, rs, HAND_VIEWS[v], HAND_LOOKAT), f"{name} - shells - {v}") for v in ("palm", "back", "3/4 palm")]
        Image.fromarray(np.concatenate(tiles, 1)).save(RENDER_DIR / f"hand_{name}_shells.jpg", quality=88)
        print("wrote", RENDER_DIR / f"hand_{name}_shells.jpg")
    if shells:
        rs.close()


def compare_hands(size):
    poses = json.loads((HERE / "data" / "hand_poses.json").read_text())
    rows = []
    for model, pose in (("AH_Right_5F", V.DEMO_POSES_5F["OpenHand"]), ("AH_Right_5F_Slim", V.DEMO_POSES_5F["OpenHand"]),
                        ("AH_Right_5F_3S", {k: tuple(v) for k, v in poses["OpenHand"].items()})):
        h, r = make_hand(model, size)
        settle(h, pose)
        rows.append(np.concatenate([label(shoot(h.m, h.d, r, HAND_VIEWS[v], HAND_LOOKAT), f"{model} - {v}")
                                    for v in ("palm", "back", "3/4 palm")], 1))
        r.close()
    Image.fromarray(np.concatenate(rows, 0)).save(RENDER_DIR / "hand_compare_OpenHand.jpg", quality=88)
    print("wrote", RENDER_DIR / "hand_compare_OpenHand.jpg")


def fingers(size):
    sys.path.insert(0, str(HERE))
    from verify_finger import roles
    f_s = json.loads((REPO / "cad" / "five_finger" / "ultimate_finger" / "design.json").read_text())["predicted"]["f_straight_deg"]
    models = [("stock 2-phalanx", REPO / "Demo/AHSimulation/three_segment/model/scene_stock.xml", stock_roles(), -37.5),
              ("3-phalanx concept", REPO / "Demo/AHSimulation/three_segment/model/scene_3seg.xml", three_seg_roles(), -56.0),
              ("ultimate", HERE / "model" / "scene_uf.xml", roles(), f_s)]
    lookat = np.array([0.0, 0.02, 0.135])
    rows = []
    for title, path, rl, fs in models:
        f = Finger(path, rl)
        f.m.vis.global_.offwidth, f.m.vis.global_.offheight = size * 2, size * 2
        hide_helpers(f.m)
        r = mujoco.Renderer(f.m, size, int(size * 4 / 3))
        row = []
        for t, fv in ((f"straight f = {fs:.0f}", fs), ("f = 0", 0.0), ("f = +45", 45.0), ("full f = +90", 90.0)):
            f.reset()
            f.move(*flexion_ctrl(fv), ramp=0.6, hold=0.4, check=False)
            row.append(label(shoot(f.m, f.d, r, FINGER_VIEWS["side"], lookat), f"{title} - {t}"))
        rows.append(np.concatenate(row, 1))
        if title == "ultimate":
            tiles = []
            for t, fv in ((f"straight f = {fs:.0f}", fs), ("f = +30", 30.0)):
                f.reset()
                f.move(*flexion_ctrl(fv), ramp=0.6, hold=0.4, check=False)
                for v in ("3/4 palm", "3/4 back"):
                    tiles.append(label(shoot(f.m, f.d, r, FINGER_VIEWS[v], lookat), f"ultimate - {t} - {v}"))
            Image.fromarray(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)).save(
                RENDER_DIR / "finger_closeup.jpg", quality=88)
        r.close()
    Image.fromarray(np.concatenate(rows, 0)).save(RENDER_DIR / "finger_compare_side.jpg", quality=88)
    print("wrote", RENDER_DIR / "finger_compare_side.jpg", RENDER_DIR / "finger_closeup.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--poses", default="Peace,MiddleFinger,Pinched,OpenHand,CloseHand,SpreadHand,Perfect,ClenchHand")
    ap.add_argument("--only", choices=["fingers", "hands", "compare"], default=None)
    ap.add_argument("--shells", action="store_true", help="also render each pose with the shells of scene_shells.xml")
    args = ap.parse_args()
    RENDER_DIR.mkdir(exist_ok=True)
    if args.only in (None, "fingers"):
        fingers(args.size)
    if args.only in (None, "hands"):
        hands(args.size, args.poses.split(","), args.shells)
    if args.only in (None, "compare"):
        compare_hands(args.size)


if __name__ == "__main__":
    main()
