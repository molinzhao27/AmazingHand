#!/usr/bin/env python3
"""Offscreen renders (MuJoCo): one-finger models (stock 2-phalanx vs 3-phalanx), the 5-digit hands
(stock AH_Right_5F vs model/hand5_3seg at the re-mapped demo poses) and an exploded view of the new
CAD parts.

Writes renders/finger_*.jpg, renders/hand_*.jpg, and figures/render_*.jpg in cad/five_finger/three_segment.

Usage: python render_finger.py [--size 420]
"""

import argparse
import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from finger_common import (DESIGN_JSON, FIG_DIR, MODEL_DIR, RENDER_DIR, Finger, flexion_ctrl, stock_roles,
                           three_seg_roles)

VIEWS = {"side (index side)": (270, -5, 0.24), "3/4 palm": (215, -25, 0.25), "3/4 back": (330, -20, 0.25),
         "palm": (180, -8, 0.24)}
LOOKAT = np.array([0.0, 0.02, 0.135])


def make(name, size):
    roles = stock_roles() if name == "stock" else three_seg_roles()
    f = Finger(MODEL_DIR / f"scene_{name}.xml", roles)
    f.m.vis.global_.offwidth = max(f.m.vis.global_.offwidth, size * 2)
    f.m.vis.global_.offheight = max(f.m.vis.global_.offheight, size * 2)
    return f, mujoco.Renderer(f.m, size, int(size * 4 / 3))


def shoot(f, r, view, lookat=LOOKAT):
    az, el, dist = VIEWS[view]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    r.update_scene(f.d, cam, opt)
    return r.render()


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(im)


def pose_set(f_straight):
    return [("straight (f = %.0f)" % f_straight, f_straight, 0.0), ("f = 0", 0.0, 0.0), ("f = +45", 45.0, 0.0),
            ("full flexion f = +90", 90.0, 0.0), ("abducted, a = +30", f_straight + 10, 30.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--exploded", action="store_true")
    ap.add_argument("--hands", action="store_true")
    args = ap.parse_args()
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    fs3 = json.loads(DESIGN_JSON.read_text())["drive"]["f_straight_deg"]
    sheets = {}
    for name, fstr in (("stock", -37.5), ("3seg", fs3)):
        f, r = make(name, args.size)
        rows = []
        for title, fv, a in pose_set(fstr):
            f.reset()
            f.move(*flexion_ctrl(fv, a), ramp=0.6, hold=0.4, check=False)
            m = f.measure()
            ang = f"MCP {m['mcp']:.0f} PIP {m['pip']:.0f}" + (f" DIP {m['dip']:.0f}" if "dip" in m else "")
            rows.append(np.concatenate([label(shoot(f, r, v), f"{name}: {title}: {ang} - {v}") for v in VIEWS], 1))
        sheet = np.concatenate(rows, 0)
        Image.fromarray(sheet).save(RENDER_DIR / f"finger_{name}_poses.jpg", quality=88)
        sheets[name] = (f, r)
        print("wrote", RENDER_DIR / f"finger_{name}_poses.jpg")
    # side-by-side comparison, side view, 4 poses
    tiles = []
    for name, fstr in (("stock", -37.5), ("3seg", fs3)):
        f, r = sheets[name]
        row = []
        for title, fv, a in pose_set(fstr)[:4]:
            f.reset()
            f.move(*flexion_ctrl(fv, a), ramp=0.6, hold=0.4, check=False)
            row.append(label(shoot(f, r, "side (index side)"), f"{'stock 2-phalanx' if name == 'stock' else '3-phalanx'}"
                                                                  f" - {title}"))
        tiles.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(RENDER_DIR / "finger_compare_side.jpg", quality=88)
    Image.fromarray(np.concatenate(tiles, 0)).save(FIG_DIR / "render_compare_side.jpg", quality=85)
    print("wrote", RENDER_DIR / "finger_compare_side.jpg")


def hands(size):
    """Stock 5-digit hand vs the 3-phalanx hand at the demo poses (re-mapped for the new fingers)."""
    import sys

    from finger_common import HERE, SIM_5F
    from verify_3seg_hand import THREE_SEG, remap
    sys.path.insert(0, str(HERE.parent / "five_finger"))
    import verify_5f_model as V
    from render_5f_model import LOOKAT as HL
    from render_5f_model import VIEWS as HV

    f_s = json.loads(DESIGN_JSON.read_text())["drive"]["f_straight_deg"]
    models = {"stock 5-digit": V.Hand(SIM_5F), "3-phalanx 5-digit": V.Hand(MODEL_DIR / "hand5_3seg")}
    rend = {}
    for k, h in models.items():
        h.m.vis.global_.offwidth, h.m.vis.global_.offheight = size * 2, size * 2
        for b in range(h.m.nbody):  # hide mocap target spheres and the floor
            if h.m.body_mocapid[b] >= 0:
                for g in range(h.m.body_geomadr[b], h.m.body_geomadr[b] + h.m.body_geomnum[b]):
                    h.m.geom_rgba[g, 3] = 0
        fl = mujoco.mj_name2id(h.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if fl >= 0:
            h.m.geom_rgba[fl, 3] = 0
        rend[k] = mujoco.Renderer(h.m, size, int(size * 4 / 3))

    def settle(h, pose):
        h.reset()
        target = V.pose_ctrl(h, pose)
        start = h.d.ctrl.copy()
        for k in range(800):
            h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / 400)
            mujoco.mj_step(h.m, h.d)

    def shot(h, r, view):
        az, el, dist = HV[view]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = HL
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        r.update_scene(h.d, cam, opt)
        return r.render()

    names = ("OpenHand", "CloseHand", "SpreadHand", "ClenchHand", "Perfect")
    for name in names:
        rows = []
        for k, h in models.items():
            pose = V.DEMO_POSES_5F[name]
            if "3-phalanx" in k:
                pose = {d: (remap(c, f_s) if d in THREE_SEG else c) for d, c in pose.items()}
            settle(h, pose)
            rows.append(np.concatenate([label(shot(h, rend[k], v), f"{k}: {name} - {v}")
                                        for v in ("palm", "pinky side", "3/4 palm")], 1))
        Image.fromarray(np.concatenate(rows, 0)).save(RENDER_DIR / f"hand_{name}.jpg", quality=86)
        print("wrote", RENDER_DIR / f"hand_{name}.jpg")
    # one overview for the report: palm view, open / close for both hands
    tiles = []
    for k, h in models.items():
        row = []
        for name in ("OpenHand", "SpreadHand", "CloseHand"):
            pose = V.DEMO_POSES_5F[name]
            if "3-phalanx" in k:
                pose = {d: (remap(c, f_s) if d in THREE_SEG else c) for d, c in pose.items()}
            settle(h, pose)
            row.append(label(shot(h, rend[k], "3/4 palm" if name != "CloseHand" else "pinky side"), f"{k}: {name}"))
        tiles.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(FIG_DIR / "render_hands.jpg", quality=85)
    print("wrote", FIG_DIR / "render_hands.jpg")


def hand_shells(size):
    """The 3-phalanx hand with the palm/back shells regenerated around it (regen_palm_shells.py)."""
    import sys

    from finger_common import HERE
    from verify_3seg_hand import THREE_SEG, remap, shell_set
    sys.path.insert(0, str(HERE.parent / "five_finger"))
    import verify_5f_model as V
    from render_5f_model import LOOKAT as HL
    from render_5f_model import PAD_RGBA, SHELL_RGBA
    from render_5f_model import VIEWS as HV

    f_s = json.loads(DESIGN_JSON.read_text())["drive"]["f_straight_deg"]
    extra = shell_set("regen")
    h = V.Hand(MODEL_DIR / "hand5_3seg", extra)
    for name in extra:
        g = h.part_geom[name]
        h.m.geom_matid[g] = -1
        h.m.geom_rgba[g] = SHELL_RGBA.get(name, PAD_RGBA)
    h.m.vis.global_.offwidth, h.m.vis.global_.offheight = size * 2, size * 2
    r = mujoco.Renderer(h.m, size, int(size * 4 / 3))
    tiles = []
    for name, view in (("OpenHand", "3/4 palm"), ("SpreadHand", "palm"), ("CloseHand", "pinky side"), ("CloseHand", "3/4 palm")):
        pose = {d: (remap(c, f_s) if d in THREE_SEG else c) for d, c in V.DEMO_POSES_5F[name].items()}
        h.reset()
        target = V.pose_ctrl(h, pose)
        start = h.d.ctrl.copy()
        for k in range(800):
            h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / 400)
            mujoco.mj_step(h.m, h.d)
        az, el, dist = HV[view]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = HL
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        r.update_scene(h.d, cam, opt)
        tiles.append(label(r.render(), f"3-phalanx hand, regenerated shells: {name} - {view}"))
    Image.fromarray(np.concatenate(tiles, 1)).save(FIG_DIR / "render_hand_shells.jpg", quality=85)
    print("wrote", FIG_DIR / "render_hand_shells.jpg")


def exploded(size):
    """Exploded view of the new parts in the straight pose, rendered from the CAD STLs."""
    from finger_common import CAD_DIR, REPO
    design = json.loads(DESIGN_JSON.read_text())
    stl = design["cad"]["stl"]
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = size * 4, size * 2
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.65, 0.65, 0.65]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    psi_s = np.radians(design["drive"]["psi_straight_deg"])
    col = dict(link3s=[0.95, 0.5, 0.2, 1], p1=[0.85, 0.85, 0.85, 1], p2=[0.75, 0.78, 0.8, 1], p3=[0.85, 0.85, 0.85, 1],
               l2=[0.2, 0.55, 0.95, 1], p1_shell=[1, 1, 1, 1], p2_pad=[0.35, 0.36, 0.38, 1], p3_shell=[1, 1, 1, 1])
    explode = dict(link3s=(0, 0, -14), p1=(0, 0, 0), p2=(0, 6, 14), p3=(0, 14, 0), l2=(0, 6, -26), p1_shell=(0, 0, 26),
                   p2_pad=(0, 6, -40), p3_shell=(0, 16, 20))
    for k, fname in stl.items():
        spec.add_mesh(name=k, file=str(CAD_DIR / "stl" / fname), scale=[1e-3] * 3)
        pos = np.array(explode[k], float) * 1e-3
        quat = [1, 0, 0, 0]
        if k == "link3s":  # Link frame -> straight pose: rotation about x through the Link pivot (0, 0, -6)
            c, s_ = np.cos(-psi_s / 2), np.sin(-psi_s / 2)
            quat = [c, s_, 0, 0]
            R = np.array([[1, 0, 0], [0, np.cos(-psi_s), -np.sin(-psi_s)], [0, np.sin(-psi_s), np.cos(-psi_s)]])
            OL = np.array([0, 0, -6e-3])
            pos = pos + OL - R @ OL
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=k, pos=pos, quat=quat, rgba=col[k])
    for name, fn, pos in (("gimbal", "Gimbal", (0, 0, 0)),):
        spec.add_mesh(name=name, file=str(REPO / "cad" / "stl" / f"Amazing Hand Parts - {fn}.stl"), scale=[1e-3] * 3)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=[-0.35e-3, 0, 0],
                                rgba=[0.5, 0.5, 0.5, 1])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, size, size * 2)
    tiles = []
    for az, el, dist in ((180, 0, 0.17), (215, -30, 0.2)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0.048, -0.004]
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(d, cam)
        tiles.append(r.render())
    img = np.concatenate(tiles, 0)
    Image.fromarray(img).save(FIG_DIR / "render_exploded.jpg", quality=88)
    print("wrote", FIG_DIR / "render_exploded.jpg")


if __name__ == "__main__":
    import sys
    if "--exploded" in sys.argv:
        exploded(480)
    elif "--hands" in sys.argv:
        hands(400)
        hand_shells(360)
    else:
        main()
        exploded(480)
        hands(400)
        hand_shells(360)
