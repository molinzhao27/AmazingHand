#!/usr/bin/env python3
"""Renders of AH_Right_5F_Wrist (sim, IK + servo outer loop) and CAD previews of the wrist parts.

Writes
  Demo/AHSimulation/wrist/renders/wrist_<nn>_<pose>.jpg   one sheet per wrist pose: OpenHand / CloseHand x 4 views
  Demo/AHSimulation/wrist/renders/wrist_mechanism.jpg     cut-away (back cover transparent), several wrist poses
  Demo/AHSimulation/wrist/renders/wrist_overview.jpg      hero shots
  cad/five_finger/wrist/previews/parts_print_orientation.jpg, exploded.jpg

Usage: python render_wrist_model.py [--size 420] [--no-cad]
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_wrist_model import settle_closed_loop  # noqa: E402
from wrist_sim import HAND_POSES, REPO, WristModel, wd  # noqa: E402

OUT = HERE / "renders"
CAD_DIR = REPO / "cad" / "five_finger" / "wrist"
VIEWS = {"3/4 palm": (140, -15, 0.66), "index side": (90, -8, 0.66), "back": (0, -12, 0.66), "3/4 back": (-40, -20, 0.66)}
LOOKAT = np.array([0.0, 0.0, -0.06])
WRIST_POSES = {
    "neutral": (0, 0), "palm-ward flexion 60": (60, 0), "extension 60": (-60, 0),
    "radial deviation 25": (0, 25), "ulnar deviation 25": (0, -25),
    "flexion 60 + radial 15": (60, 15), "extension 60 + ulnar 15": (-60, -15), "flexion 40 + ulnar 25": (40, -25),
}


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(im)


def prepare(w, size, back_alpha=None):
    m = w.m
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 2 * size)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 2 * size)
    for b in range(m.nbody):            # hide the mocap target spheres
        if m.body_mocapid[b] >= 0:
            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
                m.geom_rgba[g, 3] = 0
    if back_alpha is not None:
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
                if name in ("wrist_r_forearm_back",):
                    m.geom_rgba[g, 3] = back_alpha
    return mujoco.Renderer(m, size, int(size * 4 / 3))


def shoot(w, r, view, lookat=LOOKAT, dist=None):
    az, el, d0 = VIEWS[view] if isinstance(view, str) else view
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist or d0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    r.update_scene(w.d, cam, opt)
    return r.render()


def pose(w, hand, wrist_deg):
    w.reset()
    w.move(w.ctrl_for(HAND_POSES[hand]), 0.5, 0.5)
    settle_closed_loop(w, np.radians(wrist_deg), w.ctrl_for(HAND_POSES[hand]), t=2.5)


def render_sim(size):
    OUT.mkdir(exist_ok=True)
    w = WristModel("scene_shells.xml")
    r = prepare(w, size)
    for k, (name, q) in enumerate(WRIST_POSES.items()):
        rows = []
        for hand in ("OpenHand", "CloseHand"):
            pose(w, hand, q)
            got = np.degrees(w.wrist_angles())
            rows.append(np.concatenate([label(shoot(w, r, v), f"{name} ({got[0]:+.1f}, {got[1]:+.1f} deg) - {hand} - {v}")
                                        for v in VIEWS], 1))
        f = OUT / f"wrist_{k:02d}_{name.replace(' + ', '_').replace(' ', '_')}.jpg"
        Image.fromarray(np.concatenate(rows, 0)).save(f, quality=88)
        print("wrote", f.relative_to(REPO))
    # mechanism cut-away
    w2 = WristModel("scene.xml")
    r2 = prepare(w2, size, back_alpha=0.18)
    tiles = []
    for name in ("neutral", "palm-ward flexion 60", "extension 60", "flexion 60 + radial 15"):
        pose(w2, "OpenHand", WRIST_POSES[name])
        for view in ((60, -10, 0.42), (-30, -15, 0.42)):
            tiles.append(label(shoot(w2, r2, view, lookat=np.array([0, 0, -0.085])), f"{name} - back cover transparent"))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)], 0)
    Image.fromarray(sheet).save(OUT / "wrist_mechanism.jpg", quality=88)
    print("wrote", (OUT / "wrist_mechanism.jpg").relative_to(REPO))
    # hero shots
    tiles = []
    for hand, q, view in (("OpenHand", (0, 0), (150, -12, 0.72)), ("CloseHand", (45, 15), (120, -18, 0.72)),
                          ("OpenHand", (-45, -20), (200, -10, 0.72))):
        pose(w, hand, q)
        tiles.append(label(shoot(w, r, view), f"{hand}, wrist ({q[0]:+d}, {q[1]:+d}) deg"))
    Image.fromarray(np.concatenate(tiles, 1)).save(OUT / "wrist_overview.jpg", quality=90)
    print("wrote", (OUT / "wrist_overview.jpg").relative_to(REPO))


# ----------------------------------------------------------------------------- CAD previews
PARTS = ["R-Wrist_Hand_Adapter", "Wrist_Gimbal_Ring", "R-Wrist_Fork", "R-Forearm_Front", "R-Forearm_Back",
         "R-Wrist_Servo_Frame", "Wrist_Horn_Arm", "Elbow_Flange"]
UP = {"R-Wrist_Hand_Adapter": (0, -1, 0), "Wrist_Gimbal_Ring": (0, 1, 0), "R-Wrist_Fork": (0, 1, 0), "R-Forearm_Front": (0, 1, 0),
      "R-Forearm_Back": (0, 1, 0), "R-Wrist_Servo_Frame": (0, 0, 1), "Wrist_Horn_Arm": (1, 0, 0), "Elbow_Flange": (0, 1, 0)}


def quat_from_to(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    v, c = np.cross(a, b), np.dot(a, b)
    if c < -0.999999:
        return np.array([0, 1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 0, 1.0, 0])
    q = np.array([1 + c, *v])
    return q / np.linalg.norm(q)


def cad_scene(placements, size):
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 3 * size, 2 * size
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.worldbody.add_light(pos=[0.4, -0.4, 0.8], dir=[-0.4, 0.4, -1], diffuse=[0.45, 0.45, 0.45])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=[0.92, 0.93, 0.95, 1])
    for k, (path, pos, quat, rgba) in enumerate(placements):
        spec.add_mesh(name=f"m{k}", file=str(path), scale=[1e-3] * 3)
        b = spec.worldbody.add_body(pos=pos, quat=quat)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{k}", rgba=rgba)
    m = spec.compile()
    return m, mujoco.MjData(m)


def render_cad(size):
    import trimesh
    prev = CAD_DIR / "previews"
    prev.mkdir(exist_ok=True)
    # print orientation: rotate each part so its print-up is +z, sit it on the plate, lay out on a grid
    placements, x = [], 0.0
    for name in PARTS:
        path = CAD_DIR / "stl" / f"{name}.stl"
        t = trimesh.load(path)
        # UP is in W; the STLs are in the CAD frame (W turned by HAND_YAW), and the adapter prints on its flange top (CAD y)
        up = (0, -1, 0) if name == "R-Wrist_Hand_Adapter" else wd.R_W2CAD @ np.asarray(UP[name], float)
        q = quat_from_to(up, (0, 0, 1))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        v = t.vertices @ R.reshape(3, 3).T / 1000
        lo, hi = v.min(0), v.max(0)
        pos = np.array([x - lo[0], -0.5 * (lo[1] + hi[1]), -lo[2]])
        placements.append((path, pos, q, [0.86, 0.87, 0.89, 1] if "Forearm" in name else [0.35, 0.55, 0.85, 1]))
        x += (hi[0] - lo[0]) + 0.02
    m, d = cad_scene(placements, size)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, size, int(size * 2.4))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [x / 2, 0, 0.04]
    cam.azimuth, cam.elevation, cam.distance = 90, -28, 0.52
    r.update_scene(d, cam)
    img = label(r.render(), "printed parts in their print orientation (bed = floor): " + ", ".join(PARTS))
    Image.fromarray(img).save(prev / "parts_print_orientation.jpg", quality=88)
    # exploded assembly in the hand's frame (forearm parts at their place, gimbal parts lifted)
    off = {"R-Wrist_Hand_Adapter": 0.075, "Wrist_Gimbal_Ring": 0.045, "R-Wrist_Fork": 0.015, "R-Forearm_Back": -0.0,
           "R-Wrist_Servo_Frame": 0.0, "Elbow_Flange": -0.03}
    lat = {"R-Forearm_Back": (0.0, 0.0, 0.07), "R-Wrist_Servo_Frame": (-0.0, 0.0, -0.075)}
    placements = []
    cols = {"R-Wrist_Hand_Adapter": [0.35, 0.55, 0.85, 1], "Wrist_Gimbal_Ring": [0.9, 0.55, 0.2, 1], "R-Wrist_Fork": [0.6, 0.62, 0.66, 1],
            "R-Forearm_Front": [0.88, 0.88, 0.9, 1], "R-Forearm_Back": [0.95, 0.95, 0.96, 1], "R-Wrist_Servo_Frame": [0.4, 0.62, 0.42, 1],
            "Elbow_Flange": [0.3, 0.3, 0.32, 1], "Wrist_Horn_Arm": [0.9, 0.78, 0.2, 1]}
    # forearm upright (as in the sim): CAD -> root -> world
    Rq = np.zeros(4)
    mujoco.mju_mulQuat(Rq, wd.WORLD_QUAT, wd.PLATE_QUAT)
    Rw = np.zeros(9)
    mujoco.mju_quat2Mat(Rw, wd.WORLD_QUAT)
    Rw = Rw.reshape(3, 3)
    for name in PARTS + ["ref/STS3215_motor1", "ref/STS3215_motor2", "ref/Wrist_Horn_Arm_motor2_placed"]:
        path = CAD_DIR / "stl" / f"{name}.stl"
        dz = off.get(name, 0.0)
        dl = np.array(lat.get(name, (0, 0, 0)))
        # CAD z (back) = root -x; CAD y (up) = root z
        pos = Rw @ (wd.PLATE_POS + np.array([-dl[2], 0, 0]) + dz * wd.dir_w_to_root([0, 1, 0])) + np.array([0, 0, 0.29])
        rgba = cols.get(name, [0.15, 0.15, 0.17, 1] if "STS" in name else [0.9, 0.78, 0.2, 1])
        placements.append((path, pos, Rq, rgba))
    m, d = cad_scene(placements, size)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, int(size * 1.4), int(size * 1.4))
    tiles = []
    for az in (135, 45):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.2]
        cam.azimuth, cam.elevation, cam.distance = az, -18, 0.62
        r.update_scene(d, cam)
        tiles.append(r.render())
    Image.fromarray(label(np.concatenate(tiles, 1), "exploded: hand adapter (blue), ring (orange), fork, forearm front/back, servo frame (green), "
                                                     "2 x STS3215 + horn arms, elbow flange")).save(prev / "exploded.jpg", quality=88)
    print("wrote", (prev / "parts_print_orientation.jpg").relative_to(REPO), (prev / "exploded.jpg").relative_to(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--no-cad", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    args = ap.parse_args()
    if not args.no_sim:
        render_sim(args.size)
    if not args.no_cad:
        render_cad(args.size)


if __name__ == "__main__":
    main()
