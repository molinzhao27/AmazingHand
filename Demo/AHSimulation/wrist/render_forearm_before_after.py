#!/usr/bin/env python3
"""Before/after render of the forearm restyle: the hand on the forearm at the neutral pose (keyframe), with the
forearm parts of a git revision (before) and of the working tree (after), palm side, radial side and 3/4.

The "before" forearm (R-Forearm_Front / _Back, R-Wrist_Fork, Elbow_Flange) is read from `git show <rev>:...` and swapped
into the same sim model, so the hand, the shells and the camera are identical in both rows.

Usage: python render_forearm_before_after.py [--before-rev HEAD] [--model AH_Right_5F_Slim_Wrist]
                                             [--out ../five_finger/renders/before_after_forearm.jpg]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wrist_sim import REPO, wd  # noqa: E402

PARTS = {"wrist_r_forearm_front": "R-Forearm_Front", "wrist_r_forearm_back": "R-Forearm_Back",
         "wrist_r_wrist_fork": "R-Wrist_Fork", "wrist_elbow_flange": "Elbow_Flange"}
# (label, camera direction in W (from the target towards the camera), elevation deg)
VIEWS = [("palm side", (0, 0, -1), -4), ("radial side", (-1, 0, 0), -4), ("3/4 palm, ulnar", (0.55, 0, -0.84), -14)]


def render(model, swap, size):
    spec = mujoco.MjSpec.from_file(str(model / "scene_shells.xml"))
    for msh in spec.meshes:
        if msh.name in swap:
            msh.file = str(swap[msh.name])
    m = spec.compile()
    m.vis.global_.offwidth, m.vis.global_.offheight = 4 * size, 4 * size
    m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"), 3] = 0
    for b in range(m.nbody):                  # hide the mocap target spheres
        if m.body_mocapid[b] >= 0:
            m.geom_rgba[m.body_geomadr[b]:m.body_geomadr[b] + m.body_geomnum[b], 3] = 0
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    Rw = np.zeros(9)
    mujoco.mju_quat2Mat(Rw, wd.WORLD_QUAT)
    Rw = Rw.reshape(3, 3)
    r = mujoco.Renderer(m, int(size * 1.45), size)
    tiles = []
    for _, v, el in VIEWS:
        # the camera looks along -v: azimuth of -v in the world
        f = Rw @ wd.dir_w_to_root(-np.asarray(v, float))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = Rw @ wd.w_to_root([0, -20, -5])
        cam.azimuth, cam.elevation, cam.distance = np.degrees(np.arctan2(f[1], f[0])), el, 0.70
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        r.update_scene(d, cam, opt)
        tiles.append(r.render())
    return tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-rev", default="HEAD")
    ap.add_argument("--model", default="AH_Right_5F_Slim_Wrist")
    ap.add_argument("--out", type=Path, default=HERE.parent / "five_finger" / "renders" / "before_after_forearm.jpg")
    ap.add_argument("--size", type=int, default=480)
    args = ap.parse_args()
    model = HERE.parent / "AHSimulation" / args.model / "mjcf"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        swap = {}
        for mesh, stem in PARTS.items():
            blob = subprocess.run(["git", "show", f"{args.before_rev}:cad/five_finger/wrist/stl/{stem}.stl"], cwd=REPO,
                                  check=True, capture_output=True).stdout
            (Path(tmp) / f"{stem}.stl").write_bytes(blob)
            swap[mesh] = Path(tmp) / f"{stem}.stl"
        for tag, sw in (("before", swap), ("after", {})):
            tiles = []
            for (name, _, _), img in zip(VIEWS, render(model, sw, args.size)):
                im = Image.fromarray(img)
                ImageDraw.Draw(im).text((8, 6), f"{tag} - {name}", fill=(255, 255, 255))
                tiles.append(np.asarray(im))
            rows.append(np.concatenate(tiles, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(args.out, quality=88)
    print("wrote", args.out.relative_to(REPO))


if __name__ == "__main__":
    main()
