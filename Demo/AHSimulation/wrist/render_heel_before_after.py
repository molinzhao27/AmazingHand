#!/usr/bin/env python3
"""Before/after render of the slim wrist connector: the heel of the hand and the wrist gimbal at the neutral pose,
without the palm shells, palm side, radial side and back.

"before" swaps the stock wrist interface (AH_Right/mjcf/assets/r_wrist_interface.stl) back in as the hand root's mesh
and the d56 hand adapter of a git revision (`git show <rev>:cad/five_finger/wrist/stl/R-Wrist_Hand_Adapter.stl`; the
default ff5a133 is the last one with the d56 flange) in as the adapter's mesh. "after" is the model as built (the
adapter's interface plate in the hand, its d42 neck on the wrist). Everything else is the same model.

Usage: python render_heel_before_after.py [--before-rev ff5a133] [--model AH_Right_5F_Wrist]
                                          [--out ../five_finger/renders/before_after_heel.jpg]
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

# (label, camera direction in W (from the target towards the camera), elevation deg)
VIEWS = [("palm side", (0, 0, -1), -5), ("radial side", (-1, 0, 0), -5), ("back", (0, 0, 1), -5)]


def render(model, swap, size):
    spec = mujoco.MjSpec.from_file(str(model / "scene.xml"))
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
    r = mujoco.Renderer(m, size, size)
    tiles = []
    for _, v, el in VIEWS:
        f = Rw @ wd.dir_w_to_root(-np.asarray(v, float))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = Rw @ wd.w_to_root([0, 40, -5])
        cam.azimuth, cam.elevation, cam.distance = np.degrees(np.arctan2(f[1], f[0])), el, 0.30
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        r.update_scene(d, cam, opt)
        tiles.append(r.render())
    return tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-rev", default="ff5a133", help="a revision with the d56 adapter")
    ap.add_argument("--model", default="AH_Right_5F_Wrist")
    ap.add_argument("--out", type=Path, default=HERE.parent / "five_finger" / "renders" / "before_after_heel.jpg")
    ap.add_argument("--size", type=int, default=420)
    args = ap.parse_args()
    model = HERE.parent / "AHSimulation" / args.model / "mjcf"
    args.out = args.out.resolve()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "R-Wrist_Hand_Adapter.stl"
        old.write_bytes(subprocess.run(["git", "show", f"{args.before_rev}:cad/five_finger/wrist/stl/R-Wrist_Hand_Adapter.stl"],
                                       cwd=REPO, check=True, capture_output=True).stdout)
        before = {"wrist_r_wrist_hand_adapter_neck": old,
                  "r_wrist_interface": HERE.parent / "AHSimulation" / "AH_Right" / "mjcf" / "assets" / "r_wrist_interface.stl"}
        for tag, sw in (("before (stock wrist interface + d56 adapter)", before), ("after (one adapter, d42 neck)", {})):
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
