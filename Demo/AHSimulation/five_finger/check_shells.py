#!/usr/bin/env python3
"""Clearance between the palm/top shells and every digit (they are not in the MuJoCo export).

The shells are added to the palm at the plate's pose (all CAD parts share one part-studio frame),
each digit is settled on an N x N grid over its two motors and the moving parts are compared with
each shell; palm-fixed parts (servos, finger frames) and the plate are compared once.

Usage: python check_shells.py --model AH_Right   --shells stock   (stock hand, stock shells)
       python check_shells.py --model AH_Right_5F --shells 5f     (cad/five_finger shells)
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_5f_model import SIM_DIR, Group, Hand, fmt_pose, geom_label, group_distance, log, posed, settle_grid  # noqa: E402

REPO = HERE.parent.parent.parent
SHELLS = {
    "stock": {"palm_shell": REPO / "cad/stl/Amazing Hand Parts - R-Palm_Shell.stl",
              "top_shell": REPO / "cad/stl/Amazing Hand Parts - R-Top_Shell.stl"},
    "5f": {"palm_shell": REPO / "cad/five_finger/stl/R-Palm_Shell_5F.stl",
           "top_shell": REPO / "cad/five_finger/stl/R-Top_Shell_5F.stl"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="AH_Right_5F")
    ap.add_argument("--shells", choices=SHELLS, default="5f")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--coarse", type=float, default=1.0)
    ap.add_argument("--fine", type=float, default=0.3)
    args = ap.parse_args()

    h = Hand(SIM_DIR / args.model / "mjcf", SHELLS[args.shells])
    coarse, fine = args.coarse * 1e-3, args.fine * 1e-3
    touch = 0.75 * fine
    log(f"model {args.model} with {args.shells} shells; digits {h.digits}")
    poses, _ = settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, args.grid))
    keys = list(poses)
    shells = {n: {sp: Group(h, [h.part_geom[n]], sp) for sp in (coarse, fine)} for n in SHELLS[args.shells]}
    mujoco.mj_kinematics(h.m, h.d)
    shell_pose = {n: g[fine].poses(h.d) for n, g in shells.items()}

    log("palm-fixed parts vs shells (bolted parts may touch by design):")
    fixed = {"plate": [h.part_geom["r_hand_plate"]], "wrist interface": [h.part_geom["r_wrist_interface"]]}
    fixed.update({f"{dig} servos+frames": h.geoms[dig][1] for dig in h.digits})
    for fname, geoms in fixed.items():
        gf = Group(h, geoms, fine)
        out = []
        for n, g in shells.items():
            dist, who = group_distance(gf, gf.poses(h.d), g[fine], shell_pose[n], 0.005)
            out.append(f"{n} " + (f"{dist * 1e3:.2f} mm [{geom_label(h, who[0])}]" if who else "> 5 mm")
                       + (" TOUCH/OVERLAP" if dist < touch else ""))
        log(f"  {fname:<22} " + ", ".join(out))

    log(f"moving parts vs shells over the {len(keys)} grid poses:")
    for dig in h.digits:
        g = {sp: Group(h, h.geoms[dig][0], sp) for sp in (coarse, fine)}
        pa = posed(h, g[coarse], poses)
        for n, gs in shells.items():
            res = []
            for p in pa:
                dist, who = group_distance(g[coarse], p, gs[coarse], shell_pose[n], 0.005)
                if touch <= dist < 0.003:
                    dist, who = group_distance(g[fine], p, gs[fine], shell_pose[n], dist + coarse)
                res.append((dist, who))
            i = int(np.argmin([r[0] for r in res]))
            hits = [keys[k] for k, r in enumerate(res) if r[0] < touch]
            log(f"  {dig:<6} vs {n:<10} min {res[i][0] * 1e3:5.2f} mm at {fmt_pose(keys[i])}"
                + (f" [{geom_label(h, res[i][1][0])}]" if res[i][1] else "")
                + f"; touching at {len(hits)}/{len(res)} poses" + (f": {', '.join(fmt_pose(k) for k in hits[:8])}" if hits else ""))


if __name__ == "__main__":
    main()
