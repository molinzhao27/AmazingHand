#!/usr/bin/env python3
"""Thumb vs forearm: where the over-extended thumb reaches the forearm, and the interlock rule.

For thumb flexion (m1 - m2)/2 from -90 to -50 deg (abduction -30 / 0 / +30, other fingers in OpenHand)
and palm-ward wrist flexion 30..60 deg (deviation across the operating ROM), prints the minimum
clearance between the hand (with 5F shells) and the forearm parts. Backs wrist_design.thumb_flexion_min().

Usage: python check_thumb_rule.py      (writes reports/thumb_vs_forearm.txt)
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_wrist_model as V  # noqa: E402
from verify_5f_model import Group, group_distance  # noqa: E402
from wrist_sim import DEMO_POSES_5F, WristModel, wd  # noqa: E402


def main():
    w = WristModel("scene_shells.xml")
    m, d = w.m, w.d
    hr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_interface")
    hand = [g for g in range(m.ngeom) if V._under(m, m.geom_bodyid[g], hr) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    fa = V.body_geoms(m, ["wrist_forearm"], ("f683",))
    H = type("H", (), {"m": m})()
    A, B = Group(H, hand, 1e-3), Group(H, fa, 1e-3)
    lines = ["Min clearance (mm) hand (with 5F shells) vs forearm parts, over the operating-ROM deviation range,",
             "for palm-ward wrist flexion 30 / 40 / 50 / 60 deg (other fingers OpenHand; samples every 1 mm):"]
    for flex in (-90, -80, -70, -60, -50):
        for abd in (-30, 0, 30):
            m1, m2 = abd + flex, abd - flex
            if abs(m1) > 90 or abs(m2) > 90:
                continue
            c = w.ctrl_for(DEMO_POSES_5F["OpenHand"])
            c[w.finger_ctrl["thumb"]] = np.radians([m1, m2])
            w.reset()
            w.move(c, 0.4, 0.3)
            q = d.qpos.copy()
            row = []
            for p in (30, 40, 50, 60):
                lim = np.degrees(wd.rom_dev_limit(np.radians(p)))
                best = 99.0
                for s in np.linspace(-lim, lim, 7):
                    d.qpos[:] = q
                    V.set_wrist_pose(w, np.radians(p), np.radians(s))
                    dist, _ = group_distance(A, A.poses(d), B, B.poses(d), 0.02)
                    best = min(best, dist * 1e3)
                row.append(f"{p} deg: {best:5.1f}")
            lines.append(f"  thumb abduction {abd:+3d}, flexion {flex:+3d} (m1 {m1:+3d}, m2 {m2:+3d}):  " + "   ".join(row))
    lim = np.degrees(wd.thumb_flexion_min(np.radians(60.0)))
    lines.append(f"Rule (wrist_design.thumb_flexion_min, PALM_EXT = {wd.PALM_EXT:g} mm): " + (
        "no limit, the thumb clears the forearm everywhere." if lim <= -90 else
        f"thumb flexion >= {lim:.0f} deg whenever wrist flexion > 40 deg (OpenHand uses -35)."))
    out = "\n".join(lines)
    print(out)
    (HERE / "reports" / "thumb_vs_forearm.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
