#!/usr/bin/env python3
"""Mass model of the 5-digit AmazingHand (AH_Right_5F + 5F shells and grip pads).

The MJCF inertials are not usable for wrist torques (the palm root has 1e-9 kg and the onshape
finger inertials are rough), so every mesh of the hand is given a material and its mass properties
are integrated from the triangle mesh (trimesh, watertight meshes), at the pose of the model.

Assumptions (stated in the report and README):
  * printed PLA parts: effective density 1.15 g/cm3 (PLA 1.24 g/cm3 at ~93 % fill: the parts are small
    and mostly walls). This value is calibrated so that the same model gives the stock hand (AH_Right +
    stock shells) the 400 g stated in the README (1.00 g/cm3 would give 375 g);
  * TPU parts (finger shells, palm shell, grip pads): 1.10 g/cm3 (TPU 1.20, thin walls, ~90 % fill);
  * steel (pins, screws, rods, washers, ball joints' balls): 7.85 g/cm3;
  * igus bushings: 1.46 g/cm3;
  * SCS0009 servos: 13.2 g each (reseller spec), distributed over the servo mesh volume;
  * hardware that is not modelled (servo and frame screws, 10 servo cables + bus cable, shell screws):
    +30 g lumped at the palm plate's centroid.

Usage: python hand_mass_model.py [--model AH_Right_5F|AH_Right] [--no-shells]
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "five_finger"))
from check_shells import shells  # noqa: E402
from verify_5f_model import DEMO_POSES_5F, SIM_DIR, Hand, pose_ctrl  # noqa: E402

DENS = {"pla": 1.15, "tpu": 1.10, "steel": 7.85, "igus": 1.46}       # g/cm3
SCS0009_G = 13.2
MISC_G = 30.0


def material(mesh_name, geom_name):
    n = (geom_name or "") + "|" + mesh_name
    if "scs0009" in n:
        return "servo"
    if any(k in n for k in ("shell", "pad")) and "top_shell" not in n:
        return "tpu"
    if "bushing" in n:
        return "igus"
    if any(k in n for k in ("pin", "screw", "std00", "m2_rod", "washer", "rotule_ball", "spacer")):
        return "steel"
    return "pla"


def mesh_props(m, g, cache):
    """(volume m3, com in geom frame, inertia about com for unit density, in geom frame) of geom g's mesh."""
    mid = int(m.geom_dataid[g])
    if mid not in cache:
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        t = trimesh.Trimesh(v, f, process=False)
        if t.volume < 0:
            t.invert()
        cache[mid] = (t.volume, t.center_mass.copy(), t.moment_inertia.copy())
    return cache[mid]


def hand_geoms(m):
    """Every mesh geom that belongs to the hand (the subtree of the hand root, plus extra palm parts)."""
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_interface")
    def under(b):
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False
    return [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and under(m.geom_bodyid[g])]


def mass_model(m, d, geoms=None):
    """Mass properties of the hand at the current pose: dict(mass kg, com (3,) world, inertia (3,3) about
    com in world axes, parts list). Uses d.geom_xpos / geom_xmat (call mj_kinematics first)."""
    geoms = hand_geoms(m) if geoms is None else geoms
    cache, parts = {}, []
    M, S, I_O = 0.0, np.zeros(3), np.zeros((3, 3))
    servo_vol = None
    for g in geoms:
        vol, c, I1 = mesh_props(m, g, cache)
        mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
        mat = material(mesh, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        if mat == "servo":
            mass = SCS0009_G / 1000.0
            servo_vol = vol
        else:
            mass = DENS[mat] * 1000.0 * vol           # g/cm3 -> kg/m3
        R = d.geom_xmat[g].reshape(3, 3)
        cw = d.geom_xpos[g] + R @ c
        Ic = mass / vol * (R @ I1 @ R.T)
        I_O += Ic + mass * (cw @ cw * np.eye(3) - np.outer(cw, cw))
        M += mass
        S += mass * cw
        parts.append((mesh, mat, mass, vol))
    # unmodelled hardware and cables at the plate centroid
    plate = [g for g in geoms if "hand_plate" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])][0]
    vol, c, _ = mesh_props(m, plate, cache)
    cw = d.geom_xpos[plate] + d.geom_xmat[plate].reshape(3, 3) @ c
    mm = MISC_G / 1000.0
    I_O += mm * (cw @ cw * np.eye(3) - np.outer(cw, cw))
    M += mm
    S += mm * cw
    parts.append(("misc screws + cables (lumped)", "misc", mm, 0.0))
    com = S / M
    I_c = I_O - M * (com @ com * np.eye(3) - np.outer(com, com))
    return dict(mass=M, com=com, inertia=I_c, inertia_origin=I_O, parts=parts)


def summarize(parts):
    by = {}
    for mesh, mat, mass, vol in parts:
        key = (mat, mesh)
        n, ms = by.get(key, (0, 0.0))
        by[key] = (n + 1, ms + mass)
    per_mat = {}
    for (mat, _), (n, ms) in by.items():
        per_mat[mat] = per_mat.get(mat, 0.0) + ms
    return by, per_mat


def settle(h, pose):
    h.reset()
    if pose is None:
        return
    target = pose_ctrl(h, pose)
    for k in range(600):
        h.d.ctrl[:] = target * min(1.0, (k + 1) / 300)
        mujoco.mj_step(h.m, h.d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="AH_Right_5F")
    ap.add_argument("--no-shells", action="store_true")
    args = ap.parse_args()
    five = args.model.endswith("_5F")
    extra = None if args.no_shells else shells("5f" if five else "stock", "R")
    h = Hand(SIM_DIR / args.model / "mjcf", extra)
    sys.path.insert(0, str(HERE.parent.parent.parent / "cad" / "five_finger" / "wrist"))
    import wrist_design as wd
    to_cad = lambda x: (wd.R_CAD2ROOT.T @ (x - wd.PLATE_POS)) * 1000
    for name, pose in [("keyframe", None), ("OpenHand", DEMO_POSES_5F["OpenHand"]), ("CloseHand", DEMO_POSES_5F["CloseHand"])]:
        settle(h, pose)
        mujoco.mj_kinematics(h.m, h.d)
        r = mass_model(h.m, h.d)
        print(f"{args.model} {'+ shells ' if extra else ''}[{name}]: mass {r['mass'] * 1000:.1f} g, "
              f"CoM root {np.round(r['com'] * 1000, 1)} mm = CAD {np.round(to_cad(r['com']), 1)} mm, "
              f"{np.linalg.norm(to_cad(r['com']) - wd.C_CAD):.1f} mm from the wrist centre")
        if name == "keyframe":
            by, per_mat = summarize(r["parts"])
            for (mat, mesh), (n, ms) in sorted(by.items(), key=lambda kv: -kv[1][1]):
                print(f"    {mat:6s} {n:3d} x {mesh:60s} {ms * 1000:7.1f} g")
            print("    per material: " + ", ".join(f"{k} {v * 1000:.1f} g" for k, v in per_mat.items()))
            print("    inertia about CoM (root axes, kg m2):\n", np.round(r["inertia"], 7))


if __name__ == "__main__":
    main()
