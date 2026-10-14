#!/usr/bin/env python3
"""Verify the ultimate finger (model/finger_uf.xml) against its design, the stock finger and the 3-phalanx concept.

Reuses the checks of three_segment/verify_3seg_finger.py (imported, unchanged):
  1. structure, masses;  2. kinematics along the flexion line (sim vs planar design, coupling, ranges);
  3. stability (each actuator alone, range corners, 60 abrupt random full-range steps): NaN, warnings, loop errors;
  4. 25 x 25 grid over (m1, m2): joint ranges, loop errors, motor tracking, fingertip workspace;
  5. self-collision of every part over the full joint range (exact mesh booleans, manifold3d): 13 x 13 grid + the
     flexion line; pins are excused only against the two parts they join (explicit list, so the A pin is checked in
     P2's slot); the stock finger is the baseline;
and adds
  6. statics and strength for the thinner parts: holdable fingertip force at SCS0009 stall, every pin force from
     MuJoCo's constraint forces (frictionless), pin-hole bearing in each part, pin bending/shear, Link-arm bending
     along the arm, L2 axial stress and buckling, phalanx bending at the critical sections; the servo torque limit
     that keeps all of them below the material limits, per material.

Writes reports/verify_finger.txt, data/verify_finger.npz, renders/../figures in cad/five_finger/ultimate_finger/figures.
Usage: python verify_finger.py [--grid 13] [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import manifold3d as mf
import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "three_segment"))
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "slim_fingers"))
import plotstyle as ps  # noqa: E402
import verify_3seg_finger as V3  # noqa: E402
from finger_common import SERVO_STALL_NM, Finger, flexion_ctrl, stock_roles  # noqa: E402
from linkage import Design  # noqa: E402

CAD = REPO / "cad" / "five_finger" / "ultimate_finger"
MODEL_DIR = HERE / "model"
REPORT_DIR, DATA_DIR, FIG_DIR = HERE / "reports", HERE / "data", CAD / "figures"
LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


V3.out = out   # the imported checks log into this report

NEW_MESHES = {"uf_link", "uf_p1", "uf_p1_shell", "uf_p1_pad", "uf_p2", "uf_p2_shell", "uf_p2_pad", "uf_p3", "uf_p3_shell",
              "uf_p3_pad", "uf_l2", "uf_pin_mcp", "uf_pin_pip", "uf_pin_a", "uf_pin_p4", "uf_pin_dip", "uf_pin_b"}
PIN_BODIES = {"uf_pin_mcp": {"p1", "std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default"},
              "uf_pin_pip": {"p1", "p2"}, "uf_pin_a": {"p1", "l2"}, "uf_pin_p4": {"p2", "rotule_ball"},
              "uf_pin_dip": {"p2", "p3"}, "uf_pin_b": {"p3", "l2"}}
V3.STOCK_EQUIV.update({"uf_link": "link", "uf_p1": "proximal", "uf_p1_shell": "proximal_shell", "uf_p1_pad": "proximal_shell",
                       "uf_p2": "distal", "uf_p2_shell": "distal_shell", "uf_p2_pad": "distal_shell", "uf_p3": "distal",
                       "uf_p3_shell": "distal_shell", "uf_p3_pad": "distal_shell", "uf_l2": "link"})


def roles():
    return dict(frame=("r_wrist_interface", "finger_frame_1"),
                gimbal=("std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default", "gimbal"),
                link=("rotule_ball", "uf_link"), P1=("p1", None), P2=("p2", None), P3=("p3", None), L2=("l2", None))


class Parts(V3.Parts):
    """V3.Parts with the new pins (mesh geoms 'uf_pin_*') excused only against the two parts each one joins."""

    def __init__(self, f, new_meshes):
        super().__init__(f, new_meshes)
        m = f.m
        bname = self.bname
        for it in self.items:
            if it["name"].startswith("uf_pin"):
                it["pin"] = True
        pairs = []
        for i, a in enumerate(self.items):
            for j, b in enumerate(self.items):
                if j <= i or a["body"] == b["body"] or not (a["new"] or b["new"]):
                    continue
                pin = a if a["name"].startswith("uf_pin") else b if b["name"].startswith("uf_pin") else None
                if pin is not None:
                    other = b if pin is a else a
                    if bname(other["body"]) in PIN_BODIES[pin["name"]] or other["name"].startswith("uf_pin"):
                        continue
                elif (a["pin"] or b["pin"]) and (a["body"], b["body"]) in self.adj:
                    continue
                pairs.append((i, j))
        self.pairs = pairs
        del m


V3.Parts = Parts

# ---------------------------------------------------------------- strength model
# lateral layout of each pin joint: (part, wall thickness per side mm, number of walls, wall centre |x| mm)
PIN_JOINTS = {
    "PIP": dict(spec="2x12", holder=("P1 fork", 1.75, 2, 5.125), runner=("P2", 2.25, 2, 2.875)),
    "A": dict(spec="2x12", holder=("P1 fork", 1.75, 2, 5.125), runner=("L2 eye", 3.0, 1, 0.0)),
    "P4": dict(spec="2x8", holder=("P2", 2.25, 2, 2.875), runner=("Link eye", 3.0, 1, 0.0)),
    "DIP": dict(spec="2x12", holder=("P2 fork", 1.6, 2, 5.05), runner=("P3", 2.25, 2, 2.875)),
    "B": dict(spec="2x8", holder=("P3", 2.25, 2, 2.875), runner=("L2 eye", 3.0, 1, 0.0)),
}
MATERIALS = {  # design limits (MPa) = typical FDM datasheet strength in the layer plane / ~2 (static, dry)
    "PLA": dict(bearing=25.0, bending=25.0, E=3500.0),
    "PETG": dict(bearing=20.0, bending=22.0, E=2100.0),
    "PA-CF": dict(bearing=40.0, bending=40.0, E=6500.0),
}
PIN_LIMIT = dict(bending=600.0, shear=300.0)   # hardened dowel ISO 8734 (~1500 MPa UTS) with margin


def pin_stress(name, F):
    j = PIN_JOINTS[name]
    out_ = {}
    for role in ("holder", "runner"):
        part, t, n, xc = j[role]
        out_[part] = F / (2.0 * t * n)   # projected-area bearing: F / (d * t * n)
    (_, th, nh, xh), (_, tr, nr, xr) = j["holder"], j["runner"]
    if nr == 1:   # centred eye between two walls: uniform load over the eye, reactions at the wall centres
        arm = xh - tr / 4 if nh == 2 else xr
    else:
        arm = abs(xh - xr)
    M = F / 2 * arm
    out_["pin bending"] = M / (np.pi * 2.0 ** 3 / 32)
    out_["pin shear"] = F / (2 * np.pi * 1.0 ** 2)
    return out_


def site_world(f, body, yz_mm, x_mm=0.0):
    b = mujoco.mj_name2id(f.m, mujoco.mjtObj.mjOBJ_BODY, body)
    p = np.array([x_mm, yz_mm[0], yz_mm[1]]) * 1e-3
    return f.d.xpos[b] + f.d.xmat[b].reshape(3, 3) @ p, f.d.xmat[b].reshape(3, 3)


def joint_force(f, body):
    """Force (N, world) the parent applies to `body`'s subtree through its joint (= the pin force on that body)."""
    b = mujoco.mj_name2id(f.m, mujoco.mjtObj.mjOBJ_BODY, body)
    return f.d.cfrc_int[b, 3:].copy()


def bending_about_x(R, r_from, r_to, F):
    """Moment (N mm) about the body's x axis of force F (world, N) at r_to about point r_from (world, m)."""
    return float((R.T @ np.cross(r_to - r_from, F))[0]) * 1e3


def section_props(mesh_path, y):
    sys.path.insert(0, str(REPO / "cad" / "five_finger" / "slim_fingers"))
    import make_slim_fingers as SF
    m = trimesh.load(mesh_path)
    poly = SF.section_poly(m, y)
    p = SF.section_props(poly)
    zs = np.asarray(poly.exterior.coords)[:, 1] if hasattr(poly, "exterior") else np.concatenate(
        [np.asarray(g.exterior.coords)[:, 1] for g in poly.geoms])
    c = max(zs.max() - p["zc"], p["zc"] - zs.min())
    return p["area"], p["Ixx"], c


def arm_profile(design):
    """Link arm: (s from the Link pivot along the arm, in-plane height h) from the CAD outline (Link frame)."""
    sys.path.insert(0, str(CAD))
    m = trimesh.load(CAD / "stl" / "UF_Link.stl")
    OL = np.array([0.0, -6.0])
    P4L = np.array(design["geometry"]["P4_in_link_frame"])
    u = (P4L - OL) / np.linalg.norm(P4L - OL)
    n = np.array([-u[1], u[0]])
    rows = []
    for s in (8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 38.0, 41.0):
        c = OL + s * u
        sec = m.section(plane_origin=[0, c[0], c[1]], plane_normal=[0, u[0], u[1]])
        if sec is None:
            continue
        v = sec.vertices
        v = v[np.abs(v[:, 0]) <= 1.6]
        h = (v[:, 1:] - c) @ n
        rows.append((s, float(np.ptp(h)), float(np.ptp(v[:, 0]))))
    return rows, OL, P4L, u


def statics(f3, fsx, design, f_s):
    out("\n-- 6. statics and strength (thin parts)")
    # 6a. holdable fingertip force at stall (Jacobian transpose), straight .. full travel
    F = np.arange(-60, 90.01, 2.5)
    up, _ = V3.sweep_line(f3, F)
    hold = []
    for r in up:
        J = r["J"]
        n_pad = -r["R_pad"][:, 2]
        hold.append((r["f"], SERVO_STALL_NM / np.abs(J.T @ n_pad).max()))
    hold = np.array(hold)
    # 6b. loads for a 5 N push on the fingertip pad, per pose
    arm_rows, OL, P4L, u = arm_profile(design)
    stl = CAD / "stl"
    sec_p1 = section_props(stl / "UF_P1_Proximal.stl", 25.0)
    sec_p2 = section_props(stl / "UF_P2_Middle.stl", 0.5 * (design["geometry"]["PIP"][0] + design["geometry"]["DIP"][0]))
    y3 = design["geometry"]["DIP"][0] + 4.0
    sec_p3 = section_props(stl / "UF_P3_Distal.stl", y3)
    g = design["geometry"]
    fvals = [round(f_s), -30, 0, 30, 45, 60, 75]
    rows = []
    out("  loads for a 5 N push on the fingertip pad, frictionless pins (MuJoCo constraint forces), straight .. f = +75:")
    out("     f   motor torque  |  pin forces (N): MCP    P4   PIP     A   DIP     B   | Link-arm moment at s=8 / 20 / 35 (N mm)"
        " | L2 axial (N) | P1 / P2 / P3 bending moment (N mm)")
    for fv in fvals:
        tau, fc, jf = V3.loads_for_pad_force(f3, fv, 5.0)
        # re-run the same hold to read vectors (loads_for_pad_force restores friction afterwards; redo in place)
        m, d = f3.m, f3.d
        fl, dmp = m.dof_frictionloss.copy(), m.dof_damping.copy()
        m.dof_frictionloss[:] = 0.0
        m.dof_damping[:] = np.maximum(dmp, 2e-4)
        f3.reset()
        f3.move(*flexion_ctrl(fv), ramp=0.5, hold=0.3, check=False)
        pad = f3.frame("P3")
        nvec = pad[:3, 2]
        b = m.site_bodyid[f3.tip]
        d.xfrc_applied[b, :3] = 5.0 * nvec
        d.xfrc_applied[b, 3:] = np.cross(d.site_xpos[f3.tip] - d.xipos[b], 5.0 * nvec)
        for _ in range(1500):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        F_p4_on_p2 = joint_force(f3, "p2")
        F_pip_on_p1 = joint_force(f3, "p1")
        F_dip_on_p3 = joint_force(f3, "p3")
        F_a_on_l2 = joint_force(f3, "l2")
        # Link arm: force of P2 on the Link at P4 = -F_p4_on_p2; moment about arm sections
        # the Link's CAD frame (body pose x the geom pose as written in the XML; the compiled geom frame of a mesh is
        # its inertia frame, not the CAD frame)
        TL = f3.frame("link")
        Rg, pg = TL[:3, :3], TL[:3, 3]

        def link_pt(yz):
            return pg + Rg @ (np.array([0.0, yz[0], yz[1]]) * 1e-3)
        r_p4 = link_pt(P4L)
        Ms = {}
        for s, h, t in arm_rows:
            Ms[s] = abs(float((Rg.T @ np.cross(r_p4 - link_pt(OL + s * u), -F_p4_on_p2))[0]) * 1e3)
        F_axial = float(np.dot(-F_p4_on_p2, Rg @ np.r_[0, u]))
        # phalanges: moments at their critical sections from the forces distal of them
        rP1 = {k: site_world(f3, "p1", g[k])[0] for k in ("PIP", "A")}
        R1 = site_world(f3, "p1", [0, 0])[1]
        s1 = site_world(f3, "p1", [25.0, 0.0])[0]
        M1 = bending_about_x(R1, s1, rP1["PIP"], F_pip_on_p1) + bending_about_x(R1, s1, rP1["A"], -F_a_on_l2)
        R2 = site_world(f3, "p2", [0, 0])[1]
        s2 = site_world(f3, "p2", [0.5 * (g["PIP"][0] + g["DIP"][0]), 0.0])[0]
        M2 = bending_about_x(R2, s2, site_world(f3, "p2", g["DIP"])[0], -F_dip_on_p3)
        R3 = site_world(f3, "p3", [0, 0])[1]
        s3 = site_world(f3, "p3", [y3, 0.0])[0]
        M3 = bending_about_x(R3, s3, d.site_xpos[f3.tip], 5.0 * nvec)
        d.xfrc_applied[:] = 0
        m.dof_frictionloss[:], m.dof_damping[:] = fl, dmp
        pins = dict(MCP=fc["closing_3 (1)_1"], P4=np.linalg.norm(F_p4_on_p2), PIP=np.linalg.norm(F_pip_on_p1),
                    A=np.linalg.norm(F_a_on_l2), DIP=np.linalg.norm(F_dip_on_p3), B=fc["closing_dip (1)_1"])
        rows.append(dict(f=fv, tau=float(np.abs(tau).max()), pins=pins, arm=Ms, arm_axial=F_axial, L2=pins["A"],
                         M1=abs(M1), M2=abs(M2), M3=abs(M3)))
        out(f"  {fv:+5d}   {np.abs(tau).max():.3f} N m     |  {pins['MCP']:5.1f} {pins['P4']:5.1f} {pins['PIP']:5.1f} "
            f"{pins['A']:5.1f} {pins['DIP']:5.1f} {pins['B']:5.1f}  | {Ms[8.0]:6.0f} / {Ms[20.0]:5.0f} / {Ms[35.0]:5.0f}"
            f"           | {pins['A']:5.1f}        | {abs(M1):5.0f} / {abs(M2):4.0f} / {abs(M3):4.0f}")
    out("  (stock finger for comparison: " + "; ".join(
        f"f {fv:+d}: P4 {V3.loads_for_pad_force(fsx, fv, 5.0)[2]['parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999']:.0f} N"
        for fv in (-38, 0, 30)) + ")")
    # 6c. stresses per 5 N and per unit motor torque
    out("\n  section properties used: Link arm 3.0 mm thick, in-plane height h(s) from the CAD: "
        + ", ".join(f"s {s:.0f}: {h:.2f}" for s, h, _ in arm_rows))
    out(f"  P1 at y 25: A {sec_p1[0]:.1f} mm2, I {sec_p1[1]:.0f} mm4, c {sec_p1[2]:.2f}; P2 mid-shaft: A {sec_p2[0]:.1f}, I {sec_p2[1]:.0f}, "
        f"c {sec_p2[2]:.2f}; P3 at y {y3:.1f}: A {sec_p3[0]:.1f}, I {sec_p3[1]:.0f}, c {sec_p3[2]:.2f}")
    L2len = g["coupler_L2"]
    I_l2 = min(3.5 * 3.0 ** 3 / 12, 3.0 * 3.5 ** 3 / 12)
    checks = []   # (label, material class, stress per 5 N pad at each f)
    for r in rows:
        st = {}
        for pn in ("PIP", "A", "P4", "DIP", "B"):
            for k, v in pin_stress(pn, r["pins"][pn]).items():
                st[f"{pn}: {k}"] = v
        for s, h, t in arm_rows:
            st[f"Link arm s={s:.0f}"] = 6 * r["arm"][s] / (3.0 * h ** 2) + abs(r["arm_axial"]) / (3.0 * h)
        st["L2 axial"] = r["L2"] / (3.0 * 3.5)
        st["P1 bending y25"] = r["M1"] * sec_p1[2] / sec_p1[1]
        st["P2 bending mid"] = r["M2"] * sec_p2[2] / sec_p2[1]
        st["P3 bending"] = r["M3"] * sec_p3[2] / sec_p3[1]
        r["stress"] = st
    keys = list(rows[0]["stress"])
    out("\n  stresses (MPa) for a 5 N fingertip force, by pose (bearing = F / (d t n); pins: hardened dowels):")
    out("  " + f"{'':<28}" + "".join(f"f {r['f']:+4d}  " for r in rows))
    for k in keys:
        out("  " + f"{k:<28}" + "".join(f"{r['stress'][k]:7.1f} " for r in rows))
    return rows, hold, keys, (L2len, I_l2)


def torque_limit(rows, hold, keys, l2info, f_s):
    """Servo torque limit so that every stress stays below the material limit for contact up to f_max."""
    out("\n  -- servo torque limit")
    out("  At a torque limit tau_lim the pad force is F(f) = tau_lim / k(f), k = motor torque per N of pad force (from the 5 N case);"
        " every stress scales with F. The limit for contact anywhere in straight..f_max is the minimum over f.")
    part_of = {}
    for k in keys:
        if k.startswith("Link arm"):
            part_of[k] = "Link"
        elif "pin" in k:
            part_of[k] = "pin"
        elif k.startswith("L2"):
            part_of[k] = "L2"
        elif "P1" in k:
            part_of[k] = "P1"
        elif "P2" in k:
            part_of[k] = "P2"
        elif "P3" in k:
            part_of[k] = "P3"
        elif "Link eye" in k:
            part_of[k] = "Link"
        elif "L2 eye" in k:
            part_of[k] = "L2"
        else:
            part_of[k] = "?"
    plans = {"all PLA": dict(Link="PLA", L2="PLA", P1="PLA", P2="PLA", P3="PLA"),
             "all PETG": dict(Link="PETG", L2="PETG", P1="PETG", P2="PETG", P3="PETG"),
             "recommended: Link, L2, P1, P2 in PA-CF; P3 in PLA": dict(Link="PA-CF", L2="PA-CF", P1="PA-CF", P2="PA-CF",
                                                                       P3="PLA")}
    res = {}
    for plan, mats in plans.items():
        lims = []
        for r in rows:
            k_ = r["tau"] / 5.0   # N m per N of pad force
            worst, who = np.inf, None
            for key, sv in r["stress"].items():
                part = part_of[key]
                if part == "pin":
                    lim = PIN_LIMIT["shear"] if "shear" in key else PIN_LIMIT["bending"]
                else:
                    mat = MATERIALS[mats[part]]
                    lim = mat["bearing"] if (":" in key) else mat["bending"]
                F_allow = lim / (sv / 5.0)
                if F_allow < worst:
                    worst, who = F_allow, key
            lims.append((r["f"], worst, who, worst * k_))
        res[plan] = lims
        out(f"  {plan}:")
        for f, Fa, who, tl in lims:
            out(f"    contact at f {f:+4d}: allowable pad force {Fa:6.1f} N (limited by {who}) -> motor torque {tl:.3f} N m "
                f"= {100 * tl / SERVO_STALL_NM:5.1f} % of stall")
    out("\n  one fixed limit (SCS0009 EEPROM max_torque_limit, addr 16, 0..1000 = 0..100 %) that keeps every part below its "
        "limit for contact anywhere from straight up to f_max, and the fingertip force it then allows:")
    out("    plan                                              f_max   limit   pad force straight / f=0 / f=+30 / f=+45 (N)")
    for plan, lims in res.items():
        for fmax in (30, 45, 60, 75):
            sel = [(f, Fa, tl) for f, Fa, _, tl in lims if f <= fmax]
            tl = min(min(t for _, _, t in sel), SERVO_STALL_NM)
            pf = {r["f"]: min(tl / (r["tau"] / 5.0), Fa) for r, (f, Fa, _, _) in zip(rows, lims)}
            out(f"    {plan[:48]:<48}  {fmax:+4d}   {100 * tl / SERVO_STALL_NM:5.1f} %   "
                f"{pf[rows[0]['f']]:5.1f} / {pf[0]:5.1f} / {pf[30]:5.1f} / {pf[45]:5.1f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=13)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-statics", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    design = json.loads((CAD / "design.json").read_text())
    g = design["geometry"]
    dz = Design(g["L1"], g["L2"], g["L3_tip"], g["P4"], g["A"], g["B"], np.radians(design["drive"]["psi_straight_deg"]), g["tip"][1])
    f3 = Finger(MODEL_DIR / "scene_uf.xml", roles())
    fs = Finger(MODEL_DIR / "scene_stock.xml", stock_roles())
    m = f3.m
    out("== ultimate finger: verification (Demo/AHSimulation/ultimate_finger/model/finger_uf.xml)")
    if not args.only_statics:
        out("\n-- 1. structure")
        out(f"  nbody {m.nbody}, njnt {m.njnt}, nq {m.nq}, nv {m.nv}, nu {m.nu}, neq {m.neq}, ngeom {m.ngeom} "
            f"(stock finger: nbody {fs.m.nbody}, nq {fs.m.nq}, neq {fs.m.neq})")
        for obj, num in ((mujoco.mjtObj.mjOBJ_BODY, m.nbody), (mujoco.mjtObj.mjOBJ_JOINT, m.njnt), (mujoco.mjtObj.mjOBJ_SITE, m.nsite)):
            names = [x for x in (mujoco.mj_id2name(m, obj, i) for i in range(num)) if x]
            assert len(names) == len(set(names))
        out("  names unique; actuators: " + ", ".join(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} "
                                                     f"ctrlrange {np.degrees(m.actuator_ctrlrange[i]).round(1)}" for i in range(m.nu)))
        out("  loop closures: " + "; ".join(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, m.eq_obj1id[e])} <-> "
                                             f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, m.eq_obj2id[e])}" for e in range(m.neq)))
        mass = lambda ff, names: sum(ff.m.body_mass[mujoco.mj_name2id(ff.m, mujoco.mjtObj.mjOBJ_BODY, x)] for x in names)  # noqa
        m3 = mass(f3, ["rotule_ball", "p1", "p2", "p3", "l2"])
        ms = mass(fs, ["rotule_ball", "parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999",
                       "parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b"])
        out(f"  moving finger mass beyond the gimbal (Link + phalanges + coupler + shells + pads + pins): {m3 * 1e3:.1f} g "
            f"(stock {ms * 1e3:.1f} g; 3-phalanx concept 23.5 g)")
        f3.reset()
        out(f"  keyframe (settled at ctrl = 0): max loop error {f3.eq_err().max() * 1e3:.4f} mm")

        out("\n-- 2. kinematics along the flexion line (m1 = f, m2 = -f), settled, up and down")
        F = np.arange(-90, 90.01, 2.5)
        up3, dn3 = V3.sweep_line(f3, F)
        ups, _ = V3.sweep_line(fs, F)
        A = {k: np.array([r[k] for r in up3]) for k in ("psi", "mcp", "pip", "dip", "abd", "eq")}
        S = {k: np.array([r[k] for r in ups]) for k in ("psi", "mcp", "pip", "abd")}
        sol = dz.solve(np.radians(A["psi"]))
        err = {k: np.abs(np.degrees(sol[k]) - A[k]).max() for k in ("mcp", "pip", "dip")}
        hyst = max(np.abs(np.array([r[k] for r in dn3]) - A[k]).max() for k in ("mcp", "pip", "dip"))
        out(f"  sim vs planar design: max |dMCP| {err['mcp']:.3f}, |dPIP| {err['pip']:.3f}, |dDIP| {err['dip']:.3f} deg; up/down "
            f"hysteresis {hyst:.3f} deg; loop error max {A['eq'].max():.4f} mm")
        out(f"  drive unchanged: Link angle psi(f) vs stock max {np.abs(A['psi'] - S['psi']).max():.3f} deg; abduction "
            f"{np.abs(A['abd'] - S['abd']).max():.3f} deg")
        f_s = float(np.interp(0, A["mcp"], F))
        out(f"  straight finger (MCP = 0) at f = {f_s:.1f} deg; there PIP {np.interp(f_s, F, A['pip']):.2f}, "
            f"DIP {np.interp(f_s, F, A['dip']):.2f} deg")
        out("      f     psi     MCP     PIP     DIP  total | stock MCP  PIP")
        for fv in (-90, -60, round(f_s / 2.5) * 2.5, -30, 0, 30, 45, 60, 75, 90):
            i = int(np.argmin(np.abs(F - fv)))
            out(f"  {F[i]:6.1f} {A['psi'][i]:7.2f} {A['mcp'][i]:7.2f} {A['pip'][i]:7.2f} {A['dip'][i]:7.2f} "
                f"{A['mcp'][i] + A['pip'][i] + A['dip'][i]:6.1f} | {S['mcp'][i]:8.2f} {S['pip'][i]:6.2f}")
        sel = F >= f_s
        out(f"  ranges straight -> full: MCP {A['mcp'][-1]:.1f}, PIP {A['pip'][-1]:.1f}, DIP {A['dip'][-1]:.1f} deg "
            f"(stock {S['mcp'][-1]:.1f} / {S['pip'][-1]:.1f}); coupling slopes PIP/MCP "
            f"{np.polyfit(A['mcp'][sel], A['pip'][sel], 1)[0]:.3f}, DIP/PIP {np.polyfit(A['pip'][sel], A['dip'][sel], 1)[0]:.3f}")
        i60 = int(np.argmin(np.abs(F + 60)))
        out(f"  at the hyperextension clamp f = -60: MCP {A['mcp'][i60]:.1f}, PIP {A['pip'][i60]:.1f}, DIP {A['dip'][i60]:.1f}")

        out("\n-- 3. stability")
        st3 = V3.stability(f3)
        # loop 2 (DIP coupler) must stay on its assembly branch: DIP within its kinematic range over 60 abrupt steps
        f3.reset()
        rng = np.random.default_rng(7)
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        dips = []
        for _ in range(60):
            f3.d.ctrl[:] = rng.uniform(lo, hi)
            for k in range(300):
                mujoco.mj_step(m, f3.d)
                if k % 10 == 0:
                    dips.append(f3.measure()["dip"])
        out(f"  branch check, 60 abrupt random steps (DIP sampled every 10 ms): DIP {min(dips):.1f}..{max(dips):.1f} deg "
            f"(kinematic range -16.8..54.4, soft stops at -22 / +62; the other branch of loop 2 lies below -47 for any PIP "
            f">= -25): {'stays on its branch' if min(dips) > -45 else 'FLIPPED'}; settled DIP after the last step "
            f"{f3.measure()['dip']:.1f}")
        out("   baseline, stock finger:")
        V3.stability(fs)

        out("\n-- 4. grid over (m1, m2) and fingertip workspace")
        n25 = 13 if args.quick else 25
        ang = np.linspace(-90, 90, n25)
        g3 = V3.settle_grid(f3, ang)
        gs = V3.settle_grid(fs, ang)
        for lab, gg in (("ultimate", g3), ("stock", gs)):
            v = {k: np.array([r[k] for r in gg.values() if k in r]) for k in ("abd", "mcp", "pip", "dip", "eq", "track")}
            T = np.array([r["tip"] for r in gg.values()])
            out(f"  {lab:<9} {n25}x{n25}: abduction {v['abd'].min():.1f}..{v['abd'].max():.1f}, MCP {v['mcp'].min():.1f}..{v['mcp'].max():.1f}, "
                f"PIP {v['pip'].min():.1f}..{v['pip'].max():.1f}" + (f", DIP {v['dip'].min():.1f}..{v['dip'].max():.1f}" if len(v["dip"]) else "")
                + f" deg; loop err settled max {v['eq'].max():.4f} mm; motor tracking max {v['track'].max():.3f} deg; tip hull "
                f"{ConvexHull(T).volume / 1e3:.1f} cm3")

        out(f"\n-- 5. self-collision (exact mesh booleans, manifold3d), {args.grid}x{args.grid} grid + flexion line")
        angs = np.linspace(-90, 90, args.grid)
        gis = V3.settle_grid(fs, angs)
        posess = [(k, r["qpos"], r["mcp"]) for k, r in gis.items()] + [(f"f {r['f']:+.1f}", r["qpos"], r["mcp"]) for r in ups]
        rowss = V3.interference(fs, posess, V3.STOCK_FINGER_MESHES, "stock baseline: Link/Proximal/Distal (+shells) vs every other part")
        V3.report_interference(rowss)
        gi3 = V3.settle_grid(f3, angs)
        poses3 = [(k, r["qpos"], r["mcp"]) for k, r in gi3.items()] + [(f"f {r['f']:+.1f}", r["qpos"], r["mcp"]) for r in up3]
        rows3 = V3.interference(f3, poses3, NEW_MESHES, "ultimate: new parts (cores, shells, pads, pins) vs every other part")
        hits = V3.report_interference(rows3, n_show=30, stock_rows=rowss)
        # working range = not hyperextended past the clamp f >= -60
        clamp_bad = [r for r in rows3 if r[0] > 0.05]
        out(f"    overlapping pairs in total: {len(clamp_bad)}; in the working range (MCP >= -3 deg): {len(hits)}")
    else:
        f_s = design["predicted"]["f_straight_deg"]
        st3, hits, err = [], [], dict(mcp=0, pip=0, dip=0)
    rows, hold, keys, l2info = statics(f3, fs, design, f_s)
    res = torque_limit(rows, hold, keys, l2info, f_s)
    out("\n  holdable fingertip force at stall (normal to the pad): " + ", ".join(
        f"f {fv:+.0f}: {np.interp(fv, hold[:, 0], hold[:, 1]):.1f} N" for fv in (f_s, 0, 30, 45, 60, 75)))
    Ecr = {k: np.pi ** 2 * v["E"] * l2info[1] / l2info[0] ** 2 for k, v in MATERIALS.items()}
    out(f"  L2 buckling (pinned-pinned, {l2info[0]:.1f} mm, 3.0 x 3.5 mm): " + ", ".join(f"{k} {v:.0f} N" for k, v in Ecr.items()))
    np.savez(DATA_DIR / "verify_finger.npz", hold=hold, rows=np.array([json.dumps({k: (v if not isinstance(v, dict) else
                                                                                       {kk: float(vv) for kk, vv in v.items()})
                                                                                   for k, v in r.items()}) for r in rows]))
    figures(rows, hold, keys, res, f_s)
    if not args.only_statics:
        verdict = (max(err.values()) < 0.5 and not hits and all(not s[4] for s in st3))
        out(f"\nVERDICT: {'PASS' if verdict else 'CHECK'} - sim matches the design (<0.5 deg), stable without warnings, "
            f"{'no' if not hits else len(hits)} new-part interference(s) in the working range  [{time.time() - t0:.0f} s]")
    (REPORT_DIR / ("verify_finger_statics.txt" if args.only_statics else "verify_finger.txt")).write_text("\n".join(LINES) + "\n")


def figures(rows, hold, keys, res, f_s):
    import matplotlib.pyplot as plt
    ps.setup()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(hold[:, 0], hold[:, 1], color=ps.C[1], label="holdable at SCS0009 stall")
    for i, (plan, lims) in enumerate(res.items()):
        a = np.array([(f, Fa) for f, Fa, _, _ in lims])
        ax[0].plot(a[:, 0], a[:, 1], "o-", color=ps.C[[0, 2, 3, 4][i]], lw=1.4, ms=4, label=f"allowable: {plan}")
    ax[0].set_ylim(0, 60)
    ax[0].axvline(f_s, color=ps.INK2, lw=1)
    ax[0].set_xlabel("flexion command f (deg)")
    ax[0].set_ylabel("fingertip force (N)")
    ax[0].set_title("Fingertip force: servo stall vs part strength")
    ax[0].legend(fontsize=7)
    for i, (plan, lims) in enumerate(res.items()):
        a = np.array([(f, 100 * tl / SERVO_STALL_NM) for f, _, _, tl in lims])
        ax[1].plot(a[:, 0], a[:, 1], "o-", color=ps.C[[0, 2, 3, 4][i]], lw=1.4, ms=4, label=plan)
    ax[1].axhline(100, color=ps.INK2, lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("flexion command f at contact (deg)")
    ax[1].set_ylabel("servo torque that reaches the part limit (% of stall)")
    ax[1].set_title("Torque limit: contact at f needs at most this")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strength.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
