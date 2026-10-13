#!/usr/bin/env python3
"""Verify the 3-phalanx finger prototype (model/finger_3seg.xml) against its design and the stock finger.

  1. structure: loads, names, actuators, equality constraints, masses (vs the stock finger);
  2. kinematics: settled flexion sweep (up and down) -> MCP/PIP/DIP vs the planar design and the
     targets; the drive (Link angle psi, abduction) vs the stock finger (the drive is unchanged);
  3. stability: each actuator alone over its full range, range corners, 60 abrupt random steps:
     no NaN, no MuJoCo warnings, loop-closure errors (transient / settled), motor tracking;
  4. grid over (m1, m2): joint ranges, abduction, settled loop errors, fingertip workspace vs stock;
  5. self-collision: every new part posed on a 13 x 13 grid + the dense flexion line, checked against
     every other part of the finger (stock and new) with exact mesh booleans (manifold3d): overlap
     volume (interference) and minimum gap (clearance). Pins are excluded only against the two
     parts they join. The same check on the stock finger is the baseline;
  6. statics: holdable fingertip force at SCS0009 stall (Jacobian transpose through all loops) along
     the flexion line, and pin/coupler loads for a 5 N pad force from MuJoCo's constraint forces.

Writes reports/verify_3seg.txt, data/verify_3seg.npz and figures/verify_*.png.
Usage: python verify_3seg_finger.py [--grid 13] [--quick]
"""

import argparse
import json
import time

import manifold3d as mf
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

import plotstyle as ps
from finger_common import (DATA_DIR, DESIGN_JSON, FIG_DIR, MODEL_DIR, REPORT_DIR, SERVO_STALL_NM, Finger,
                           flexion_ctrl, log, stock_roles, three_seg_roles)
from linkage import Design

LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    log(s)


NEW_BODIES_3S = ["rotule_ball", "p1", "p2", "p3", "l2"]  # rotule_ball = Link body (new arm)
NEW_MESHES_3S = {"link3s", "p1", "p1_shell", "p2", "p2_pad", "p3", "p3_shell", "l2"}
STOCK_FINGER_MESHES = {"link", "proximal", "proximal_shell", "distal", "distal_shell"}


# ---------------------------------------------------------------- 3. stability
def run_sequence(f, segments, label, stats):
    m, d = f.m, f.d
    dt = m.opt.timestep
    vmax, e_dyn, e_set, track, n = 0.0, 0.0, 0.0, 0.0, 0
    for target, ramp, hold in segments:
        start = d.ctrl.copy()
        nr, nh = int(round(ramp / dt)), int(round(hold / dt))
        for k in range(nr + nh):
            a = 1.0 if k >= nr else (k + 1) / nr
            d.ctrl[:] = start + (target - start) * a
            mujoco.mj_step(m, d)
            n += 1
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                raise RuntimeError(f"{label}: non-finite state")
            vmax = max(vmax, np.abs(d.qvel).max())
            e_dyn = max(e_dyn, f.eq_err().max())
        e_set = max(e_set, f.eq_err().max())
        track = max(track, np.abs(d.actuator_length - d.ctrl).max())
    warn = {mujoco.mjtWarning(i).name: d.warning[i].number for i in range(len(d.warning)) if d.warning[i].number}
    stats.append((label, e_dyn, e_set, track, warn))
    out(f"  {label:<40} {n * dt:6.1f} s  max|qvel| {vmax:6.1f} rad/s  loop err max {e_dyn * 1e3:6.3f} mm "
        f"(settled {e_set * 1e3:6.4f})  |q_motor - ctrl| settled {np.degrees(track):5.2f} deg  warnings {warn or 'none'}")


def stability(f):
    m = f.m
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    stats = []
    f.reset()
    segs = []
    for a in range(m.nu):
        for target in (hi[a], lo[a], 0.0):
            c = np.zeros(m.nu)
            c[a] = target
            segs.append((c, 1.0, 0.5))
    run_sequence(f, segs, "each actuator alone 0 -> max -> min -> 0", stats)
    f.reset()
    run_sequence(f, [(np.array([c1 * hi[0], c2 * hi[1]]), 1.5, 0.5) for c1, c2 in
                     [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1), (0, 0)]], "range corners", stats)
    f.reset()
    rng = np.random.default_rng(1)
    run_sequence(f, [(rng.uniform(lo, hi), 0.0, 0.6) for _ in range(60)], "60 abrupt random full-range steps", stats)
    return stats


# ---------------------------------------------------------------- grids
def settle_grid(f, angles, jac=False):
    f.reset()
    grid = [(a1, a2) for i, a1 in enumerate(angles) for a2 in (angles if i % 2 == 0 else angles[::-1])]
    res = {}
    for a1, a2 in grid:
        f.move(np.radians(a1), np.radians(a2), ramp=0.15, hold=0.25, check=False)
        r = f.measure()
        r["qpos"] = f.d.qpos.copy()
        r["track"] = np.degrees(np.abs(f.d.actuator_length - f.d.ctrl).max())
        res[(a1, a2)] = r
    return dict(sorted(res.items()))


def sweep_line(f, fs, a=0.0):
    f.reset()
    f.move(*flexion_ctrl(fs[0], a), ramp=0.5, hold=0.5, check=False)
    res = []
    for k, fv in enumerate(list(fs) + list(fs[::-1])):
        f.move(*flexion_ctrl(fv, a), ramp=0.05, hold=0.25, check=False)
        r = f.measure()
        J, resid, _ = f.motor_jacobian()
        r.update(f=fv, up=k < len(fs), J=J, resid=resid, qpos=f.d.qpos.copy(),
                 R_pad=f.frame("P3" if "P3" in f.cad else "P2")[:3, :3].copy())
        res.append(r)
    return [r for r in res if r["up"]], [r for r in res if not r["up"]][::-1]


# ---------------------------------------------------------------- 5. interference
class Parts:
    """Every geom of the model as a manifold in its geom frame (mm)."""

    def __init__(self, f, new_meshes):
        m = f.m
        self.f, self.items = f, []
        for gi in range(m.ngeom):
            b = m.geom_bodyid[gi]
            if b == 0:
                continue
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gi]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] * 1e3
                fc = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                man = mf.Manifold(mf.Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=fc.astype(np.uint32)))
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
                pin = name.startswith("parallel_pin")
            elif m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, h = m.geom_size[gi, 0] * 1e3, m.geom_size[gi, 1] * 1e3
                man = mf.Manifold.cylinder(2 * h, r, r, 48, True)
                name, pin = "pin", True
            else:
                continue
            assert man.status() == mf.Error.NoError, name
            bb = man.bounding_box()
            lo, hi = np.array(bb[:3]), np.array(bb[3:])
            self.items.append(dict(g=gi, body=b, name=name, man=man, new=(name in new_meshes) or
                                   (pin and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) in NEW_BODIES_3S
                                    and name == "pin"), pin=pin, c=(lo + hi) / 2, r=np.linalg.norm(hi - lo) / 2))
        # adjacency (bodies joined by a joint or a loop closure): pins are excused against these
        adj = set()
        for b in range(1, m.nbody):
            p = m.body_parentid[b]
            if p > 0:
                adj |= {(b, p), (p, b)}
        for e in range(m.neq):
            b1, b2 = m.site_bodyid[m.eq_obj1id[e]], m.site_bodyid[m.eq_obj2id[e]]
            adj |= {(b1, b2), (b2, b1)}
        self.adj = adj
        bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731
        self.pairs = []
        for i, a in enumerate(self.items):
            for j, b in enumerate(self.items):
                if j <= i or a["body"] == b["body"] or not (a["new"] or b["new"]):
                    continue
                if (a["pin"] or b["pin"]) and (a["body"], b["body"]) in adj:
                    continue
                self.pairs.append((i, j))
        self.bname = bname

    def check(self, gap_search=3.0):
        """Per pair: (overlap mm^3, min gap mm) at the current pose (gap only below gap_search)."""
        d = self.f.d
        T = {}
        for k, it in enumerate(self.items):
            R, p = d.geom_xmat[it["g"]].reshape(3, 3), d.geom_xpos[it["g"]] * 1e3
            T[k] = (R, p)
        res = {}
        for i, j in self.pairs:
            a, b = self.items[i], self.items[j]
            ca = T[i][1] + T[i][0] @ a["c"]
            cb = T[j][1] + T[j][0] @ b["c"]
            if np.linalg.norm(ca - cb) > a["r"] + b["r"] + gap_search:
                continue
            ma = a["man"].transform(np.c_[T[i][0], T[i][1]])
            mb = b["man"].transform(np.c_[T[j][0], T[j][1]])
            ov = (ma ^ mb).volume()
            gap = 0.0 if ov > 1e-6 else ma.min_gap(mb, gap_search)
            res[(i, j)] = (ov, gap)
        return res


def interference(f, poses, new_meshes, label, mcp_min=-3.0):
    """poses: [(key, qpos, mcp)]. Overlaps are split into the working range (MCP >= mcp_min) and
    hyperextension beyond it (where the stock finger has its stop against the finger frame)."""
    parts = Parts(f, new_meshes)
    worst = {}
    t0 = time.time()
    for key, q, mcp in poses:
        f.d.qpos[:] = q
        mujoco.mj_forward(f.m, f.d)
        for pair, (ov, gap) in parts.check().items():
            w = worst.setdefault(pair, dict(ov=0.0, gap=np.inf, n_ov=0, n_work=0, at_ov=None, at_gap=None,
                                            mcp_onset=np.inf, ov_work=0.0))
            if ov > w["ov"]:
                w["ov"], w["at_ov"] = ov, key
            if ov > 0.05:
                w["n_ov"] += 1
                w["mcp_onset"] = min(w["mcp_onset"], mcp) if mcp < mcp_min else w["mcp_onset"]
                if mcp >= mcp_min:
                    w["n_work"] += 1
                    w["ov_work"] = max(w["ov_work"], ov)
            if gap < w["gap"]:
                w["gap"], w["at_gap"] = gap, key
    out(f"  {label}: {len(poses)} poses, {len(parts.pairs)} part pairs checked [{time.time() - t0:.0f} s]")
    rows = []
    for (i, j), w in worst.items():
        a, b = parts.items[i], parts.items[j]
        rows.append((w["ov"], w["gap"], f"{a['name']}@{parts.bname(a['body'])}", f"{b['name']}@{parts.bname(b['body'])}",
                     w["n_ov"], w["at_ov"], w["at_gap"], w["n_work"], w["ov_work"], len(poses)))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return rows


def fmt_key(k):
    return k if isinstance(k, str) else f"(m1 {k[0]:+.0f}, m2 {k[1]:+.0f})"


STOCK_EQUIV = {"link3s": "link", "p1": "proximal", "p1_shell": "proximal_shell", "p2": "distal", "p3": "distal",
               "p3_shell": "distal_shell", "p2_pad": "distal_shell", "l2": "link"}


def mesh_pair(r):
    a, b = r[2].split("@")[0], r[3].split("@")[0]
    return frozenset({STOCK_EQUIV.get(a, a), STOCK_EQUIV.get(b, b)})


def report_interference(rows, n_show=14, stock_rows=None):
    """Lists overlapping pairs; returns those that overlap in the working range and are not
    stock-identical artefacts (a stock pair with the same parts overlapping in every pose)."""
    hits = [r for r in rows if r[0] > 0.05]
    out(f"    pairs with overlap > 0.05 mm^3 in some pose: {len(hits)}")
    artefacts = {mesh_pair(r) for r in (stock_rows or []) if r[0] > 0.05 and r[4] == r[9]}
    bad = []
    for r in hits[:n_show]:
        tag = ""
        if stock_rows is not None:
            if mesh_pair(r) in artefacts and r[4] == r[9]:
                tag = "  -> same as stock (overlaps in every pose: mesh artefact of the stock ball joints)"
            elif r[7] == 0:
                tag = "  -> only in hyperextension (MCP < -3 deg): the stop against the palm-fixed parts, as stock"
            else:
                tag = f"  -> IN THE WORKING RANGE ({r[7]} poses, up to {r[8]:.2f} mm^3)"
                bad.append(r)
        out(f"      {r[2]:<38} x {r[3]:<38} max {r[0]:7.2f} mm^3 in {r[4]:3d}/{r[9]} poses{tag}")
    close = sorted([r for r in rows if r[0] <= 0.05], key=lambda r: r[1])[:n_show]
    out("    closest non-overlapping pairs (min gap over all poses; 0.25 mm = the lateral running gap between layers):")
    for r in close:
        out(f"      {r[2]:<38} x {r[3]:<38} min gap {r[1]:6.3f} mm (at {fmt_key(r[6])})")
    return bad if stock_rows is not None else hits


# ---------------------------------------------------------------- 6. statics
def loads_for_pad_force(f, fv, F=5.0):
    """Hold the finger at flexion fv and push the fingertip pad with F (N) towards the back of the
    finger (normal to the distal pad); returns motor torques and constraint forces at the closures."""
    m, d = f.m, f.d
    fl, dmp = m.dof_frictionloss.copy(), m.dof_damping.copy()
    m.dof_frictionloss[:] = 0.0  # frictionless pins for the statics (the model's generic 0.1 N m would carry load)
    m.dof_damping[:] = np.maximum(dmp, 2e-4)
    f.reset()
    f.move(*flexion_ctrl(fv), ramp=0.5, hold=0.3, check=False)
    pad = f.frame("P3" if "P3" in f.cad else "P2")
    n = pad[:3, 2]  # +z of the distal = dorsal: the object pushes the pad towards the back of the finger
    b = m.site_bodyid[f.tip]
    d.xfrc_applied[b, :3] = F * n
    d.xfrc_applied[b, 3:] = np.cross(d.site_xpos[f.tip] - d.xipos[b], F * n)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    mujoco.mj_rnePostConstraint(m, d)  # body interaction forces incl. the loop-closure (connect) forces
    tau = d.actuator_force.copy()
    joint = {}
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        joint[name] = np.linalg.norm(d.cfrc_int[b, 3:])
    forces = {}
    for e in range(m.neq):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, m.eq_obj1id[e])
        rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == e]
        forces[name] = np.linalg.norm(d.efc_force[rows]) if rows else 0.0
    d.xfrc_applied[:] = 0
    m.dof_frictionloss[:], m.dof_damping[:] = fl, dmp
    return tau, forces, joint


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=13, help="interference grid size per motor")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    design = json.loads(DESIGN_JSON.read_text())
    dz = Design.from_json(design)
    f3 = Finger(MODEL_DIR / "scene_3seg.xml", three_seg_roles())
    fs = Finger(MODEL_DIR / "scene_stock.xml", stock_roles())
    m = f3.m

    out("== 3-phalanx finger prototype: verification (model/finger_3seg.xml)")
    out("\n-- 1. structure")
    out(f"  nbody {m.nbody}, njnt {m.njnt}, nq {m.nq}, nv {m.nv}, nu {m.nu}, neq {m.neq} (stock finger: nbody {fs.m.nbody}, "
        f"nq {fs.m.nq}, neq {fs.m.neq})")
    for obj, num in ((mujoco.mjtObj.mjOBJ_BODY, m.nbody), (mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
                     (mujoco.mjtObj.mjOBJ_SITE, m.nsite)):
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
    out(f"  moving finger mass beyond the gimbal (Link + phalanges + coupler + shells): {m3 * 1e3:.1f} g (stock {ms * 1e3:.1f} g)")
    f3.reset()
    out(f"  keyframe (settled at ctrl = 0): max loop error {f3.eq_err().max() * 1e3:.4f} mm")

    # ---- 2. kinematics
    out("\n-- 2. kinematics along the flexion line (m1 = f, m2 = -f), settled, up and down")
    F = np.arange(-90, 90.01, 2.5)
    up3, dn3 = sweep_line(f3, F)
    ups, dns = sweep_line(fs, F)
    A = {k: np.array([r[k] for r in up3]) for k in ("psi", "mcp", "pip", "dip", "abd", "eq")}
    S = {k: np.array([r[k] for r in ups]) for k in ("psi", "mcp", "pip", "abd")}
    sol = dz.solve(np.radians(A["psi"]))
    err = {k: np.abs(np.degrees(sol[k]) - A[k]).max() for k in ("mcp", "pip", "dip")}
    hyst = max(np.abs(np.array([r[k] for r in dn3]) - A[k]).max() for k in ("mcp", "pip", "dip"))
    out(f"  sim vs planar design (given the sim's own Link angle): max |dMCP| {err['mcp']:.3f}, |dPIP| {err['pip']:.3f}, "
        f"|dDIP| {err['dip']:.3f} deg; up/down hysteresis {hyst:.3f} deg; loop error max {A['eq'].max():.4f} mm")
    out(f"  drive unchanged: Link angle psi(f) differs from the stock finger by max {np.abs(A['psi'] - S['psi']).max():.3f} deg; "
        f"abduction by max {np.abs(A['abd'] - S['abd']).max():.3f} deg")
    f_s = float(np.interp(0, A["mcp"], F))
    out(f"  straight finger (MCP = 0) at f = {f_s:.1f} deg (design {design['drive']['f_straight_deg']}); there PIP "
        f"{np.interp(f_s, F, A['pip']):.2f}, DIP {np.interp(f_s, F, A['dip']):.2f} deg")
    out("      f     psi     MCP     PIP     DIP  MCP+PIP+DIP | stock MCP  PIP  MCP+PIP")
    for fv in (-90, -75, round(f_s / 2.5) * 2.5, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90):
        i = int(np.argmin(np.abs(F - fv)))
        out(f"  {F[i]:6.1f} {A['psi'][i]:7.2f} {A['mcp'][i]:7.2f} {A['pip'][i]:7.2f} {A['dip'][i]:7.2f} "
            f"{A['mcp'][i] + A['pip'][i] + A['dip'][i]:9.2f}   | {S['mcp'][i]:8.2f} {S['pip'][i]:6.2f} {S['mcp'][i] + S['pip'][i]:7.2f}")
    sel = F >= f_s
    out(f"  ranges straight -> full flexion: MCP 0 -> {A['mcp'][-1]:.1f} (target 90), PIP 0 -> {A['pip'][-1]:.1f} (target 100), "
        f"DIP 0 -> {A['dip'][-1]:.1f} (target 70) deg; total curl {A['mcp'][-1] + A['pip'][-1] + A['dip'][-1]:.0f} deg "
        f"(stock {S['mcp'][-1] + S['pip'][-1]:.0f}, human fist ~260)")
    out(f"  coupling: PIP/MCP slope {np.polyfit(A['mcp'][sel], A['pip'][sel], 1)[0]:.3f}, DIP/PIP slope "
        f"{np.polyfit(A['pip'][sel], A['dip'][sel], 1)[0]:.3f}; DIP/PIP at full flexion {A['dip'][-1] / A['pip'][-1]:.3f}")
    out(f"  hyperextension end (f = -90): MCP {A['mcp'][0]:.1f}, PIP {A['pip'][0]:.1f}, DIP {A['dip'][0]:.1f} deg "
        f"(stock MCP {S['mcp'][0]:.1f}, PIP {S['pip'][0]:.1f})")

    # ---- 3. stability
    out("\n-- 3. stability (3-phalanx)")
    st3 = stability(f3)
    out("   baseline, stock finger:")
    stability(fs)

    # ---- 4. grid / workspace
    out("\n-- 4. grid over (m1, m2) and fingertip workspace")
    n25 = 13 if args.quick else 25
    ang = np.linspace(-90, 90, n25)
    g3 = settle_grid(f3, ang)
    gs = settle_grid(fs, ang)
    T3 = np.array([r["tip"] for r in g3.values()])
    Ts = np.array([r["tip"] for r in gs.values()])
    for lab, g, T in (("3-phalanx", g3, T3), ("stock", gs, Ts)):
        v = {k: np.array([r[k] for r in g.values() if k in r]) for k in ("abd", "psi", "mcp", "pip", "dip", "eq", "track")}
        hull = ConvexHull(T)
        out(f"  {lab:<10} {n25}x{n25}: abduction {v['abd'].min():.1f}..{v['abd'].max():.1f}, MCP {v['mcp'].min():.1f}.."
            f"{v['mcp'].max():.1f}, PIP {v['pip'].min():.1f}..{v['pip'].max():.1f}" +
            (f", DIP {v['dip'].min():.1f}..{v['dip'].max():.1f}" if len(v["dip"]) else "") +
            f" deg; loop err settled max {v['eq'].max():.4f} mm; motor tracking max {v['track'].max():.3f} deg")
        out(f"             tip workspace: x {T[:, 0].min():.1f}..{T[:, 0].max():.1f}, y {T[:, 1].min():.1f}..{T[:, 1].max():.1f}, "
            f"z {T[:, 2].min():.1f}..{T[:, 2].max():.1f} mm; convex hull {hull.volume / 1e3:.1f} cm^3; max reach from MCP "
            f"{np.linalg.norm(T, axis=1).max():.1f} mm; closest approach to the MCP {np.linalg.norm(T, axis=1).min():.1f} mm")
    for lab, g in (("3-phalanx", g3), ("stock", gs)):
        T = np.array([r["tip"] for r in g.values() if r["mcp"] >= -3.0])
        out(f"  {lab:<10} working range only (MCP >= -3 deg, i.e. not hyperextended): {len(T)} grid poses, tip convex hull "
            f"{ConvexHull(T).volume / 1e3:.1f} cm^3, lateral x {T[:, 0].min():.1f}..{T[:, 0].max():.1f} mm")
    tip_line3 = np.array([r["tip"] for r in up3])
    tip_lines = np.array([r["tip"] for r in ups])
    out(f"  flexion line, fully flexed tip (y, z): 3-phalanx ({tip_line3[-1, 1]:.1f}, {tip_line3[-1, 2]:.1f}), stock "
        f"({tip_lines[-1, 1]:.1f}, {tip_lines[-1, 2]:.1f}) mm; enclosed area of the tip path + palm line: "
        f"3-phalanx {area_under(tip_line3):.0f}, stock {area_under(tip_lines):.0f} mm^2")

    # ---- 5. interference
    out(f"\n-- 5. self-collision (exact mesh booleans, manifold3d), {args.grid}x{args.grid} grid + flexion line")
    angs = np.linspace(-90, 90, args.grid)
    gis = settle_grid(fs, angs) if args.grid != n25 else gs
    posess = [(k, r["qpos"], r["mcp"]) for k, r in gis.items()] + [(f"f {r['f']:+.1f}", r["qpos"], r["mcp"]) for r in ups]
    rowss = interference(fs, posess, STOCK_FINGER_MESHES, "stock baseline: Link/Proximal/Distal (+shells) vs every other part")
    report_interference(rowss)
    gi3 = settle_grid(f3, angs) if args.grid != n25 else g3
    poses3 = [(k, r["qpos"], r["mcp"]) for k, r in gi3.items()] + [(f"f {r['f']:+.1f}", r["qpos"], r["mcp"]) for r in up3]
    rows3 = interference(f3, poses3, NEW_MESHES_3S, "3-phalanx: new parts vs every other part")
    hits3 = report_interference(rows3, stock_rows=rowss)
    stock_work = [r for r in rowss if r[0] > 0.05 and r[7] > 0]
    out(f"    stock baseline for comparison: {len(stock_work)} stock pairs overlap in the stock working range (MCP >= -3 deg), "
        "e.g. " + "; ".join(f"{r[2].split('@')[0]} x {r[3].split('@')[0]} ({r[7]} poses)" for r in stock_work[:4]))

    # ---- 6. statics
    out("\n-- 6. statics")
    out(f"  holdable fingertip force at SCS0009 stall ({SERVO_STALL_NM:.3f} N m), along the fingertip motion / normal to the pad:")
    force = {}
    for lab, up in (("3-phalanx", up3), ("stock", ups)):
        rows = []
        for r in up:
            J = r["J"]
            v = J[:, 0] - J[:, 1]
            n_mo = v / np.linalg.norm(v)
            n_pad = -r["R_pad"][:, 2]
            rows.append((r["f"], SERVO_STALL_NM / np.abs(J.T @ n_mo).max(), SERVO_STALL_NM / np.abs(J.T @ n_pad).max(),
                         np.linalg.norm(v) * 1e3 * np.pi / 180))
        force[lab] = np.array(rows)
    f_ss = -37.5
    for fv in (None, 0, 30, 60, 75, 90):
        cells = []
        for lab, fstr in (("3-phalanx", f_s), ("stock", f_ss)):
            fq = fstr if fv is None else fv
            i = int(np.argmin(np.abs(force[lab][:, 0] - fq)))
            cells.append(f"{lab} f {force[lab][i, 0]:+6.1f}: {force[lab][i, 1]:6.2f} / {force[lab][i, 2]:6.2f} N "
                         f"({force[lab][i, 3]:.2f} mm/deg)")
        out("    " + "   |   ".join(cells))
    out("  loads for a 5 N push on the fingertip pad (sim constraint forces; motor torque vs 0.226 N m stall):")
    pin_rows = []
    for fv in (round(f_s), 0, 30, 60):
        tau, fc, jf = loads_for_pad_force(f3, fv, 5.0)
        tau_s, fc_s, jf_s = loads_for_pad_force(fs, fv, 5.0)
        out(f"    f {fv:+4d}: 3-phalanx motor torques {np.round(tau, 3)} N m; pin forces: P4 (Link/P2) {jf['p2']:.1f}, "
            f"PIP {jf['p1']:.1f}, DIP {jf['p3']:.1f}, A/B (coupler L2) {jf['l2']:.1f} | B closure {fc['closing_dip (1)_1']:.1f}, "
            f"MCP closure {fc['closing_3 (1)_1']:.1f}, rod-2 ball {fc['closing_ball1_1']:.1f} N")
        out(f"            stock    motor torques {np.round(tau_s, 3)} N m; pin forces: P4 "
            f"{jf_s['parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999']:.1f}, PIP "
            f"{jf_s['parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b']:.1f} | MCP closure {fc_s['closing_3 (1)_1']:.1f}, "
            f"rod-2 ball {fc_s['closing_ball1_1']:.1f} N")
        pin_rows.append((fv, max(abs(tau)), jf["p2"], jf["p1"], jf["p3"], jf["l2"]))
    out("  strength check (PLA, 2 mm steel pins), loads scaled from 5 N to the pad force the servos can hold there:")
    for fv, tmax, fp4, fpip, fdip, fl2 in pin_rows:
        scale = SERVO_STALL_NM / tmax  # pad force at stall = 5 N * scale
        worst = max(fp4, fpip, fdip, fl2) * scale
        out(f"    f {fv:+4d}: pad force at stall {5 * scale:5.1f} N -> largest pin force {worst:5.1f} N; bearing stress on a "
            f"2 mm pin in a 2.85 mm PLA wall {worst / (2 * 2 * 2.85):5.1f} MPa (two walls), in the 3.8 mm Link/L2 eye "
            f"{worst / (2 * 3.8):5.1f} MPa; L2 axial stress {fl2 * scale / (4.0 * 3.8):4.1f} MPa (PLA ~40-50 MPa, ~15-25 across layers)")

    np.savez(DATA_DIR / "verify_3seg.npz", F=F, **{f"s3_{k}": v for k, v in A.items()}, **{f"st_{k}": v for k, v in S.items()},
             tip3=tip_line3, tips=tip_lines, T3=T3, Ts=Ts, force3=force["3-phalanx"], forces=force["stock"], f_s=f_s)
    figures(F, A, S, sol, tip_line3, tip_lines, T3, Ts, force, f_s, dz)
    verdict = (err["mcp"] < 0.5 and err["pip"] < 0.5 and err["dip"] < 0.5 and not hits3 and
               all(not s[4] for s in st3))
    out(f"\nVERDICT: {'PASS' if verdict else 'CHECK'} - sim matches the design (<0.5 deg), stable without warnings, "
        f"{'no' if not hits3 else len(hits3)} new-part interference(s) in the working range (MCP >= -3 deg)  "
        f"[{time.time() - t0:.0f} s]")
    (REPORT_DIR / "verify_3seg.txt").write_text("\n".join(LINES) + "\n")


def area_under(tip):
    """Area swept between the fingertip path in the flexion plane (y, z) and the straight line joining
    its ends: a simple measure of how much the finger can wrap."""
    y, z = tip[:, 1], tip[:, 2]
    return 0.5 * abs(np.dot(y, np.roll(z, 1)) - np.dot(z, np.roll(y, 1)))


def figures(F, A, S, sol, t3, ts, T3, Ts, force, f_s, dz):
    import matplotlib.pyplot as plt
    ps.setup()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, (key, lab) in enumerate((("mcp", "MCP"), ("pip", "PIP"), ("dip", "DIP"))):
        ax[0].plot(F, A[key], color=ps.C[k], label=f"{lab} sim")
        ax[0].plot(F, np.degrees(sol[key]), color=ps.INK, lw=0.8, ls=(0, (2, 2)))
    ax[0].plot([], [], color=ps.INK, lw=0.8, ls=(0, (2, 2)), label="planar design")
    for k, key in enumerate(("mcp", "pip")):
        ax[0].plot(F, S[key], color=ps.C[k], lw=1.0, alpha=0.6, ls=(0, (6, 3)))
    ax[0].plot([], [], color=ps.INK2, lw=1.0, ls=(0, (6, 3)), label="stock MCP / PIP")
    ax[0].axvline(f_s, color=ps.INK2, lw=1)
    ax[0].set_xlabel("flexion command f (deg), a = 0")
    ax[0].set_ylabel("joint angle (deg)")
    ax[0].set_title("3-phalanx prototype in MuJoCo vs design")
    ax[0].legend(loc="upper left", fontsize=7.5)
    sel = F >= f_s
    ax[1].plot(A["mcp"][sel], A["pip"][sel], color=ps.C[1], label="PIP(MCP) sim")
    ax[1].plot(A["pip"][sel], A["dip"][sel], color=ps.C[2], label="DIP(PIP) sim")
    ax[1].plot(S["mcp"][F >= -37.5], S["pip"][F >= -37.5], color=ps.C[0], label="PIP(MCP) stock")
    ax[1].plot([0, 90], [0, 100], color=ps.INK2, lw=1, ls=(0, (2, 2)), label="target PIP = 1.11 MCP")
    ax[1].plot([0, 100], [0, 70], color=ps.INK2, lw=1, ls=(0, (6, 2)), label="target DIP = 0.7 PIP")
    ax[1].scatter([90, 100], [100, 70], s=30, color=ps.INK2, zorder=5)
    ax[1].annotate("target end points", (90, 100), xytext=(-90, 2), textcoords="offset points", fontsize=8, color=ps.INK2)
    ax[1].set_xlabel("proximal joint angle (deg)")
    ax[1].set_ylabel("distal joint angle (deg)")
    ax[1].set_title("Coupling curves (straight -> full flexion)")
    ax[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "verify_coupling.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    ax[0].plot(ts[:, 1], ts[:, 2], color=ps.C[0], label="stock 2-phalanx")
    ax[0].plot(t3[:, 1], t3[:, 2], color=ps.C[1], label="3-phalanx")
    for t, c in ((ts, ps.C[0]), (t3, ps.C[1])):
        ax[0].plot(*t[-1, 1:], "o", color=c, ms=7, mec=ps.SURFACE, mew=1.5)
    ax[0].plot(0, 0, "+", color=ps.INK, ms=10)
    ax[0].annotate("MCP", (0, 0), xytext=(4, 4), textcoords="offset points", fontsize=8, color=ps.INK2)
    ax[0].annotate("full flexion", t3[-1, 1:], xytext=(6, -10), textcoords="offset points", fontsize=8, color=ps.INK2)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("y (mm, along the straight finger)")
    ax[0].set_ylabel("z (mm, dorsal +)")
    ax[0].set_title("Fingertip path, flexion line f = -90..+90")
    ax[0].legend(loc="upper right")
    for k, (T, c, lab) in enumerate(((Ts, ps.C[0], "stock"), (T3, ps.C[1], "3-phalanx"))):
        ax[1].scatter(T[:, 1], T[:, 2], s=6, color=c, alpha=0.6, label=lab, lw=0)
        ax[2].scatter(T[:, 0], T[:, 2], s=6, color=c, alpha=0.6, label=lab, lw=0)
    ax[1].set_aspect("equal")
    ax[2].set_aspect("equal")
    ax[1].set_xlabel("y (mm)")
    ax[1].set_ylabel("z (mm)")
    ax[1].set_title("Reachable fingertip points, side view (25 x 25 motor grid)")
    ax[2].set_xlabel("x (mm, lateral)")
    ax[2].set_ylabel("z (mm)")
    ax[2].set_title("Reachable fingertip points, front view")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "verify_workspace.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for lab, c in (("stock", ps.C[0]), ("3-phalanx", ps.C[1])):
        ax.plot(force[lab][:, 0], force[lab][:, 2], color=c, label=f"{lab}, normal to the distal pad")
    ax.axvline(f_s, color=ps.C[1], lw=1, alpha=0.6)
    ax.axvline(-37.5, color=ps.C[0], lw=1, alpha=0.6)
    ax.set_ylim(0, 30)
    ax.set_xlabel("flexion command f (deg), a = 0 (vertical lines: straight pose)")
    ax.set_ylabel("holdable force (N)")
    ax.set_title("Fingertip force at SCS0009 stall torque (Jacobian transpose)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "verify_force.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
