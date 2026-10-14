#!/usr/bin/env python3
"""Verify the slim-finger model (AH_Right_5F_Slim) against the stock 5-digit model (AH_Right_5F).

Both models are settled on the same N x N grid over each digit's two motors (+-90 deg) and compared
with the surface-sample distance machinery of five_finger/verify_5f_model.py (samples every --coarse
mm, refined to --fine mm below 3 mm; "touching" = sampled distance < 0.75 * fine):

  1. structure: same bodies/joints/actuators/keyframe; a quick actuator sweep is stable;
  2. each digit's own parts over its whole joint range: every changed part (proximal, distal, both
     shells, pad, shell screws) against every other part of the same digit (gimbal, link, rods,
     horns, the other phalanx, servos, finger frames, bushings) and the palm plate/wrist interface;
     a contact is NEW if that pair touches at a grid pose where the stock pair does not;
  3. neighbouring digits: ring/pinky, middle/ring, index/middle over every pair of grid poses
     (stock baseline: 887/6561 touching pose pairs for ring/pinky and middle/ring);
  4. the 5-finger palm/back shells and grip pads (cad/five_finger/stl, check_shells.py logic):
     moving parts of every digit vs each shell over the grid;
  5. demo poses: clearance between neighbouring digits.

Jobs run in parallel processes. Writes reports/verify_slim.txt (+ .json).

Usage: python verify_slim.py [--grid 9] [--jobs 12] [--skip-pairs]
"""

import argparse
import ast
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "five_finger"))
from check_shells import shells  # noqa: E402
from verify_5f_model import (DEMO_POSES_5F, SIM_DIR, Group, Hand, clearance_table, digit_geoms,  # noqa: E402
                             fmt_pose, geom_label, group_distance, posed, pose_ctrl, settle_grid)

MODELS = {"stock": "AH_Right_5F", "slim": "AH_Right_5F_Slim"}
CHANGED = {"proximal", "proximal_shell", "distal", "distal_shell", "distal_pad",
           "std00447_thermoplastique_m2_5x6__configuration_default", "thermo_screw_m2_5x5"}
SCREW_NAMES = {"std00447_thermoplastique_m2_5x6__configuration_default": "shell_screw", "thermo_screw_m2_5x5": "shell_screw"}
DIGITS = ["index", "middle", "ring", "thumb", "pinky"]


def log(*a):
    print(*a, flush=True)


def mesh_name(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])


def key_of(h, g):
    """Model-independent geom key: (body, mesh, ordinal of that mesh in the body)."""
    m = h.m
    inv = {v: k for k, v in h.part_geom.items()}
    if g in inv:
        return ("palm", inv[g], 0)
    b = m.geom_bodyid[g]
    mesh = SCREW_NAMES.get(mesh_name(m, g), mesh_name(m, g))
    same = [x for x in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
            if SCREW_NAMES.get(mesh_name(m, x), mesh_name(m, x)) == mesh]
    return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), mesh, same.index(g))


def short(key):
    body, mesh, k = key
    return mesh if body == "palm" else f"{mesh}{'#' + str(k) if k else ''}"


def hand(model, with_shells=False):
    return Hand(SIM_DIR / MODELS[model] / "mjcf", shells("5f") if with_shells else None)


# ------------------------------------------------------------------ job: own parts of one digit
def job_self(model, dig, n, coarse, fine, cutoff=3e-3):
    h = hand(model)
    m = h.m
    poses, _ = settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n))
    moving, fixed = h.geoms[dig]
    others = moving + fixed + h.palm
    mods = [g for g in moving if mesh_name(m, g) in CHANGED]
    pairs = [(a, b) for a in mods for b in others
             if m.geom_bodyid[a] != m.geom_bodyid[b] and not (b in mods and mods.index(b) < mods.index(a))]
    geoms = set(mods) | set(others)
    grp = {(g, sp): Group(h, [g], sp) for g in geoms for sp in (coarse, fine)}
    res = {p: [] for p in pairs}
    for k in poses:
        h.d.qpos[:] = poses[k]
        mujoco.mj_kinematics(m, h.d)
        pc = {g: grp[(g, coarse)].poses(h.d) for g in geoms}
        for a, b in pairs:
            d, _ = group_distance(grp[(a, coarse)], pc[a], grp[(b, coarse)], pc[b], cutoff)
            if 0.75 * fine <= d < cutoff:
                d, _ = group_distance(grp[(a, fine)], pc[a], grp[(b, fine)], pc[b], d + coarse)
            res[(a, b)].append(d)
    out = {}
    for (a, b), d in res.items():
        ka, kb = key_of(h, a), key_of(h, b)
        out[repr(tuple(sorted((ka, kb))))] = d
    return model, dig, [list(k) for k in poses], out


# ------------------------------------------------------------------ job: two digits
def job_pair(model, a, b, n, coarse, fine):
    h = hand(model)
    poses, _ = settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n))
    keys = list(poses)
    touch = 0.75 * fine
    table, who = clearance_table(h, poses, digit_geoms(h, a), digit_geoms(h, b), coarse, fine, touch)
    parts = {}
    for i, j in zip(*np.nonzero(table < touch)):
        pk = tuple(sorted((geom_label(h, who[(i, j)][0]).split(" (")[0], geom_label(h, who[(i, j)][1]).split(" (")[0])))
        parts[pk] = parts.get(pk, 0) + 1
    i, j = np.unravel_index(np.argmin(table), table.shape)
    worst = (float(table[i, j]), fmt_pose(keys[i]), fmt_pose(keys[j]),
             [geom_label(h, g) for g in who[(i, j)]] if (i, j) in who else None)
    return model, a, b, [list(k) for k in keys], table.tolist(), {" / ".join(k): v for k, v in parts.items()}, worst


# ------------------------------------------------------------------ job: palm/back shells
def job_shells(model, n, coarse, fine):
    h = hand(model, with_shells=True)
    touch = 0.75 * fine
    poses, _ = settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n))
    keys = list(poses)
    names = list(shells("5f"))
    sg = {s: {sp: Group(h, [h.part_geom[s]], sp) for sp in (coarse, fine)} for s in names}
    mujoco.mj_kinematics(h.m, h.d)
    sp_pose = {s: sg[s][fine].poses(h.d) for s in names}
    out = {}
    for dig in h.digits:
        g = {sp: Group(h, h.geoms[dig][0], sp) for sp in (coarse, fine)}
        pa = posed(h, g[coarse], poses)
        for s in names:
            res = []
            for p in pa:
                d, who = group_distance(g[coarse], p, sg[s][coarse], sp_pose[s], 0.005)
                if touch <= d < 0.003:
                    d, who = group_distance(g[fine], p, sg[s][fine], sp_pose[s], d + coarse)
                res.append((d, geom_label(h, who[0]) if who else ""))
            i = int(np.argmin([r[0] for r in res]))
            out[f"{dig}|{s}"] = (float(res[i][0]), fmt_pose(keys[i]), res[i][1], int(sum(r[0] < touch for r in res)))
    return model, out


# ------------------------------------------------------------------ job: structure + demo poses
def job_misc(model, fine):
    h = hand(model)
    m = h.m
    out = {"counts": dict(nbody=m.nbody, njnt=m.njnt, nq=m.nq, nu=m.nu, neq=m.neq, ngeom=m.ngeom, nkey=m.nkey)}
    # quick stability: every actuator 0 -> max -> min -> 0, then the demo poses
    h.reset()
    worst_eq, nan = 0.0, False
    for a in range(m.nu):
        for target in (m.actuator_ctrlrange[a, 1], m.actuator_ctrlrange[a, 0], 0.0):
            start = h.d.ctrl[a]
            for k in range(600):
                h.d.ctrl[a] = start + (target - start) * min(1.0, (k + 1) / 300)
                mujoco.mj_step(m, h.d)
            worst_eq = max(worst_eq, h.eq_err().max())
            nan |= not np.all(np.isfinite(h.d.qpos))
    warn = {mujoco.mjtWarning(i).name: h.d.warning[i].number for i in range(len(h.d.warning)) if h.d.warning[i].number}
    out["stability"] = dict(max_eq_err_mm=worst_eq * 1e3, nan=bool(nan), warnings=warn)
    # demo poses
    sp = fine
    pairs = [(a, b) for a, b in [("index", "middle"), ("middle", "ring"), ("ring", "pinky"), ("thumb", "index"),
                                 ("thumb", "middle"), ("thumb", "ring"), ("thumb", "pinky")]]
    groups = {dig: Group(h, digit_geoms(h, dig), sp) for dig in h.digits}
    demo = {}
    for name, pose in DEMO_POSES_5F.items():
        h.reset()
        target = pose_ctrl(h, pose)
        start = h.d.ctrl.copy()
        for k in range(500):
            h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / 250)
            mujoco.mj_step(m, h.d)
        row = {}
        for a, b in pairs:
            d, _ = group_distance(groups[a], groups[a].poses(h.d), groups[b], groups[b].poses(h.d), 0.005, skip_root_pair=True)
            row[f"{a}/{b}"] = float(d)
        demo[name] = row
    out["demo"] = demo
    return model, out


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--coarse", type=float, default=1.0)
    ap.add_argument("--fine", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "reports" / "verify_slim.txt")
    args = ap.parse_args()
    n, coarse, fine = args.grid, args.coarse * 1e-3, args.fine * 1e-3
    touch = 0.75 * fine
    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        log(s)

    with ProcessPoolExecutor(args.jobs) as ex:
        fut_pairs = [] if args.skip_pairs else [ex.submit(job_pair, mdl, a, b, n, coarse, fine) for mdl in MODELS
                                                for a, b in (("ring", "pinky"), ("middle", "ring"), ("index", "middle"))]
        fut_self = [ex.submit(job_self, mdl, dig, n, coarse, fine) for mdl in MODELS for dig in DIGITS]
        fut_shells = [ex.submit(job_shells, mdl, n, coarse, fine) for mdl in MODELS]
        fut_misc = [ex.submit(job_misc, mdl, fine) for mdl in MODELS]
        misc = dict(f.result() for f in fut_misc)
        selfres = {}
        for f in fut_self:
            mdl, dig, keys, res = f.result()
            selfres[(mdl, dig)] = (keys, res)
        shellres = dict(f.result() for f in fut_shells)
        pairres = {}
        for f in fut_pairs:
            mdl, a, b, keys, table, parts, worst = f.result()
            pairres[(mdl, a, b)] = (keys, np.array(table), parts, worst)

    out(f"verify_slim.py: {MODELS['slim']} vs {MODELS['stock']}, {n}x{n} grid per digit over (motor1, motor2) in [-90, 90] deg, "
        f"surface samples {args.coarse} mm / {args.fine} mm, touching = sampled distance < {touch * 1e3:.3f} mm "
        f"(mujoco {mujoco.__version__}, {time.time() - t0:.0f} s)")
    out()
    out("== 1. structure and stability")
    for mdl in MODELS:
        out(f"  {mdl:<5} {misc[mdl]['counts']}  quick sweep: max equality error {misc[mdl]['stability']['max_eq_err_mm']:.3f} mm, "
            f"NaN {misc[mdl]['stability']['nan']}, warnings {misc[mdl]['stability']['warnings'] or 'none'}")

    out()
    out("== 2. each digit's own parts over its joint range (changed parts vs everything else of the digit + palm)")
    summary = {}
    for dig in DIGITS:
        keys, rs = selfres[("stock", dig)]
        _, rl = selfres[("slim", dig)]
        allk = sorted(set(rs) | set(rl))
        out(f"  -- {dig}")
        n_new, n_gone = 0, 0
        for k in allk:
            ds = np.array(rs.get(k, [np.inf] * len(keys)))
            dl = np.array(rl.get(k, [np.inf] * len(keys)))
            ts, tl = ds < touch, dl < touch
            new = tl & ~ts
            if ds.min() < 3e-3 or dl.min() < 3e-3:
                ka, kb = ast.literal_eval(k)
                tag = f"{short(ka)} / {short(kb)}"
                note = ""
                if new.any():
                    note = "  NEW at " + ", ".join(fmt_pose(keys[i]) for i in np.flatnonzero(new)[:6])
                    n_new += int(new.sum())
                n_gone += int((ts & ~tl).sum())
                out(f"    {tag:<70} stock: min {ds.min() * 1e3:5.2f} mm, touching {int(ts.sum()):2d}/{len(keys)}   "
                    f"slim: min {dl.min() * 1e3:5.2f} mm, touching {int(tl.sum()):2d}/{len(keys)}{note}")
        summary[dig] = (n_new, n_gone)
        out(f"    => new contacts (pair x pose) in the slim model: {n_new}; stock contacts that are gone: {n_gone}")

    if pairres:
        out()
        out("== 3. neighbouring digits, every pair of grid poses")
        for a, b in (("ring", "pinky"), ("middle", "ring"), ("index", "middle")):
            for mdl in MODELS:
                keys, t, parts, worst = pairres[(mdl, a, b)]
                out(f"  {mdl:<5} {a}/{b}: touching {int((t < touch).sum())}/{t.size} pose pairs; min clearance "
                    f"{worst[0] * 1e3:.2f} mm at A{worst[1]} B{worst[2]}; within 1 mm: {int((t < 1e-3).sum())}, within 3 mm: {int((t < 3e-3).sum())}")
                if parts:
                    out("        touching parts: " + ", ".join(f"{k} x{v}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1])[:10]))
            ts, tl = pairres[("stock", a, b)][1] < touch, pairres[("slim", a, b)][1] < touch
            out(f"        slim touching set is a subset of the stock set: {bool(np.all(ts | ~tl))}; "
                f"pose pairs no longer touching: {int((ts & ~tl).sum())}; new: {int((tl & ~ts).sum())}")
        keys = pairres[("slim", "ring", "pinky")][0]
        ab = np.array([np.degrees(k[0] + k[1]) / 2 for k in keys])
        closing = ab[None, :] - ab[:, None]
        for mdl in MODELS:
            t = pairres[(mdl, "ring", "pinky")][1]
            hit = t < touch
            rows = []
            for c0 in sorted(set(np.round(closing.ravel(), 1))):
                sel = np.isclose(closing, c0)
                if hit[sel].any():
                    rows.append(f"c={c0:+.1f}: {int(hit[sel].sum())}/{int(sel.sum())}")
            out(f"  {mdl:<5} ring/pinky touching vs relative abduction c = a_pinky - a_ring (deg): " + ", ".join(rows))

    out()
    out("== 4. 5-finger palm/back shells and grip pads vs the moving parts of every digit")
    for key in shellres["stock"]:
        dig, s = key.split("|")
        a, b = shellres["stock"][key], shellres["slim"][key]
        out(f"  {dig:<6} vs {s:<10} stock: min {a[0] * 1e3:5.2f} mm at {a[1]} [{a[2][:40]}], touching {a[3]}/{n * n}   "
            f"slim: min {b[0] * 1e3:5.2f} mm at {b[1]} [{b[2][:40]}], touching {b[3]}/{n * n}")

    out()
    out("== 5. demo poses (DEMO_POSES_5F): clearance between neighbouring digits, stock -> slim (mm; > 5 mm not shown)")
    for name in DEMO_POSES_5F:
        row = []
        for pair, d0 in misc["stock"]["demo"][name].items():
            d1 = misc["slim"]["demo"][name][pair]
            if min(d0, d1) < 0.005:
                row.append(f"{pair} {d0 * 1e3:.2f} -> {d1 * 1e3:.2f}" + (" TOUCH" if d1 < touch else ""))
        out(f"  {name:<15} " + (", ".join(row) if row else "all > 5 mm"))

    out()
    out(f"done in {time.time() - t0:.0f} s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    js = dict(self_new_contacts={d: v[0] for d, v in summary.items()}, self_gone={d: v[1] for d, v in summary.items()},
              pairs={f"{m}:{a}/{b}": int((v[1] < touch).sum()) for (m, a, b), v in pairres.items()},
              shells={m: {k: v[3] for k, v in r.items()} for m, r in shellres.items()})
    args.out.with_suffix(".json").write_text(json.dumps(js, indent=1))


if __name__ == "__main__":
    main()
