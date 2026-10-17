#!/usr/bin/env python3
"""Neighbour-convergence sweep of the 3-phalanx fingers of AH_Right_5F_3S: how far may two neighbouring fingers converge
when one of them is curled? Checks converge_limit() of PythonExample/AmazingHand_Demo_5F.py.

f = (m1 - m2)/2, a = (m1 + m2)/2 (> 0 towards the index), c = a of the pinky-side finger - a of the index-side finger.
  1. every finger settled on a grid over its allowed range (f in [-60, 90], |a| <= 90 - |f|, |a| <= 30; the index and
     pinky spreading outward beyond 30 only moves them away from their neighbour, so it is left out); for index/middle,
     middle/ring and ring/pinky the min surface distance of the two fingers (moving parts, shells, pads, and their
     palm-fixed servos and frames) for every pair of grid poses, with the method of verify_hand.py /
     five_finger/verify_5f_model.py (samples every 1 mm, refined to 0.3 mm below 3 mm; "touching" < 0.225 mm);
  2. min clearance by the curled finger's f and c (either finger of a pair curled), and by the more curled finger's f;
  3. converge_limit() of the demo: the pose pairs it allows beyond 15 deg, on the grid and on a 5 x 5 local grid per
     finger (f +-3.75, a +-2.5) around the closest ones;
  4. every bus frame of the demo (start, demo(), stop) played in real time in the sim, clearance after every frame;
  5. exact distance (manifold3d min_gap on the full-resolution meshes) at the closest refined pose pairs, at the limit's
     corner and at Victory / Peace.
Memory: ~3 GB of surface samples, built once and shared copy-on-write with forked workers (~0.3 GB each); a watchdog
stops everything once the summed physical footprint goes over --max-gb (default 6). macOS (libproc).

Writes reports/neighbour_sweep.txt; the grid table is cached in $TMPDIR (--reuse analyses it again without the sweep).
Usage: python neighbour_sweep.py [--workers 6] [--max-gb 6] [--reuse]
"""

import argparse
import contextlib
import ctypes
import gc
import math
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "five_finger"))
sys.path.insert(0, str(REPO / "PythonExample"))
sys.path.insert(0, str(HERE))
import AmazingHand_Demo_5F as D  # noqa: E402
import verify_5f_model as V  # noqa: E402
from verify_hand import Tee  # noqa: E402

warnings.filterwarnings("ignore", ".*multi-threaded, use of fork.*", DeprecationWarning)  # numpy/scipy only in workers
HAND = V.SIM_DIR / "AH_Right_5F_3S" / "mjcf"
CACHE = Path(tempfile.gettempdir()) / "ah_neighbour_sweep.npz"
FINGERS = ("index", "middle", "ring", "pinky")
PAIRS = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))
F_VALUES = [-60, -56, -52.5, -45, -37.5, -30, -22.5, -15, -7.5, 0, 7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60, 67.5, 75,
            82.5, 90]
A_STEP = 5.0
COARSE, FINE, TOUCH = 1e-3, 0.3e-3, 0.225e-3
CUTOFF, REFINE_BELOW = 5e-3, 3e-3
C_BINS = [-60, 0, 15, 20, 25, 30, 35, 40, 45, 60]
THUMB = D.THUMB_HOME


# ---------------------------------------------------------------- memory guard
class _RUsage(ctypes.Structure):  # rusage_info_v0 (libproc.h)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint(pid):
    """Physical footprint of a process (bytes, Activity Monitor's "Memory"); pages a forked child still shares
    copy-on-write with the parent are not counted in the child. 0 if the process is gone."""
    ru = _RUsage()
    return ru.phys_footprint if _LIBPROC.proc_pid_rusage(int(pid), 0, ctypes.byref(ru)) == 0 else 0


def total_footprint():
    return footprint(os.getpid()) + sum(footprint(p.pid) for p in mp.active_children())


PEAK = [0]


def watchdog(limit_gb, period=0.25):
    """Kill this process and its workers once their summed physical footprint goes over limit_gb."""
    def run():
        while True:
            kids = [p.pid for p in mp.active_children()]
            tot = footprint(os.getpid()) + sum(footprint(k) for k in kids)
            PEAK[0] = max(PEAK[0], tot)
            if tot > limit_gb * 2 ** 30:
                sys.__stderr__.write(f"\nWATCHDOG: footprint {tot / 2 ** 30:.2f} GB > {limit_gb} GB, killing {len(kids)} "
                                     "workers and the main process\n")
                for k in kids:
                    with contextlib.suppress(OSError):
                        os.kill(k, 9)
                os._exit(3)
            time.sleep(period)
    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- poses
def pose_grid():
    out = []
    for f in F_VALUES:
        lim = min(D.A_INNER, D.M_MAX - abs(f))
        a_vals = sorted(set(np.round(np.arange(-np.floor(lim / A_STEP) * A_STEP, lim + 1e-9, A_STEP), 6)) | {-lim, lim})
        out += [(float(f), float(a)) for a in a_vals]
    return np.array(out)


def allowed(f, a):
    return D.F_MIN - 1e-9 <= f <= D.F_MAX + 1e-9 and abs(a) <= min(D.A_INNER, D.M_MAX - abs(f)) + 1e-9


def settle_list(h, fa, ramp=100, hold=100):
    """Every finger settled at each (f, a) in turn (as verify_5f_model.settle_grid); qpos per pose, max loop error."""
    m, d = h.m, h.d
    h.reset()
    qpos, worst = np.zeros((len(fa), m.nq)), 0.0
    for i, (f, a) in enumerate(fa):
        target = V.pose_ctrl(h, dict({k: D.finger_m(f, a) for k in FINGERS}, thumb=THUMB))
        start = d.ctrl.copy()
        for k in range(ramp + hold):
            d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / ramp)
            mujoco.mj_step(m, d)
        worst = max(worst, max(h.eq_err(h.eqs[k]).max() for k in FINGERS))
        qpos[i] = d.qpos
    return qpos, worst


def serpentine(fa):
    order = []
    for k, f in enumerate(sorted(set(fa[:, 0]))):
        idx = [i for i in np.lexsort((fa[:, 1], fa[:, 0])) if fa[i, 0] == f]
        order += idx if k % 2 == 0 else idx[::-1]
    return np.array(order)


def settle_sorted(h, fa, **kw):
    order = serpentine(fa)
    q, worst = settle_list(h, fa[order], **kw)
    qpos = np.zeros_like(q)
    qpos[order] = q
    return qpos, worst


# ---------------------------------------------------------------- distances (forked workers read W)
W = {}


def build_groups(h):
    W["groups"] = {(dig, sp): V.Group(h, V.digit_geoms(h, dig), sp) for dig in FINGERS for sp in (COARSE, FINE)}


def pose_bodies(h, qpos):
    W["posed"] = {}
    for key, g in W["groups"].items():
        out = []
        for q in qpos:
            h.d.qpos[:] = q
            mujoco.mj_kinematics(h.m, h.d)
            out.append(g.poses(h.d))
        W["posed"][key] = out


def distance(L, i, R, j):
    """Min surface distance (m, inf if > CUTOFF) of finger L at pose i and finger R at pose j, and the two geoms."""
    G, P = W["groups"], W["posed"]
    d, w = V.group_distance(G[(L, COARSE)], P[(L, COARSE)][i], G[(R, COARSE)], P[(R, COARSE)][j], CUTOFF,
                            skip_root_pair=True)
    if TOUCH <= d < REFINE_BELOW:
        d, w = V.group_distance(G[(L, FINE)], P[(L, FINE)][i], G[(R, FINE)], P[(R, FINE)][j], d + COARSE,
                                skip_root_pair=True)
    return d, w


def work_rows(task):
    (L, R), rows, n = task
    dist, who = np.full((len(rows), n), np.inf), np.full((len(rows), n, 2), -1, np.int32)
    for r, i in enumerate(rows):
        for j in range(n):
            dist[r, j], w = distance(L, i, R, j)
            if w:
                who[r, j] = w
    return (L, R), rows, dist, who


def work_list(task):
    return [(p, i, j) + distance(p[0], i, p[1], j) for p, i, j in task]


def pool_map(fn, tasks, workers):
    with mp.get_context("fork").Pool(workers) as pool:
        yield from pool.imap_unordered(fn, tasks)


# ---------------------------------------------------------------- report helpers
def mm(x):
    return f"{x * 1e3:5.2f}" if np.isfinite(x) else "  >5 "


def lab(h, g):
    return V.geom_label(h, int(g)).split(" (")[0]


def who_str(h, w):
    return f" [{lab(h, w[0])} / {lab(h, w[1])}]" if w is not None and w[0] >= 0 else ""


def grids(fa, D_):
    fL, aL = np.broadcast_to(fa[:, 0][:, None], D_.shape), np.broadcast_to(fa[:, 1][:, None], D_.shape)
    fR, aR = np.broadcast_to(fa[:, 0][None, :], D_.shape), np.broadcast_to(fa[:, 1][None, :], D_.shape)
    return fL, aL, fR, aR, aR - aL


def limit_of(fl, fr):
    return np.vectorize(D.converge_limit)(fl, fr)


def bin_table(title, rows, key, D_, c):
    V.log(f"  {title}")
    V.log("     f     " + " ".join(f"{f'({lo},{hi}]':>9}" for lo, hi in zip(C_BINS, C_BINS[1:])))
    for f0 in rows:
        cells = []
        for lo, hi in zip(C_BINS, C_BINS[1:]):
            sel = (key == f0) & (c > lo) & (c <= hi)
            cells.append(mm(D_[sel].min()) if sel.any() else "   -  ")
        V.log(f"    {f0:6.1f} " + " ".join(f"{x:>9}" for x in cells))


# ---------------------------------------------------------------- 4. the demo's frames
def record_demo():
    events = []

    class Rec(D.DryBus):
        def __init__(self):
            super().__init__(start=dict(index=(-20, 70), middle=(0, 0), ring=(0, 0), pinky=(0, 0), thumb=(0, 0)))

        def write_positions(self, ids, rad):
            events.append(("frame", {d: tuple(math.degrees(rad[D.ALL_IDS.index(i)]) - D.MiddlePos[i - 1] for i in D.IDS[d])
                                     for d in D.DIGITS}))

        def write_speeds(self, ids, rad_s):
            pass

        def torque(self, ids, on):
            pass

        def write_torque_limit(self, i, raw):
            self.limit[i] = int(raw)

    hand = D.Hand(Rec(), log=lambda *a: None)
    hand.hold = lambda s: events.append(("hold", s))
    hand.start(D.TORQUE_LIMITS["pa-cf"])
    D.demo(hand)
    hand.stop()
    return events


def play(h, events):
    """Real time: 10 sim steps (20 ms) per 50 Hz frame, the holds as they are; the state after each."""
    h.reset()
    frames, qpos = [], []
    for kind, x in events:
        if kind == "frame":
            h.d.ctrl[:] = V.pose_ctrl(h, x)
            for _ in range(400 if not frames else 10):  # first: the hand at rest where it starts
                mujoco.mj_step(h.m, h.d)
        else:
            for _ in range(int(round(x / h.m.opt.timestep))):
                mujoco.mj_step(h.m, h.d)
            x = frames[-1]
        frames.append(x)
        qpos.append(h.d.qpos.copy())
    return frames, np.array(qpos)


# ---------------------------------------------------------------- 5. exact distances
def exact_distance(h, cache, L, R, search=8.0):
    """min over the two fingers' geom pairs (not palm-fixed/palm-fixed) of manifold3d min_gap, mm (inf if > search)."""
    import manifold3d as mf

    def man(g):
        mid = int(h.m.geom_dataid[g])
        if mid not in cache:
            m = h.m
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] * 1e3
            f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            M = mf.Manifold(mf.Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=f.astype(np.uint32)))
            cache[mid] = M if M.status() == mf.Error.NoError and M.num_tri() > 0 else None
        if cache[mid] is None:
            return None
        return cache[mid].transform(np.c_[h.d.geom_xmat[g].reshape(3, 3), h.d.geom_xpos[g] * 1e3])

    fixed = set(h.geoms[L][1]) | set(h.geoms[R][1])
    ML, MR = {g: man(g) for g in V.digit_geoms(h, L)}, {g: man(g) for g in V.digit_geoms(h, R)}
    bad = [g for g, x in list(ML.items()) + list(MR.items()) if x is None]
    best, who = search, None
    for a, A in ML.items():
        if A is None:
            continue
        ba = A.bounding_box()
        for b, B in MR.items():
            if B is None or (a in fixed and b in fixed):
                continue
            bb = B.bounding_box()
            if max(max(ba[i] - bb[i + 3], bb[i] - ba[i + 3]) for i in range(3)) >= best:
                continue
            d = A.min_gap(B, best)
            if d < best:
                best, who = d, (a, b)
    return (best if who else np.inf), who, bad


def settle_pose(h, pose, steps=800):
    h.reset()
    target = V.pose_ctrl(h, pose)
    start = h.d.ctrl.copy()
    for k in range(steps):
        h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / (steps // 2))
        mujoco.mj_step(h.m, h.d)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=6.0, help="footprint cap of the main process + workers together")
    ap.add_argument("--rows", type=int, default=4, help="grid rows per task")
    ap.add_argument("--worst", type=int, default=8, help="closest newly allowed grid pairs refined per finger pair")
    ap.add_argument("--reuse", action="store_true", help=f"reuse the grid table cached in {CACHE}")
    args = ap.parse_args()
    watchdog(args.max_gb)
    with open(HERE / "reports" / "neighbour_sweep.txt", "w") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        V.log("== neighbour-convergence sweep, AH_Right_5F_3S (3-phalanx index, middle, ring, pinky on the parallel 30 mm "
              "layout): f = (m1 - m2)/2, a = (m1 + m2)/2 (> 0 towards the index), c = a_pinky_side - a_index_side")
        V.log(f"   rule under test: AmazingHand_Demo_5F.converge_limit = {D.CONVERGE_MAX:g}, rising linearly to "
              f"{D.CONVERGE_CURLED:g} as max(f_left, f_right) goes from {D.CURL_FROM:g} to {D.CURL_FULL:g}")
        h = V.Hand(HAND)
        fa = pose_grid()
        n = len(fa)
        t0 = time.time()
        qpos, worst = settle_sorted(h, fa)
        V.log(f"\n== 1. grid: {n} (f, a) poses per finger, f in {F_VALUES}, a every {A_STEP:g} deg (and +-(90 - |f|)) "
              f"within |a| <= min(30, 90 - |f|); settled in {time.time() - t0:.1f} s, max loop error {worst * 1e3:.4f} mm")
        V.log(f"   {n} x {n} = {n * n} pose pairs per finger pair, surface samples every {COARSE * 1e3:g} mm, "
              f"{FINE * 1e3:g} mm below {REFINE_BELOW * 1e3:g} mm; clearances over {CUTOFF * 1e3:g} mm are shown as '>5'")
        t0 = time.time()
        build_groups(h)
        V.log(f"   surface samples built in {time.time() - t0:.1f} s, footprint {total_footprint() / 2 ** 30:.2f} GB")
        if args.reuse and CACHE.exists() and np.array_equal(np.load(CACHE)["fa"], fa):
            z = np.load(CACHE)
            tables = {p: z["dist_" + "-".join(p)] for p in PAIRS}
            whos = {p: z["who_" + "-".join(p)] for p in PAIRS}
            V.log(f"   grid table reused from {CACHE}")
        else:
            pose_bodies(h, qpos)
            tables = {p: np.full((n, n), np.inf) for p in PAIRS}
            whos = {p: np.full((n, n, 2), -1, np.int32) for p in PAIRS}
            tasks = [(p, list(range(i, min(n, i + args.rows))), n) for p in PAIRS for i in range(0, n, args.rows)]
            t0 = time.time()
            for p, rows, dist, who in pool_map(work_rows, tasks, args.workers):
                tables[p][rows], whos[p][rows] = dist, who
            V.log(f"   {3 * n * n} pose pairs measured in {time.time() - t0:.0f} s ({args.workers} workers), "
                  f"peak footprint {PEAK[0] / 2 ** 30:.2f} GB")
            np.savez(CACHE, fa=fa, **{"dist_" + "-".join(p): tables[p] for p in PAIRS},
                     **{"who_" + "-".join(p): whos[p] for p in PAIRS})

        summary = {p: {} for p in PAIRS}
        # ---------------- 2. tables
        V.log("\n== 2. min clearance (mm) by flexion and convergence c (bins (lo, hi] in deg)")
        curl_rows = [f for f in F_VALUES if f >= 30]
        for p in PAIRS:
            L, R = p
            Dt = tables[p]
            fL, aL, fR, aR, c = grids(fa, Dt)
            cur = c <= D.CONVERGE_MAX + 1e-9
            k = np.unravel_index(np.argmin(np.where(cur, Dt, np.inf)), Dt.shape)
            summary[p]["plain"] = Dt[k]
            V.log(f"\n  {L}/{R}: inside c <= {D.CONVERGE_MAX:g} (any flexion) the min is {mm(Dt[k])} mm at {L} "
                  f"(f {fa[k[0], 0]:+.1f}, a {fa[k[0], 1]:+.1f}) / {R} (f {fa[k[1], 0]:+.1f}, a {fa[k[1], 1]:+.1f}) "
                  f"{who_str(h, whos[p][k])}; pose pairs touching (< {TOUCH * 1e3:.3f} mm) "
                  f"there: {int((Dt[cur] < TOUCH).sum())}/{int(cur.sum())}")
            bin_table(f"by f of the more curled finger, max(f_{L}, f_{R}) (rows), the other finger anywhere:",
                      F_VALUES, np.maximum(fL, fR), Dt, c)
            bin_table(f"{R} curled at f (rows), {L} anywhere:", curl_rows, fR, Dt, c)
            bin_table(f"{L} curled at f (rows), {R} anywhere:", curl_rows, fL, Dt, c)

        # ---------------- 3. the rule
        V.log(f"\n== 3. converge_limit(): the pose pairs it allows beyond c = {D.CONVERGE_MAX:g}")
        seeds = []
        for p in PAIRS:
            Dt = tables[p]
            fL, aL, fR, aR, c = grids(fa, Dt)
            lim = limit_of(fL, fR)
            new = (c > D.CONVERGE_MAX + 1e-9) & (c <= lim + 1e-9)
            Dm = np.where(new, Dt, np.inf)
            k = np.unravel_index(np.argmin(Dm), Dt.shape)
            summary[p]["grid"] = Dt[k]
            V.log(f"  {p[0]}/{p[1]}: {int(new.sum())} grid pose pairs newly allowed; min {mm(Dt[k])} mm at "
                  f"{p[0]} (f {fa[k[0], 0]:+.1f}, a {fa[k[0], 1]:+.1f}) / {p[1]} (f {fa[k[1], 0]:+.1f}, a {fa[k[1], 1]:+.1f}) "
                  f"{who_str(h, whos[p][k])}; touching: {int((Dt[new] < TOUCH).sum())}")
            for kk in np.argsort(Dm, axis=None)[:args.worst]:
                i, j = np.unravel_index(kk, Dt.shape)
                seeds.append((p, tuple(fa[i]), tuple(fa[j]), float(Dt[i, j])))
        V.log("  the same for other limits, as min clearance (mm) over the pose pairs each allows beyond 15 "
              "(cmax at f = 90, ramp start - end):")
        for x0, x1, cm in [(30, 60, 30), (45, 75, 30), (60, 75, 30), (60, 90, 30), (60, 90, 40), (30, 60, 40)]:
            parts = []
            for p in PAIRS:
                Dt = tables[p]
                fL, aL, fR, aR, c = grids(fa, Dt)
                lim = 15 + (cm - 15) * np.clip((np.maximum(fL, fR) - x0) / (x1 - x0), 0, 1)
                sel = (c > 15 + 1e-9) & (c <= lim + 1e-9)
                parts.append(f"{p[0]}/{p[1]} {mm(Dt[sel].min()) if sel.any() else '  -  '}")
            V.log(f"    cmax {cm}, f {x0}-{x1}: " + ", ".join(parts))
        # local refinement around the closest newly allowed grid pairs
        local = [(df, da) for df in (-3.75, -1.875, 0, 1.875, 3.75) for da in (-2.5, -1.25, 0, 1.25, 2.5)]
        index, poses, triples = {}, [], set()

        def pid(x):
            key = (round(x[0], 4), round(x[1], 4))
            if key not in index:
                index[key] = len(poses)
                poses.append(key)
            return index[key]

        for p, (fl, al), (fr, ar), _ in seeds:
            for dfl, dal in local:
                if not allowed(fl + dfl, al + dal):
                    continue
                for dfr, dar in local:
                    if not allowed(fr + dfr, ar + dar):
                        continue
                    c = (ar + dar) - (al + dal)
                    if D.CONVERGE_MAX + 1e-9 < c <= D.converge_limit(fl + dfl, fr + dfr) + 1e-9:
                        triples.add((p, pid((fl + dfl, al + dal)), pid((fr + dfr, ar + dar))))
        fa_loc = np.array(poses)
        q_loc, worst = settle_sorted(h, fa_loc, ramp=300, hold=200)
        pose_bodies(h, q_loc)
        triples = sorted(triples)
        res = []
        for out in pool_map(work_list, [triples[k:k + 60] for k in range(0, len(triples), 60)], args.workers):
            res += out
        V.log(f"  local refinement around the {args.worst} closest per pair (f +-3.75 in 1.875 steps, a +-2.5 in 1.25 steps, "
              f"both fingers): {len(fa_loc)} poses settled (max loop error {worst * 1e3:.4f} mm), {len(triples)} newly "
              "allowed pose pairs")
        refined = {}
        for p in PAIRS:
            rr = sorted((r for r in res if r[0] == p), key=lambda r: r[3])
            _, i, j, d, w = rr[0]
            refined[p] = (tuple(fa_loc[i]), tuple(fa_loc[j]))
            summary[p]["refined"] = d
            V.log(f"    {p[0]}/{p[1]}: min {mm(d)} mm over {len(rr)} pairs at {p[0]} (f {fa_loc[i, 0]:+.3f}, a {fa_loc[i, 1]:+.2f})"
                  f" / {p[1]} (f {fa_loc[j, 0]:+.3f}, a {fa_loc[j, 1]:+.2f}), c {fa_loc[j, 1] - fa_loc[i, 1]:+.2f}"
                  + who_str(h, w))

        # ---------------- 4. the demo's frames
        events = record_demo()
        frames, q_fr = play(h, events)
        pose_bodies(h, q_fr)
        nf = len(frames)
        res = []
        for out in pool_map(work_list, [[(p, k, k) for p in PAIRS for k in range(s, min(nf, s + 20))]
                                        for s in range(0, nf, 20)], args.workers):
            res += out
        V.log(f"\n== 4. the demo (AmazingHand_Demo_5F: start, demo(), stop): {sum(e[0] == 'frame' for e in events)} bus frames "
              f"and {sum(e[0] == 'hold' for e in events)} holds played in real time; clearance after every frame and hold")
        for p in PAIRS:
            rr = sorted((r for r in res if r[0] == p), key=lambda r: r[1])
            d = np.array([r[3] for r in rr])
            c = np.array([D.finger_fa(*fr[p[1]])[1] - D.finger_fa(*fr[p[0]])[1] for fr in frames])
            k = int(np.argmin(d))
            w = rr[k][4]
            fl, al = D.finger_fa(*frames[k][p[0]])
            fr_, ar = D.finger_fa(*frames[k][p[1]])
            more = c > D.CONVERGE_MAX + 1e-6
            summary[p]["demo"] = d[more].min() if more.any() else None
            V.log(f"  {p[0]}/{p[1]}: min {mm(d[k])} mm at frame {k} ({p[0]} f {fl:+.1f} a {al:+.1f}, {p[1]} f {fr_:+.1f} "
                  f"a {ar:+.1f})" + who_str(h, w)
                  + (f"; {int(more.sum())} frames converge by more than {D.CONVERGE_MAX:g} (up to {c.max():.1f}), min there "
                     f"{mm(d[more].min())} mm" if more.any() else f"; no frame converges by more than {D.CONVERGE_MAX:g}"))
        for name in ("Victory", "Peace"):
            pose = D.POSES[name]
            V.log(f"  {name}: middle {pose['middle']} (a {D.finger_fa(*pose['middle'])[1]:+.1f}), ring {pose['ring']}: "
                  f"converging by {D.finger_fa(*pose['ring'])[1] - D.finger_fa(*pose['middle'])[1]:.1f}, limit "
                  f"{D.converge_limit(D.finger_fa(*pose['middle'])[0], D.finger_fa(*pose['ring'])[0]):.1f}; "
                  f"clamp_hand leaves it unchanged: {D.clamp_hand(pose) == pose}")

        # ---------------- 5. exact distances (the samples freed first)
        W.clear()
        V.Group._cache.clear()
        gc.collect()
        V.log("\n== 5. exact min distance (manifold3d min_gap, full-resolution meshes, every geom pair of the two fingers but "
              "palm-fixed/palm-fixed), settled at each pose pair (the other fingers at (0, 0))")
        cache = {}
        checks = [(p, *refined[p], "closest newly allowed (refined)") for p in PAIRS]
        checks += [(("middle", "ring"), (90.0, 0.0), (-45.0, 30.0), "corner of the limit: c = 30 at f = 90"),
                   (("middle", "ring"), (-45.0, -30.0), (90.0, 0.0), "corner of the limit: c = 30 at f = 90")]
        for name in ("Victory", "Peace"):
            for p in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
                checks.append((p, D.finger_fa(*D.POSES[name][p[0]]), D.finger_fa(*D.POSES[name][p[1]]), name))
        for n_check, ((L, R), (fl, al), (fr, ar), why) in enumerate(checks):
            pose = dict({k: (0.0, 0.0) for k in FINGERS}, thumb=THUMB)
            if why in ("Victory", "Peace"):
                pose = dict(D.POSES[why])
            pose[L], pose[R] = D.finger_m(fl, al), D.finger_m(fr, ar)
            settle_pose(h, pose)
            d, w, bad = exact_distance(h, cache, L, R)
            if n_check < len(PAIRS):
                summary[(L, R)]["exact"] = d * 1e-3
            V.log(f"  {why:<34} {L} (f {fl:+.2f}, a {al:+.2f}) / {R} (f {fr:+.2f}, a {ar:+.2f}), c {ar - al:+.2f}: "
                  f"{d:.2f} mm" + (who_str(h, w) if w else " (> 8 mm)")
                  + (f"; not manifold, left out: {sorted({lab(h, g) for g in bad})}" if bad else ""))
        V.log("\n== verdict (clearance, mm; >= 1 mm required for the pose pairs converge_limit adds)")
        for p in PAIRS:
            x = summary[p]
            V.log(f"  {p[0]}/{p[1]}: inside c <= {D.CONVERGE_MAX:g} {mm(x['plain'])}; added by converge_limit: grid "
                  f"{mm(x['grid'])}, refined {mm(x['refined'])}, exact at the refined closest {mm(x['exact'])}"
                  + (f"; demo frames converging by more than {D.CONVERGE_MAX:g}: {mm(x['demo'])}" if x["demo"] is not None
                     else ""))
        worst = min(min(x["grid"], x["refined"], x["exact"]) for x in summary.values())
        V.log(f"  -> converge_limit {'HOLDS' if worst >= 1e-3 else 'DOES NOT HOLD'}: the closest pose pair it adds is "
              f"{worst * 1e3:.2f} mm apart")
        V.log(f"\n   peak footprint (main process + workers) {PEAK[0] / 2 ** 30:.2f} GB")


if __name__ == "__main__":
    main()
