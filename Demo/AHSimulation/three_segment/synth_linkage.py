#!/usr/bin/env python3
"""Synthesise the 3-phalanx coupling linkage (planar, finger design frame, mm).

Kept from the stock finger: SCS0009 servos, horns, push rods, gimbal and the Link *base* (pivot and
the two rod balls), so the drive map (m1, m2) -> (abduction, Link angle psi) is exactly the stock one
measured by analyze_stock_finger.py (data/stock_flex_line.npz).

Chosen: segment lengths (human index proportions, same MCP->fingertip length as the stock finger),
then numerically optimised (scipy differential evolution, then Nelder-Mead polish):
  loop 1  psi_s (Link-base angle at which the new finger is straight; sets the new Link arm angle)
          and P4 (Link/P2 pivot, near the PIP)  -> MCP 0..90, PIP 0..100 deg at full servo travel,
          PIP ~ 1.11 MCP, good transmission angles, no dead point over the whole servo range,
          straight pose as late in the servo travel as possible (keeps abduction room).
  loop 2  A (on P1 near the PIP) and B (on P3 near the DIP), coupler L2 = |AB|
          -> DIP ~ 0.7 PIP, transmission angles, pin/boss/link clearances in the shared centre layer.
Also evaluates the "minimal-change" variant (stock Link, 52 mm proximal and stock loop 1 kept; only a
DIP coupler added) for comparison.

Writes cad/five_finger/three_segment/design.json (single source of truth for the CAD and the sim),
reports/synthesis.txt and figures/synthesis_*.png.

Usage: python synth_linkage.py [--seed 3] [--quick]
"""

import argparse
import json
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

import plotstyle as ps
from finger_common import DATA_DIR, DESIGN_JSON, FIG_DIR, REPORT_DIR, log
from linkage import O_L, O_M, Design, check_branch, cross, rot, seg_dist

# ------------------------------------------------------------------ fixed choices
STOCK_TIP = np.array([86.847, -4.436])  # stock tip site in the straight pose (y, z)
L1, L2 = 47.0, 26.0  # MCP-PIP, PIP-DIP (human index: P1:P2:P3 ~ 1 : 0.58 : 0.39-0.45)
L3T = STOCK_TIP[0] - L1 - L2  # DIP -> tip site (the fingertip ends ~6.5 mm further, like the stock)
TARGET = dict(mcp=90.0, pip=100.0, dip=70.0)
K_PIP, K_DIP = TARGET["pip"] / TARGET["mcp"], TARGET["dip"] / TARGET["pip"]

# clearances (mm): 2 mm pins, PLA bosses r = 1 + 2 (>= 1.5 mm wall + margin), 4 mm wide links
PIN_R, BOSS_R, BAR_HALF, GAP = 1.0, 3.0, 2.0, 0.8
LINK_HALF = 2.25  # in-plane half-height of the Link arm near P4
MU_MIN, MU_MIN_HYPER = 40.0, 25.0  # transmission angle margins (deg from a dead point)

LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    log(s)


drive = np.load(DATA_DIR / "stock_flex_line.npz")
F_EVAL = np.arange(-90.0, 90.01, 1.0)
PSI_EVAL = np.radians(np.interp(F_EVAL, drive["f"], drive["psi"]))


def f_of_psi(psi_rad):
    return float(np.interp(np.degrees(psi_rad), drive["psi"], drive["f"]))


def margin(mu):
    return np.minimum(mu, 180.0 - mu)


# ------------------------------------------------------------------ loop 1
def make1(x, L1=L1):
    psi_s, dy, dz = x
    return Design(L1, L2, L3T, [L1 + dy, dz], [L1 - 4, -5], [L1 + L2 - 3, 6], np.radians(psi_s), STOCK_TIP[1])


def cost1(x, detail=False):
    dz_ = make1(x)
    if not 4.8 <= dz_.c <= 8.5:
        return 1e4 + 100 * abs(dz_.c - 6)
    s = dz_.solve(PSI_EVAL)
    p1_ok = s["ok"] & (np.sign(cross(s["P4"] - O_M, s["PIP"] - O_M)) == dz_.sg1)
    if not p1_ok.all():
        return 1e4 + 100 * (~p1_ok).sum()
    mcp, pip = np.degrees(s["mcp"]), np.degrees(s["pip"])
    f_s = f_of_psi(dz_.psi_s)
    fl = F_EVAL >= f_s
    hy = ~fl
    m_fl = min(margin(s["mu_out1"][fl]).min(), margin(s["mu_in1"][fl]).min())
    m_hy = min(margin(s["mu_out1"][hy]).min(), margin(s["mu_in1"][hy]).min()) if hy.any() else 90
    c = 0.0
    c += 4.0 * max(0, TARGET["mcp"] - mcp[-1]) ** 2 + 0.3 * max(0, mcp[-1] - TARGET["mcp"] - 6) ** 2
    c += 1.0 * (pip[-1] - TARGET["pip"]) ** 2
    c += 0.5 * np.mean((pip[fl] - K_PIP * mcp[fl]) ** 2)
    c += 1.5 * max(0, -30.0 - f_s) ** 2 * 0.05 - 0.6 * f_s  # straight pose late in the travel
    c += 5.0 * max(0, MU_MIN - m_fl) ** 2 + 5.0 * max(0, MU_MIN_HYPER - m_hy) ** 2
    dm, dp = np.diff(mcp[fl]), np.diff(pip[fl])
    c += 50.0 * (np.sum(np.clip(-dm, 0, None)) + np.sum(np.clip(-dp, 0, None)))
    clr = seg_dist(s["PIP"], np.broadcast_to(O_L, s["PIP"].shape), s["P4"]).min() - (PIN_R + LINK_HALF)
    c += 200.0 * max(0, GAP - clr) ** 2
    if detail:
        return dict(cost=c, f_s=f_s, mcp90=mcp[-1], pip90=pip[-1], m_fl=m_fl, m_hy=m_hy, s=s, clr=clr)
    return c


# ------------------------------------------------------------------ loop 2
def make2(x1, x2, L1=L1, L2=L2):
    psi_s, dy, dz = x1
    ay, az, by, bz = x2
    return Design(L1, L2, STOCK_TIP[0] - L1 - L2, [L1 + dy, dz], [L1 + ay, az], [L1 + L2 + by, bz], np.radians(psi_s),
                  STOCK_TIP[1])


def clearances(dz_, s):
    """Minimum clearances (mm) over the motion, all in the shared centre layer of P2's slot."""
    A, B = s["A_p2"], s["B_p2"]  # P2 frame
    P4 = np.broadcast_to(dz_.P4_0, A.shape)
    PIP = np.broadcast_to(dz_.PIP0, A.shape)
    DIP = np.broadcast_to(dz_.DIP0, A.shape)
    out = dict(
        A_vs_P4boss=np.linalg.norm(A - P4, axis=-1).min() - (PIN_R + BOSS_R),
        L2_vs_PIPpin=seg_dist(PIP, A, B).min() - (PIN_R + BAR_HALF),
        L2_vs_DIPpin=seg_dist(DIP, A, B).min() - (PIN_R + BAR_HALF),
        L2_vs_P4boss=seg_dist(P4, A, B).min() - (BOSS_R + BAR_HALF),
    )
    # the Link arm (gimbal frame) against the A pin and against L2
    Ag, Bg, P4g = s["A"], s["B"], s["P4"]
    OL = np.broadcast_to(O_L, Ag.shape)
    out["A_vs_link"] = seg_dist(Ag, OL, P4g).min() - (PIN_R + LINK_HALF)
    out["PIP_vs_link"] = seg_dist(s["PIP"], OL, P4g).min() - (PIN_R + LINK_HALF)
    ts = np.linspace(0, 1, 15)
    dmin = np.inf
    for t in ts:  # segment-segment distance by sampling L2
        q = Ag + t * (Bg - Ag)
        dmin = min(dmin, seg_dist(q, OL, P4g).min())
    out["L2_vs_link"] = dmin - BAR_HALF - LINK_HALF
    return out


def cost2(x2, x1, detail=False, L1=L1, L2=L2, sel_from=None):
    dz_ = make2(x1, x2, L1, L2)
    if not (4.5 <= dz_.rA <= 8.0 and 4.5 <= dz_.rB <= 9.0):
        return 1e4 + 100 * (abs(dz_.rA - 6) + abs(dz_.rB - 7))
    if not (-7.5 <= dz_.A0[1] <= 7.0 and -7.0 <= dz_.B0[1] <= 9.5):
        return 1e4
    s = dz_.solve(PSI_EVAL)
    ok = s["ok"] & check_branch(dz_, s)
    if not ok.all():
        return 1e4 + 100 * (~ok).sum()
    pip, dip = np.degrees(s["pip"]), np.degrees(s["dip"])
    f_s = f_of_psi(dz_.psi_s) if sel_from is None else sel_from
    fl = F_EVAL >= f_s
    hy = ~fl
    m_fl = min(margin(s["mu_out2"][fl]).min(), margin(s["mu_in2"][fl]).min())
    m_hy = min(margin(s["mu_out2"][hy]).min(), margin(s["mu_in2"][hy]).min()) if hy.any() else 90
    cl = clearances(dz_, s)
    c = 0.0
    c += 1.0 * np.mean((dip[fl] - K_DIP * pip[fl]) ** 2)
    c += 2.0 * max(0, K_DIP * pip[-1] - 3 - dip[-1]) ** 2
    c += 5.0 * max(0, MU_MIN - m_fl) ** 2 + 5.0 * max(0, MU_MIN_HYPER - m_hy) ** 2
    c += 200.0 * sum(max(0, GAP - v) ** 2 for v in cl.values())
    dd = np.diff(dip[fl])
    c += 50.0 * np.sum(np.clip(-dd, 0, None))
    c += 0.05 * (dz_.rA + dz_.rB)  # compact
    if detail:
        return dict(cost=c, m_fl=m_fl, m_hy=m_hy, cl=cl, s=s, dip90=dip[-1])
    return c


def optimise(fun, bounds, args=(), seed=0, quick=False):
    r = differential_evolution(fun, bounds, args=args, seed=seed, tol=1e-8, polish=False,
                               maxiter=150 if quick else 600, popsize=25 if quick else 40, mutation=(0.5, 1.0),
                               recombination=0.8)
    p = minimize(fun, r.x, args=args, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-8, maxiter=4000))
    return p.x if p.fun <= r.fun else r.x


def tradeoff(x1_best):
    """Re-optimise P4 for a range of straight-pose positions: the reach / abduction-room trade-off."""
    out("\n-- trade-off: straight pose later in the servo travel = more abduction room but less flexion")
    out("   f_straight  psi_s   MCP90  PIP90  MCP+PIP  min transmission  abduction cmd room  ~abduction at straight")
    rows = []
    for psi_s in (-22.0, -18.0, -14.0, -10.0, -6.0, -2.0, 0.0):
        fun = lambda x: cost1([psi_s, x[0], x[1]])  # noqa: E731
        r = minimize(fun, x1_best[1:], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=3000))
        d = cost1([psi_s, *r.x], detail=True)
        room = 90 - abs(d["f_s"])
        rows.append((d["f_s"], psi_s, d["mcp90"], d["pip90"], d["m_fl"], room))
        out(f"   {d['f_s']:9.1f}  {psi_s:6.1f}  {d['mcp90']:6.1f}  {d['pip90']:5.1f}  {d['mcp90'] + d['pip90']:7.1f}  "
            f"{min(d['m_fl'], d['m_hy']):16.1f}  {room:12.1f} deg  {0.465 * room:12.1f} deg")
    np.save(DATA_DIR / "synthesis_tradeoff.npy", np.array(rows))


def describe(dz_, s, label):
    mcp, pip, dip = (np.degrees(s[k]) for k in ("mcp", "pip", "dip"))
    f_s = f_of_psi(dz_.psi_s)
    out(f"\n== {label}")
    out(f"  segments: P1 MCP-PIP {dz_.L1:.2f}, P2 PIP-DIP {dz_.L2:.2f}, P3 DIP-tip site {dz_.L3t:.2f} mm "
        f"(MCP->tip {dz_.L1 + dz_.L2 + dz_.L3t:.2f}; ratios 1 : {dz_.L2 / dz_.L1:.2f} : {(dz_.L3t + 6.5) / dz_.L1:.2f} "
        "to the fingertip end incl. the TPU cap)")
    out(f"  loop 1: Link pivot O_L {O_L}, MCP O_M {O_M}; P4 {np.round(dz_.P4_0, 3)} (PIP->P4 {dz_.c:.2f} mm); "
        f"Link pivot->P4 {dz_.L_link:.2f} mm")
    out(f"          straight pose at Link angle psi_s = {np.degrees(dz_.psi_s):.2f} deg (stock: 0) -> f_straight = {f_s:.1f} deg "
        f"(stock -37.5); new Link arm rotated {-np.degrees(dz_.psi_s):.2f} deg vs the stock arm about the Link pivot; "
        f"P4 in the Link frame {np.round(dz_.link_arm_in_link_frame(), 3)}")
    out(f"  loop 2: A {np.round(dz_.A0, 3)} (PIP->A {dz_.rA:.2f}), B {np.round(dz_.B0, 3)} (DIP->B {dz_.rB:.2f}), "
        f"coupler L2 = |AB| {dz_.dAB:.2f} mm")
    fl = F_EVAL >= f_s
    out(f"  at f = +90 (full servo travel): MCP {mcp[-1]:.1f}, PIP {pip[-1]:.1f}, DIP {dip[-1]:.1f} deg "
        f"(targets {TARGET['mcp']:.0f} / {TARGET['pip']:.0f} / {TARGET['dip']:.0f})")
    out(f"  at f = -90: MCP {mcp[0]:.1f}, PIP {pip[0]:.1f}, DIP {dip[0]:.1f} deg (hyperextension end of the servo travel)")
    if fl.sum() > 3:
        out(f"  coupling slopes over straight..full: PIP/MCP {np.polyfit(mcp[fl], pip[fl], 1)[0]:.3f}, "
            f"DIP/PIP {np.polyfit(pip[fl], dip[fl], 1)[0]:.3f}; rms dev. from PIP=1.11MCP "
            f"{np.sqrt(np.mean((pip[fl] - K_PIP * mcp[fl]) ** 2)):.2f} deg, from DIP=0.7PIP "
            f"{np.sqrt(np.mean((dip[fl] - K_DIP * pip[fl]) ** 2)):.2f} deg")
    for k in ("mu_in1", "mu_out1", "mu_in2", "mu_out2"):
        out(f"  transmission {k:<8} flexion range {s[k][fl].min():6.1f}..{s[k][fl].max():6.1f} deg, "
            f"whole servo range {s[k].min():6.1f}..{s[k].max():6.1f} (dead point = 0/180)")
    cl = clearances(dz_, s)
    out("  centre-layer clearances over the whole servo range (mm, > 0 = clear): " +
        ", ".join(f"{k} {v:.2f}" for k, v in cl.items()))
    return dict(f_s=f_s, mcp=mcp, pip=pip, dip=dip, cl=cl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out("== 3-phalanx coupling synthesis (planar, mm, finger design frame: y distal, z dorsal)")
    out(f"drive map: stock Link angle psi(f) at a = 0 from analyze_stock_finger.py, f = -90..+90 deg -> "
        f"psi {np.degrees(PSI_EVAL[0]):.1f}..{np.degrees(PSI_EVAL[-1]):.1f} deg")
    out(f"fixed: P1 {L1}, P2 {L2}, P3 (to tip site) {L3T:.3f} mm; targets at f = +90: {TARGET}; "
        f"coupling targets PIP = {K_PIP:.3f} MCP, DIP = {K_DIP:.2f} PIP")

    # loop 1
    b1 = [(-30.0, 12.0), (-9.0, 6.0), (-9.0, 9.5)]
    x1 = optimise(cost1, b1, seed=args.seed, quick=args.quick)
    r1 = cost1(x1, detail=True)
    out(f"\nloop 1 optimum x = psi_s {x1[0]:.3f} deg, P4 - PIP = ({x1[1]:.3f}, {x1[2]:.3f}) mm; cost {r1['cost']:.3f}; "
        f"f_straight {r1['f_s']:.1f}, MCP90 {r1['mcp90']:.1f}, PIP90 {r1['pip90']:.1f}, margins {r1['m_fl']:.1f} / "
        f"{r1['m_hy']:.1f} deg  [{time.time() - t0:.0f} s]")
    # loop 2
    b2 = [(-9.0, 3.0), (-8.0, 2.0), (-8.0, 4.0), (-2.0, 9.5)]
    x2 = optimise(cost2, b2, args=(x1,), seed=args.seed, quick=args.quick)
    r2 = cost2(x2, x1, detail=True)
    out(f"loop 2 optimum x = A - PIP ({x2[0]:.3f}, {x2[1]:.3f}), B - DIP ({x2[2]:.3f}, {x2[3]:.3f}) mm; cost {r2['cost']:.3f}"
        f"  [{time.time() - t0:.0f} s]")
    tradeoff(x1)
    # round to 0.05 mm / 0.05 deg (CAD-friendly) and re-evaluate
    x1r, x2r = np.round(np.array(x1) / 0.05) * 0.05, np.round(np.array(x2) / 0.05) * 0.05
    dz_ = make2(x1r, x2r)
    s = dz_.solve(PSI_EVAL)
    assert (s["ok"] & check_branch(dz_, s)).all()
    res = describe(dz_, s, "RECOMMENDED: 3 phalanges, 2 servos, new Link arm + DIP coupler (dimensions rounded to 0.05)")

    # minimal-change variant: stock loop 1 (Link, 52 mm proximal, P4) + DIP coupler only
    L1m = 52.0
    L2m = 21.0
    x1m = np.array([0.0, 49.216 - 52.0, 5.315])
    x2m = optimise(lambda x: cost2(x, x1m, L1=L1m, L2=L2m), b2, seed=args.seed, quick=True)
    dzm = make2(x1m, np.round(x2m / 0.05) * 0.05, L1m, L2m)
    sm = dzm.solve(PSI_EVAL)
    resm = describe(dzm, sm, "ALTERNATIVE (minimal change): stock Link + stock 52 mm proximal geometry, DIP coupler added")

    # ------------------------------------------------------------------ json
    DESIGN_JSON.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(DESIGN_JSON.read_text()) if DESIGN_JSON.exists() else {}
    design = dict(
        units="mm, deg", frame="finger design frame of the stock CAD: origin MCP/abduction-axis intersection, "
                               "x lateral (pin axes), y distal along the straight finger, z dorsal; points are (y, z) "
                               "in the straight pose",
        geometry=dict(L1=dz_.L1, L2=dz_.L2, L3_tip=round(dz_.L3t, 3), L3_end=round(dz_.L3t + 4.65, 3),  # core end; the TPU cap adds ~1.9 mm (stock tip at 93.4)
                      O_M=O_M.tolist(), O_L=O_L.tolist(), PIP=dz_.PIP0.tolist(), DIP=dz_.DIP0.tolist(),
                      P4=dz_.P4_0.round(3).tolist(), A=dz_.A0.round(3).tolist(), B=dz_.B0.round(3).tolist(),
                      tip=dz_.TIP0.round(3).tolist(), P4_in_link_frame=dz_.link_arm_in_link_frame().round(4).tolist(),
                      link_length=round(dz_.L_link, 3), crank_P2=round(dz_.c, 3), crank_A=round(dz_.rA, 3),
                      crank_B=round(dz_.rB, 3), coupler_L2=round(dz_.dAB, 3)),
        drive=dict(psi_straight_deg=round(float(np.degrees(dz_.psi_s)), 3), f_straight_deg=round(res["f_s"], 2),
                   f_full_deg=90.0, stock_f_straight_deg=-37.5),
        predicted=dict(mcp_at_f90=round(res["mcp"][-1], 2), pip_at_f90=round(res["pip"][-1], 2),
                       dip_at_f90=round(res["dip"][-1], 2), mcp_at_fm90=round(res["mcp"][0], 2),
                       pip_at_fm90=round(res["pip"][0], 2), dip_at_fm90=round(res["dip"][0], 2)),
        minimal_variant=dict(L1=L1m, L2=L2m, A=dzm.A0.round(3).tolist(), B=dzm.B0.round(3).tolist(),
                             mcp_at_f90=round(resm["mcp"][-1], 2), pip_at_f90=round(resm["pip"][-1], 2),
                             dip_at_f90=round(resm["dip"][-1], 2)),
        pins=dict(diameter=2.0, note="stock parallel pins: 2x16 (MCP, PIP, A: through the 16 mm P1 fork), "
                                     "2x10 (Link pivot, P4), 2x14 (DIP, B: through the 14 mm P3 fork)"),
    )
    if "cad" in old:
        design["cad"] = old["cad"]  # written by make_three_segment.py
    DESIGN_JSON.write_text(json.dumps(design, indent=2) + "\n")
    out(f"\nwrote {DESIGN_JSON}")

    # ------------------------------------------------------------------ figures
    import matplotlib.pyplot as plt
    ps.setup()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for k, (lab, col) in enumerate((("MCP", ps.C[0]), ("PIP", ps.C[1]), ("DIP", ps.C[2]))):
        key = lab.lower()
        ax[0].plot(F_EVAL, res[key], color=col, label=f"{lab} (3-phalanx)")
    ax[0].plot(drive["f"], drive["mcp"], color=ps.C[0], lw=1.2, ls=(0, (4, 2)), label="MCP (stock)")
    ax[0].plot(drive["f"], drive["pip"], color=ps.C[1], lw=1.2, ls=(0, (4, 2)), label="PIP (stock)")
    ax[0].axvline(res["f_s"], color=ps.INK2, lw=1)
    ax[0].axvline(-37.5, color=ps.GRID, lw=1)
    ax[0].annotate(f"straight\nf = {res['f_s']:.1f}", (res["f_s"], -25), xytext=(4, 0), textcoords="offset points",
                   fontsize=8, color=ps.INK2)
    ax[0].set_xlabel("flexion command f = (m1 - m2)/2 (deg), a = 0")
    ax[0].set_ylabel("joint angle (deg)")
    ax[0].set_title("Joint angles over the servo travel (planar model)")
    ax[0].legend(loc="lower right", fontsize=7.5)
    fl = F_EVAL >= res["f_s"]
    ax[1].plot(res["mcp"][fl], res["pip"][fl], color=ps.C[1], label="PIP(MCP), 3-phalanx")
    ax[1].plot(res["pip"][fl], res["dip"][fl], color=ps.C[2], label="DIP(PIP), 3-phalanx")
    ax[1].plot(drive["mcp"][drive["f"] >= -37.5], drive["pip"][drive["f"] >= -37.5], color=ps.C[0],
               label="PIP(MCP), stock")
    ax[1].plot([0, 90], [0, 100], color=ps.INK2, lw=1, ls=(0, (2, 2)), label="target PIP = 1.11 MCP")
    ax[1].plot([0, 100], [0, 70], color=ps.INK2, lw=1, ls=(0, (6, 2)), label="target DIP = 0.7 PIP")
    ax[1].set_xlabel("proximal joint angle (deg)")
    ax[1].set_ylabel("distal joint angle (deg)")
    ax[1].set_title("Coupling curves (straight -> full flexion)")
    ax[1].legend(loc="upper left", fontsize=7.5)
    for k, (key, lab) in enumerate((("mu_in1", "Link / P2"), ("mu_out1", "P2 / P1"), ("mu_in2", "P1 crank / L2"),
                                    ("mu_out2", "L2 / P3 crank"))):
        ax[2].plot(F_EVAL, margin(s[key]), color=ps.C[k], label=lab)
    ax[2].axhspan(0, MU_MIN, color=ps.C[4], alpha=0.08, lw=0)
    ax[2].axvline(res["f_s"], color=ps.INK2, lw=1)
    ax[2].set_ylim(0, 92)
    ax[2].set_xlabel("flexion command f (deg)")
    ax[2].set_ylabel("distance from a dead point (deg)")
    ax[2].set_title(f"Transmission angles (shaded: < {MU_MIN:.0f} deg)")
    ax[2].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthesis_coupling.png", dpi=130)
    plt.close(fig)
    synthesis_diagram(dz_, FIG_DIR / "synthesis_kinematic_diagram.png")
    out(f"figures: synthesis_coupling.png, synthesis_kinematic_diagram.png  [{time.time() - t0:.0f} s]")
    (REPORT_DIR / "synthesis.txt").write_text("\n".join(LINES) + "\n")


def synthesis_diagram(dz_, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    ps.setup()
    fs = [dz_.psi_s, np.radians(np.interp(0.0, drive["f"], drive["psi"])), PSI_EVAL[-1]]
    titles = ["straight", "f = 0", "full servo travel f = +90"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, psi, title in zip(axes, fs, titles):
        s = dz_.solve_psi(psi)
        P4, PIP, DIP, A, B, TIP = s["P4"], s["PIP"], s["DIP"], s["A"], s["B"], s["TIP"]
        base = O_L + rot(np.array([3.596, -11.051]) - O_L, psi)  # rod-ball centre (stock link frame)
        ax.plot(*np.c_[O_M, O_L], color=ps.INK2, lw=5, alpha=0.35, solid_capstyle="round")
        ax.plot(*np.c_[O_L, base], color=ps.C[0], lw=2)
        ax.plot(*np.c_[O_L, P4], color=ps.C[0], lw=3, label="Link (input; stock base)")
        ax.plot(*np.c_[O_M, PIP], color=ps.C[1], lw=4, label="P1 proximal")
        ax.plot(*np.c_[PIP, A], color=ps.C[1], lw=2)
        ax.plot(*np.c_[PIP, DIP], color=ps.C[2], lw=4, label="P2 middle")
        ax.plot(*np.c_[PIP, P4], color=ps.C[2], lw=2)
        ax.plot(*np.c_[DIP, TIP], color=ps.C[3], lw=4, label="P3 distal")
        ax.plot(*np.c_[DIP, B], color=ps.C[3], lw=2)
        ax.plot(*np.c_[A, B], color=ps.C[4], lw=2.5, label="L2 DIP coupler")
        for pt, lab, off in ((O_M, "MCP", (-22, 4)), (O_L, "Link pivot", (-50, -4)), (PIP, "PIP", (4, 4)),
                             (P4, "P4", (4, -2)), (DIP, "DIP", (4, 4)), (A, "A", (4, -8)), (B, "B", (4, 2))):
            ax.add_patch(Circle(pt, 1.0, color=ps.SURFACE, ec=ps.INK, lw=1.1, zorder=5))
            ax.annotate(lab, pt, xytext=off, textcoords="offset points", fontsize=8, color=ps.INK2)
        ax.plot(*TIP, "o", color=ps.C[3], ms=6, mec=ps.SURFACE, mew=1.5, zorder=6)
        ax.set_title(f"{title}: psi {np.degrees(psi):.1f}, MCP {np.degrees(s['mcp']):.1f}, PIP {np.degrees(s['pip']):.1f}, "
                     f"DIP {np.degrees(s['dip']):.1f}", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xlim(-40, 95)
        ax.set_ylim(-80, 25)
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm, dorsal +)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("3-phalanx finger, flexion plane: loop 1 gimbal-Link-P2-P1 (crossed), loop 2 P2-P1-L2-P3 (crossed)",
                 color=ps.INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
