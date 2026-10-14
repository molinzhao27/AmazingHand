#!/usr/bin/env python3
"""Six-bar PIP study for the 3-phalanx AmazingHand finger (planar; no CAD, no MuJoCo edits).

Question
--------
With the stock drive left exactly as it is (two SCS0009 servos, horns, rods, gimbal and Link base,
i.e. the measured Link-base angle psi(f) in three_segment/data/stock_flex_line.npz, and the stock
ground pivots MCP O_M = (0, 0) and Link pivot O_L = (0, -6) on the gimbal), can extra *passive*
links in the finger raise PIP from 78 deg (current crossed four-bar, design.json) to ~100 deg at
MCP ~85-90 deg (full servo travel f = +90), with DIP ~0.7 PIP and acceptable transmission?

Frame and conventions are those of three_segment/linkage.py: points (y, z) in mm in the finger
design frame, straight pose; y distal, z dorsal; flexion positive turns +y towards -z,
R(t) = [[c, s], [-s, c]]. P1 MCP->PIP 47, P2 PIP->DIP 26, P3 DIP->tip site 13.85 mm (unchanged).

Layouts (all 1-DOF, Link = only input; with a binary ground these are the useful six-bars)
  4bar   current topology (gimbal-Link-P2-P1, crossed), re-optimised under the same rules: baseline
  watt   (a) Watt six-bar. Loop 1 gimbal-Link-R-P1: the Link drives a rocker R pivoted on P1 at Q
         (Link/R pin P4) -> MCP. Stage 2 on P1: R (crank Q-E) -> coupler C (E-F) -> P2 (crank
         PIP-F) maps R's rotation relative to P1 to PIP with a shaped gain.
  steph  (b) Stephenson six-bar. Link -> binary K (P4-Q) -> P1 gives MCP; a second Link pin G ->
         binary D (G-H) -> P2 gives PIP (P2 is driven by the Link/P1 relative motion).
  vp     (c) Stephenson six-bar, "virtual P4": P2 hangs from the Link on two binary links
         X (U1-V1) and Y (U2-V2) instead of the single pin P4, so the instant centre of P2 relative
         to the Link moves (a physical P2 crank must be >= 6 mm, a virtual one need not be).
Loop 2 (DIP) is the existing crossed four-bar P2-P1-L2-P3 (A on P1, B on P3); it only sees the
PIP angle, so it is re-synthesised for DIP = 0.7 PIP of each new design.

Rules (all checked over the whole servo travel f = -90..+90 unless stated)
  * assembles on the straight-pose branch everywhere (closed-form circle intersections, fixed
    branch sign; for vp the input psi(alpha) must be monotonic over a single branch),
  * MCP and PIP monotonic over straight..full; MCP(f=+90) in [87, 91]; hyperextension >= -35,
  * transmission angles (distance from a dead point 0/180) >= MU_FL in the flexion range and
    >= MU_ALL over the whole travel, both ends of every four-bar (or equivalent four-bar),
  * pins on the same part >= 6 mm apart (2 mm pins, 1.5 mm walls), new pivots |z| <= 8 mm from the
    phalanx centre line and within the phalanx (checked in the host phalanx frame over the motion),
  * new bars and the Link arm clear the PIP pin (and P1's new pin) in-plane by >= 0.5 mm
    (bar half-width 2, pin radius 1), since those pins run through every layer,
  * straight pose no earlier than the current design (f_straight >= -56: same abduction room).
Searches (scipy differential evolution, deterministic seeds, + bounded Nelder-Mead polish, like
three_segment/synth_linkage.py; independent runs are spread over a process pool):
  maxpip   maximise PIP(+90) for each flexion-range transmission limit MU_LEVELS (trade-off curve),
  target   PIP(+90) 100-104 at MCP 88-90, curl no worse than the current design (rms <= 10 deg from
           PIP = 1.12 MCP), then maximise the transmission margins  -> the recommended designs,
  target6  the same with a near-proportional curl (rms <= 6 deg): sensitivity only.
The target designs are rounded to 0.05 mm, get a re-synthesised loop 2 (DIP = 0.7 PIP), and are
verified with an independent Newton solver on the generic body/pin model, which also gives the
pin-play (backlash) stack and tip-speed / PIP-gain ratios (force capacity per Link torque).

Usage:  python sixbar_study.py [--quick] [--workers 16] [--seeds 1 2 3 4]
        (full run ~10-15 min on 16 cores; --quick is a ~1 min smoke test with a small budget)
Writes sixbar_report.txt and sixbar_study.png next to this file.
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TS = REPO / "Demo" / "AHSimulation" / "three_segment"
sys.path.insert(0, str(TS))
from linkage import Design, O_L, O_M, ang, circ_int, cross, rot, seg_dist, vec_angle, wrap  # noqa: E402
import plotstyle as ps  # noqa: E402  (sets the Agg backend)

# ------------------------------------------------------------------ fixed data
DESIGN = json.loads((REPO / "cad" / "five_finger" / "three_segment" / "design.json").read_text())
L1, L2 = DESIGN["geometry"]["L1"], DESIGN["geometry"]["L2"]
PIP0, DIP0 = np.array([L1, 0.0]), np.array([L1 + L2, 0.0])
TIP0 = np.array(DESIGN["geometry"]["tip"])
_drive = np.load(TS / "data" / "stock_flex_line.npz")
F_EVAL = np.arange(-90.0, 90.01, 1.0)
PSI = np.radians(np.interp(F_EVAL, _drive["f"], _drive["psi"]))
_mono = np.argmin(_drive["psi"])  # psi(f) has a 0.07 deg dip at f = -87.5; use the monotonic part
_PSI_M, _F_M = _drive["psi"][_mono:], _drive["f"][_mono:]
PSI_S_CUR = DESIGN["drive"]["psi_straight_deg"]  # -14.25 deg -> straight at f = -56

VIOLET = "#7a4fd0"  # second new link in the sketches (the 5-colour palette is taken by Link/P1/P2/P3/first new link)
PIN_R, BAR_HALF, GAP = 1.0, 2.0, 0.5  # pin radius, link half-width, in-plane gap (mm)
PIN_MIN, Z_ENV = 6.0, 8.0  # same-part pin spacing, |z| envelope of new pivots (mm)
MCP_LO, MCP_HI, HYPER_MIN = 87.0, 91.0, -35.0
PIP_CAP = 140.0
PLAY = 0.05  # radial pin play per joint (mm), as in the README's backlash estimate
MU_LEVELS = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)  # flexion-range transmission limits swept
VP_LEVELS = (0.0, 25.0, 30.0, 35.0, 40.0)  # (c) cannot reach the target even unconstrained: fewer levels
MU_FL_REC, MU_ALL_REC = 40.0, 25.0  # the recommended-design rule (same as synth_linkage.py)
K_TARGET = 100.0 / 89.0  # target curl PIP ~1.12 MCP (synth_linkage.py: 1.11)
# allowed rms deviation from that curl: 'target' = no worse than the current design (10.2 deg), 'target6' = near-proportional
TARGET_RMS = {"target": 10.0, "target6": 6.0}

# loop 2 search box: A - PIP (y, z) palm side, B - DIP (y, z) dorsal side (crossed, like the current design)
B_LOOP2 = [(-10.0, 4.0), (-8.0, 0.0), (-10.0, 5.0), (0.0, 8.0)]
ALPHA = np.radians(np.arange(-170.0, 170.01, 0.5))  # vp: driving parameter grid (rocker X angle)
I0 = int(np.argmin(np.abs(ALPHA)))

LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def f_of_psi(psi_deg):
    return float(np.interp(psi_deg, _PSI_M, _F_M))


def margin(mu):
    return np.minimum(mu, 180.0 - mu)


def nrm(v):
    return float(np.hypot(v[0], v[1]))


def line_h(p, q):
    """Homogeneous line through points p, q (..., 2)."""
    one = np.ones(np.broadcast_shapes(np.shape(p)[:-1], np.shape(q)[:-1]) + (1,))
    P = np.concatenate([np.broadcast_to(p, one.shape[:-1] + (2,)), one], -1)
    Q = np.concatenate([np.broadcast_to(q, one.shape[:-1] + (2,)), one], -1)
    return np.cross(P, Q)


def hdir(ic, p):
    """w * (IC - p) for a homogeneous point IC: a direction from p to IC that stays defined when IC
    is at infinity (then it is the common direction of the two lines)."""
    return ic[..., :2] - ic[..., 2:3] * p


def env_v(pts, ylo, yhi):
    """Envelope violation (mm) of points (..., 2) in their host phalanx frame."""
    pts = np.atleast_2d(pts)
    return float(max(0.0, np.abs(pts[:, 1]).max() - Z_ENV) + max(0.0, ylo - pts[:, 0].min()) +
                 max(0.0, pts[:, 0].max() - yhi))


def to_p1(p, mcp):
    """Ground-frame points -> P1 frame (straight-pose coordinates); O_M is the origin."""
    return rot(p, -mcp)


def clr(pin, a, b):
    return float(seg_dist(pin, a, b).min() - (PIN_R + BAR_HALF))


def fail(n):
    return dict(ok=False, nbad=float(n))


# ------------------------------------------------------------------ layouts (closed-form position analysis)
def solve_4bar(x, psi, full=False):
    """Current topology: Link pin P4 on P2; P2 is the coupler of gimbal-Link-P2-P1."""
    psi_s, P4_0 = np.radians(x[0]), np.asarray(x[1:3])
    c = nrm(P4_0 - PIP0)
    sg = np.sign(cross(P4_0, PIP0))
    P4 = O_L + rot(P4_0 - O_L, psi - psi_s)
    PIP, ok = circ_int(O_M, L1, P4, c, sg)
    if not ok.all():
        return fail((~ok).sum())
    mcp = -wrap(ang(PIP))
    pip = wrap(-wrap(ang(P4 - PIP) - ang(P4_0 - PIP0)) - mcp)
    P4_1, OL_1 = to_p1(P4, mcp), to_p1(np.broadcast_to(O_L, P4.shape), mcp)
    s = dict(ok=True, mcp=mcp, pip=pip,
             mu=dict(link_P2crank=vec_angle(P4 - O_L, PIP - P4), P2crank_P1=vec_angle(PIP, P4 - PIP)),
             pairs=dict(P2_crank_P4_PIP=c),
             env=dict(P4=env_v(P4_1, 10, L1 + 6)),
             clear=dict(link_vs_PIPpin=clr(PIP0, OL_1, P4_1)))
    s["p1pins"], s["p2pins"] = {}, dict(P4=P4_0)
    s["bars_p1"] = dict(Link=(OL_1, P4_1))
    if full:
        s["pts"] = dict(P4=P4)
        s["bars"] = [("Link", "O_L", "P4", 0)]
        s["joints"] = [("g", "link", O_L), ("g", "p1", O_M), ("link", "p2", P4_0), ("p1", "p2", PIP0)]
        s["geom"] = dict(P4=P4_0)
    return s


def solve_watt(x, psi, full=False):
    """(a) Watt: Link -(P4)- R -(Q on P1); R -(E)- C -(F)- P2."""
    psi_s = np.radians(x[0])
    P4_0, Q0, E0, F0 = (np.asarray(x[i:i + 2]) for i in (1, 3, 5, 7))
    rQ, cR, rE = nrm(Q0), nrm(P4_0 - Q0), nrm(E0 - Q0)
    rF, cC = nrm(F0 - PIP0), nrm(F0 - E0)
    sg1 = np.sign(cross(P4_0, Q0))
    sg2 = np.sign(cross(E0 - PIP0, F0 - PIP0))
    P4 = O_L + rot(P4_0 - O_L, psi - psi_s)
    Q, ok1 = circ_int(O_M, rQ, P4, cR, sg1)
    if not ok1.all():
        return fail(2 * (~ok1).sum())
    mcp = -wrap(ang(Q) - ang(Q0))
    thR = -wrap(ang(P4 - Q) - ang(P4_0 - Q0))
    rel = wrap(thR - mcp)  # R relative to P1
    E = Q0 + rot(E0 - Q0, rel)  # P1 frame
    F, ok2 = circ_int(PIP0, rF, E, cC, sg2)  # P1 frame
    if not ok2.all():
        return fail((~ok2).sum())
    pip = -wrap(ang(F - PIP0) - ang(F0 - PIP0))
    P4_1, OL_1 = to_p1(P4, mcp), to_p1(np.broadcast_to(O_L, P4.shape), mcp)
    Qb = np.broadcast_to(Q0, E.shape)
    s = dict(ok=True, mcp=mcp, pip=pip,
             mu=dict(link_R=vec_angle(P4 - O_L, Q - P4), R_P1=vec_angle(Q, P4 - Q),
                     Rcrank_C=vec_angle(E - Q0, F - E), C_P2crank=vec_angle(F - E, F - PIP0)),
             pairs=dict(R_P4_Q=cR, R_Q_E=rE, R_P4_E=nrm(E0 - P4_0), C_E_F=cC, P1_Q_PIP=nrm(Q0 - PIP0),
                        P2_F_PIP=rF, P2_F_DIP=nrm(F0 - DIP0)),
             env=dict(Q=env_v(Q0, 10, L1 + 2), P4=env_v(P4_1, 10, L1 + 6), E=env_v(E, 10, L1 + 6),
                      F=env_v(F0, L1 - 8, L1 + L2)),
             clear=dict(link_vs_PIPpin=clr(PIP0, OL_1, P4_1), R_QP4_vs_PIPpin=clr(PIP0, Qb, P4_1),
                        R_QE_vs_PIPpin=clr(PIP0, Qb, E), C_vs_PIPpin=clr(PIP0, E, F),
                        link_vs_Qpin=clr(Q0, OL_1, P4_1), C_vs_Qpin=clr(Q0, E, F)))
    s["p1pins"] = dict(Q=np.broadcast_to(Q0, E.shape), E=E)  # pins living on P1 or moving within it
    s["p2pins"] = dict(F=F0)
    s["bars_p1"] = dict(Link=(OL_1, P4_1), R_QP4=(Qb, P4_1), R_QE=(Qb, E), C=(E, F))
    if full:
        s["pts"] = dict(P4=P4, Q=Q, E=rot(E, mcp), F=rot(F, mcp))
        s["bars"] = [("Link", "O_L", "P4", 0), ("R", "Q", "P4", 4), ("R", "Q", "E", 4), ("C", "E", "F", 3)]
        s["joints"] = [("g", "link", O_L), ("g", "p1", O_M), ("link", "r", P4_0), ("r", "p1", Q0),
                       ("r", "c", E0), ("c", "p2", F0), ("p1", "p2", PIP0)]
        s["geom"] = dict(P4=P4_0, Q=Q0, E=E0, F=F0)
    return s


def solve_steph(x, psi, full=False):
    """(b) Stephenson: Link -(P4)- K -(Q)- P1 for MCP; Link -(G)- D -(H)- P2 for PIP."""
    psi_s = np.radians(x[0])
    P4_0, Q0, G0, H0 = (np.asarray(x[i:i + 2]) for i in (1, 3, 5, 7))
    rQ, cK = nrm(Q0), nrm(P4_0 - Q0)
    rH, cD = nrm(H0 - PIP0), nrm(H0 - G0)
    sg1 = np.sign(cross(P4_0, Q0))
    sg2 = np.sign(cross(G0 - PIP0, H0 - PIP0))
    d = psi - psi_s
    P4, G = O_L + rot(P4_0 - O_L, d), O_L + rot(G0 - O_L, d)
    Q, ok1 = circ_int(O_M, rQ, P4, cK, sg1)
    if not ok1.all():
        return fail(2 * (~ok1).sum())
    mcp = -wrap(ang(Q) - ang(Q0))
    PIP = rot(PIP0, mcp)
    H, ok2 = circ_int(PIP, rH, G, cD, sg2)
    if not ok2.all():
        return fail((~ok2).sum())
    pip = wrap(-wrap(ang(H - PIP) - ang(H0 - PIP0)) - mcp)
    # instant centre of the Link relative to P1 (Kennedy: on line O_M-O_L and on line P4-Q)
    ic = np.cross(line_h(O_M, O_L), line_h(P4, Q))
    P4_1, G_1, OL_1 = to_p1(P4, mcp), to_p1(G, mcp), to_p1(np.broadcast_to(O_L, P4.shape), mcp)
    H_1 = to_p1(H, mcp)
    Qb = np.broadcast_to(Q0, P4.shape)
    s = dict(ok=True, mcp=mcp, pip=pip,
             mu=dict(link_K=vec_angle(P4 - O_L, Q - P4), K_P1=vec_angle(Q, P4 - Q),
                     linkrelP1_D=vec_angle(hdir(ic, G), H - G), D_P2crank=vec_angle(H - G, H - PIP)),
             pairs=dict(link_P4_G=nrm(G0 - P4_0), K_P4_Q=cK, D_G_H=cD, P1_Q_PIP=nrm(Q0 - PIP0), P2_H_PIP=rH,
                        P2_H_DIP=nrm(H0 - DIP0)),
             env=dict(Q=env_v(Q0, 10, L1 + 2), P4=env_v(P4_1, 10, L1 + 6), G=env_v(G_1, 10, L1 + 6),
                      H=env_v(H0, L1 - 8, L1 + L2)),
             clear=dict(linkP4_vs_PIPpin=clr(PIP0, OL_1, P4_1), linkG_vs_PIPpin=clr(PIP0, OL_1, G_1),
                        K_vs_PIPpin=clr(PIP0, P4_1, Qb), D_vs_PIPpin=clr(PIP0, G_1, H_1),
                        linkP4_vs_Qpin=clr(Q0, OL_1, P4_1), linkG_vs_Qpin=clr(Q0, OL_1, G_1),
                        D_vs_Qpin=clr(Q0, G_1, H_1)))
    s["p1pins"] = dict(Q=Qb, P4=P4_1, G=G_1)
    s["p2pins"] = dict(H=H0)
    s["bars_p1"] = dict(Link_P4=(OL_1, P4_1), Link_G=(OL_1, G_1), K=(P4_1, Qb), D=(G_1, H_1))
    if full:
        s["pts"] = dict(P4=P4, Q=Q, G=G, H=H)
        s["bars"] = [("Link", "O_L", "P4", 0), ("Link", "O_L", "G", 0), ("K", "P4", "Q", 4), ("D", "G", "H", 3)]
        s["joints"] = [("g", "link", O_L), ("g", "p1", O_M), ("link", "k", P4_0), ("k", "p1", Q0),
                       ("link", "d", G0), ("d", "p2", H0), ("p1", "p2", PIP0)]
        s["geom"] = dict(P4=P4_0, Q=Q0, G=G0, H=H0)
    return s


def solve_vp(x, psi, full=False):
    """(c) virtual P4: P2 hangs from the Link on X (U1-V1) and Y (U2-V2).
    Solved in the Link frame by the rocker angle alpha of X (closed form), then psi(alpha) is inverted
    (it must be monotonic on one branch over the whole travel, otherwise the Link would hit a limit)."""
    psi_s = np.radians(x[0])
    U1, U2, V1_0, V2_0 = (np.asarray(x[i:i + 2]) for i in (1, 3, 5, 7))
    rY, cV = nrm(V2_0 - U2), nrm(V2_0 - V1_0)
    sg = np.sign(cross(V1_0 - U2, V2_0 - U2))
    V1 = U1 + rot(V1_0 - U1, ALPHA)
    V2, oka = circ_int(U2, rY, V1, cV, sg)
    beta = -wrap(ang(V2 - V1) - ang(V2_0 - V1_0))  # P2 flexion relative to the Link
    PIPL = V1 + rot(PIP0 - V1_0, beta)
    sgo = np.sign(cross(PIP0 - O_L, O_M - O_L))
    OML, okb = circ_int(O_L, 6.0, PIPL, L1, sgo)  # MCP pivot seen from the Link
    ok = oka & okb
    if not ok[I0]:
        return fail(200)
    delta = wrap(ang(OML - O_L) - ang(O_M - O_L))  # Link flexion relative to the gimbal
    dd = np.diff(delta)
    sgn = np.sign(dd[I0]) if dd[I0] != 0 else 1.0
    good = ok[:-1] & ok[1:] & (dd * sgn > 1e-7)
    bad = np.flatnonzero(~good)
    r = bad[bad >= I0].min() if (bad >= I0).any() else len(ALPHA) - 1
    lft = bad[bad < I0].max() + 1 if (bad < I0).any() else 0
    seg = slice(lft, r + 1)
    dl = delta[seg]
    lo, hi = psi_s + dl.min(), psi_s + dl.max()
    unc = max(0.0, lo - psi.min()) + max(0.0, psi.max() - hi)
    if unc > 0:
        return fail(np.degrees(unc) / 2 + 1)
    order = np.argsort(dl)
    dsort = dl[order]

    def at(v):  # resample alpha-grid quantities at the requested psi
        v = v[seg][order]
        if v.ndim == 1:
            return np.interp(psi - psi_s, dsort, v)
        return np.stack([np.interp(psi - psi_s, dsort, v[:, k]) for k in range(v.shape[1])], -1)

    mcp_a = wrap(delta - ang(PIPL - OML))
    pip_a = wrap(delta + beta - mcp_a)
    ic = np.cross(line_h(U1, V1), line_h(U2, V2))  # instant centre of P2 relative to the Link
    mu_a = dict(link_eqcrank=vec_angle(hdir(ic, O_L), -hdir(ic, PIPL)),
                eqcrank_P1=vec_angle(PIPL - OML, hdir(ic, PIPL)),
                X_P2=vec_angle(V1 - U1, V2 - V1), Y_P2=vec_angle(V2 - U2, V1 - V2))
    mcp, pip, dl_i = at(mcp_a), at(pip_a), psi - psi_s
    mu = {k: at(np.nan_to_num(v)) for k, v in mu_a.items()}
    # Link pins in the P1 frame: ground = O_L + rot(U - O_L, delta); P1 = rot(ground, -mcp)
    U1_1 = to_p1(O_L + rot(U1 - O_L, dl_i), mcp)
    U2_1 = to_p1(O_L + rot(U2 - O_L, dl_i), mcp)
    OL_1 = to_p1(np.broadcast_to(O_L, U1_1.shape), mcp)
    V1_1, V2_1 = PIP0 + rot(V1_0 - PIP0, pip), PIP0 + rot(V2_0 - PIP0, pip)
    s = dict(ok=True, mcp=mcp, pip=pip, mu=mu,
             pairs=dict(link_U1_U2=nrm(U2 - U1), X_U1_V1=nrm(V1_0 - U1), Y_U2_V2=rY, P2_V1_V2=cV,
                        P2_V1_PIP=nrm(V1_0 - PIP0), P2_V2_PIP=nrm(V2_0 - PIP0), P2_V1_DIP=nrm(V1_0 - DIP0),
                        P2_V2_DIP=nrm(V2_0 - DIP0)),
             env=dict(U1=env_v(U1_1, 10, L1 + 6), U2=env_v(U2_1, 10, L1 + 6), V1=env_v(V1_0, L1 - 8, L1 + L2),
                      V2=env_v(V2_0, L1 - 8, L1 + L2)),
             clear=dict(linkU1_vs_PIPpin=clr(PIP0, OL_1, U1_1), linkU2_vs_PIPpin=clr(PIP0, OL_1, U2_1),
                        X_vs_PIPpin=clr(PIP0, U1_1, V1_1), Y_vs_PIPpin=clr(PIP0, U2_1, V2_1)))
    s["p1pins"] = dict(U1=U1_1, U2=U2_1)
    s["p2pins"] = dict(V1=V1_0, V2=V2_0)
    s["bars_p1"] = dict(Link_U1=(OL_1, U1_1), Link_U2=(OL_1, U2_1), X=(U1_1, V1_1), Y=(U2_1, V2_1))
    if full:
        g = lambda p1: rot(p1, mcp)  # noqa: E731
        s["pts"] = dict(U1=g(U1_1), U2=g(U2_1), V1=g(V1_1), V2=g(V2_1))
        s["bars"] = [("Link", "O_L", "U1", 0), ("Link", "O_L", "U2", 0), ("X", "U1", "V1", 4), ("Y", "U2", "V2", 3)]
        s["joints"] = [("g", "link", O_L), ("g", "p1", O_M), ("link", "x", U1), ("x", "p2", V1_0),
                       ("link", "y", U2), ("y", "p2", V2_0), ("p1", "p2", PIP0)]
        s["geom"] = dict(U1=U1, U2=U2, V1=V1_0, V2=V2_0)
    return s


SOLVERS = dict(**{"4bar": solve_4bar}, watt=solve_watt, steph=solve_steph, vp=solve_vp)
PS_B = (PSI_S_CUR, 4.0)  # straight pose no earlier than the current design (f_s >= -56: same abduction room)
LAYOUTS = {
    "4bar": dict(label="4-bar, current topology (baseline)", pnames=["psi_s", "P4y", "P4z"],
                 bounds=[PS_B, (35, 60), (-8, 8)], parts=0, pins=0),
    "watt": dict(label="(a) Watt six-bar", pnames=["psi_s", "P4y", "P4z", "Qy", "Qz", "Ey", "Ez", "Fy", "Fz"],
                 bounds=[PS_B, (28, 53), (-8, 8), (28, 49), (-8, 8), (28, 53), (-8, 8), (39, 62), (-8, 8)],
                 parts=2, pins=3),
    "steph": dict(label="(b) Stephenson six-bar", pnames=["psi_s", "P4y", "P4z", "Qy", "Qz", "Gy", "Gz", "Hy", "Hz"],
                  bounds=[PS_B, (10, 53), (-8, 8), (10, 49), (-8, 8), (10, 53), (-8, 8), (39, 62), (-8, 8)],
                  parts=2, pins=3),
    "vp": dict(label="(c) virtual-P4 six-bar", pnames=["psi_s", "U1y", "U1z", "U2y", "U2z", "V1y", "V1z", "V2y", "V2z"],
               bounds=[PS_B, (25, 53), (-8, 8), (25, 53), (-8, 8), (39, 62), (-8, 8), (39, 62), (-8, 8)],
               parts=2, pins=3),
}


# ------------------------------------------------------------------ cost
def score(s, x, mode, mu_fl, mu_all):
    mcp, pip = np.degrees(s["mcp"]), np.degrees(s["pip"])
    f_s = f_of_psi(x[0])
    fl = F_EVAL >= f_s - 1e-9
    M = margin(np.array(list(s["mu"].values())))
    m_fl, m_all = M[:, fl].min(), M.min()
    vdeg = [max(0, mu_fl - m_fl), max(0, mu_all - m_all), max(0, MCP_LO - mcp[-1]), max(0, mcp[-1] - MCP_HI),
            np.clip(-np.diff(mcp[fl]), 0, None).sum() + np.clip(-np.diff(pip[fl]), 0, None).sum(),
            max(0, HYPER_MIN - mcp.min()), max(0, HYPER_MIN - pip.min())]
    vmm = [max(0, PIN_MIN - d) for d in s["pairs"].values()] + list(s["env"].values()) + \
          [max(0, GAP - c) for c in s["clear"].values()]
    pen = sum(3 * v + 30 * v * v for v in vdeg) + sum(20 * v + 200 * v * v for v in vmm)
    pip90 = pip[-1]
    rms = float(np.sqrt(np.mean((pip[fl] - K_TARGET * mcp[fl]) ** 2)))
    if mode == "maxpip":
        c = -min(pip90, PIP_CAP)
    else:  # "target": PIP 100-104 at MCP 88-90, near-proportional curl, then maximise the transmission margins
        v = [max(0, 100.0 - pip90), max(0, pip90 - 104.0), max(0, 88.0 - mcp[-1]), max(0, mcp[-1] - 90.0),
             max(0, rms - TARGET_RMS[mode])]
        c = sum(3 * u + 30 * u * u for u in v) - 0.3 * min(m_fl, 70.0) - 0.15 * min(m_all, 50.0)
    return c + pen, dict(f_s=f_s, mcp90=mcp[-1], pip90=pip90, m_fl=m_fl, m_all=m_all, pen=pen, vdeg=vdeg, vmm=vmm,
                         rms=rms)


def nx(lay):
    return len(LAYOUTS[lay]["bounds"])


def bounds_for(lay, mode):
    """Target modes synthesise loop 2 (A, B) jointly: A and the six-bar's new P1 pins compete for the
    same corner of P1 next to the PIP, so loop 2 cannot simply be added afterwards."""
    return LAYOUTS[lay]["bounds"] + (B_LOOP2 if mode.startswith("target") else [])


def cost(x, lay, mode, mu_fl, mu_all):
    x = np.asarray(x, float)
    n = nx(lay)
    s = SOLVERS[lay](x[:n], PSI)
    if not s["ok"]:
        return 1e7 + 1e4 * s["nbad"]  # above any constraint penalty of an assembling design
    c, info = score(s, x[:n], mode, mu_fl, mu_all)
    if len(x) > n:
        c += cost2(x[n:], s["pip"], F_EVAL >= info["f_s"] - 1e-9, s["p1pins"], s["p2pins"], s["bars_p1"])
    return c


def de_task(task):
    """One serial DE run + bounded Nelder-Mead polish; tasks run in parallel over seeds/levels."""
    lay, mode, mu_fl, mu_all, seed, (pop, it) = task
    b = bounds_for(lay, mode)
    args = (lay, mode, mu_fl, mu_all)
    t = time.time()
    r = differential_evolution(cost, b, args=args, seed=seed, tol=1e-9, polish=False, maxiter=it, popsize=pop,
                               mutation=(0.5, 1.0), recombination=0.8, init="sobol")
    x, c = r.x, float(r.fun)
    for _ in range(2):
        p = minimize(cost, x, args=args, method="Nelder-Mead", bounds=b,
                     options=dict(xatol=1e-6, fatol=1e-9, maxiter=8000, adaptive=True))
        if p.fun < c:
            x, c = p.x, float(p.fun)
    return task, np.asarray(x), c, time.time() - t


def round_design(lay, x, mode, mu_fl, mu_all):
    """Round to 0.05 mm / 0.05 deg (CAD-friendly). Rounding alone can tip a design that sits exactly on
    a constraint, so search the 3^n rounding neighbourhood (six-bar part, then loop 2), then greedy passes."""
    b = np.array(bounds_for(lay, mode))
    x0 = np.clip(np.round(np.asarray(x) / 0.05) * 0.05, b[:, 0], b[:, 1])
    f = lambda xx: cost(np.clip(xx, b[:, 0], b[:, 1]), lay, mode, mu_fl, mu_all)  # noqa: E731
    best, cb = x0, f(x0)
    for sl in (slice(0, nx(lay)), slice(nx(lay), len(x0))):
        k = len(range(*sl.indices(len(x0))))
        if k == 0:
            continue
        base = best.copy()
        for d in np.array(np.meshgrid(*[[-0.05, 0.0, 0.05]] * k, indexing="ij")).reshape(k, -1).T:
            xx = base.copy()
            xx[sl] += d
            c = f(xx)
            if c < cb - 1e-12:
                best, cb = xx, c
    for _ in range(4):
        improved = False
        for i in range(len(best)):
            for d in (-0.05, 0.05):
                xx = best.copy()
                xx[i] += d
                c = f(xx)
                if c < cb - 1e-12:
                    best, cb, improved = xx, c, True
        if not improved:
            break
    return np.round(np.clip(best, b[:, 0], b[:, 1]), 4), cb


def evaluate(lay, x, mu_fl=MU_FL_REC, mu_all=MU_ALL_REC, mode="maxpip"):
    s = SOLVERS[lay](np.asarray(x, float), PSI, full=True)
    if not s["ok"]:
        return s, None
    c, info = score(s, x, mode, mu_fl, mu_all)
    info["cost"] = c
    info["feasible"] = max(info["vdeg"]) <= 0.2 and max(info["vmm"], default=0.0) <= 0.05  # tolerance of the penalty method
    return s, info


# ------------------------------------------------------------------ loop 2 (DIP) re-synthesis
def solve_loop2(y, pip):
    """Crossed four-bar on P2: P1 crank PIP-A, coupler L2 = AB, P3 crank DIP-B (as linkage.Design)."""
    A0, B0 = PIP0 + y[:2], DIP0 + y[2:4]
    rB, dAB = nrm(B0 - DIP0), nrm(B0 - A0)
    sg = np.sign(cross(A0 - DIP0, B0 - DIP0))
    A = PIP0 + rot(A0 - PIP0, -pip)  # P2 frame
    B, ok = circ_int(DIP0, rB, A, dAB, sg)
    dip = -wrap(ang(B - DIP0) - ang(B0 - DIP0))
    return dict(ok=ok, A0=A0, B0=B0, A=A, B=B, dip=dip,
                mu=dict(P1crank_L2=vec_angle(A - PIP0, B - A), L2_P3crank=vec_angle(B - A, B - DIP0)))


def loop2_rules(L, pip, fl, p1pins, p2pins, bars_p1):
    """Rule values of loop 2: margins, lengths, A-Q spacing (same part P1), hard clearances of L2 against
    the P2 pins and of the A pin (it runs through every layer) against every bar moving in P1, and soft
    L2-vs-bar overlaps (those only mean the parts must sit in different layers)."""
    A0, B0, A, B = L["A0"], L["B0"], L["A"], L["B"]
    pipd, dip = np.degrees(pip), np.degrees(L["dip"])
    M = margin(np.array(list(L["mu"].values())))
    r = dict(m_fl=M[:, fl].min(), m_all=M.min(), rA=nrm(A0 - PIP0), rB=nrm(B0 - DIP0),
             dip_short=max(0, 0.7 * pipd[-1] - 2 - dip[-1]), mono=np.clip(-np.diff(dip[fl]), 0, None).sum(),
             env=max(0, abs(A0[1]) - Z_ENV) + max(0, abs(B0[1]) - Z_ENV))
    if "Q" in p1pins:
        r["A_Q"] = nrm(A0 - p1pins["Q"][0])
    clear = dict(L2_vs_PIPpin=clr(PIP0, A, B), L2_vs_DIPpin=clr(DIP0, A, B))
    for k, p in p2pins.items():
        clear[f"L2_vs_{k}pin"] = clr(p, A, B)
    for k, (a, b) in bars_p1.items():
        clear[f"Apin_vs_{k}"] = clr(A0, a, b)
    Bp1 = PIP0 + rot(B - PIP0, pip)  # L2 in the P1 frame
    overlap = {k: min(seg_dist(A0 + t_ * (Bp1 - A0), a, b).min() for t_ in np.linspace(0, 1, 5)) - (2 * BAR_HALF + GAP)
               for k, (a, b) in bars_p1.items()}
    return r, clear, overlap


def cost2(y, pip, fl, p1pins, p2pins, bars_p1):
    L = solve_loop2(y, pip)
    if not L["ok"].all():
        return 1e7 + 1e4 * (~L["ok"]).sum()
    r, clear, overlap = loop2_rules(L, pip, fl, p1pins, p2pins, bars_p1)
    v = [max(0, MU_FL_REC - r["m_fl"]), max(0, MU_ALL_REC - r["m_all"]), r["mono"], r["dip_short"]]
    vmm = [max(0, PIN_MIN - r["rA"]), max(0, r["rA"] - 10), max(0, PIN_MIN - r["rB"]), max(0, r["rB"] - 10), r["env"]]
    if "A_Q" in r:
        vmm.append(max(0, PIN_MIN - r["A_Q"]))
    vmm += [max(0, GAP - c) for c in clear.values()]
    vmm += [0.2 * max(0, -o) for o in overlap.values()]  # soft: a different layer also fixes it
    pipd, dip = np.degrees(pip), np.degrees(L["dip"])
    c = np.mean((dip[fl] - 0.7 * pipd[fl]) ** 2) + 0.05 * (r["rA"] + r["rB"]) - 0.1 * min(r["m_fl"], 60)
    return c + sum(3 * x + 30 * x * x for x in v) + sum(20 * x + 200 * x * x for x in vmm)


def synth_loop2(pip, f_s, s, seed=3, quick=False):
    """DE (3 seeds) + Nelder-Mead for A, B; rounded to 0.05 mm (best of the 3^4 rounding neighbourhood)."""
    fl = F_EVAL >= f_s - 1e-9
    b2 = B_LOOP2
    args = (pip, fl, s.get("p1pins", {}), s.get("p2pins", {}), s.get("bars_p1", {}))
    best, cb = None, np.inf
    for sd in (seed, seed + 100, seed + 200)[:1 if quick else 3]:
        r = differential_evolution(cost2, b2, args=args, seed=sd, tol=1e-9, polish=False, maxiter=150 if quick else 400,
                                   popsize=20, mutation=(0.5, 1.0), recombination=0.8, init="sobol")
        p = minimize(cost2, r.x, args=args, method="Nelder-Mead", bounds=b2,
                     options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
        y = p.x if p.fun <= r.fun else r.x
        y0 = np.round(y / 0.05) * 0.05
        for d in np.array(np.meshgrid(*[[-0.05, 0.0, 0.05]] * 4, indexing="ij")).reshape(4, -1).T:
            c = cost2(y0 + d, *args)
            if c < cb:
                best, cb = y0 + d, c
    return best, solve_loop2(best, pip)


# ------------------------------------------------------------------ generic body/pin model (verification)
def rmat(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def drmat(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[-s, c], [-c, -s]])


class Generic:
    """Planar bodies (pose t, theta; straight pose = identity) joined by pins given as straight-pose
    points. The Link angle is prescribed; Newton with continuation along the servo travel."""

    def __init__(self, joints, psi_s):
        self.joints = joints
        self.bodies = sorted({b for j in joints for b in j[:2]} - {"g"})
        self.ix = {b: 3 * k for k, b in enumerate(self.bodies)}
        self.n = 3 * len(self.bodies)
        self.psi_s = psi_s
        assert 2 * len(joints) + 1 == self.n, (len(joints), self.n)

    def world(self, q, b, p):
        if b == "g":
            return np.asarray(p, float)
        k = self.ix[b]
        return q[k:k + 2] + rmat(q[k + 2]) @ p

    def res_jac(self, q, psi):
        r = np.zeros(self.n)
        J = np.zeros((self.n, self.n))
        for m, (bi, bj, p) in enumerate(self.joints):
            r[2 * m:2 * m + 2] = self.world(q, bi, p) - self.world(q, bj, p)
            for b, sgn in ((bi, 1.0), (bj, -1.0)):
                if b == "g":
                    continue
                k = self.ix[b]
                J[2 * m:2 * m + 2, k:k + 2] += sgn * np.eye(2)
                J[2 * m:2 * m + 2, k + 2] += sgn * drmat(q[k + 2]) @ p
        kl = self.ix["link"] + 2
        r[-1] = q[kl] - (psi - self.psi_s)
        J[-1, kl] = 1.0
        return r, J

    def newton(self, q, psi):
        for _ in range(30):
            r, J = self.res_jac(q, psi)
            dq = np.linalg.solve(J, -r)
            q = q + dq
            if np.abs(dq).max() < 1e-12:
                break
        r, J = self.res_jac(q, psi)
        return q, np.abs(r).max()

    def path(self, psis, i_s):
        """Solve at psis (servo-travel order) by continuation outwards from index i_s (straight)."""
        Q = np.zeros((len(psis), self.n))
        err = np.zeros(len(psis))
        q, _ = self.newton(np.zeros(self.n), self.psi_s)
        for rng in (range(i_s, len(psis)), range(i_s, -1, -1)):
            qq = q.copy()
            for i in rng:
                qq, err[i] = self.newton(qq, psis[i])
                Q[i] = qq
        return Q, err

    def theta(self, Q, b):
        return Q[:, self.ix[b] + 2]

    def sens(self, q, psi, sel):
        """Pin-play and speed analysis at one pose. sel(q) -> (value, d value / d q) row(s).
        Returns (worst-case output play for PLAY radial play in every pin, d value / d psi)."""
        _, J = self.res_jac(q, psi)
        Ji = np.linalg.inv(J)
        val, g = sel(q)
        g = np.atleast_2d(g)
        play = 0.0
        for m in range(len(self.joints)):
            G = g @ Ji[:, 2 * m:2 * m + 2]  # d value / d (pin offset), rows x 2
            play += np.linalg.norm(G, 2) * PLAY
        dpsi = g @ Ji[:, -1]
        return play, dpsi


def generic_check(lay, x, s, L2):
    """Independent Newton solution of the full finger (incl. loop 2) and pin-play / tip-speed ratios."""
    joints = list(s["joints"]) + [("p1", "l2", L2["A0"]), ("l2", "p3", L2["B0"]), ("p2", "p3", DIP0)]
    gm = Generic(joints, np.radians(x[0]))
    i_s = int(np.argmin(np.abs(PSI - np.radians(x[0]))))
    Q, err = gm.path(PSI, i_s)
    mcp = gm.theta(Q, "p1")
    pip = gm.theta(Q, "p2") - mcp
    dip = gm.theta(Q, "p3") - gm.theta(Q, "p2")
    rows = {}
    for f in (0.0, 45.0, 90.0):
        i = int(np.argmin(np.abs(F_EVAL - f)))
        q = Q[i]

        def sel_ang(q, a="p2", b="p1"):
            g = np.zeros(gm.n)
            g[gm.ix[a] + 2] += 1
            if b:
                g[gm.ix[b] + 2] -= 1
            return None, g

        def sel_tip(q):
            k = gm.ix["p3"]
            g = np.zeros((2, gm.n))
            g[:, k:k + 2] = np.eye(2)
            g[:, k + 2] = drmat(q[k + 2]) @ TIP0
            return None, g

        p_mcp, _ = gm.sens(q, PSI[i], lambda q: sel_ang(q, "p1", None))
        p_pip, d_pip = gm.sens(q, PSI[i], sel_ang)
        p_dip, _ = gm.sens(q, PSI[i], lambda q: sel_ang(q, "p3", "p2"))
        p_tip, v_tip = gm.sens(q, PSI[i], sel_tip)
        rows[f] = dict(play_mcp=np.degrees(p_mcp), play_pip=np.degrees(p_pip), play_dip=np.degrees(p_dip),
                       play_tip=p_tip, dpip_dpsi=float(d_pip[0]), vtip=float(np.linalg.norm(v_tip)))
    return dict(mcp=mcp, pip=pip, dip=dip, err=err.max(), rows=rows)


# ------------------------------------------------------------------ helpers for the report
def full_pose(lay, x, s, L2):
    """Ground-frame points of every pin for all f (for the sketches and the layer count)."""
    mcp, pip, dip = s["mcp"], s["pip"], L2["dip"]
    P = dict(O_M=np.broadcast_to(O_M, (len(PSI), 2)), O_L=np.broadcast_to(O_L, (len(PSI), 2)))
    P["PIP"] = rot(PIP0, mcp)
    th2 = mcp + pip
    P["DIP"] = P["PIP"] + rot(DIP0 - PIP0, th2)
    P["A"] = rot(L2["A0"], mcp)
    P["B"] = P["PIP"] + rot(L2["B"] - PIP0, th2)
    P["TIP"] = P["DIP"] + rot(TIP0 - DIP0, th2 + dip)
    P.update(s["pts"])
    return P


def seg_seg_min(a, b, c, d):
    return min(seg_dist(a + t * (b - a), c, d).min() for t in np.linspace(0, 1, 21))


def layer_count(bars, P):
    """Bars (centre-region parts) that overlap in-plane at any time (incl. at a shared pin) cannot
    share a layer; greedy colouring of that conflict graph = number of layers they need."""
    parts = {}
    for name, a, b, _ in bars:
        parts.setdefault(name, []).append((a, b))
    names = list(parts)
    conf = {n: set() for n in names}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            d = min(seg_seg_min(P[a], P[b], P[c], P[e]) for a, b in parts[n1] for c, e in parts[n2])
            if d < 2 * BAR_HALF + GAP:
                conf[n1].add(n2)
                conf[n2].add(n1)
    col = {}
    for n in sorted(names, key=lambda n: -len(conf[n])):
        used = {col[m] for m in conf[n] if m in col}
        col[n] = min(set(range(len(names) + 1)) - used)
    return 1 + max(col.values()), {n: sorted(conf[n]) for n in names}


def fidx(f):
    return int(np.argmin(np.abs(F_EVAL - f)))


NEW_PARTS = {"4bar": "", "watt": "R (ternary rocker on P1), C (coupler)", "steph": "K (MCP coupler), D (PIP coupler)",
             "vp": "X, Y (two P2 hangers)"}
NEW_PINS = {"4bar": "", "watt": "Q, E, F", "steph": "Q, G, H (P4 now joins Link-K)", "vp": "U1, U2, V1, V2 replace P4"}


def current_design():
    """design.json evaluated in this framework (4bar layout + its own loop 2)."""
    g = DESIGN["geometry"]
    x = np.array([PSI_S_CUR, *g["P4"]])
    s, info = evaluate("4bar", x)
    L = solve_loop2(np.r_[np.array(g["A"]) - PIP0, np.array(g["B"]) - DIP0], s["pip"])
    L["rules"] = loop2_rules(L, s["pip"], F_EVAL >= info["f_s"] - 1e-9, s["p1pins"], s["p2pins"], s["bars_p1"])
    ref = Design.from_json(DESIGN).solve(PSI)  # three_segment/linkage.py, the model the CAD and sim were built from
    dev = max(np.abs(ref[k] - v).max() for k, v in (("mcp", s["mcp"]), ("pip", s["pip"]), ("dip", L["dip"])))
    return x, s, info, L, np.degrees(dev)


def finish(lay, x, mode, mu_fl, mu_all, seed, quick):
    """Round, re-evaluate, (target modes: loop 2 came with the joint synthesis; otherwise synthesise it
    now), check the loop-2 rules, verify with the Newton model."""
    n = nx(lay)
    xr, _ = round_design(lay, x, mode, mu_fl, mu_all)
    s, info = evaluate(lay, xr[:n], mu_fl, mu_all, mode)
    if info is None or (not info["feasible"] and evaluate(lay, x[:n], mu_fl, mu_all, mode)[1]["feasible"]):
        xr = np.asarray(x)  # rounding broke it: keep the unrounded optimum (reported as such)
        s, info = evaluate(lay, xr[:n], mu_fl, mu_all, mode)
    if len(xr) > n:
        y = xr[n:]
    else:
        y, _ = synth_loop2(s["pip"], info["f_s"], s, seed=seed, quick=quick)
    L = solve_loop2(y, s["pip"])
    L["rules"] = loop2_rules(L, s["pip"], F_EVAL >= info["f_s"] - 1e-9, s["p1pins"], s["p2pins"], s["bars_p1"])
    g = generic_check(lay, xr[:n], s, L)
    return dict(lay=lay, x=xr[:n], y=y, s=s, info=info, L=L, g=g)


def overall(d):
    """Min transmission margins over every loop incl. loop 2, and whether every rule holds (tolerances of the
    penalty method: 0.2 deg, 0.05 mm)."""
    fl = F_EVAL >= d["info"]["f_s"] - 1e-9
    M = margin(np.array(list(d["s"]["mu"].values()) + list(d["L"]["mu"].values())))
    m_fl, m_all = M[:, fl].min(), M.min()
    mu_fl = MU_FL_REC if d["lay"] != "vp" else 25.0
    mu_all = MU_ALL_REC
    r2, c2, _ = d["L"]["rules"]
    ok = (d["info"]["feasible"] and m_fl >= mu_fl - 0.2 and m_all >= mu_all - 0.2 and r2["dip_short"] <= 0.2 and
          r2["rA"] >= PIN_MIN - 0.05 and r2["rB"] >= PIN_MIN - 0.05 and r2.get("A_Q", 99) >= PIN_MIN - 0.05 and
          r2["env"] <= 0.05 and min(c2.values()) >= GAP - 0.05)
    return m_fl, m_all, bool(ok)


def describe(tag, d, ref=None):
    lay, x, s, info, L, g = d["lay"], d["x"], d["s"], d["info"], d["L"], d["g"]
    out(f"\n== {tag}: {LAYOUTS[lay]['label']}")
    out("  parameters: " + ", ".join(f"{n} {v:.2f}" for n, v in zip(LAYOUTS[lay]["pnames"], x)))
    out(f"  drive: straight at Link angle psi_s = {x[0]:.2f} deg -> f_straight = {info['f_s']:.1f} deg (current design "
        f"-56.0); abduction-command room when straight +-{90 - abs(info['f_s']):.0f} deg")
    geom = s["geom"]
    out("  pivots (y, z) mm, straight pose: " + ", ".join(f"{k} ({v[0]:.2f}, {v[1]:.2f})" for k, v in geom.items()) +
        f", A ({L['A0'][0]:.2f}, {L['A0'][1]:.2f}), B ({L['B0'][0]:.2f}, {L['B0'][1]:.2f}); "
        "fixed: O_M (0, 0), O_L (0, -6), PIP (47, 0), DIP (73, 0)")
    lk = [k for k in geom if k in ("P4", "G", "U1", "U2")]
    out("  Link pins in the stock Link frame (cf. design.json P4_in_link_frame): " + ", ".join(
        f"{k} {np.round(O_L + rot(geom[k] - O_L, -np.radians(x[0])), 3).tolist()} (|O_L-{k}| {nrm(geom[k] - O_L):.2f})"
        for k in lk))
    out("  pin-to-pin lengths (mm): " + ", ".join(f"{k} {v:.2f}" for k, v in s["pairs"].items()) +
        f"; loop 2: L2 {nrm(L['B0'] - L['A0']):.2f}, PIP-A {nrm(L['A0'] - PIP0):.2f}, DIP-B {nrm(L['B0'] - DIP0):.2f}")
    mcp, pip, dip = np.degrees(s["mcp"]), np.degrees(s["pip"]), np.degrees(L["dip"])
    for f in (-90.0, round(info["f_s"]), 0.0, 45.0, 90.0):
        i = fidx(f)
        out(f"  f = {F_EVAL[i]:+4.0f} (psi {np.degrees(PSI[i]):5.1f}):  MCP {mcp[i]:6.1f}  PIP {pip[i]:6.1f}  DIP {dip[i]:6.1f}"
            f"   (P2 vs ground {mcp[i] + pip[i]:6.1f})")
    fl = F_EVAL >= info["f_s"] - 1e-9
    out(f"  coupling straight..full: PIP/MCP slope {np.polyfit(mcp[fl], pip[fl], 1)[0]:.3f}, rms dev. from "
        f"PIP = {K_TARGET:.3f} MCP {info['rms']:.1f} deg; DIP/PIP slope {np.polyfit(pip[fl], dip[fl], 1)[0]:.3f}, "
        f"rms dev. from DIP = 0.7 PIP {np.sqrt(np.mean((dip[fl] - 0.7 * pip[fl]) ** 2)):.2f} deg")
    out("  transmission, distance from a dead point (deg): flexion range / whole travel")
    for k, v in list(s["mu"].items()) + list(L["mu"].items()):
        m = margin(v)
        out(f"    {k:<16} {m[fl].min():5.1f} / {m.min():5.1f}")
    pr = min(s["pairs"].values())
    out(f"  rules: min same-part pin spacing {pr:.2f} mm (>= {PIN_MIN}); envelope violation "
        f"{sum(s['env'].values()):.2f} mm; in-plane clearances (mm, >= {GAP}): " +
        ", ".join(f"{k} {v:.2f}" for k, v in s["clear"].items()))
    r2, c2, o2 = L["rules"]
    out(f"  loop 2 rules: P1 crank PIP-A {r2['rA']:.2f}, P3 crank DIP-B {r2['rB']:.2f} mm (>= {PIN_MIN})" +
        (f", A-Q spacing on P1 {r2['A_Q']:.2f} mm (>= {PIN_MIN})" if "A_Q" in r2 else "") +
        f"; hard clearances (mm, >= {GAP}): " + ", ".join(f"{k} {v:.2f}" for k, v in c2.items()))
    ov = [k for k, v in o2.items() if v < 0]
    out("  L2 overlaps in-plane with (=> different layer): " + (", ".join(ov) if ov else "none"))
    P = full_pose(lay, x, s, L)
    nl, conf = layer_count(s["bars"] + [("L2", "A", "B", 2)], P)
    out(f"  centre-region layers needed by the bars (overlap graph colouring): {nl}; overlaps: " +
        "; ".join(f"{k}-{','.join(v)}" for k, v in conf.items() if v))
    out(f"  Newton check (generic body/pin model incl. loop 2): residual {g['err']:.1e}, max |closed form - Newton| "
        f"MCP {np.abs(np.degrees(g['mcp']) - mcp).max():.1e}, PIP {np.abs(np.degrees(g['pip']) - pip).max():.1e}, "
        f"DIP {np.abs(np.degrees(g['dip']) - dip).max():.1e} deg")
    out(f"  pin play ({PLAY} mm radial in every finger pin, worst-case linear stack) and speed ratios:")
    for f, r in g["rows"].items():
        extra = ""
        if ref is not None:
            rr = ref["g"]["rows"][f]
            extra = (f"; vs current: tip force per Link torque x{rr['vtip'] / r['vtip']:.2f}, PIP torque per Link torque "
                     f"x{rr['dpip_dpsi'] / r['dpip_dpsi']:.2f}")
        out(f"    f = {f:+3.0f}: MCP +-{r['play_mcp']:.2f}, PIP +-{r['play_pip']:.2f}, DIP +-{r['play_dip']:.2f} deg, "
            f"tip +-{r['play_tip']:.2f} mm; dPIP/dpsi {r['dpip_dpsi']:.2f}, tip speed {r['vtip']:.0f} mm/rad{extra}")
    if LAYOUTS[lay]["parts"]:
        out(f"  cost vs the current design: +{LAYOUTS[lay]['parts']} parts ({NEW_PARTS[lay]}), +{LAYOUTS[lay]['pins']} "
            f"pins ({NEW_PINS[lay]}): {7 + LAYOUTS[lay]['pins']} finger pin joints instead of 7")
    d.update(nl=nl, P=P)


# ------------------------------------------------------------------ figure
def sketch(ax, d, i, title):
    from matplotlib.patches import Circle
    s, L, P = d["s"], d["L"], d["P"]
    at = {k: v[i] for k, v in P.items()}
    mcp, pip, dip = (np.degrees(a[i]) for a in (s["mcp"], s["pip"], L["dip"]))
    psi = PSI[i]
    base = O_L + rot(np.array([3.596, -11.051]) - O_L, psi)  # rod-ball centre (stock Link frame)
    ax.plot(*np.c_[O_M, O_L], color=ps.INK2, lw=6, alpha=0.35, solid_capstyle="round")
    ax.plot(*np.c_[O_L, base], color=ps.C[0], lw=2)
    foot = lambda p0: rot(np.array([p0[0], 0.0]), s["mcp"][i])  # noqa: E731  (P1 axis point under a P1 pin)
    ax.plot(*np.c_[O_M, at["PIP"]], color=ps.C[1], lw=4.5, label="P1 proximal")
    ax.plot(*np.c_[foot(L["A0"]), at["A"]], color=ps.C[1], lw=2)
    if "Q" in s["geom"]:
        ax.plot(*np.c_[foot(s["geom"]["Q"]), at["Q"]], color=ps.C[1], lw=2)
    ax.plot(*np.c_[at["PIP"], at["DIP"]], color=ps.C[2], lw=4.5, label="P2 middle")
    for k in s.get("p2pins", {}):
        ax.plot(*np.c_[at["PIP"], at[k]], color=ps.C[2], lw=2)
    ax.plot(*np.c_[at["DIP"], at["TIP"]], color=ps.C[3], lw=4.5, label="P3 distal")
    ax.plot(*np.c_[at["DIP"], at["B"]], color=ps.C[3], lw=2)
    seen = set()
    for name, a, b, col in s["bars"]:
        lab = ("Link (input)" if name == "Link" else f"{name} (new)") if name not in seen else None
        seen.add(name)
        ax.plot(*np.c_[at[a], at[b]], color=VIOLET if col == 3 else ps.C[col], lw=3 if name == "Link" else 2.5,
                label=lab, solid_capstyle="round")
    ax.plot(*np.c_[at["A"], at["B"]], color=ps.INK2, lw=2, label="L2 DIP coupler")
    pins = ["O_M", "O_L", "PIP", "DIP", "A", "B"] + list(s["geom"])
    for k in pins:
        ax.add_patch(Circle(at[k], 1.0, color=ps.SURFACE, ec=ps.INK, lw=1.0, zorder=5))
        ax.annotate(k, at[k], xytext=(3, 3), textcoords="offset points", fontsize=7, color=ps.INK2, zorder=6)
    ax.plot(*at["TIP"], "o", color=ps.C[3], ms=6, mec=ps.SURFACE, mew=1.5, zorder=6)
    ax.set_title(f"{title}: MCP {mcp:.0f}, PIP {pip:.0f}, DIP {dip:.0f}", fontsize=9.5)
    ax.set_aspect("equal")
    ax.set_xlim(-30, 95)
    ax.set_ylim(-75, 25)
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm, dorsal +)")


def make_figure(cur, finals, rec, other, table, path):
    import matplotlib.pyplot as plt
    ps.setup()
    fig = plt.figure(figsize=(17, 15.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.95, 0.95], hspace=0.3, wspace=0.22, top=0.955, bottom=0.04)
    # (1) joint angles vs f
    ax = fig.add_subplot(gs[0, 0])
    sty = {"cur": (0, (4, 2)), rec: "-", other: (0, (1, 1.5))}
    names = {"cur": "current 4-bar", rec: LAYOUTS[rec]["label"].split(" six")[0], other: LAYOUTS[other]["label"].split(" six")[0]}
    for key, d in (("cur", cur), (rec, finals[rec]), (other, finals[other])):
        for j, (lab, arr) in enumerate((("MCP", d["s"]["mcp"]), ("PIP", d["s"]["pip"]), ("DIP", d["L"]["dip"]))):
            ax.plot(F_EVAL, np.degrees(arr), color=ps.C[j], ls=sty[key], lw=2.2 if key == rec else 1.5,
                    label=f"{lab}, {names[key]}")
    ax.axvline(cur["info"]["f_s"], color=ps.INK2, lw=1)
    ax.axhline(100, color=ps.GRID, lw=1.2)
    ax.set_xlabel("flexion command f (deg), a = 0")
    ax.set_ylabel("joint angle (deg)")
    ax.set_title("Joint angles over the servo travel")
    ax.legend(loc="upper left", fontsize=7, ncol=1)
    # (2) trade-off
    ax = fig.add_subplot(gs[0, 1])
    for j, lay in enumerate(("4bar", "watt", "steph", "vp")):
        mus = sorted(m for (l_, m) in table if l_ == lay)
        v = np.array([table[(lay, m)] for m in mus], float)
        ax.plot(mus, np.minimum(v, PIP_CAP), "o-", color=ps.C[j], label=LAYOUTS[lay]["label"], lw=2, ms=5)
    ax.plot([cur["info"]["m_fl"]], [cur["info"]["pip90"]], "s", color=ps.INK, ms=7, label="current design (P2 crank 5.48 mm)")
    ax.axhline(100, color=ps.INK2, lw=1, ls=(0, (4, 2)))
    ax.axhline(PIP_CAP, color=ps.GRID, lw=1)
    ps.note(ax, f"MCP(+90) in [{MCP_LO:.0f}, {MCP_HI:.0f}], straight at f >= -56,\nwhole-travel limit min(mu, 25); "
                f"PIP capped at {PIP_CAP:.0f} in the search", xy=(0.01, 0.12))
    ax.set_xlabel("required transmission margin in the flexion range (deg)")
    ax.set_ylabel("best PIP at f = +90 (deg)")
    ax.set_title("Best achievable PIP vs transmission limit")
    ax.set_ylim(0, PIP_CAP + 8)
    ax.legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(0.0, 0.2))
    # (3) transmission margins of the recommended design
    ax = fig.add_subplot(gs[0, 2])
    d = finals[rec]
    for j, (k, v) in enumerate(list(d["s"]["mu"].items()) + list(d["L"]["mu"].items())):
        ax.plot(F_EVAL, margin(v), color=ps.C[j % 5], ls="-" if j < 5 else (0, (3, 1.5)), label=k)
    ax.axhspan(0, MU_FL_REC, color=ps.C[4], alpha=0.07, lw=0)
    ax.axhline(MU_ALL_REC, color=ps.C[4], lw=1, ls=(0, (2, 2)))
    ax.axvline(d["info"]["f_s"], color=ps.INK2, lw=1)
    ax.set_ylim(0, 92)
    ax.set_xlabel("flexion command f (deg)")
    ax.set_ylabel("distance from a dead point (deg)")
    ax.set_title(f"Transmission, {names[rec]} (shaded < {MU_FL_REC:.0f})")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    for row, key in ((1, rec), (2, other)):
        d = finals[key]
        for col, (i, t) in enumerate(((fidx(d["info"]["f_s"]), "straight"), (fidx(0), "f = 0"), (fidx(90), "f = +90"))):
            ax = fig.add_subplot(gs[row, col])
            sketch(ax, d, i, f"{names[key]}, {t}")
            if col == 0:
                ax.legend(loc="lower left", fontsize=7)
    fig.suptitle("Six-bar PIP study (planar, stock drive psi(f), stock gimbal pivots): can PIP reach ~100 deg at MCP ~90?",
                 color=ps.INK, fontsize=12.5, y=0.995)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small DE budget, fewer seeds and levels (smoke test)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    run(args)


def run(args):
    t0 = time.time()
    quick = args.quick
    seeds = args.seeds[:2] if quick else args.seeds
    big, small = ((12, 150), (12, 80)) if quick else ((20, 700), (20, 200))
    levels = (25.0, 40.0, 55.0) if quick else MU_LEVELS
    vp_levels = (0.0, 25.0) if quick else VP_LEVELS
    tasks = [(lay, m, MU_FL_REC, MU_ALL_REC, sd, big) for m in TARGET_RMS for lay in ("watt", "steph") for sd in seeds]
    tasks += [("vp", "maxpip", mu, min(mu, MU_ALL_REC), sd, big) for mu in vp_levels for sd in seeds]
    tasks += [(lay, "maxpip", mu, min(mu, MU_ALL_REC), sd, big) for mu in levels for lay in ("watt", "steph")
              for sd in seeds]
    tasks += [("4bar", "maxpip", mu, min(mu, MU_ALL_REC), sd, small) for mu in levels for sd in seeds[:2]]

    out("== Six-bar PIP study for the 3-phalanx finger (planar, mm, finger design frame: y distal, z dorsal)")
    out(f"drive (fixed): stock Link-base angle psi(f), f = -90..+90 -> psi {np.degrees(PSI[0]):.1f}..{np.degrees(PSI[-1]):.1f}"
        f" deg; ground pivots O_M (0, 0), O_L (0, -6); P1 {L1}, P2 {L2}, P3 to tip site {TIP0[0] - L1 - L2:.2f} mm")
    out(f"rules: one assembly branch over the whole travel; MCP, PIP monotonic straight..full; MCP(+90) in "
        f"[{MCP_LO:.0f}, {MCP_HI:.0f}]; hyperextension >= {HYPER_MIN:.0f}; straight pose at f >= -56 (psi_s >= {PSI_S_CUR}, "
        f"same abduction room as the current design); same-part pins >= {PIN_MIN} mm apart; new pivots |z| <= {Z_ENV} "
        f"mm in their phalanx; bars clear the PIP pin (and P1's new pin) by >= {GAP} mm in-plane")
    out(f"transmission: every four-bar (or equivalent four-bar) angle >= mu in the flexion range and >= min(mu, "
        f"{MU_ALL_REC:.0f}) over the whole travel; recommended designs use mu = {MU_FL_REC:.0f} / {MU_ALL_REC:.0f} like "
        "synth_linkage.py")
    out(f"search: {len(tasks)} DE runs (popsize {big[0]}, {big[1]} generations, seeds {seeds}) + bounded Nelder-Mead "
        f"polish, run in parallel ({args.workers} workers); designs rounded to 0.05 mm / 0.05 deg")
    out("\n-- topology: with a binary ground (only O_M and O_L may be used) and the Link as the only input, the 1-DOF "
        "six-bars that\n   drive both P1 and P2 are: Watt (a) [R pinned to P1 and Link, C to P2], Stephenson (b) "
        "[Link ternary: K to P1, D to P2],\n   Stephenson (c) [P2 hung from the Link on two binaries X, Y], and a Stephenson "
        "with a floating ternary between Link and P2\n   tied to P1 (not studied: it is (c) with one hanger moved onto P1). "
        "Every option adds 2 parts and 3 pin joints.")

    res = {}
    with Pool(args.workers) as pool:
        for k, (task, x, c, dt) in enumerate(pool.imap_unordered(de_task, tasks)):
            key = task[:4]
            if key not in res or c < res[key][1]:
                res[key] = (x, c)
            print(f"  [{k + 1}/{len(tasks)}] {task[0]:<5} {task[1]:<6} mu {task[2]:4.0f} seed {task[4]}: cost {c:9.3f} "
                  f"({dt:.0f} s, total {time.time() - t0:.0f} s)", flush=True)

    # ---- current design under these rules
    xc, sc, ic, Lc, dev = current_design()
    cur = dict(lay="4bar", x=xc, s=sc, info=ic, L=Lc, g=generic_check("4bar", xc, sc, Lc))
    describe("CURRENT design.json (crossed four-bar), for reference", cur)
    out(f"  this framework vs three_segment/linkage.Design on design.json: max |difference| {dev:.1e} deg")
    out(f"  note: its P2 crank PIP-P4 is 5.48 mm (< the {PIN_MIN} mm pin rule used here), which is why the re-optimised "
        "4-bar baseline below is a little lower")

    # ---- trade-off table (monotone envelope: a design that meets a stricter limit also meets a looser one)
    found = {}
    for (lay, mode, mu, mall), (x, _) in res.items():
        if mode != "maxpip":
            continue
        s, info = evaluate(lay, x, mu, mall)
        if info is not None and info["feasible"]:
            found[(lay, mu)] = (info["pip90"], info["mcp90"], x)
    table = {}
    out(f"\n-- best PIP at f = +90 vs the transmission limit (MCP(+90) in [{MCP_LO:.0f}, {MCP_HI:.0f}], straight at "
        f"f >= -56; search capped at PIP {PIP_CAP:.0f};\n   six-bar + P1/P2 drive only, loop 2 not included here; "
        "'-' = no design found that meets all rules; value = best over the levels >= mu)")
    out("   mu flex / whole   4-bar   (a) Watt  (b) Stephenson  (c) virtual-P4      [MCP(+90) of each]")
    for mu in sorted(set(levels) | set(vp_levels)):
        row, mrow = [], []
        for lay in ("4bar", "watt", "steph", "vp"):
            cands = [v for (l_, m), v in found.items() if l_ == lay and m >= mu]
            searched = any(l_ == lay and m == mu for (l_, _, m, _) in res)
            if not cands or not searched:
                row.append("      -" if searched else "    n/a")
                mrow.append("  -  ")
                if searched:
                    table[(lay, mu)] = np.nan
                continue
            pip90, mcp90, _ = max(cands, key=lambda v: v[0])
            table[(lay, mu)] = pip90
            row.append(f"{pip90:7.1f}" + ("+" if pip90 >= PIP_CAP - 0.5 else " "))
            mrow.append(f"{mcp90:5.1f}")
        out(f"   {mu:4.0f} / {min(mu, MU_ALL_REC):4.0f}     " + "  ".join(f"{r:>8}" for r in row) + "     [" +
            " ".join(mrow) + "]")
    for lay in ("watt", "steph"):
        ok = [m for (l_, m), v in table.items() if l_ == lay and np.isfinite(v) and v >= 100.0]
        out(f"   {LAYOUTS[lay]['label']}: PIP >= 100 reachable up to a flexion-range margin of "
            f"{max(ok) if ok else float('nan'):.0f} deg (of the levels tried)")

    # ---- the PIP-100 designs (target mode, loop 2 synthesised jointly) + the best (c)
    finals = {}
    for lay in ("watt", "steph"):
        x, _ = res[(lay, "target", MU_FL_REC, MU_ALL_REC)]
        finals[lay] = finish(lay, x, "target", MU_FL_REC, MU_ALL_REC, seeds[0], quick)
    vp_mu = 25.0
    vp_c = [v for (l_, m), v in found.items() if l_ == "vp" and m >= vp_mu]
    x_vp = max(vp_c, key=lambda v: v[0])[2] if vp_c else res[("vp", "maxpip", vp_mu, vp_mu)][0]
    finals["vp"] = finish("vp", x_vp, "maxpip", vp_mu, vp_mu, seeds[0], quick)
    for lay, tag in (("watt", "DESIGN (a), target PIP 100, max transmission, loop 2 synthesised jointly"),
                     ("steph", "DESIGN (b), target PIP 100, max transmission, loop 2 synthesised jointly"),
                     ("vp", f"DESIGN (c), best PIP found with mu >= {vp_mu:.0f} (cannot reach the target)")):
        describe(tag, finals[lay], ref=cur)

    out("\n-- sensitivity: the same PIP-100 designs when the curl must stay near-proportional (rms <= 6 deg from "
        f"PIP = {K_TARGET:.2f} MCP; current design: {cur['info']['rms']:.1f})")
    for lay in ("watt", "steph"):
        key = (lay, "target6", MU_FL_REC, MU_ALL_REC)
        s6, i6 = evaluate(lay, res[key][0][:nx(lay)], MU_FL_REC, MU_ALL_REC, "target6")
        if i6 is None:
            out(f"   {LAYOUTS[lay]['label']}: no assembling design found")
            continue
        out(f"   {LAYOUTS[lay]['label']}: MCP {i6['mcp90']:.1f}, PIP {i6['pip90']:.1f}, rms {i6['rms']:.1f}, transmission "
            f"margin {i6['m_fl']:.1f} / {i6['m_all']:.1f} (flexion / whole){'' if i6['feasible'] else ' - violates a rule'}"
            f"  vs {finals[lay]['info']['m_fl']:.1f} / {finals[lay]['info']['m_all']:.1f} at rms "
            f"{finals[lay]['info']['rms']:.1f}")

    # ---- comparison and pick
    out("\n-- summary (f = +90 unless stated)")
    out("   design                       MCP    PIP    DIP  f_str  margin flex/whole: drive | incl. loop 2   play PIP@45  "
        "tip force@45  layers  +parts/+pins  all rules  binding angle")
    rows = [("current 4-bar", cur)] + [(LAYOUTS[k]["label"], finals[k]) for k in ("watt", "steph", "vp")]
    for name, d in rows:
        if "nl" not in d:
            P = full_pose(d["lay"], d["x"], d["s"], d["L"])
            d["nl"] = layer_count(d["s"]["bars"] + [("L2", "A", "B", 2)], P)[0]
            d["P"] = P
        r45 = d["g"]["rows"][45.0]
        tf = cur["g"]["rows"][45.0]["vtip"] / r45["vtip"]
        d["m_fl"], d["m_all"], d["rules_ok"] = overall(d)
        fl = F_EVAL >= d["info"]["f_s"] - 1e-9
        mus = {k: margin(v)[fl].min() for k, v in list(d["s"]["mu"].items()) + list(d["L"]["mu"].items())}
        out(f"   {name:<28} {np.degrees(d['s']['mcp'][-1]):5.1f}  {np.degrees(d['s']['pip'][-1]):5.1f}  "
            f"{np.degrees(d['L']['dip'][-1]):5.1f}  {d['info']['f_s']:5.1f}       {d['info']['m_fl']:5.1f} / {d['info']['m_all']:4.1f}"
            f" | {d['m_fl']:5.1f} / {d['m_all']:4.1f}       +-{r45['play_pip']:.2f}       x{tf:.2f}       {d['nl']}     "
            f"+{LAYOUTS[d['lay']]['parts']}/+{LAYOUTS[d['lay']]['pins']}       {'yes' if d['rules_ok'] else 'no ':<3}"
            f"        {min(mus, key=mus.get)}")
    out("   (drive = gimbal-Link-...-P1/P2 linkage only; play = worst-case stack of 0.05 mm radial play in every finger "
        "pin; tip force = per unit Link torque along the tip motion)")
    out("   current: fails only the 6 mm pin rule (P2 crank 5.48)." + (
        "" if finals["vp"]["rules_ok"] else " (c): loop 2 was added after the drive synthesis, not jointly as for "
        "(a)/(b), and clashes in-plane; (c) is no candidate anyway (PIP < 80)."))
    l2m = {k: (finals[k]["L"]["rules"][0]["m_fl"], finals[k]["L"]["rules"][0]["m_all"],
               np.degrees(np.ptp(finals[k]["s"]["pip"]))) for k in ("watt", "steph")}
    out("   loop 2 (DIP coupler) margins flexion / whole at PIP 100: " + ", ".join(
        f"{LAYOUTS[k]['label']} {a:.1f} / {b:.1f} for a {c:.0f} deg PIP swing" for k, (a, b, c) in l2m.items()) +
        f" (current: {cur['L']['rules'][0]['m_fl']:.1f} / {cur['L']['rules'][0]['m_all']:.1f} for "
        f"{np.degrees(np.ptp(cur['s']['pip'])):.0f} deg).\n   So loop 2 is at the 40 / 25 rule too: PIP well beyond "
        "~100 would also need a re-thought DIP drive (or a lower DIP ratio), not just a better six-bar.")
    cand = [k for k in ("watt", "steph") if finals[k]["rules_ok"] and finals[k]["info"]["pip90"] >= 99.5]
    def rank(k):  # drive transmission (loop 2 caps both alike), then layers and the worst-case PIP play
        d = finals[k]
        play = max(r["play_pip"] for r in d["g"]["rows"].values())
        return d["info"]["m_fl"] + 0.5 * d["info"]["m_all"] - 5 * d["nl"] - 2 * play
    rec = max(cand or ["watt", "steph"], key=rank)
    out("   ranking score (drive margin flex + 0.5 drive margin whole - 5 x layers - 2 x worst PIP play over the travel): "
        + ", ".join(f"{LAYOUTS[k]['label']} {rank(k):.1f}" for k in ("watt", "steph")))
    other = "steph" if rec == "watt" else "watt"
    out(f"   -> best layout for PIP 100: {LAYOUTS[rec]['label']} (see the ranking above; build / no-build advice is in "
        "the study summary, not decided here)")
    make_figure(cur, finals, rec, other, table, HERE / "sixbar_study.png")
    out(f"\nfigure: sixbar_study.png   [total {time.time() - t0:.0f} s]")
    (HERE / "sixbar_report.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
