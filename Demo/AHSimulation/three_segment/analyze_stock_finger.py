#!/usr/bin/env python3
"""Kinematics, workspace and fingertip force of the stock 2-phalanx AmazingHand finger.

Works on model/finger_stock.xml (the index finger cut out of AH_Right_5F by build_finger_models.py;
first checked against finger1 of the full 5-digit model). Settles the finger on a grid of the two
motor commands and measures, between the CAD frames of the parts (see finger_common.py):
abduction, Link angle psi, MCP and PIP, the fingertip (site tip1) and the loop-closure errors.

Then it
  * identifies the coupling: fits the planar crossed four-bar gimbal / Link / Distal / Proximal
    (pivots read from the CAD frames) and compares its prediction with the sim;
  * tabulates the drive map (m1, m2) -> (abduction, psi) that the 3-segment design reuses;
  * computes the fingertip Jacobian through every loop (constraint null space) and the force the
    two SCS0009 can hold at the fingertip (stall 0.226 N m) along the flexion direction.

Outputs: data/stock_grid.npz, data/stock_flex_line.npz, reports/stock_finger_analysis.txt,
figures in cad/five_finger/three_segment/figures/ (stock_*.png).

Usage: python analyze_stock_finger.py [--step 7.5]
"""

import argparse
import time

import mujoco
import numpy as np

import plotstyle as ps
from finger_common import (DATA_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR, SERVO_STALL_NM, SIM_5F, STOCK_NAMES, Finger,
                           flexion_ctrl, log, stock_roles)

LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    log(s)


def crosscheck_with_full_hand(f):
    """finger_stock.xml must move exactly like finger1 of AH_Right_5F."""
    spec = mujoco.MjSpec.from_file(str(SIM_5F / "scene.xml"))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip1")
    a1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger1_motor1")
    worst = 0.0
    for c in [(0, 0), (60, -60), (-80, 30), (20, 70), (90, -90), (-90, -10)]:
        target = np.radians(c)
        start = d.ctrl[a1:a1 + 2].copy()
        for k in range(500):
            d.ctrl[a1:a1 + 2] = start + (target - start) * min(1, (k + 1) / 250)
            mujoco.mj_step(m, d)
        f.move(*target, ramp=0.5, hold=0.5, check=False)
        worst = max(worst, np.linalg.norm(d.site_xpos[tip] - f.d.site_xpos[f.tip]))
    return worst * 1e3


def settle_grid(f, angles):
    f.reset()
    grid = [(a1, a2) for i, a1 in enumerate(angles) for a2 in (angles if i % 2 == 0 else angles[::-1])]
    rows = []
    for a1, a2 in grid:
        f.move(np.radians(a1), np.radians(a2), ramp=0.15, hold=0.25, check=False)
        r = f.measure()
        J, resid, _ = f.motor_jacobian()
        rows.append((a1, a2, r["abd"], r["psi"], r["mcp"], r["pip"], *r["tip"], r["eq"], *J.T.ravel(), resid))
    rows.sort()
    return np.array(rows)


def sweep_flexion(f, fs, a=0.0, up_and_down=True):
    f.reset()
    f.move(*flexion_ctrl(fs[0], a), ramp=0.5, hold=0.5, check=False)
    res = []
    seq = list(fs) + (list(fs[::-1]) if up_and_down else [])
    for k, fv in enumerate(seq):
        f.move(*flexion_ctrl(fv, a), ramp=0.05, hold=0.25, check=False)
        r = f.measure()
        J, resid, V = f.motor_jacobian()
        res.append(dict(f=fv, up=k < len(fs), J=J, resid=resid, R_p2=f.frame("P2")[:3, :3].copy(),
                        R_frame=f.frame("frame")[:3, :3].copy(), **r))
    return res


# ---------------------------------------------------------------- planar four-bar model
def planar_pivots(f):
    """Pivot coordinates (y, z) in mm in the finger design frame, read from the part CAD frames."""
    f.reset()
    n = STOCK_NAMES
    d, m = f.d, f.m

    def body(name):
        return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)]

    def site(name):
        return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)]

    p = dict(
        O_M_g=f.to_frame("gimbal", site("closing_3 (1)_2")), O_L_g=f.to_frame("gimbal", body(n["gimbal"])),
        O_L_l=f.to_frame("link", body(n["gimbal"])), P4_l=f.to_frame("link", body(n["distal"])),
        P4_d=f.to_frame("P2", body(n["distal"])), PIP_d=f.to_frame("P2", body(n["proximal"])),
        PIP_p=f.to_frame("P1", body(n["proximal"])), O_M_p=f.to_frame("P1", site("closing_3 (1)_1")),
        ball1_l=f.to_frame("link", body(n["link"])), ball2_l=f.to_frame("link", site("closing_ball1_2")),
        tip_d=f.to_frame("P2", site("tip1")),
    )
    return {k: v * 1e3 for k, v in p.items()}


def fourbar_predict(piv, psi_deg):
    """MCP and PIP (deg) from the Link angle psi, crossed branch, planar (y, z)."""
    yz = lambda v: np.array([v[1], v[2]])  # noqa: E731
    O_M, O_L = yz(piv["O_M_g"]), yz(piv["O_L_g"])
    P4_0, PIP_0 = yz(piv["P4_d"]), yz(piv["PIP_d"])
    L1 = np.linalg.norm(PIP_0 - O_M)
    c = np.linalg.norm(P4_0 - PIP_0)
    out_mcp, out_pip = [], []
    for psi in np.radians(np.atleast_1d(psi_deg)):
        R = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])  # flexion in (y, z)
        P4 = O_L + R @ (P4_0 - O_L)
        dvec = P4 - O_M
        dist = np.linalg.norm(dvec)
        a = (L1 ** 2 - c ** 2 + dist ** 2) / (2 * dist)
        h = np.sqrt(max(L1 ** 2 - a ** 2, 0))
        base = O_M + a * dvec / dist
        perp = np.array([-dvec[1], dvec[0]]) / dist
        cands = [base + h * perp, base - h * perp]
        # crossed branch: the one closest to the straight pose rotated by ~psi
        guess = O_M + R @ (PIP_0 - O_M)
        PIP = min(cands, key=lambda q: np.linalg.norm(q - guess))
        mcp = -np.arctan2(PIP[1] - O_M[1], PIP[0] - O_M[0]) + np.arctan2(PIP_0[1] - O_M[1], PIP_0[0] - O_M[0])
        ang_now = np.arctan2(*(P4 - PIP)[::-1])
        ang_0 = np.arctan2(*(P4_0 - PIP_0)[::-1])
        pip = -(ang_now - ang_0) - mcp
        out_mcp.append(np.degrees(mcp))
        out_pip.append(np.degrees((pip + np.pi) % (2 * np.pi) - np.pi))
    return np.array(out_mcp), np.array(out_pip)


def kinematic_diagram(piv, line, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    ps.setup()
    yz = lambda v: np.array([v[1], v[2]])  # noqa: E731
    O_M, O_L = yz(piv["O_M_g"]), yz(piv["O_L_g"])
    P4_0, PIP_0, TIP_0 = yz(piv["P4_d"]), yz(piv["PIP_d"]), yz(piv["tip_d"])
    B1 = yz(piv["ball1_l"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    poses = [("straight (f = %.1f deg)" % line["f_straight"], line["psi_straight"]),
             ("export pose, ctrl = 0 (f = 0)", line["psi_f0"]), ("fully flexed (f = +90 deg)", line["psi_f90"])]
    for ax, (title, psi) in zip(axes, poses):
        mcp, pip = fourbar_predict(piv, psi)
        R = lambda t: np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])  # noqa: E731
        ps_, m_, p_ = np.radians(psi), np.radians(mcp[0]), np.radians(pip[0])
        P4 = O_L + R(ps_) @ (P4_0 - O_L)
        B = O_L + R(ps_) @ (B1 - O_L)
        PIP = O_M + R(m_) @ (PIP_0 - O_M)
        TIP = PIP + R(m_ + p_) @ (TIP_0 - PIP_0)
        # ground (gimbal)
        ax.plot(*np.c_[O_M, O_L], color=ps.INK2, lw=5, solid_capstyle="round", alpha=0.35)
        ax.plot(*np.c_[O_L, B], color=ps.C[0], lw=2)  # link lever (to the rod balls)
        ax.plot(*np.c_[O_L, P4], color=ps.C[0], lw=3, label="Link (input, driven by the 2 rods)")
        ax.plot(*np.c_[O_M, PIP], color=ps.C[1], lw=4, label="Proximal (rocker)")
        ax.plot(*np.c_[PIP, P4], color=ps.C[2], lw=3)
        ax.plot(*np.c_[PIP, TIP], color=ps.C[2], lw=4, label="Distal (coupler)")
        for pt, lab, off in ((O_M, "MCP", (-9, 4)), (O_L, "Link pivot", (-14, -7)), (PIP, "PIP", (2, 3)),
                             (P4, "P4", (2, -6)), (B, "rod balls", (-14, -9))):
            ax.add_patch(Circle(pt, 1.3, color=ps.SURFACE, ec=ps.INK, lw=1.2, zorder=5))
            ax.annotate(lab, pt, xytext=off, textcoords="offset points", fontsize=8, color=ps.INK2)
        ax.plot(*TIP, "o", color=ps.C[2], ms=6, mec=ps.SURFACE, mew=1.5, zorder=6)
        ax.set_title(f"{title}\npsi {psi:.1f}, MCP {mcp[0]:.1f}, PIP {pip[0]:.1f} deg")
        ax.set_aspect("equal")
        ax.set_xlim(-25, 95)
        ax.set_ylim(-80, 30)
        ax.set_xlabel("y (mm, along the straight finger)")
        ax.set_ylabel("z (mm, dorsal +)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Stock finger, flexion plane: a crossed four-bar  gimbal (ground) - Link - Distal - Proximal",
                 color=ps.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=float, default=7.5)
    args = ap.parse_args()
    for p in (DATA_DIR, REPORT_DIR, FIG_DIR):
        p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = Finger(MODEL_DIR / "scene_stock.xml", stock_roles())
    out("== Stock AmazingHand finger (index, from AH_Right_5F) - kinematics, workspace, force")
    out(f"model: nq={f.m.nq} nv={f.m.nv} neq={f.m.neq} (connect constraints) nu={f.m.nu}, timestep {f.m.opt.timestep} s")
    err = crosscheck_with_full_hand(f)
    out(f"cross-check vs finger1 of AH_Right_5F at 6 poses: max tip difference {err:.4f} mm")

    piv = planar_pivots(f)
    out("\n-- pivots in the finger design frame (mm; x lateral, y distal, z dorsal), straight pose")
    for k, v in piv.items():
        out(f"  {k:<8} {np.round(v, 3)}")
    yz = lambda v: np.array([v[1], v[2]])  # noqa: E731
    L_ground = np.linalg.norm(yz(piv["O_L_g"]) - yz(piv["O_M_g"]))
    L_link = np.linalg.norm(yz(piv["P4_l"]) - yz(piv["O_L_l"]))
    L_prox = np.linalg.norm(yz(piv["PIP_p"]) - yz(piv["O_M_p"]))
    L_crank = np.linalg.norm(yz(piv["P4_d"]) - yz(piv["PIP_d"]))
    out(f"  four-bar: ground MCP-LinkPivot {L_ground:.2f} mm, Link {L_link:.2f}, Proximal {L_prox:.2f}, "
        f"Distal crank PIP-P4 {L_crank:.2f}")
    out(f"  tip site: {np.linalg.norm(yz(piv['tip_d']) - yz(piv['PIP_d'])):.2f} mm from the PIP; "
        f"MCP->tip {L_prox + np.linalg.norm(yz(piv['tip_d']) - yz(piv['PIP_d'])):.2f} mm along the straight finger")

    # ---- flexion line
    fs = np.arange(-90, 90.01, 2.5)
    line = sweep_flexion(f, fs)
    up = [r for r in line if r["up"]]
    down = [r for r in line if not r["up"]][::-1]
    F = np.array([r["f"] for r in up])
    psi = np.array([r["psi"] for r in up])
    mcp = np.array([r["mcp"] for r in up])
    pip = np.array([r["pip"] for r in up])
    abd = np.array([r["abd"] for r in up])
    hyst = max(np.abs(np.array([r["mcp"] for r in down]) - mcp).max(), np.abs(np.array([r["pip"] for r in down]) - pip).max())
    pm, pp = fourbar_predict(piv, psi)
    f_straight = float(np.interp(0.0, mcp, F))
    out("\n-- pure flexion sweep (m1 = f, m2 = -f), f = -90..+90 deg, up and down")
    out(f"  straight finger (MCP = 0) at f = {f_straight:.1f} deg; PIP there {np.interp(f_straight, F, pip):.2f} deg")
    for fv in (-90, -60, -45, -37.5, -30, 0, 30, 45, 60, 75, 90):
        i = int(np.argmin(np.abs(F - fv)))
        out(f"  f {F[i]:6.1f}: psi {psi[i]:7.2f}  MCP {mcp[i]:7.2f}  PIP {pip[i]:7.2f}  MCP+PIP {mcp[i] + pip[i]:7.2f}  "
            f"abd {abd[i]:5.2f} deg  tip {np.round(up[i]['tip'], 1)} mm  eq {up[i]['eq']:.4f} mm")
    out(f"  range over the drive: MCP {mcp.min():.1f}..{mcp.max():.1f}, PIP {pip.min():.1f}..{pip.max():.1f}, "
        f"psi {psi.min():.1f}..{psi.max():.1f} deg; up/down hysteresis {hyst:.3f} deg")
    sel = F >= f_straight
    out(f"  from straight to fully flexed: MCP 0 -> {mcp[-1]:.1f}, PIP {np.interp(f_straight, F, pip):.1f} -> "
        f"{pip[-1]:.1f} deg (PIP/MCP slope {np.polyfit(mcp[sel], pip[sel], 1)[0]:.3f}); Link turns "
        f"{psi[-1] - np.interp(f_straight, F, psi):.1f} deg for MCP {mcp[-1]:.1f} deg")
    out(f"  planar crossed four-bar prediction from psi vs sim: max |dMCP| {np.abs(pm - mcp).max():.3f} deg, "
        f"max |dPIP| {np.abs(pp - pip).max():.3f} deg  -> the coupling IS this four-bar")
    out("  instant-centre estimate for a crossed four-bar with short ground g and short crank c: "
        f"PIP/MCP ~ g/c = {L_ground / L_crank:.3f}, MCP/psi ~ 1")

    # ---- abduction lines
    out("\n-- abduction (m1 = a + f, m2 = a - f)")
    abd_lines = {}
    for fv in (round(f_straight / 2.5) * 2.5, 0.0, 45.0):
        amax = 90 - abs(fv)
        As = np.arange(-amax, amax + 0.01, 5.0)
        f.reset()
        f.move(*flexion_ctrl(fv, As[0]), ramp=0.5, hold=0.4, check=False)
        rr = []
        for a in As:
            f.move(*flexion_ctrl(fv, a), ramp=0.05, hold=0.25, check=False)
            rr.append(f.measure())
        ab = np.array([r["abd"] for r in rr])
        mc = np.array([r["mcp"] for r in rr])
        abd_lines[fv] = (As, ab, mc)
        out(f"  f = {fv:6.1f}: a {As[0]:.0f}..{As[-1]:.0f} -> abduction {ab.min():.1f}..{ab.max():.1f} deg "
            f"(slope {np.polyfit(As, ab, 1)[0]:.3f}), MCP varies {mc.min():.1f}..{mc.max():.1f} deg")

    # ---- grid
    angles = np.arange(-90, 90.01, args.step)
    g = settle_grid(f, angles)
    cols = "m1 m2 abd psi mcp pip tx ty tz eq j1x j1y j1z j2x j2y j2z resid".split()
    G = {c: g[:, i] for i, c in enumerate(cols)}
    np.savez(DATA_DIR / "stock_grid.npz", **G, step=args.step)
    out(f"\n-- grid {len(angles)}x{len(angles)} over (m1, m2), step {args.step} deg")
    out(f"  abduction range {G['abd'].min():.1f}..{G['abd'].max():.1f} deg; psi {G['psi'].min():.1f}..{G['psi'].max():.1f}; "
        f"MCP {G['mcp'].min():.1f}..{G['mcp'].max():.1f}; PIP {G['pip'].min():.1f}..{G['pip'].max():.1f}")
    out(f"  settled loop-closure error max {G['eq'].max():.4f} mm; Jacobian null-space residual max {G['resid'].max():.2e}")
    A = np.c_[(G["m1"] + G["m2"]) / 2, (G["m1"] - G["m2"]) / 2, np.ones(len(G["m1"]))]
    ca = np.linalg.lstsq(A, G["abd"], rcond=None)[0]
    out(f"  linear fit abduction ~ {ca[0]:.3f} a + {ca[1]:.3f} f + {ca[2]:.2f} (a=(m1+m2)/2, f=(m1-m2)/2); "
        f"rms {np.sqrt(np.mean((A @ ca - G['abd']) ** 2)):.2f} deg")
    tips = np.c_[G["tx"], G["ty"], G["tz"]]
    from scipy.spatial import ConvexHull
    hull = ConvexHull(tips)
    out(f"  fingertip (site tip1) workspace: x {tips[:, 0].min():.1f}..{tips[:, 0].max():.1f}, y {tips[:, 1].min():.1f}.."
        f"{tips[:, 1].max():.1f}, z {tips[:, 2].min():.1f}..{tips[:, 2].max():.1f} mm; convex hull of the reachable "
        f"tip surface {hull.volume / 1e3:.1f} cm^3")

    # ---- force at the fingertip along the flexion motion (a = 0)
    out("\n-- fingertip force from servo torque (Jacobian transpose through all loops), a = 0")
    out(f"  SCS0009 stall torque {SERVO_STALL_NM:.4f} N m (2.3 kg.cm @ 6 V). F = force on the tip along n;")
    out("  motor torques tau = J^T F; the holdable force is stall / max_i |J_i . n| (J in m/rad).")
    force = []
    for r in up:
        J = r["J"]  # world, m/rad per motor
        v = J[:, 0] - J[:, 1]  # tip velocity for df (m1 = f, m2 = -f)
        n_motion = v / np.linalg.norm(v)
        n_pad = -r["R_p2"][:, 2]  # palmar normal of the distal
        fm = SERVO_STALL_NM / np.abs(J.T @ n_motion).max()
        fp = SERVO_STALL_NM / np.abs(J.T @ n_pad).max()
        force.append((r["f"], np.linalg.norm(v) * 1e3 * np.pi / 180, fm, fp))
    force = np.array(force)
    for fv in (f_straight, 0, 45, 75, 90):
        i = int(np.argmin(np.abs(force[:, 0] - fv)))
        out(f"  f {force[i, 0]:6.1f}: tip speed {force[i, 1]:.3f} mm/deg of f; holdable force along the motion "
            f"{force[i, 2]:.2f} N, normal to the distal pad {force[i, 3]:.2f} N")
    np.savez(DATA_DIR / "stock_flex_line.npz", f=F, psi=psi, mcp=mcp, pip=pip, abd=abd,
             tip=np.array([r["tip"] for r in up]), force=force, f_straight=f_straight)

    # ---- figures
    import matplotlib.pyplot as plt
    ps.setup()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(F, psi, color=ps.C[0], label="Link angle psi")
    ax[0].plot(F, mcp, color=ps.C[1], label="MCP")
    ax[0].plot(F, pip, color=ps.C[2], label="PIP")
    ax[0].axvline(f_straight, color=ps.INK2, lw=1)
    ax[0].annotate(f"straight finger\nf = {f_straight:.1f}", (f_straight, -25), xytext=(6, 0), textcoords="offset points",
                   fontsize=8, color=ps.INK2)
    ax[0].set_xlabel("flexion command f = (m1 - m2)/2 (deg), abduction a = 0")
    ax[0].set_ylabel("angle (deg)")
    ax[0].set_title("Joint angles vs servo command")
    ax[0].legend(loc="upper left")
    ax[1].plot(mcp, pip, color=ps.C[0], label="sim (MuJoCo, settled)")
    ax[1].plot(pm, pp, color=ps.C[1], lw=1.2, ls=(0, (4, 2)), label="planar crossed four-bar")
    ax[1].plot([0, 90], [0, 100], color=ps.INK2, lw=1, label="human-like target PIP = 1.1 MCP")
    ax[1].set_xlabel("MCP (deg)")
    ax[1].set_ylabel("PIP (deg)")
    ax[1].set_title("Stock coupling curve PIP(MCP)")
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stock_coupling.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, (fv, (As, ab, mc)) in enumerate(abd_lines.items()):
        ax[0].plot(As, ab, color=ps.C[k], label=f"f = {fv:.1f}")
    ax[0].set_xlabel("abduction command a = (m1 + m2)/2 (deg)")
    ax[0].set_ylabel("gimbal abduction (deg)")
    ax[0].set_title("Abduction vs servo command")
    ax[0].legend(loc="upper left")
    ax[1].plot(force[:, 0], force[:, 2], color=ps.C[0], label="along the fingertip motion")
    ax[1].plot(force[:, 0], force[:, 3], color=ps.C[1], label="normal to the distal pad")
    ax[1].axvline(f_straight, color=ps.INK2, lw=1)
    ax[1].set_ylim(0, min(30, force[:, 2:].max() * 1.1))
    ax[1].set_xlabel("flexion command f (deg), a = 0")
    ax[1].set_ylabel("holdable fingertip force (N)")
    ax[1].set_title("Fingertip force at SCS0009 stall (0.226 N m)")
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stock_abduction_force.png", dpi=130)
    plt.close(fig)

    # command maps: abduction (diverging) and MCP / PIP (sequential) over (m1, m2)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    n = len(angles)
    M1 = G["m1"].reshape(n, n)
    M2 = G["m2"].reshape(n, n)
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#104281"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (key, title, cmap, norm) in enumerate((("abd", "abduction (deg)", div, TwoSlopeNorm(0, -40, 40)),
                                                 ("mcp", "MCP (deg)", seq, None), ("pip", "PIP (deg)", seq, None))):
        Z = G[key].reshape(n, n)
        cs = ax[k].contourf(M1, M2, Z, levels=16, cmap=cmap, norm=norm)
        cl = ax[k].contour(M1, M2, Z, levels=8, colors=ps.INK, linewidths=0.4, alpha=0.5)
        ax[k].clabel(cl, fontsize=7, fmt="%.0f")
        fig.colorbar(cs, ax=ax[k], shrink=0.85)
        ax[k].plot([-90, 90], [90, -90], color=ps.INK2, lw=0.8, ls=(0, (3, 2)))
        ax[k].annotate("pure flexion (a = 0)", (60, -60), xytext=(-20, 12), textcoords="offset points", fontsize=7.5,
                       color=ps.INK2)
        ax[k].set_aspect("equal")
        ax[k].set_xlabel("motor 1 command m1 (deg)")
        ax[k].set_ylabel("motor 2 command m2 (deg)")
        ax[k].set_title(f"stock finger: {title}")
        ax[k].grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stock_command_maps.png", dpi=130)
    plt.close(fig)

    line_info = dict(f_straight=f_straight, psi_straight=float(np.interp(f_straight, F, psi)),
                     psi_f0=float(np.interp(0, F, psi)), psi_f90=float(psi[-1]))
    kinematic_diagram(piv, line_info, FIG_DIR / "stock_kinematic_diagram.png")
    out(f"\nfigures: {', '.join(p.name for p in sorted(FIG_DIR.glob('stock_*.png')))}  [{time.time() - t0:.0f} s]")
    (REPORT_DIR / "stock_finger_analysis.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
