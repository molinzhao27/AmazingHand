#!/usr/bin/env python3
"""Verify AH_Right_5F_Wrist (5-digit hand on the powered wrist and forearm base).

  1. structure: compiles, names unique, actuator order, ranges, keyframe, masses;
  2. stability: wrist actuators swept through their full ctrlrange (alone, box corners), IK-driven ROM
     boundary, 60 abrupt random steps of all 12 actuators - fingers open and closed: no NaN, no warnings;
     loop-closure (equality) errors of the two push rods and of the fingers;
  3. ROM: linkage workspace (analytic), achieved wrist angles in the sim open-loop (servo P sag under
     gravity) and with the servo-angle outer loop the controller should run;
  4. torque: static per-servo torque over the ROM for six mounting orientations of the forearm, with
     and without 200 g at the fingertips (sim, cross-checked against the analytic Jacobian model), and a
     dynamic estimate from fast minimum-jerk moves;
  5. clearance: hand (with 5F shells and pads; keyframe / OpenHand / CloseHand / thumb over a 7x7 grid)
     vs forearm, fork, ring, push rods, horns; and the wrist parts against each other, over the ROM.

Usage: python verify_wrist_model.py [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wrist_sim import DEMO_POSES_5F, HAND_POSES, MJCF, PAYLOAD_CAD, WristModel, aid, jid, wd  # noqa: E402
from verify_5f_model import Group, group_distance  # noqa: E402

STALL = 2.94           # N.m, STS3215 C018 at 12 V (30 kg.cm), the servo in the model
RATED = 0.98           # N.m, its "rated" (continuous) torque, 10 kg.cm
ALT = {"STS3215 C001 @ 6 V (hand rail)": 1.62, "STS3215 C001 @ 7.4 V": 1.91, "STS3215 C018 @ 12 V": 2.94, "STS3250 @ 12 V": 4.9}
G = 9.81
ORIENT = {             # gravity direction in the root frame for six ways of mounting the forearm
    "forearm vertical, hand up": (0, 0, -1),
    "forearm vertical, hand hanging down": (0, 0, 1),
    "forearm horizontal, palm down": (1, 0, 0),
    "forearm horizontal, palm up": (-1, 0, 0),
    "forearm horizontal, thumb up": (0, -1, 0),
    "forearm horizontal, thumb down": (0, 1, 0),
}
OUT = {}
LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


# ----------------------------------------------------------------------------- helpers
def rom_grid(flex_step, n_dev):
    """(phi, psi) in deg on the coupled operating ROM: n_dev deviation values per flexion value."""
    out = []
    for p in np.arange(-60, 60.01, flex_step):
        lim = np.degrees(wd.rom_dev_limit(np.radians(p)))
        out += [(float(p), float(round(s, 2))) for s in np.linspace(-lim, lim, n_dev)]
    return out


ROM_BOUNDARY = [(60, 0), (60, 15), (40, 25), (0, 25), (-40, 25), (-60, 15), (-60, 0), (-60, -15), (-40, -25), (0, -25),
                (40, -25), (60, -15), (60, 0), (0, 0)]

def outer_loop_step(w, target_theta, gain=0.1):
    """Servo-angle outer loop (what the controller does with Present_Position): nudge the servo goal by
    the remaining angle error so the P-controlled servo ends where the IK wants it, despite the load."""
    m, d = w.m, w.d
    err = target_theta - w.servo_angles()
    lo, hi = m.actuator_ctrlrange[w.a_wrist, 0], m.actuator_ctrlrange[w.a_wrist, 1]
    d.ctrl[w.a_wrist] = np.clip(d.ctrl[w.a_wrist] + gain * err, lo, hi)


def settle_closed_loop(w, wrist, hand_ctrl=None, t=2.5, period=10):
    th = wd.servo_ik(*wrist)
    if hand_ctrl is not None:
        w.d.ctrl[:] = hand_ctrl
    w.d.ctrl[w.a_wrist] = th
    n = int(t / w.m.opt.timestep)
    for k in range(n):
        mujoco.mj_step(w.m, w.d)
        if k % period == period - 1 and k > n // 4:
            outer_loop_step(w, th)
    return th


def eq_errors(m, d, eqs):
    return np.array([np.linalg.norm(d.site_xpos[m.eq_obj1id[e]] - d.site_xpos[m.eq_obj2id[e]]) for e in eqs])


def warnings(d):
    return {mujoco.mjtWarning(i).name: d.warning[i].number for i in range(len(d.warning)) if d.warning[i].number}


# ----------------------------------------------------------------------------- 1. structure
def check_structure(w):
    m = w.m
    log("== 1. structure")
    log(f"  nbody={m.nbody} njnt={m.njnt} nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq} ngeom={m.ngeom} nkey={m.nkey} "
        f"nmocap={m.nmocap} timestep={m.opt.timestep}")
    for obj, n in [(mujoco.mjtObj.mjOBJ_BODY, m.nbody), (mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
                   (mujoco.mjtObj.mjOBJ_SITE, m.nsite), (mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)]:
        names = [x for x in (mujoco.mj_id2name(m, obj, i) for i in range(n)) if x]
        assert len(names) == len(set(names))
    acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    ref = mujoco.MjModel.from_xml_path(str(MJCF.parent.parent / "AH_Right_5F" / "mjcf" / "scene.xml"))
    ref_acts = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(ref.nu)]
    assert acts[:10] == ref_acts and acts[10:] == ["wrist_motor1", "wrist_motor2"]
    log(f"  names unique; actuators: 10 finger actuators in the AH_Right_5F order, then {acts[10:]}")
    for i in w.a_wrist:
        log(f"  {acts[i]}: ctrlrange {np.degrees(m.actuator_ctrlrange[i]).round(1)} deg, kp {m.actuator_gainprm[i, 0]:.3g} N.m/rad, "
            f"forcerange {m.actuator_forcerange[i]} N.m; joint damping {m.dof_damping[m.jnt_dofadr[m.actuator_trnid[i, 0]]]:.3g}, "
            f"armature {m.dof_armature[m.jnt_dofadr[m.actuator_trnid[i, 0]]]:.3g}, frictionloss {m.dof_frictionloss[m.jnt_dofadr[m.actuator_trnid[i, 0]]]:.3g}")
    for name in ("wrist_flexion", "wrist_deviation"):
        j = jid(m, name)
        log(f"  {name}: qpos[{m.jnt_qposadr[j]}], range {np.degrees(m.jnt_range[j]).round(1)} deg, axis (root) {m.jnt_axis[j]}")
    a0 = m.jnt_qposadr[jid(m, "finger1_motor1")]
    assert np.allclose(w.m.key_qpos[0][a0:a0 + ref.nq], ref.key_qpos[0], atol=2e-4)
    log(f"  hand joints at qpos[{a0}:{a0 + ref.nq}] in the AH_Right_5F order; keyframe matches the AH_Right_5F key to "
        f"{np.abs(w.m.key_qpos[0][a0:a0 + ref.nq] - ref.key_qpos[0]).max():.1e}")
    w.reset()
    eqs = [e for e in range(m.neq)]
    log(f"  keyframe: max equality error {eq_errors(m, w.d, eqs).max() * 1e3:.4f} mm; wrist at "
        f"({np.degrees(w.wrist_angles()[0]):+.2f}, {np.degrees(w.wrist_angles()[1]):+.2f}) deg")
    ring = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_gimbal_ring")
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_interface")
    sub = lambda top: [b for b in range(m.nbody) if _under(m, b, top)]
    mh = m.body_mass[sub(hand)].sum()
    mw_ = m.body_mass[sub(ring)].sum()
    log(f"  masses: hand {mh * 1000:.1f} g (lumped mass model + finger bodies), hand + adapter + ring "
        f"{mw_ * 1000:.1f} g, each push rod {m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist_rod1')] * 1000:.1f} / "
        f"{m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wrist_rod2')] * 1000:.1f} g")
    mujoco.mj_forward(m, w.d)
    com = w.d.subtree_com[ring]
    log(f"  CoM of everything the wrist carries: {np.round((com - wd.C_ROOT) * 1000, 1)} mm from the wrist centre (root axes), "
        f"{np.linalg.norm(com - wd.C_ROOT) * 1000:.1f} mm")
    OUT["moving_mass_g"] = float(mw_ * 1000)
    OUT["hand_mass_g"] = float(mh * 1000)


def _under(m, b, top):
    while b > 0:
        if b == top:
            return True
        b = m.body_parentid[b]
    return False


# ----------------------------------------------------------------------------- 2. stability
def run_segments(w, segments, label, closed_loop_targets=None):
    m, d = w.m, w.d
    wrist_eq = [e for e in range(m.neq) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) in ("wrist_rod1_closure", "wrist_rod2_closure")]
    finger_eq = [e for e in range(m.neq) if e not in wrist_eq]
    vmax, ew, ef, ew_set, steps = 0.0, 0.0, 0.0, 0.0, 0
    lim_hits = 0
    for k, (target, ramp, hold) in enumerate(segments):
        start = d.ctrl.copy()
        nr, nh = int(round(ramp / m.opt.timestep)), int(round(hold / m.opt.timestep))
        th = None if closed_loop_targets is None else closed_loop_targets[k]
        for s in range(nr + nh):
            a = 1.0 if s >= nr else (s + 1) / nr
            if th is None or s < nr:
                d.ctrl[:] = start + (target - start) * a
            elif s % 10 == 9:
                outer_loop_step(w, th)
            mujoco.mj_step(m, d)
            steps += 1
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                raise RuntimeError(f"{label}: non-finite state at step {steps}")
            vmax = max(vmax, np.abs(d.qvel).max())
            ew = max(ew, eq_errors(m, d, wrist_eq).max())
            ef = max(ef, eq_errors(m, d, finger_eq).max())
        ew_set = max(ew_set, eq_errors(m, d, wrist_eq).max())
        q = w.wrist_angles()
        lim_hits += int(abs(q[0]) > wd.FLEX_LIMIT - 0.01 or abs(q[1]) > wd.DEV_LIMIT - 0.01)
    warn = warnings(d)
    log(f"  {label:<52} {steps * m.opt.timestep:6.1f} s  max|qvel| {vmax:6.2f} rad/s  push-rod closure err max "
        f"{ew * 1e3:.3f} mm (settled {ew_set * 1e3:.3f})  finger closures max {ef * 1e3:.3f} mm  "
        f"segments ending on a wrist stop {lim_hits}/{len(segments)}  warnings {warn or 'none'}")
    return dict(time=steps * m.opt.timestep, vmax=vmax, rod_err=ew, rod_err_settled=ew_set, finger_err=ef, warnings=warn)


def check_stability(w):
    log("== 2. stability sweeps (fingers open = OpenHand, closed = CloseHand)")
    m = w.m
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    res = {}
    for hand in ("OpenHand", "CloseHand"):
        base = w.ctrl_for(HAND_POSES[hand])
        w.reset()
        w.move(base, 0.5, 0.5)
        segs = []
        for a in w.a_wrist:
            for t, ramp, hold in [(hi[a], 1.5, 0.5), (lo[a], 3.0, 0.5), (0.0, 1.5, 0.5)]:
                c = base.copy()
                c[a] = t
                segs.append((c, ramp, hold))
        res[f"alone_{hand}"] = run_segments(w, segs, f"{hand}: each wrist servo alone 0 -> +75 -> -75 -> 0 deg")
        w.reset()
        w.move(base, 0.5, 0.5)
        segs = []
        for s1, s2 in [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1), (0, 0)]:
            c = base.copy()
            c[w.a_wrist] = [s1 * hi[w.a_wrist[0]], s2 * hi[w.a_wrist[1]]]
            segs.append((c, 1.5, 0.5))
        res[f"corners_{hand}"] = run_segments(w, segs, f"{hand}: both wrist servos around their +-75 deg box")
        w.reset()
        w.move(base, 0.5, 0.5)
        segs, ths = [], []
        for phi, psi in ROM_BOUNDARY:
            q = np.radians([phi, psi])
            segs.append((w.ctrl_for(None, q, base), 1.0, 0.8))
            ths.append(wd.servo_ik(*q))
        res[f"rom_{hand}"] = run_segments(w, segs, f"{hand}: IK around the coupled ROM boundary (outer loop)", ths)
    w.reset()
    rng = np.random.default_rng(1)
    segs = [(rng.uniform(lo, hi), 0.0, 0.6) for _ in range(60)]
    res["random"] = run_segments(w, segs, "60 abrupt random steps of all 12 actuators")
    OUT["stability"] = {k: {kk: (vv if not isinstance(vv, float) else round(vv, 6)) for kk, vv in v.items()} for k, v in res.items()}


# ----------------------------------------------------------------------------- 3. ROM
def check_rom(w):
    log("== 3. range of motion")
    # analytic workspace of the linkage within the servo ctrlrange
    grid = [(p, s) for p in range(-70, 71, 1) for s in range(-35, 36, 1)]
    ok = {}
    for p, s in grid:
        th = wd.servo_ik(np.radians(p), np.radians(s))
        ok[(p, s)] = bool(np.all(np.isfinite(th)) and np.all(np.abs(th) <= wd.SERVO_LIMIT))
    rect = all(ok[(p, s)] for p in range(-60, 61) for s in range(-25, 26))
    th_rect = max(np.abs(wd.servo_ik(np.radians(p), np.radians(s))).max() for p, s in rom_grid(2.5, 21))
    pf = max(p for p in range(0, 71) if ok[(p, 0)])
    pe = min(p for p in range(-70, 1) if ok[(p, 0)])
    sr = max(s for s in range(0, 36) if ok[(0, s)])
    su = min(s for s in range(-35, 1) if ok[(0, s)])
    log("  operating ROM (coupled, enforced by wrist_design.clamp_wrist): flexion -60..+60 deg, deviation +-25 deg up to "
        "+-40 deg of flexion, tapering to +-15 deg at +-60 deg")
    log(f"  linkage (analytic): even the whole +-60 x +-25 deg rectangle is reachable: {rect}; max |servo angle| needed on the "
        f"operating ROM {np.degrees(th_rect):.1f} deg (ctrlrange +-{np.degrees(wd.SERVO_LIMIT):.0f}); within +-75 deg servo travel the "
        f"linkage alone reaches flexion {pe}..+{pf} deg, deviation {su}..+{sr} deg - the joint stops (+-65 / +-30 deg) end the travel first")
    worst_mis = {}
    for p, s in rom_grid(5, 11):
            for k, v in wd.misalignment(np.radians(p), np.radians(s)).items():
                worst_mis[k] = max(worst_mis.get(k, 0), v)
    log("  ball-link misalignment over the ROM (angle between rod and the plane normal to the stud): "
        + ", ".join(f"rod {i} {end} {v:.1f} deg" for (i, end), v in sorted(worst_mis.items())))
    OUT["misalignment_deg"] = {f"rod{i}_{e}": round(v, 1) for (i, e), v in worst_mis.items()}
    # why the gimbal is compact: how close the rods come to the flexion axis, and to the fork lugs
    def rod_axis(grid):
        best_axis, best_lug = 1e9, 1e9
        for p, s in grid:
            th = wd.servo_ik(np.radians(p), np.radians(s))
            for k, i in enumerate((1, 2)):
                a, b = wd.ball_bottom(i, th[k]), wd.ball_top(i, np.radians(p), np.radians(s))
                P = a + np.linspace(0, 1, 400)[:, None] * (b - a)
                rad = np.where((np.linalg.norm(P - b, axis=1) < 11) | (np.linalg.norm(P - a, axis=1) < 11), 3.25, 1.5)
                r = np.hypot(P[:, 1], P[:, 2])
                near = np.abs(P[:, 0]) <= 26
                best_axis = min(best_axis, (r - rad)[near].min())
                dx = np.maximum(0, np.maximum(wd.LUG_IN - np.abs(P[:, 0]), np.abs(P[:, 0]) - wd.LUG_OUT))
                dr = np.maximum(0, r - wd.LUG_R)
                best_lug = min(best_lug, (np.hypot(dx, dr) - rad).min())
        return best_axis, best_lug
    stops = [(p, s) for p in np.linspace(-65, 65, 27) for s in np.linspace(-30, 30, 13)]
    ax_rom, lug_rom = rod_axis(rom_grid(2.5, 13))
    ax_st, lug_st = rod_axis(stops)
    log(f"  push rods vs the flexion axis (rod / link-neck surface, |x| <= 26 mm): min {ax_rom:.2f} mm on the ROM, {ax_st:.2f} mm "
        f"over the stop rectangle -> nothing on that axis may sit at |x| ~ 16..26 mm; clearance to the compact fork lugs "
        f"(|x| {wd.LUG_IN}..{wd.LUG_OUT}, r {wd.LUG_R}): {lug_rom:.2f} mm on the ROM, {lug_st:.2f} mm at the stops")
    OUT["rod_vs_fork_lugs_mm"] = (round(float(lug_rom), 2), round(float(lug_st), 2))
    # sim: open loop and closed loop, hand up (gravity as in the scene), OpenHand
    base = w.ctrl_for(HAND_POSES["OpenHand"])
    rows = []
    for p, s in [(pp, ss) for pp in (-60, -40, 0, 40, 60) for ss in (-1, 0, 1)]:
            s = round(float(s * np.degrees(wd.rom_dev_limit(np.radians(p)))), 1)
            q = np.radians([p, s])
            w.reset()
            w.move(base, 0.4, 0.2)
            w.move(w.ctrl_for(None, q, base), 0.8, 0.8)
            open_q = w.wrist_angles().copy()
            settle_closed_loop(w, q, t=3.0)
            closed_q = w.wrist_angles().copy()
            rows.append((p, s, np.degrees(open_q - q), np.degrees(closed_q - q)))
    eo = np.array([np.abs(r[2]).max() for r in rows])
    ec = np.array([np.abs(r[3]).max() for r in rows])
    log(f"  sim, hand up, OpenHand, 15 poses on the ROM: |achieved - commanded| open loop (servo goal = IK, P sag) "
        f"max {eo.max():.2f} deg (worst at {rows[int(eo.argmax())][:2]}); with the servo-angle outer loop max {ec.max():.2f} deg")
    for p, s, a, b in rows:
        if abs(p) == 60 or (p == 0 and s == 0):
            log(f"    command ({p:+3.0f}, {s:+3.0f}) deg: open loop error ({a[0]:+5.2f}, {a[1]:+5.2f}), outer loop ({b[0]:+5.2f}, {b[1]:+5.2f}) deg")
    OUT["rom"] = dict(rectangle_reachable=rect, servo_max_deg=round(float(np.degrees(th_rect)), 1), open_loop_err_max_deg=round(float(eo.max()), 2),
                      closed_loop_err_max_deg=round(float(ec.max()), 2), linkage_flex=(pe, pf), linkage_dev=(su, sr))


# ----------------------------------------------------------------------------- 4. torque
def analytic_static(w, gdir, payload):
    """Per-servo static torque from the Jacobian and the sim's mass distribution (no friction)."""
    m, d = w.m, w.d
    ring = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_gimbal_ring")
    adapter = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_hand_adapter")
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    g = np.asarray(gdir, float) * G
    out = []
    for body, axis_name in ((ring, "wrist_flexion"), (adapter, "wrist_deviation")):
        j = jid(m, axis_name)
        ax = d.xaxis[j]
        anchor = d.xanchor[j]
        M = d.subtree_com[body] - anchor
        mass = m.body_subtreemass[body]
        out.append(-np.dot(np.cross(M, mass * g), ax))   # joint torque needed to hold the load
    return np.array(out)


def check_torque(w_plain, w_pay, quick):
    log("== 4. wrist torque budget (model servo STS3215 C018 @ 12 V: stall %.2f N.m, rated %.2f N.m)" % (STALL, RATED))
    grid = rom_grid(15 if quick else 10, 5)
    summary = {}
    for wname, w in (("hand", w_plain), ("hand + 200 g at the fingertips", w_pay)):
        m, d = w.m, w.d
        base = w.ctrl_for(HAND_POSES["OpenHand"])
        worst_all = (0, None)
        for oname, gdir in ORIENT.items():
            m.opt.gravity[:] = np.asarray(gdir, float) * G
            worst, worst_an, cross = (0, None), 0.0, []
            w.reset()
            w.move(base, 0.4, 0.3)
            for p, s in grid:
                q = np.radians([p, s])
                settle_closed_loop(w, q, base, t=2.5)
                tau = np.abs(d.actuator_force[w.a_wrist])
                if tau.max() > worst[0]:
                    worst = (tau.max(), (p, s))
                th = w.servo_angles()
                qq = w.wrist_angles()
                # analytic: tau_servo = J^-T tau_joint, at the achieved pose
                J = wd.jacobian(*qq)
                tj = analytic_static(w, gdir, None)
                ta = np.linalg.solve(J.T, tj)
                worst_an = max(worst_an, np.abs(ta).max())
                cross.append(np.abs(np.abs(ta) - tau).max())
            m.opt.gravity[:] = (0, 0, -G)
            summary[(wname, oname)] = (worst[0], worst[1], worst_an, max(cross))
            if worst[0] > worst_all[0]:
                worst_all = (worst[0], (oname, worst[1]))
            log(f"  {wname:<32} {oname:<36} max per-servo static torque {worst[0]:.3f} N.m ({100 * worst[0] / STALL:4.1f} % of stall) "
                f"at {worst[1]}; analytic {worst_an:.3f} N.m (max |sim - analytic| {max(cross):.3f})")
        log(f"  -> {wname}: worst {worst_all[0]:.3f} N.m = {100 * worst_all[0] / STALL:.0f} % of stall, "
            f"{worst_all[0] / RATED:.2f} x rated, at {worst_all[1]}; as % of the stall torque of: "
            + ", ".join(f"{k} {100 * worst_all[0] / v:.0f} %" for k, v in ALT.items()))
        OUT[f"static_worst_{'payload' if 'g at' in wname else 'hand'}"] = dict(torque=round(float(worst_all[0]), 3), where=str(worst_all[1]))
    # neutral pose, most common use
    for wname, w in (("hand", w_plain), ("hand + 200 g", w_pay)):
        m, d = w.m, w.d
        vals = []
        for oname, gdir in ORIENT.items():
            m.opt.gravity[:] = np.asarray(gdir, float) * G
            w.reset()
            w.move(w.ctrl_for(HAND_POSES["OpenHand"]), 0.4, 0.3)
            settle_closed_loop(w, (0.0, 0.0), t=2.5)
            vals.append(np.abs(d.actuator_force[w.a_wrist]).max())
            m.opt.gravity[:] = (0, 0, -G)
        log(f"  {wname}, neutral wrist: max per-servo torque over the six orientations {max(vals):.3f} N.m ({100 * max(vals) / STALL:.0f} % of stall)")
        OUT[f"static_neutral_{'payload' if '200' in wname else 'hand'}"] = round(float(max(vals)), 3)

    # dynamic: minimum-jerk moves, forearm horizontal palm down (worst gravity for flexion) and hand up
    log("  dynamic: minimum-jerk wrist moves (servo model: kp 17.11 N.m/rad, force limit 2.94 N.m, damping 0.56 N.m.s/rad "
        "-> no-load speed 5.25 rad/s; datasheet 4.7 rad/s)")
    for wname, w in (("hand", w_plain), ("hand + 200 g", w_pay)):
        m, d = w.m, w.d
        for oname in ("forearm horizontal, palm down", "forearm vertical, hand up"):
            m.opt.gravity[:] = np.asarray(ORIENT[oname], float) * G
            for (q0, q1), T in [(((-60, 0), (60, 0)), 2.0), (((-60, 0), (60, 0)), 1.2), (((-60, 0), (60, 0)), 0.8), (((0, -25), (0, 25)), 0.6)]:  # noqa
                w.reset()
                w.move(w.ctrl_for(HAND_POSES["OpenHand"]), 0.3, 0.2)
                settle_closed_loop(w, np.radians(q0), t=2.5)
                a0 = np.radians(q0)
                a1 = np.radians(q1)
                n = int(T / m.opt.timestep)
                peak, sat, vpk, err = 0.0, 0, 0.0, 0.0
                for k in range(n + int(0.4 / m.opt.timestep)):
                    t = min(1.0, (k + 1) / n)
                    sfac = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
                    qd = a0 + (a1 - a0) * sfac
                    d.ctrl[w.a_wrist] = wd.servo_ik(*qd)
                    mujoco.mj_step(m, d)
                    f = np.abs(d.actuator_force[w.a_wrist])
                    peak = max(peak, f.max())
                    sat += int(f.max() > STALL - 1e-3)
                    vpk = max(vpk, np.abs(d.qvel[[m.jnt_dofadr[jid(m, 'wrist_motor1')], m.jnt_dofadr[jid(m, 'wrist_motor2')]]]).max())
                    err = max(err, np.abs(np.degrees(w.wrist_angles() - qd)).max())
                vmax_cmd = 1.875 * np.abs(a1 - a0).max() / T
                amax_cmd = 5.774 * np.abs(a1 - a0).max() / T ** 2
                log(f"    {wname:<13} {oname:<32} {q0}->{q1} deg in {T:.1f} s (peak {vmax_cmd:.1f} rad/s, {amax_cmd:.0f} rad/s2 at the wrist): "
                    f"peak servo torque {peak:.2f} N.m ({100 * peak / STALL:.0f} % stall), at the limit for {sat * m.opt.timestep * 1000:.0f} ms, "
                    f"peak servo speed {vpk:.1f} rad/s, max tracking error {err:.1f} deg")
            m.opt.gravity[:] = (0, 0, -G)
    # inertia about the axes
    for wname, ww in (("hand", w_plain), ("hand + 200 g", w_pay)):
        ww.reset()
        Mf = np.zeros((ww.m.nv, ww.m.nv))
        mujoco.mj_fullM(ww.m, ww.d, Mf)
        i_f, i_d = Mf[ww.v_flex, ww.v_flex], Mf[ww.v_dev, ww.v_dev]
        log(f"  inertia about the wrist axes ({wname}, fingers at the keyframe): flexion {i_f * 1e4:.1f}e-4 kg.m2, "
            f"deviation {i_d * 1e4:.1f}e-4 kg.m2 -> at 20 rad/s2 {i_f * 20:.3f} / {i_d * 20:.3f} N.m at the joint")
        OUT[f"inertia_{'payload' if '200' in wname else 'hand'}"] = (float(i_f), float(i_d))


# ----------------------------------------------------------------------------- 5. clearance
def set_wrist_pose(w, phi, psi):
    """Set the wrist joints and the linkage kinematically (IK), keeping the finger joints as they are."""
    m, d = w.m, w.d
    th = wd.servo_ik(phi, psi)
    d.qpos[w.q_flex], d.qpos[w.q_dev] = phi, psi
    for k, i in enumerate((1, 2)):
        d.qpos[w.q_motor[k]] = th[k]
        # rod: minimal rotation from the neutral direction to the current one, expressed in the horn frame
        u0 = wd.P_TOP_HAND[i] - wd.ball_bottom(i, 0.0)
        u1 = wd.ball_top(i, phi, psi) - wd.ball_bottom(i, th[k])
        u0, u1 = u0 / np.linalg.norm(u0), u1 / np.linalg.norm(u1)
        v = np.cross(u0, u1)
        c = np.dot(u0, u1)
        q_w = np.array([1 + c, *v])
        q_w /= np.linalg.norm(q_w)
        # W -> root: conjugate by R_CAD2ROOT
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q_w)
        Rr = wd.R_CAD2ROOT @ R.reshape(3, 3) @ wd.R_CAD2ROOT.T
        ax = wd.dir_w_to_root([-wd.SIDE[i], 0, 0])
        Rh = np.zeros(9)
        qh = np.array([np.cos(th[k] / 2), *(np.sin(th[k] / 2) * ax)])
        mujoco.mju_quat2Mat(Rh, qh)
        Rb = Rh.reshape(3, 3).T @ Rr
        qb = np.zeros(4)
        mujoco.mju_mat2Quat(qb, Rb.ravel())
        j = jid(m, f"wrist_rod{i}_ball")
        d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 4] = qb
    mujoco.mj_kinematics(m, d)
    eqs = [e for e in range(m.neq) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) in ("wrist_rod1_closure", "wrist_rod2_closure")]
    err = eq_errors(m, d, eqs).max()
    assert err < 1e-6, f"kinematic wrist pose ({np.degrees(phi):.1f}, {np.degrees(psi):.1f}): closure error {err * 1e3:.4f} mm"


def body_geoms(m, names, exclude_mesh=()):
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
            if not any(e in mesh for e in exclude_mesh):
                out.append(g)
    return out


def check_clearance(quick):
    log("== 5. clearance over the wrist ROM (surface samples every 1 mm, 0.3 mm below 3 mm; < 0.23 mm = touching)")
    w = WristModel("scene_shells.xml")
    m, d = w.m, w.d
    hand_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_interface")
    hand = [g for g in range(m.ngeom) if _under(m, m.geom_bodyid[g], hand_root) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    groups = {
        "hand (+ shells, pads)": hand,
        "hand adapter": body_geoms(m, ["wrist_hand_adapter"], ("f683", "stud")),
        "adapter studs": body_geoms(m, ["wrist_hand_adapter"], ("adapter", "f683")),
        "gimbal ring": body_geoms(m, ["wrist_gimbal_ring"], ("pin",)),
        "forearm (shells, fork, frame, servos)": body_geoms(m, ["wrist_forearm"], ("f683",)),
        "fork": [g for g in body_geoms(m, ["wrist_forearm"]) if "fork" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])],
        "rod 1 (links + rod)": body_geoms(m, ["wrist_rod1"]),
        "rod 2 (links + rod)": body_geoms(m, ["wrist_rod2"]),
        "horn arm 1": body_geoms(m, ["wrist_horn1"]),
        "horn arm 2": body_geoms(m, ["wrist_horn2"]),
        "servo 1": [g for g in body_geoms(m, ["wrist_forearm"]) if "sts3215_motor1" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])],
        "servo 2": [g for g in body_geoms(m, ["wrist_forearm"]) if "sts3215_motor2" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])],
        "forearm w/o servos": [g for g in body_geoms(m, ["wrist_forearm"], ("f683",)) if "sts3215" not in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])],
    }
    pairs = [("hand (+ shells, pads)", "forearm (shells, fork, frame, servos)"), ("hand (+ shells, pads)", "gimbal ring"),
             ("hand (+ shells, pads)", "rod 1 (links + rod)"), ("hand (+ shells, pads)", "rod 2 (links + rod)"),
             ("hand adapter", "forearm (shells, fork, frame, servos)"), ("adapter studs", "forearm (shells, fork, frame, servos)"),
             ("hand adapter", "gimbal ring"), ("gimbal ring", "fork"),
             ("rod 1 (links + rod)", "forearm w/o servos"), ("rod 2 (links + rod)", "forearm w/o servos"),
             ("rod 1 (links + rod)", "servo 1"), ("rod 2 (links + rod)", "servo 2"), ("rod 1 (links + rod)", "servo 2"), ("rod 2 (links + rod)", "servo 1"),
             ("rod 1 (links + rod)", "hand adapter"), ("rod 2 (links + rod)", "hand adapter"),
             ("rod 1 (links + rod)", "gimbal ring"), ("rod 2 (links + rod)", "gimbal ring"),
             ("rod 1 (links + rod)", "rod 2 (links + rod)"), ("rod 1 (links + rod)", "horn arm 2"), ("rod 2 (links + rod)", "horn arm 1"),
             ("horn arm 1", "forearm w/o servos"), ("horn arm 2", "forearm w/o servos"), ("horn arm 1", "horn arm 2"),
             ("horn arm 1", "servo 2"), ("horn arm 2", "servo 1")]
    coarse, fine = 1e-3, 0.3e-3
    touch = 0.75 * fine
    G_ = {sp: {k: Group(type("H", (), {"m": m})(), v, sp) for k, v in groups.items()} for sp in (coarse, fine)}
    hand_states = {}
    for name in ("keyframe", "OpenHand", "CloseHand", "SpreadHand"):
        w.reset()
        if name != "keyframe":
            w.move(w.ctrl_for(DEMO_POSES_5F[name]), 0.6, 0.6)
        hand_states[name] = d.qpos.copy()
    wgrid = rom_grid(15, 5) if quick else rom_grid(10, 11)
    # outside the operating ROM: the hard stops alone, and the corners of the stop rectangle
    wgrid_lim = [(-65, 0), (65, 0), (0, -30), (0, 30), (-65, -30), (-65, 30), (65, -30), (65, 30)]
    res = {pr: (np.inf, None) for pr in pairs}
    n_touch = {pr: 0 for pr in pairs}
    t0 = time.time()

    def measure(states, grid, tag):
        for hname, q in states.items():
            for p, s in grid:
                d.qpos[:] = q
                set_wrist_pose(w, np.radians(p), np.radians(s))
                for a, b in pairs:
                    if a.startswith("hand (+") or b.startswith("hand (+") or hname == "keyframe":
                        ga, gb = G_[coarse][a], G_[coarse][b]
                        dist, who = group_distance(ga, ga.poses(d), gb, gb.poses(d), 0.005)
                        if touch <= dist < 0.003:
                            ga, gb = G_[fine][a], G_[fine][b]
                            dist, who = group_distance(ga, ga.poses(d), gb, gb.poses(d), dist + coarse)
                        if dist < res[(a, b)][0]:
                            res[(a, b)] = (dist, (tag, hname, p, s, who))
                        n_touch[(a, b)] += int(dist < touch)

    measure(hand_states, wgrid, "ROM")
    lim = {}
    for a, b in pairs:
        lim[(a, b)] = res[(a, b)]
    res_rom = dict(res)
    res.update({pr: (np.inf, None) for pr in pairs})
    n_touch_rom = dict(n_touch)
    measure({"keyframe": hand_states["keyframe"], "CloseHand": hand_states["CloseHand"]}, wgrid_lim, "stops")
    res_stop = dict(res)
    n_touch = n_touch_rom
    log(f"  {len(wgrid)} wrist poses on the operating ROM x hand poses {list(hand_states)}, plus 8 poses at the hard stops "
        f"(+-65 / +-30 deg and their combined corners, outside the operating ROM) [{time.time() - t0:.0f} s]")
    rows = []
    for a, b in pairs:
        dist, info = res_rom[(a, b)]
        ds, infs = res_stop[(a, b)]
        txt = "> 5 mm" if not np.isfinite(dist) else f"{dist * 1e3:5.2f} mm at {info[1]} wrist ({info[2]:+.0f}, {info[3]:+.1f})"
        txts = "> 5 mm" if not np.isfinite(ds) else f"{ds * 1e3:5.2f} mm at ({infs[2]:+.0f}, {infs[3]:+.0f})"
        log(f"  {a:<26} vs {b:<38} ROM: min {txt}; touching poses {n_touch[(a, b)]} | stops: {txts}")
        rows.append((a, b, dist, ds))
    OUT["clearance_rom_min_mm"] = {f"{a} / {b}": (round(r * 1e3, 2) if np.isfinite(r) else None) for a, b, r, _ in rows}
    OUT["clearance_stops_min_mm"] = {f"{a} / {b}": (round(r * 1e3, 2) if np.isfinite(r) else None) for a, b, _, r in rows}

    # thumb over its own range, on a coarser wrist grid
    t0 = time.time()
    thumb_q = []
    angles = np.radians(np.linspace(-90, 90, 4 if quick else 7))
    for a1 in angles:
        for a2 in angles:
            w.reset()
            c = w.ctrl_for(DEMO_POSES_5F["OpenHand"])
            c[w.finger_ctrl["thumb"]] = (a1, a2)
            w.move(c, 0.4, 0.3)
            thumb_q.append(d.qpos.copy())
    wposes = [(p, f * np.degrees(wd.rom_dev_limit(np.radians(p)))) for p in (-60, -40, -20, 0, 20, 40, 60) for f in (-1, 0, 1)]
    best, touching = (np.inf, None), []
    for k, q in enumerate(thumb_q):
        a1, a2 = np.degrees(angles[k // len(angles)]), np.degrees(angles[k % len(angles)])
        for p, s in wposes:
            d.qpos[:] = q
            set_wrist_pose(w, np.radians(p), np.radians(s))
            for b in ("forearm (shells, fork, frame, servos)", "rod 1 (links + rod)", "rod 2 (links + rod)", "gimbal ring"):
                ga, gb = G_[coarse]["hand (+ shells, pads)"], G_[coarse][b]
                dist, who = group_distance(ga, ga.poses(d), gb, gb.poses(d), 0.005)
                if dist < best[0]:
                    best = (dist, ((float(a1), float(a2)), (p, round(float(s), 1)), b))
                if dist < 0.0015:
                    touching.append(((float(a1), float(a2)), (p, round(float(s), 1)), round(dist * 1e3, 2)))
    log(f"  thumb over a {len(angles)}x{len(angles)} grid of its motors (other fingers OpenHand) x {len(wposes)} wrist poses on the ROM "
        f"boundary and axes: hand vs forearm / rods / ring min " + ("> 5 mm" if not np.isfinite(best[0]) else f"{best[0] * 1e3:.2f} mm at {best[1]}")
        + f" [{time.time() - t0:.0f} s]")
    if touching:
        thumbs = sorted({t[0] for t in touching})
        log(f"    thumb poses (m1, m2) within 1.5 mm of the forearm somewhere on the ROM: {thumbs}")
        log("    (abduction (m1+m2)/2, flexion (m1-m2)/2 of those: " + ", ".join(f"({(a + b) / 2:+.0f}, {(a - b) / 2:+.0f})" for a, b in thumbs) + ")")
        log(f"    at wrist poses: {sorted({t[1] for t in touching})}")
    OUT["thumb_touching"] = touching
    OUT["thumb_min_mm"] = None if not np.isfinite(best[0]) else round(best[0] * 1e3, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-clearance", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    log(f"model: {MJCF.relative_to(MJCF.parents[3])}/scene.xml (mujoco {mujoco.__version__})")
    w = WristModel()
    check_structure(w)
    check_stability(w)
    check_rom(WristModel())
    check_torque(WristModel(), WristModel(payload_g=200.0), args.quick)
    if not args.skip_clearance:
        check_clearance(args.quick)
    log(f"total {time.time() - t0:.0f} s")
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "reports" / "verify_AH_Right_5F_Wrist.txt").write_text("\n".join(LINES) + "\n")
    (HERE / "reports" / "verify_AH_Right_5F_Wrist.json").write_text(json.dumps(OUT, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
