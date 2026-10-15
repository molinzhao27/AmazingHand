#!/usr/bin/env python3
"""Verify the 5-digit AmazingHand MJCF (AH_Right_5F).

Checks
  1. the model compiles, names are unique, counts are as expected, the keyframe matches the model;
  2. the copy is exact: at equal commands the pinky moves like the ring finger in its own base frame;
  3. stability: every actuator is swept through its full ctrlrange (one by one, all together,
     demo poses, abrupt random steps); no NaN, no MuJoCo warnings, bounded velocities;
  4. every equality constraint's site pair stays together (max error, transient and settled);
  5. interference: each finger is settled on an N x N grid over its two motors and the meshes of
     two digits are compared for every pair of grid poses (nearest-neighbour distance between
     surface samples: every --coarse mm, refined to --fine mm where closer than 3 mm). Pairs: pinky/ring (new), middle/ring (stock
     baseline: same relative placement), pinky/thumb, and every digit against the palm plate.

Usage: python verify_5f_model.py [--grid 9] [--coarse 1.0] [--fine 0.3] [--model AH_Right_5F]
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_5f_model import F2_PARTS, F3_PARTS, F5_PARTS, LAYOUT, parse, part_geoms  # noqa: E402

SIM_DIR = HERE.parent / "AHSimulation"

CHAINS = {
    "index": ("finger1", ["custom_servo_horn", "rotule_ball_2"]),
    "middle": ("finger2", ["custom_servo_horn_2", "rotule_ball_4"]),
    "ring": ("finger3", ["custom_servo_horn_3", "rotule_ball_6"]),
    "thumb": ("finger4", ["custom_servo_horn_4", "rotule_ball_8"]),
    "pinky": ("finger5", ["custom_servo_horn_5", "rotule_ball_10"]),
}
FIXED_PARTS = {
    "index": ["scs0009", "scs0009_4", "finger_frame_1", "finger_frame_2_4", "bushing_0608_04_2", "bushing_0608_04_6"],
    "middle": F2_PARTS,
    "ring": F3_PARTS,
    "thumb": ["scs0009_2", "scs0009_8", "finger_frame_2_3", "finger_frame_1_4", "bushing_0608_04_5", "bushing_0608_04_8"],
    "pinky": F5_PARTS,
}
PALM_PARTS = ["r_hand_plate", "r_wrist_interface"]
ROOT_BODY = 1  # r_wrist_interface: palm plate, servos, finger frames


# PythonExample/AmazingHand_Demo.py poses (right hand, degrees, (motor1, motor2)) with a pinky added.
# Abduction ~ (motor1 + motor2) / 2 (> 0 towards the index side), flexion ~ (motor1 - motor2) / 2.
# The thumb commands are for the slim thumb mount (layout.THUMB_MOUNT, FIVE_FINGER.md "Thumb mount"), all inside its
# envelope (layout.thumb_command_ok, which keeps it out of the closed palm shell): open / spread (extended along the
# radial side, out of the palm plane), along the palm's edge (ClenchHand), resting on the side of the curled index
# (CloseHand, Fuck) or on the curled middle (Index_Pointing), tip on the index fingertip (Perfect: OK sign; Pinched:
# pinch, index at f = 35 deg), swung across the front of the palm towards the curled ring and pinky (Victory: the palm
# shell leaves no room to fold it into the palm). Perfect is set for the right hand (0.31 mm pad gap); the stock left export is not an exact mirror,
# so on AH_Left_5F the same command stops 3.2 mm short (the thumb would run into the left index with less). The stock mount's thumb commands were OpenHand (-35, 35), CloseHand (90, -90), SpreadHand
# (-90, -4), ClenchHand (-4, 90), Index_Pointing and Victory (90, -90), Perfect (65, 12), Pinched and Fuck (0, -75)
# (Pinched with the index curled).
DEMO_POSES_5F = {
    "OpenHand": dict(index=(-35, 35), middle=(-35, 35), ring=(-35, 35), pinky=(-35, 35), thumb=(-52.5, 45)),
    "CloseHand": dict(index=(90, -90), middle=(90, -90), ring=(90, -90), pinky=(90, -90), thumb=(-16, 66.5)),
    "SpreadHand": dict(index=(4, 90), middle=(-32, 32), ring=(-68, 18), pinky=(-90, -4), thumb=(-36, 71)),
    # ClenchHand converges the fingers onto each other. The stock commands (index (-60, 0), middle (-35, 35),
    # ring/pinky (0, 70)) close the stock fan; on the parallel layout (cad/five_finger/layout.py) the fingers start
    # parallel and 3.3 mm closer, so about half the convergence brings the fingertips together (1.5-2.3 mm apart).
    "ClenchHand": dict(index=(-35, 25), middle=(-30, 40), ring=(-20, 50), pinky=(-27, 43), thumb=(-50, 37)),
    "Index_Pointing": dict(index=(-40, 40), middle=(90, -90), ring=(90, -90), pinky=(90, -90), thumb=(-7.5, 30)),
    "Perfect": dict(index=(50, -50), middle=(0, 0), ring=(-20, 20), pinky=(-20, 20), thumb=(-13, 69.5)),
    "Victory": dict(index=(-15, 65), middle=(-65, 15), ring=(90, -90), pinky=(90, -90), thumb=(64.5, 57.5)),
    "Pinched": dict(index=(35, -35), middle=(90, -90), ring=(90, -90), pinky=(90, -90), thumb=(-3, 57.5)),
    "Fuck": dict(index=(90, -90), middle=(-35, 35), ring=(90, -90), pinky=(90, -90), thumb=(-16, 66.5)),
}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ model bookkeeping
class Hand:
    def __init__(self, mjcf_dir, extra_parts=None):
        """extra_parts: {name: STL path in mm, in the CAD part-studio frame} added to the palm,
        e.g. the shells, which the onshape export does not contain."""
        self.dir = mjcf_dir
        if extra_parts:
            spec = mujoco.MjSpec.from_file(str(mjcf_dir / "scene.xml"))
            root_name = parse(mjcf_dir / "robot.xml").getroot().find("worldbody/body").get("name")
            palm = spec.body(root_name)
            plate = [g for g in palm.geoms if g.meshname == root_name[0] + "_hand_plate"][0]
            for name, path in extra_parts.items():
                spec.add_mesh(name=name, file=str(Path(path).resolve()), scale=[1e-3] * 3)
                palm.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=plate.pos,
                              quat=plate.quat, contype=0, conaffinity=0, group=2, rgba=[0.9, 0.9, 0.9, 0.5])
            self.m = spec.compile()
        else:
            self.m = mujoco.MjModel.from_xml_path(str(mjcf_dir / "scene.xml"))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.digits = [k for k, (f, _) in CHAINS.items()
                       if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}_motor1") >= 0]

        # root-body geoms in XML order -> part names from the "Part ..." comments
        root_xml = parse(mjcf_dir / "robot.xml").getroot().find("worldbody/body")
        self.side = "left" if root_xml.get("name").startswith("l_") else "right"
        names = [n for n, _ in sorted(part_geoms(root_xml).items(), key=lambda kv: list(root_xml).index(kv[1][1]))]
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_xml.get("name"))
        assert root == ROOT_BODY and len(names) + len(extra_parts or ()) == m.body_geomnum[root]
        self.part_geom = {n: m.body_geomadr[root] + i for i, n in enumerate(names)}
        for name in extra_parts or ():
            self.part_geom[name] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if self.side == "left":  # left parts are numbered differently: use the mirrored right part's name
            five = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger5_motor1") >= 0
            ref = Hand(SIM_DIR / ("AH_Right_5F" if five else "AH_Right") / "mjcf")
            mine, theirs = self._centroids(), ref._centroids()
            mapped = {name: self.part_geom[name] for name in extra_parts or ()}  # keep their own names
            for name, (mesh, c) in mine.items():
                if name in (extra_parts or ()):
                    continue
                dist, rname = min((np.linalg.norm(c * [1, -1, 1] - rc), rn) for rn, (rmesh, rc) in theirs.items()
                                  if rmesh.removeprefix("r_") == mesh.removeprefix("l_"))
                assert dist < 1e-3, f"no mirrored right part for {name} ({dist * 1e3:.2f} mm)"
                mapped[rname] = self.part_geom[name]
            assert len(mapped) == len(mine), (len(mapped), len(mine))
            self.part_geom = mapped

        self.geoms, self.qidx, self.ctrl, self.eqs = {}, {}, {}, {}
        for dig in self.digits:
            f, tops = CHAINS[dig]
            bodies = set()
            for top in tops:
                b0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, top)
                bodies |= {b for b in range(m.nbody) if self._under(b, b0)}
            moving = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]
            fixed = [self.part_geom[p] for p in FIXED_PARTS[dig]]
            self.geoms[dig] = (moving, fixed)
            jnts = [j for j in range(m.njnt) if m.jnt_bodyid[j] in bodies]
            self.qidx[dig] = np.concatenate([np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + _qlen(m, j)) for j in jnts])
            self.ctrl[dig] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}_motor{k}") for k in (1, 2)]
            site_ids = {s for s in range(m.nsite) if m.site_bodyid[s] in bodies}
            self.eqs[dig] = [e for e in range(m.neq) if m.eq_obj1id[e] in site_ids or m.eq_obj2id[e] in site_ids]
        self.palm = [self.part_geom[p] for p in PALM_PARTS]
        assert all(m.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT and m.eq_objtype[e] == mujoco.mjtObj.mjOBJ_SITE
                   for e in range(m.neq))

    def _centroids(self):
        """Mesh name and world centroid of each root part's vertices (root parts do not depend on qpos)."""
        mujoco.mj_kinematics(self.m, self.d)
        out = {}
        for name, g in self.part_geom.items():
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            out[name] = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_MESH, mid),
                         self.d.geom_xpos[g] + self.d.geom_xmat[g].reshape(3, 3) @ v.mean(0))
        return out

    def _under(self, b, top):
        while b > 0:
            if b == top:
                return True
            b = self.m.body_parentid[b]
        return False

    def eq_err(self, eqs=None):
        eqs = range(self.m.neq) if eqs is None else eqs
        x = self.d.site_xpos
        return np.array([np.linalg.norm(x[self.m.eq_obj1id[e]] - x[self.m.eq_obj2id[e]]) for e in eqs])

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)


def _qlen(m, j):
    return {mujoco.mjtJoint.mjJNT_BALL: 4, mujoco.mjtJoint.mjJNT_FREE: 7}.get(m.jnt_type[j], 1)


# ------------------------------------------------------------------ 1. structure
def check_structure(h):
    m = h.m
    log("== 1. structure")
    log(f"  nbody={m.nbody} njnt={m.njnt} nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq} nsite={m.nsite} "
        f"ngeom={m.ngeom} nkey={m.nkey} nmocap={m.nmocap}")
    for obj, n in [(mujoco.mjtObj.mjOBJ_BODY, m.nbody), (mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
                   (mujoco.mjtObj.mjOBJ_SITE, m.nsite), (mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)]:
        names = [mujoco.mj_id2name(m, obj, i) for i in range(n)]
        names = [x for x in names if x]
        assert len(names) == len(set(names)), f"duplicate names in {obj}"
    log("  names unique: bodies, joints, sites, actuators")
    log(f"  actuators: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]}")
    for i in range(m.nu):
        assert np.allclose(m.actuator_ctrlrange[i], [-np.pi / 2, np.pi / 2], atol=1e-4)
    log("  every ctrlrange = +-1.5708 rad")
    h.reset()
    errs = h.eq_err()
    log(f"  keyframe 'zero': qpos size {m.key_qpos.shape[1]} == nq {m.nq}; max equality error {errs.max() * 1e3:.4f} mm")
    for k in range(m.nmocap):
        b = np.flatnonzero(m.body_mocapid == k)[0]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        tip = "tip" + name[len("finger"):name.index("_")]
        dist = np.linalg.norm(h.d.mocap_pos[k] - h.d.site(tip).xpos)
        log(f"  mocap {name} -> site {tip}: {dist * 1e3:.3f} mm apart at the keyframe")


# ------------------------------------------------------------------ 2. copy check
def base_frame(h, digit):
    """Pose of the digit's finger frame (its first fixed 'finger_frame_1*' part)."""
    g = h.part_geom[[p for p in FIXED_PARTS[digit] if p.startswith("finger_frame_1")][0]]
    return h.d.geom_xpos[g].copy(), h.d.geom_xmat[g].reshape(3, 3).copy()


def check_copy(h):
    log("== 2. pinky is an exact copy of the ring finger")
    rng = np.random.default_rng(0)
    worst = 0.0
    for c in [(0, 0), (np.pi / 2, -np.pi / 2), (-np.pi / 2, np.pi / 2), (np.pi / 2, np.pi / 2), (-np.pi / 2, -np.pi / 2)] + \
            [tuple(rng.uniform(-np.pi / 2, np.pi / 2, 2)) for _ in range(5)]:
        h.reset()
        for dig in ("ring", "pinky"):
            h.d.ctrl[h.ctrl[dig]] = c
        for _ in range(750):
            mujoco.mj_step(h.m, h.d)
        rel = {}
        for dig, tip in (("ring", "tip3"), ("pinky", "tip5")):
            p, R = base_frame(h, dig)
            rel[dig] = R.T @ (h.d.site(tip).xpos - p)
        worst = max(worst, np.linalg.norm(rel["ring"] - rel["pinky"]))
    log(f"  tip in own finger frame, ring vs pinky, 10 commands: max difference {worst * 1e3:.4f} mm "
        "(only gravity acts differently on the two)")


# ------------------------------------------------------------------ 3./4. stability + constraints
def run_sequence(h, segments, label, stats):
    """segments: list of (target ctrl (nu,), ramp_s, hold_s); ramp=0 is an abrupt step."""
    m, d = h.m, h.d
    dt = m.opt.timestep
    vmax = 0.0
    err_dyn = np.zeros(m.neq)
    err_set = np.zeros(m.neq)
    track = 0.0
    n_steps = 0
    t0 = time.time()
    for target, ramp, hold in segments:
        start = d.ctrl.copy()
        nr, nh = int(round(ramp / dt)), int(round(hold / dt))
        for k in range(nr + nh):
            a = 1.0 if k >= nr else (k + 1) / nr
            d.ctrl[:] = start + (target - start) * a
            mujoco.mj_step(m, d)
            n_steps += 1
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                raise RuntimeError(f"{label}: non-finite state at step {n_steps}")
            vmax = max(vmax, np.abs(d.qvel).max())
            err_dyn = np.maximum(err_dyn, h.eq_err())
        err_set = np.maximum(err_set, h.eq_err())
        q_act = d.actuator_length  # = motor joint angle for these position actuators
        track = max(track, np.abs(q_act - d.ctrl).max())
    warnings = {mujoco.mjtWarning(i).name: d.warning[i].number for i in range(len(d.warning)) if d.warning[i].number}
    stats.append((label, n_steps * dt, vmax, err_dyn, err_set, track, warnings))
    log(f"  {label:<38} {n_steps * dt:6.1f} s sim  max|qvel| {vmax:7.2f} rad/s  "
        f"eq err max {err_dyn.max() * 1e3:6.3f} mm (settled {err_set.max() * 1e3:6.3f} mm)  "
        f"max |q_motor - ctrl| settled {np.degrees(track):5.2f} deg  warnings {warnings or 'none'}  "
        f"[{time.time() - t0:.1f} s wall]")


def pose_ctrl(h, pose):
    """Right-hand pose (deg) -> ctrl; on a left hand the abduction is mirrored: (m1, m2) -> (-m2, -m1)."""
    c = np.zeros(h.m.nu)
    for dig, (a1, a2) in pose.items():
        if dig in h.digits:
            c[h.ctrl[dig]] = np.radians([a1, a2] if h.side == "right" else [-a2, -a1])
    return c


def check_stability(h):
    log("== 3./4. stability and equality-constraint error while sweeping the actuators")
    m = h.m
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    zero = np.zeros(m.nu)
    stats = []

    h.reset()
    segs = []
    for a in range(m.nu):
        for target, ramp, hold in [(hi[a], 1.0, 0.5), (lo[a], 2.0, 0.5), (0.0, 1.0, 0.5)]:
            c = zero.copy()
            c[a] = target
            segs.append((c, ramp, hold))
    run_sequence(h, segs, "each actuator alone 0->max->min->0", stats)

    h.reset()
    segs = []
    for c1, c2 in [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1), (0, 0)]:  # corners of every (motor1, motor2) box
        c = np.zeros(m.nu)
        for dig in h.digits:
            c[h.ctrl[dig]] = [c1 * hi[h.ctrl[dig][0]], c2 * hi[h.ctrl[dig][1]]]
        segs.append((c, 1.5, 0.5))
    run_sequence(h, segs, "all digits around their range corners", stats)

    h.reset()
    segs = [(pose_ctrl(h, pose), 0.3, 0.7) for _ in range(2) for pose in DEMO_POSES_5F.values()]
    run_sequence(h, segs, "demo poses (DEMO_POSES_5F), x2", stats)

    h.reset()
    rng = np.random.default_rng(1)
    segs = [(rng.uniform(lo, hi), 0.0, 0.6) for _ in range(60)]
    run_sequence(h, segs, "60 abrupt random full-range steps", stats)

    err_dyn = np.max([s[3] for s in stats], axis=0)
    err_set = np.max([s[4] for s in stats], axis=0)
    log("  per digit (all sequences): max equality error transient / settled")
    for dig in h.digits:
        e = h.eqs[dig]
        log(f"    {dig:<7} {err_dyn[e].max() * 1e3:6.3f} mm / {err_set[e].max() * 1e3:6.3f} mm")
    worst = int(np.argmax(err_dyn))
    s1 = mujoco.mj_id2name(h.m, mujoco.mjtObj.mjOBJ_SITE, h.m.eq_obj1id[worst])
    s2 = mujoco.mj_id2name(h.m, mujoco.mjtObj.mjOBJ_SITE, h.m.eq_obj2id[worst])
    log(f"  worst pair overall: '{s1}' / '{s2}' {err_dyn[worst] * 1e3:.3f} mm")
    return stats


# ------------------------------------------------------------------ 5. interference
def sample_mesh(m, mesh_id, h):
    """Points on the mesh surface (mesh frame) such that every surface point is within ~h of one.

    Edges are sampled every h. A point inside a triangle is at most its inradius r from an edge,
    so only triangles with r > h/2 also get an interior barycentric grid.
    """
    va, vn = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
    fa, fn = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
    v = m.mesh_vert[va:va + vn].astype(np.float64)
    faces = m.mesh_face[fa:fa + fn]
    tri = v[faces]
    pts = [v]

    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    a, b = v[edges[:, 0]], v[edges[:, 1]]
    n_seg = np.ceil(np.linalg.norm(b - a, axis=1) / h).astype(int)
    for k in np.unique(n_seg[n_seg > 1]):
        s = np.arange(1, k)[None, :, None] / k
        sel = n_seg == k
        pts.append((a[sel][:, None] + s * (b[sel] - a[sel])[:, None]).reshape(-1, 3))

    lengths = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    inradius = 2 * area / np.maximum(lengths.sum(1), 1e-12)
    k_all = np.where(inradius > h / 2, np.ceil(lengths.max(1) / h), 0).astype(int)
    for k in np.unique(k_all[k_all > 2]):
        ij = np.array([(i, j) for i in range(1, k) for j in range(1, k - i)], float) / k  # interior only
        w = np.column_stack([1 - ij.sum(1), ij])  # barycentric
        pts.append(np.einsum("bk,tkd->tbd", w, tri[k_all == k]).reshape(-1, 3))
    return np.concatenate(pts)


class BodyCloud:
    """Surface samples of some geoms of one body, in body coordinates, split into voxel chunks."""

    def __init__(self, pts, owner, voxel=0.008):
        key = np.floor(pts / voxel).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        order = np.argsort(inv.ravel(), kind="stable")
        self.pts, self.owner, inv = pts[order], owner[order], inv.ravel()[order]
        starts = np.flatnonzero(np.r_[True, inv[1:] != inv[:-1]])
        self.chunks = np.column_stack([starts, np.r_[starts[1:], len(inv)]])
        self.cc = np.array([self.pts[a:b].mean(0) for a, b in self.chunks])
        self.cr = np.array([np.linalg.norm(self.pts[a:b] - c, axis=1).max() for (a, b), c in zip(self.chunks, self.cc)])
        self.lo, self.hi = self.pts.min(0), self.pts.max(0)
        self.center, self.radius = (self.lo + self.hi) / 2, np.linalg.norm(self.hi - self.lo) / 2
        self.tree = cKDTree(self.pts)


class Group:
    """Surface samples of a set of geoms (one BodyCloud per body); poses() reads the bodies from MjData."""

    _cache = {}

    def __init__(self, h, geoms, spacing):
        m = h.m
        by_body = {}
        for g in geoms:
            by_body.setdefault(int(m.geom_bodyid[g]), []).append(g)
        self.bodies, self.clouds = list(by_body), []
        for gs in by_body.values():
            pts, own = [], []
            for g in gs:
                key = (id(m), int(m.geom_dataid[g]), spacing)
                if key not in Group._cache:
                    Group._cache[key] = sample_mesh(m, m.geom_dataid[g], spacing)
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                pts.append(m.geom_pos[g] + Group._cache[key] @ R.reshape(3, 3).T)
                own.append(np.full(len(Group._cache[key]), g))
            self.clouds.append(BodyCloud(np.concatenate(pts), np.concatenate(own)))

    def poses(self, d):
        return [(d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in self.bodies]


def group_distance(ga, pa, gb, pb, cutoff, skip_root_pair=False):
    """Min distance between two posed Groups (inf if > cutoff) and the two geoms involved.
    skip_root_pair ignores palm-fixed parts against palm-fixed parts (their distance never changes)."""
    best, who = cutoff, None
    cand = []
    for a, ba, (ta, Ra) in zip(ga.clouds, ga.bodies, pa):
        ca_w = ta + Ra @ a.center
        for b, bb, (tb, Rb) in zip(gb.clouds, gb.bodies, pb):
            if skip_root_pair and ba == ROOT_BODY and bb == ROOT_BODY:
                continue
            lower = np.linalg.norm(tb + Rb @ b.center - ca_w) - a.radius - b.radius
            if lower < best:
                cand.append((lower, a, ta, Ra, b, tb, Rb))
    for lower, a, ta, Ra, b, tb, Rb in sorted(cand, key=lambda c: c[0]):  # nearest first: best shrinks early
        if lower > best:
            break
        Rab, tab = Ra.T @ Rb, Ra.T @ (tb - ta)  # b coordinates -> a coordinates
        cc = b.cc @ Rab.T + tab
        gap = np.linalg.norm(np.maximum(0, np.maximum(a.lo - cc, cc - a.hi)), axis=1) - b.cr
        sel = np.flatnonzero(gap < best)
        if len(sel) == 0:
            continue
        idx = np.concatenate([np.arange(*b.chunks[k]) for k in sel])
        dist, j = a.tree.query(b.pts[idx] @ Rab.T + tab, distance_upper_bound=best)
        k = int(np.argmin(dist))
        if dist[k] < best:
            best, who = dist[k], (a.owner[j[k]], b.owner[idx[k]])
    return (best, who) if who else (np.inf, None)


def settle_grid(h, angles):
    """Settle every digit at each (motor1, motor2) grid point; returns qpos per point and eq error."""
    m, d = h.m, h.d
    h.reset()
    grid = [(a1, a2) for i, a1 in enumerate(angles) for a2 in (angles if i % 2 == 0 else angles[::-1])]  # serpentine
    out = {}
    worst = {dig: 0.0 for dig in h.digits}
    for a1, a2 in grid:
        target = np.zeros(m.nu)
        for dig in h.digits:
            target[h.ctrl[dig]] = (a1, a2)
        start = d.ctrl.copy()
        for k in range(200):  # 0.2 s ramp + 0.2 s hold
            d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / 100)
            mujoco.mj_step(m, d)
        for dig in h.digits:
            worst[dig] = max(worst[dig], h.eq_err(h.eqs[dig]).max())
        out[(a1, a2)] = d.qpos.copy()
    return dict(sorted(out.items())), worst


def posed(h, group, poses):
    """Body poses of a Group for every grid pose."""
    out = []
    for q in poses.values():
        h.d.qpos[:] = q
        mujoco.mj_kinematics(h.m, h.d)
        out.append(group.poses(h.d))
    return out


def digit_geoms(h, dig):
    return h.geoms[dig][0] + h.geoms[dig][1]


def clearance_table(h, poses, geoms_a, geoms_b, coarse, fine, touch, cutoff=0.005, refine_below=0.003):
    """Min surface distance for every (pose of A, pose of B); pose pairs closer than refine_below
    on the coarse samples are recomputed on the fine samples."""
    out = {}
    for spacing in (coarse, fine):
        ga, gb = Group(h, geoms_a, spacing), Group(h, geoms_b, spacing)
        pa, pb = posed(h, ga, poses), posed(h, gb, poses)
        if spacing == coarse:
            todo = {(i, j): cutoff for i in range(len(pa)) for j in range(len(pb))}
        else:  # below `touch` the pair certainly touches (sampled distance >= true distance);
            # the fine distance cannot exceed the coarse one by more than the coarse sampling gap
            todo = {ij: dist + coarse for ij, (dist, _) in out.items() if touch <= dist < refine_below}
        for (i, j), bound in todo.items():
            out[(i, j)] = group_distance(ga, pa[i], gb, pb[j], bound, skip_root_pair=True)
    table = np.full((len(poses), len(poses)), np.inf)
    for (i, j), (dist, _) in out.items():
        table[i, j] = dist
    return table, {k: w for k, (_, w) in out.items()}


def geom_label(h, g):
    m = h.m
    inv = {v: k for k, v in h.part_geom.items()}
    if g in inv:
        return inv[g]
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
    return f"{mesh} ({body})"


def fmt_pose(k):
    return f"({np.degrees(k[0]):+3.0f},{np.degrees(k[1]):+3.0f})"


def report_pair(h, title, keys, table, who, touch):
    i, j = np.unravel_index(np.argmin(table), table.shape)
    if not np.isfinite(table[i, j]):
        log(f"  {title}: > 5 mm apart in all {table.size} pose pairs")
        return
    contact = table < touch
    log(f"  {title}: min clearance {table[i, j] * 1e3:.2f} mm at A{fmt_pose(keys[i])} B{fmt_pose(keys[j])} "
        f"[{geom_label(h, who[(i, j)][0])} / {geom_label(h, who[(i, j)][1])}]; "
        f"pose pairs touching or overlapping (< {touch * 1e3:.2f} mm): {int(contact.sum())}/{table.size}")
    if contact.any():
        parts = {}
        for i, j in zip(*np.nonzero(contact)):
            key = tuple(sorted((geom_label(h, who[(i, j)][0]).split(" (")[0], geom_label(h, who[(i, j)][1]).split(" (")[0])))
            parts[key] = parts.get(key, 0) + 1
        log("    touching parts: " + ", ".join(f"{a}/{b} x{n}" for (a, b), n in sorted(parts.items(), key=lambda x: -x[1])))


def check_interference(h, n_grid, coarse, fine):
    log(f"== 5. interference: {n_grid}x{n_grid} grid per digit over (motor1, motor2) in [-90, 90] deg "
        f"({180 / (n_grid - 1):.1f} deg steps); surface samples every {coarse} mm, {fine} mm below 3 mm clearance")
    t0 = time.time()
    poses, worst = settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n_grid))
    keys = list(poses)
    log(f"  settled {len(poses)} grid poses ({time.time() - t0:.1f} s); max equality error at the grid poses: "
        + ", ".join(f"{k} {v * 1e3:.3f} mm" for k, v in worst.items()))
    coarse, fine = coarse * 1e-3, fine * 1e-3
    touch = 0.75 * fine  # sampled distance below this: surfaces touch or cross
    tables = {}
    for a, b in [("ring", "pinky"), ("middle", "ring"), ("thumb", "pinky"), ("thumb", "ring")]:
        if a in h.digits and b in h.digits:
            t0 = time.time()
            fa, fb = Group(h, h.geoms[a][1], fine), Group(h, h.geoms[b][1], fine)
            mujoco.mj_kinematics(h.m, h.d)
            dist, who = group_distance(fa, fa.poses(h.d), fb, fb.poses(h.d), 0.005)
            log(f"  {a} vs {b}, palm-fixed parts only (servos, finger frames): "
                + (f"{dist * 1e3:.2f} mm [{geom_label(h, who[0])} / {geom_label(h, who[1])}]" if who else "> 5 mm"))
            table, who = clearance_table(h, poses, digit_geoms(h, a), digit_geoms(h, b), coarse, fine, touch)
            tables[(a, b)] = table
            report_pair(h, f"{a} (A) vs {b} (B)", keys, table, who, touch)
            if a == "thumb" and LAYOUT.THUMB_ENVELOPE.get(LAYOUT.THUMB_MOUNT) is not None:
                inside = np.array([LAYOUT.thumb_command_ok(*np.degrees(k), side=h.side) for k in keys])
                t_in = table[inside]
                log(f"    thumb inside its command envelope (layout.thumb_command_ok, {int(inside.sum())} of {len(keys)}): "
                    + (f"min clearance {np.min(t_in) * 1e3:.2f} mm" if np.isfinite(np.min(t_in)) else "> 5 mm apart")
                    + f", pose pairs touching {int((t_in < touch).sum())}/{t_in.size}")
            log(f"    [{time.time() - t0:.0f} s]")
    if ("ring", "pinky") in tables:
        # closing = how far the two fingers abduct towards each other (motor-angle units, deg)
        ab = np.array([np.degrees(k[0] + k[1]) / 2 for k in keys])
        fl = np.array([np.degrees(k[0] - k[1]) / 2 for k in keys])
        closing = ab[None, :] - ab[:, None]  # pinky abduction towards ring minus ring abduction towards... ring(A)=rows
        t = tables[("ring", "pinky")]
        hit = t < touch
        log(f"  ring/pinky contact vs relative abduction c = a_pinky - a_ring (a = (m1+m2)/2, > 0 = towards index): "
            f"touching pairs have c >= {closing[hit].min():.1f} deg" if hit.any() else "  ring/pinky never touch")
        for c0 in sorted(set(np.round(closing.ravel(), 1))):
            sel = np.isclose(closing, c0)
            if hit[sel].any():
                log(f"    c = {c0:+6.1f} deg: {int(hit[sel].sum())}/{int(sel.sum())} pose pairs touch "
                    f"(min clearance of the others {np.min(t[sel & ~hit], initial=np.inf) * 1e3:.2f} mm)")
    if ("ring", "pinky") in tables and ("middle", "ring") in tables:
        new, old = tables[("ring", "pinky")], tables[("middle", "ring")]
        both = np.isfinite(new) & np.isfinite(old)
        log(f"  pinky/ring vs stock middle/ring: same set of pose pairs within 5 mm: "
            f"{bool(np.array_equal(np.isfinite(new), np.isfinite(old)))}; max |clearance difference| "
            f"{np.max(np.abs(new[both] - old[both])) * 1e3:.3f} mm; same touching set: "
            f"{bool(np.array_equal(new < touch, old < touch))}")
    # moving parts only: servos and finger frames are bolted to the plate and touch it by design
    gp = {sp: Group(h, h.palm, sp) for sp in (coarse, fine)}
    mujoco.mj_kinematics(h.m, h.d)
    pp = gp[fine].poses(h.d)
    for dig in h.digits:
        res = []
        g = {sp: Group(h, h.geoms[dig][0], sp) for sp in (coarse, fine)}
        for p in posed(h, g[coarse], poses):
            dist, who = group_distance(g[coarse], p, gp[coarse], pp, 0.005)
            if touch <= dist < 0.003:
                dist, who = group_distance(g[fine], p, gp[fine], pp, dist + coarse)
            res.append((dist, who))
        i = int(np.argmin([r[0] for r in res]))
        n_touch = sum(r[0] < touch for r in res)
        who = res[i][1]
        log(f"  {dig:<6} moving parts vs palm (plate + wrist interface): "
            + (f"min clearance {res[i][0] * 1e3:5.2f} mm at {fmt_pose(keys[i])} [{geom_label(h, who[0])} / {geom_label(h, who[1])}]"
               if who else "> 5 mm")
            + f"; grid poses touching: {n_touch}/{len(res)}")
        if dig == "thumb" and LAYOUT.THUMB_ENVELOPE.get(LAYOUT.THUMB_MOUNT) is not None:
            inside = [LAYOUT.thumb_command_ok(*np.degrees(k), side=h.side) for k in keys]
            res_in = [r for r, ok in zip(res, inside) if ok]
            j = int(np.argmin([r[0] for r in res_in]))
            log(f"         inside the thumb's command envelope (layout.thumb_command_ok, {len(res_in)} of the {len(res)} grid "
                f"poses): touching {sum(r[0] < touch for r in res_in)}, min clearance {res_in[j][0] * 1e3:.2f} mm")
    return tables, poses


def check_demo_poses(h, spacing):
    log(f"== 6. demo poses with a pinky (DEMO_POSES_5F): clearance between neighbouring digits "
        f"(samples every {spacing} mm)")
    sp = spacing * 1e-3
    pairs = [(a, b) for a, b in [("index", "middle"), ("middle", "ring"), ("ring", "pinky"), ("thumb", "index"),
                                 ("thumb", "middle"), ("thumb", "ring"), ("thumb", "pinky")]
             if a in h.digits and b in h.digits]
    groups = {dig: Group(h, digit_geoms(h, dig), sp) for dig in h.digits}
    for name, pose in DEMO_POSES_5F.items():
        h.reset()
        target = pose_ctrl(h, pose)
        start = h.d.ctrl.copy()
        for k in range(500):
            h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / 250)
            mujoco.mj_step(h.m, h.d)
        out = []
        for a, b in pairs:
            dist, who = group_distance(groups[a], groups[a].poses(h.d), groups[b], groups[b].poses(h.d), 0.005,
                                       skip_root_pair=True)
            if dist < 0.005:
                out.append(f"{a}/{b} {dist * 1e3:.2f} mm" + (" TOUCH" if dist < 0.75 * sp else ""))
        log(f"  {name:<15} " + (", ".join(out) if out else "all neighbouring digits > 5 mm apart"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="AH_Right_5F",
                    help="folder under AHSimulation/: AH_Right_5F, AH_Left_5F, or AH_Right / AH_Left for the stock baseline")
    ap.add_argument("--grid", type=int, default=9, help="grid points per motor for the interference check")
    ap.add_argument("--coarse", type=float, default=1.0, help="surface sampling spacing in mm (all pose pairs)")
    ap.add_argument("--fine", type=float, default=0.3, help="surface sampling spacing in mm (pose pairs closer than 3 mm)")
    ap.add_argument("--skip-interference", action="store_true")
    args = ap.parse_args()

    h = Hand(SIM_DIR / args.model / "mjcf")
    log(f"model: {h.dir.relative_to(HERE.parent.parent.parent)}  (mujoco {mujoco.__version__}, digits: {h.digits})")
    check_structure(h)
    if "pinky" in h.digits:
        check_copy(h)
    check_stability(h)
    if not args.skip_interference:
        check_interference(h, args.grid, args.coarse, args.fine)
    check_demo_poses(h, args.fine)


if __name__ == "__main__":
    main()
