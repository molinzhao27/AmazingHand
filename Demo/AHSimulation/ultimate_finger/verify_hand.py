#!/usr/bin/env python3
"""Verify the 5-digit hand AH_Right_5F_3S (ultimate fingers on index/middle/ring/pinky, slim styled thumb).

Runs the checks of five_finger/verify_5f_model.py (imported unchanged, so the numbers compare 1:1 with the stock
5-digit hand: 887/6561 touching pose pairs for ring/pinky and middle/ring, 973 for index/middle):
  1. structure; 2. every converted finger moves like the standalone model (model/finger_uf.xml);
  3./4. stability sweeps (each actuator, range corners, demo poses, 60 abrupt random steps) and loop closures;
  5. neighbour-finger clearance on a 9 x 9 grid per digit (all 6561 pose pairs), incl. index/middle; digits vs palm;
  6. the demo poses re-mapped to the new straight pose, plus the two extra poses PEACE and MIDDLE_FINGER: clearance of
     every digit pair (touching is fine, interpenetration is reported with its overlap volume and depth), the thumb's
     position relative to the curled fingers;
  7. thumb/index pinch: the thumb command that brings the two fingertip pads together;
  8. the 5-digit palm/back shells and pads (cad/five_finger/stl, fitted by make_5f_shells.py to the merged sweeps of
     AH_Right_5F and this hand) against this hand.

Writes reports/verify_hand.txt and data/hand_poses.json (the re-mapped pose table).
--only-plate runs only the plate-dependent check (moving parts of every digit vs the palm plate and the wrist
interface on the --grid grid) into reports/verify_hand_plate.txt; --only-shells only section 8 (palm-fixed parts and
every digit vs the shells in --shells DIR, 13 x 13 grid) into reports/verify_hand_shells.txt.
Usage: python verify_hand.py [--grid 9] [--skip-grid] [--only-poses] [--only-plate] [--only-shells] [--shells DIR]
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import manifold3d as mf
import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "five_finger"))
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "three_segment"))
import verify_5f_model as V  # noqa: E402
from finger_common import Finger  # noqa: E402

CAD = REPO / "cad" / "five_finger" / "ultimate_finger"
HAND = V.SIM_DIR / "AH_Right_5F_3S" / "mjcf"
REPORT_DIR, DATA_DIR = HERE / "reports", HERE / "data"
NEW = ("index", "middle", "ring", "pinky")
F_STRAIGHT_STOCK = -37.5
TIPS = dict(index="tip1", middle="tip2", ring="tip3", thumb="tip4", pinky="tip5")
NUM = dict(index=1, middle=2, ring=3, thumb=4, pinky=5)
OWN_SHELLS = REPO / "cad" / "five_finger" / "stl"   # fitted to AH_Right_5F and AH_Right_5F_3S (make_5f_shells.py)


def shell_files(folder):
    """{name: STL} of the right-hand shells and pads in folder (names as five_finger/check_shells.shells)."""
    return {"palm_shell": folder / "R-Palm_Shell_5F.stl", "top_shell": folder / "R-Top_Shell_5F.stl",
            **{f"pad_{q.stem.split('_')[2].lower()}": q for q in sorted(folder.glob("R-Palm_Pad_*_5F.stl"))}}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def f_straight():
    return json.loads((CAD / "design.json").read_text())["predicted"]["f_straight_deg"]


def remap(pose, f_s):
    """Stock demo command -> the same intent for the new fingers (as three_segment/verify_3seg_hand.remap): flexion
    rescaled from [stock straight, 90] to [new straight, 90], hyperextension kept as an offset but clamped at
    f >= -60, abduction kept but limited by the servo range |a| + |f| <= 90."""
    m1, m2 = pose
    a, f = (m1 + m2) / 2, (m1 - m2) / 2
    if f >= F_STRAIGHT_STOCK:
        f2 = f_s + (f - F_STRAIGHT_STOCK) * (90 - f_s) / (90 - F_STRAIGHT_STOCK)
    else:
        f2 = max(f_s + (f - F_STRAIGHT_STOCK), -60.0)
    a2 = float(np.clip(a, -(90 - abs(f2)), 90 - abs(f2)))
    return round(a2 + f2, 1), round(a2 - f2, 1)


def cmd(f, a=0.0):
    return (round(a + f, 1), round(a - f, 1))


def settle(h, pose, steps=800):
    h.reset()
    target = V.pose_ctrl(h, pose)
    start = h.d.ctrl.copy()
    for k in range(steps):
        h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / (steps // 2))
        mujoco.mj_step(h.m, h.d)


# ---------------------------------------------------------------- exact overlap between two digits
class Exact:
    def __init__(self, h):
        self.h, self.cache = h, {}

    def man(self, g):
        m = self.h.m
        mid = int(m.geom_dataid[g])
        if mid not in self.cache:
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] * 1e3
            f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            self.cache[mid] = mf.Manifold(mf.Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=f.astype(np.uint32)))
        R, p = self.h.d.geom_xmat[g].reshape(3, 3), self.h.d.geom_xpos[g] * 1e3
        return self.cache[mid].transform(np.c_[R, p])

    def overlap(self, ga, gb):
        """Overlap volume (mm^3) and an overlap depth estimate (mm, thickness of the intersection) of two geom sets."""
        vol, depth, who = 0.0, 0.0, None
        A = [(g, self.man(g)) for g in ga]
        B = [(g, self.man(g)) for g in gb]
        for g1, a in A:
            ba = a.bounding_box()
            for g2, b in B:
                bb = b.bounding_box()
                if any(ba[i] > bb[i + 3] or bb[i] > ba[i + 3] for i in range(3)):
                    continue
                inter = a ^ b
                v = inter.volume()
                if v > 1e-3:
                    vol += v
                    mm = inter.to_mesh()
                    tm = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts))
                    ext = np.sort(tm.bounding_box_oriented.primitive.extents)[0] if len(tm.vertices) > 3 else 0.0
                    if ext > depth:
                        depth, who = float(ext), (g1, g2)
        return vol, depth, who


def all_pair_report(h, title, groups, exact, fine=0.3e-3):
    V.log(f"  {title}")
    digs = list(h.digits)
    worst_overlap = 0.0
    for i, a in enumerate(digs):
        for b in digs[i + 1:]:
            ga, gb = groups[a], groups[b]
            dist, who = V.group_distance(ga, ga.poses(h.d), gb, gb.poses(h.d), 0.02, skip_root_pair=True)
            if dist >= 0.02:
                V.log(f"    {a:<6}/{b:<6} > 20 mm apart")
                continue
            line = f"    {a:<6}/{b:<6} min surface distance {dist * 1e3:5.2f} mm [{V.geom_label(h, who[0]).split(' (')[0]} / " \
                   f"{V.geom_label(h, who[1]).split(' (')[0]}]"
            if dist < 0.75 * fine:
                vol, depth, w2 = exact.overlap(h.geoms[a][0], h.geoms[b][0])
                worst_overlap = max(worst_overlap, depth)
                line += (f"  TOUCH; exact overlap {vol:.2f} mm3, depth ~{depth:.2f} mm"
                         + (f" [{V.geom_label(h, w2[0]).split(' (')[0]} / {V.geom_label(h, w2[1]).split(' (')[0]}]" if w2 else "")
                         if vol > 1e-3 else "  TOUCH (no overlap volume)")
            V.log(line)
    return worst_overlap


def mesh_geoms(h, dig, names):
    m = h.m
    return [g for g in h.geoms[dig][0] if m.geom_dataid[g] >= 0 and
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) in names]


# ---------------------------------------------------------------- thumb placement search
def search_thumb(h, base_pose, target, avoid_overlap=True, step=7.5, extra=None):
    """Grid over the thumb's two motor commands with the other digits at base_pose. target: (digit, mesh names) the
    thumb's distal pad should rest on. Returns candidate list sorted by pad-to-target distance among poses where the
    thumb is >= 0.2 mm from every other digit (sampled surfaces)."""
    sp = 0.5e-3
    pad = Group_cache(h, "thumb", {"thumb_distal_pad", "thumb_distal_shell"}, sp)
    tgt = Group_cache(h, target[0], set(target[1]), sp)
    thumb_all = V.Group(h, h.geoms["thumb"][0], sp)
    others = {d: V.Group(h, h.geoms[d][0], sp) for d in h.digits if d != "thumb"}
    settle(h, dict(base_pose, thumb=(0, 0)))
    q0 = h.d.qpos.copy()
    res = []
    vals = np.arange(-90, 90.01, step)
    for i, t1 in enumerate(vals):
        for t2 in (vals if i % 2 == 0 else vals[::-1]):
            h.d.ctrl[h.ctrl["thumb"]] = np.radians([t1, t2])
            for _ in range(160):
                mujoco.mj_step(h.m, h.d)
            dpad, _ = V.group_distance(pad, pad.poses(h.d), tgt, tgt.poses(h.d), 0.03)
            dmin = min(V.group_distance(thumb_all, thumb_all.poses(h.d), o, o.poses(h.d), 0.03)[0] for o in others.values())
            res.append((float(dpad), float(dmin), (float(t1), float(t2))))
    del q0
    ok = [r for r in res if r[1] >= 0.2e-3 or not avoid_overlap]
    ok.sort(key=lambda r: r[0])
    return ok, res


def Group_cache(h, dig, names, sp):
    return V.Group(h, mesh_geoms(h, dig, names), sp)


def check_vs_single(h):
    """Each ultimate finger of the hand, in its own finger frame, against the standalone model (same commands)."""
    sys.path.insert(0, str(HERE))
    from verify_finger import roles
    single = Finger(HERE / "model" / "scene_uf.xml", roles())
    f_s = f_straight()
    worst = 0.0
    for c in [(f_s, -f_s), (0, 0), (60, -60), (90, -90), (-20, 40), (30, 80)]:
        c = np.radians(c)
        h.reset()
        for dig in NEW:
            h.d.ctrl[h.ctrl[dig]] = c
        for _ in range(750):
            mujoco.mj_step(h.m, h.d)
        single.reset()
        single.move(*c, ramp=0.5, hold=1.0, check=False)
        m, d = single.m, single.d
        g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == 1 and m.geom_dataid[i] >= 0 and
                 mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[i]) == "finger_frame_1")
        ref = d.geom_xmat[g].reshape(3, 3).T @ (d.site_xpos[single.tip] - d.geom_xpos[g])
        for dig in NEW:
            if dig == "pinky" and mujoco.mj_name2id(h.m, mujoco.mjtObj.mjOBJ_MESH, "ufpk_p1") >= 0:
                continue
            p, R = V.base_frame(h, dig)
            worst = max(worst, np.linalg.norm(R.T @ (h.d.site(TIPS[dig]).xpos - p) - ref))
    V.log(f"== 2b. every ultimate finger of the hand vs the standalone finger model (tip in its own finger frame, 6 commands): "
          f"max difference {worst * 1e3:.4f} mm")


def finger_angles(h, N):
    """MCP, PIP, DIP (deg) of ultimate finger N from the body/geom frames (all drawn in the straight pose)."""
    m, d = h.m, h.d
    from finger_common import xangle
    R = lambda name: d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)].reshape(3, 3)  # noqa: E731
    p1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"p1_{N}")
    gb = m.body_parentid[m.body_parentid[p1]]   # p1 -> p2 -> Link; the gimbal is the Link's child
    gimbal = next(b for b in range(m.nbody) if m.body_parentid[b] == gb and any(
        m.geom_dataid[g] >= 0 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) == "gimbal"
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])))
    gg = next(g for g in range(m.body_geomadr[gimbal], m.body_geomadr[gimbal] + m.body_geomnum[gimbal])
              if m.geom_dataid[g] >= 0 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) == "gimbal")
    if not hasattr(h, "_gimbal_cad"):   # geom pose as written in the XML (the compiled mesh geom frame is its inertia frame)
        h._gimbal_cad = {}
    if N not in h._gimbal_cad:
        spec = mujoco.MjSpec.from_file(str(HAND / "scene.xml"))
        gs = [x for x in spec.body(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, gimbal)).geoms if x.meshname == "gimbal"][0]
        q = np.array(gs.quat, float)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, q / np.linalg.norm(q))
        h._gimbal_cad[N] = Rq.reshape(3, 3)
    Rg = d.xmat[gimbal].reshape(3, 3) @ h._gimbal_cad[N]
    return (np.degrees(xangle(Rg.T @ R(f"p1_{N}"))), np.degrees(xangle(R(f"p1_{N}").T @ R(f"p2_{N}"))),
            np.degrees(xangle(R(f"p2_{N}").T @ R(f"p3_{N}"))))


def pinky_check(h):
    """The pinky (possibly with shortened phalanges) vs the ring: flexion line in the hand, joint angles and loop errors."""
    V.log("== 2c. ring vs pinky along the flexion line (a = 0), settled in the hand: joint angles (deg), loop errors")
    fs = f_straight()
    rows = []
    for f in (-60, fs, -30, 0, 30, 60, 90):
        settle(h, dict(ring=cmd(f), pinky=cmd(f)))
        r, p = finger_angles(h, 3), finger_angles(h, 5)
        e = max(h.eq_err(h.eqs["pinky"]).max(), h.eq_err(h.eqs["ring"]).max())
        tr = np.linalg.norm(V.base_frame(h, "ring")[1].T @ (h.d.site("tip3").xpos - V.base_frame(h, "ring")[0]))
        tp = np.linalg.norm(V.base_frame(h, "pinky")[1].T @ (h.d.site("tip5").xpos - V.base_frame(h, "pinky")[0]))
        rows.append((f, r, p, e))
        V.log(f"    f {f:+6.1f}: ring MCP/PIP/DIP {r[0]:6.1f} {r[1]:6.1f} {r[2]:6.1f} | pinky {p[0]:6.1f} {p[1]:6.1f} {p[2]:6.1f} | "
              f"tip distance from its finger frame ring {tr * 1e3:.1f} / pinky {tp * 1e3:.1f} mm | max loop error {e * 1e3:.4f} mm")
    return rows


def pose_table(f_s, extra):
    poses = {name: {dig: (remap(c, f_s) if dig in NEW else c) for dig, c in pose.items()}
             for name, pose in V.DEMO_POSES_5F.items()}
    poses.update(extra)
    return poses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--only-poses", action="store_true")
    ap.add_argument("--step", type=float, default=7.5, help="thumb search grid step (deg)")
    ap.add_argument("--only-plate", action="store_true", help="only digits vs the palm plate and wrist interface")
    ap.add_argument("--only-shells", action="store_true", help="only digits and palm-fixed parts vs the shells")
    ap.add_argument("--keep-pinch", action="store_true", help="reuse the Pinched pose of data/hand_poses.json instead of "
                    "the pinch search (thumb_refine.py searches it more thoroughly)")
    ap.add_argument("--shells", type=Path, default=None, help="folder with R-Top_Shell_5F.stl, R-Palm_Shell_5F.stl, "
                    "R-Palm_Pad_*_5F.stl (default cad/five_finger/stl)")
    args = ap.parse_args()
    # section 8 runs on --shells DIR or the default cad/five_finger/stl when it has the shells
    shells_dir = args.shells or (OWN_SHELLS if (OWN_SHELLS / "R-Palm_Shell_5F.stl").exists() else None)
    f_s = f_straight()
    report = REPORT_DIR / ("verify_hand_poses.txt" if args.only_poses else "verify_hand_plate.txt" if args.only_plate
                           else "verify_hand_shells.txt" if args.only_shells else "verify_hand.txt")
    with open(report, "w") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        V.log("== 5-digit right hand AH_Right_5F_3S: ultimate fingers (index, middle, ring, pinky), slim styled thumb")
        V.log("   checks imported unchanged from Demo/AHSimulation/five_finger/verify_5f_model.py")
        h = V.Hand(HAND)
        if args.only_plate:
            plate_check(h, args.grid)
            return
        if args.only_shells:
            assert shells_dir is not None, "no shells fitted to this hand: pass --shells DIR"
            shells_check(h, shells_dir=shells_dir)
            return
        exact = Exact(h)
        if not args.only_poses:
            V.check_structure(h)
            scaled = mujoco.mj_name2id(h.m, mujoco.mjtObj.mjOBJ_MESH, "ufpk_p1") >= 0
            if scaled:
                V.log("== 2. the pinky has shortened phalanges (meshes ufpk_*), so it is not an exact copy of the ring")
            else:
                V.check_copy(h)
            check_vs_single(h)
            pinky_check(h)
        # ---------------- poses
        straight = cmd(f_s)
        a_max = 90 - abs(f_s)
        curl = (90.0, -90.0)
        V.log(f"\n== 6. poses (straight finger at f = {f_s:.1f} deg, so (m1, m2) = {straight}; abduction room there |a| <= {a_max:.0f})")
        spread = min(30.0, a_max)   # as five_finger/hand_poses.py (SPREAD = 30): a clear V with margin to the motor range
        peace_base = dict(index=cmd(f_s, spread), middle=cmd(f_s, -spread), ring=curl, pinky=curl)
        mf_base = dict(index=curl, middle=straight, ring=curl, pinky=curl)
        V.log("  thumb placement search (thumb motors on a grid, other digits held): the thumb's distal pad onto the target "
              "phalanx, thumb >= 0.2 mm from every other digit (sampled surfaces)")
        found = {}
        for name, base, target in (("Peace", peace_base, ("ring", ["uf_p2_shell", "uf_p2_pad", "uf_p2"])),
                                   ("MiddleFinger", mf_base, ("index", ["uf_p2_shell", "uf_p2_pad", "uf_p2"]))):
            ok, allr = search_thumb(h, base, target, step=args.step)
            best = ok[0] if ok else min(allr, key=lambda r: r[0])
            V.log(f"    {name:<13} target {target[0]} middle phalanx: best thumb {best[2]} -> pad-to-target {best[0] * 1e3:.2f} mm, "
                  f"thumb to the nearest digit {best[1] * 1e3:.2f} mm; next: " +
                  ", ".join(f"{r[2]} {r[0] * 1e3:.1f}/{r[1] * 1e3:.1f}" for r in ok[1:4]))
            thumb = best[2]
            if best[0] > 5e-3:   # cannot rest on the target: tuck the thumb into the palm (as hand_poses.py does)
                thumb = (90.0, -90.0)
                V.log(f"      -> the thumb cannot reach that phalanx; the pose uses the tucked thumb {thumb}")
            found[name] = dict(base, thumb=thumb)
        # pinch: index tip pad against the thumb tip pad
        V.log("  pinch search: index flexion f_i (a = 0) x thumb grid; index P3 pad vs thumb distal pad"
              + (" - skipped (--keep-pinch): the Pinched pose of data/hand_poses.json" if args.keep_pinch else ""))
        pinch = None
        old = json.loads((DATA_DIR / "hand_poses.json").read_text()) if args.keep_pinch else {}
        for fi in ((-20, -10, 0, 10, 20, 30, 45) if not old else ()):
            base = dict(index=cmd(fi), middle=curl, ring=curl, pinky=curl)
            sp = 0.5e-3
            tp = Group_cache(h, "thumb", {"thumb_distal_pad"}, sp)
            ip = Group_cache(h, "index", {"uf_p3_pad"}, sp)
            thumb_all = V.Group(h, h.geoms["thumb"][0], sp)
            idx_other = V.Group(h, [g for g in h.geoms["index"][0] if g not in mesh_geoms(h, "index", {"uf_p3_pad", "uf_p3_shell"})], sp)
            settle(h, dict(base, thumb=(0, 0)))
            best = None
            vals = np.arange(-90, 90.01, args.step)
            for i, t1 in enumerate(vals):
                for t2 in (vals if i % 2 == 0 else vals[::-1]):
                    h.d.ctrl[h.ctrl["thumb"]] = np.radians([t1, t2])
                    for _ in range(160):
                        mujoco.mj_step(h.m, h.d)
                    dp, _ = V.group_distance(tp, tp.poses(h.d), ip, ip.poses(h.d), 0.03)
                    do, _ = V.group_distance(thumb_all, thumb_all.poses(h.d), idx_other, idx_other.poses(h.d), 0.03)
                    if do < 0.2e-3:
                        continue
                    if best is None or dp < best[0]:
                        best = (float(dp), float(do), (float(t1), float(t2)))
            V.log(f"    index f {fi:+4d}: best thumb {best[2]} -> pad-to-pad {best[0] * 1e3:.2f} mm (thumb to the rest of the index "
                  f"{best[1] * 1e3:.2f} mm)")
            if pinch is None or best[0] < pinch[0] - 1e-4:
                pinch = (best[0], fi, best[2])
        found["Pinched"] = ({d: tuple(c) for d, c in old["Pinched"].items()} if old else
                            dict(index=cmd(pinch[1]), middle=curl, ring=curl, pinky=curl, thumb=pinch[2]))
        poses = pose_table(f_s, {})
        poses["Pinched"] = found["Pinched"]
        poses["Fuck"] = dict(found["MiddleFinger"])
        poses["Peace"] = found["Peace"]
        poses["MiddleFinger"] = found["MiddleFinger"]
        V.log("\n   pose table (right hand, (m1, m2) deg; thumb stock unless stated; 'Fuck' = MiddleFinger):")
        for name, pose in poses.items():
            V.log(f"     {name:<15} " + "  ".join(f"{dig} {tuple(float(x) for x in c)}" for dig, c in pose.items()))
        (DATA_DIR / "hand_poses.json").write_text(json.dumps({k: {d: list(c) for d, c in v.items()} for k, v in poses.items()},
                                                             indent=1) + "\n")
        V.DEMO_POSES_5F = poses
        # ---------------- pose clearances (every digit pair)
        groups = {d: V.Group(h, h.geoms[d][0] + h.geoms[d][1], 0.3e-3) for d in h.digits}
        worst = {}
        for name, pose in poses.items():
            settle(h, pose)
            worst[name] = all_pair_report(h, f"{name}:", groups, exact)
        for name in ("Peace", "MiddleFinger", "Pinched"):
            settle(h, poses[name])
            V.log(f"  {name}: tips (mm, palm frame of the plate): " + ", ".join(
                f"{d} {np.round(h.d.site(TIPS[d]).xpos * 1e3, 1).tolist()}" for d in h.digits))
        if not args.only_poses:
            V.check_stability(h)
            if not args.skip_grid:
                V.check_interference(h, args.grid, 1.0, 0.3)
                im_pair(h, args.grid)
            if shells_dir is not None:
                shells_check(h, shells_dir=shells_dir)
            else:
                V.log("\n== 8. palm/back shells: none fitted to this hand yet (--shells DIR); not checked")
            plate_check(h, args.grid)


def plate_check(h, n, coarse=1e-3, fine=0.3e-3):
    """Moving parts of every digit vs the palm (plate + wrist interface), as the end of V.check_interference."""
    V.log(f"== 5b. moving parts of every digit vs the palm plate and the wrist interface ({n}x{n} grid per digit, samples every "
          f"{coarse * 1e3:.1f} mm, {fine * 1e3:.1f} mm below 3 mm); plate mesh {plate_file()}")
    poses, worst = V.settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n))
    keys = list(poses)
    touch = 0.75 * fine
    gp = {sp: V.Group(h, h.palm, sp) for sp in (coarse, fine)}
    mujoco.mj_kinematics(h.m, h.d)
    pp = gp[fine].poses(h.d)
    for dig in h.digits:
        g = {sp: V.Group(h, h.geoms[dig][0], sp) for sp in (coarse, fine)}
        res = []
        for q in V.posed(h, g[coarse], poses):
            dist, who = V.group_distance(g[coarse], q, gp[coarse], pp, 0.005)
            if touch <= dist < 0.003:
                dist, who = V.group_distance(g[fine], q, gp[fine], pp, dist + coarse)
            res.append((dist, who))
        i = int(np.argmin([r[0] for r in res]))
        hits = [keys[k] for k, r in enumerate(res) if r[0] < touch]
        V.log(f"  {dig:<6} moving parts vs palm (plate + wrist interface): min clearance {res[i][0] * 1e3:5.2f} mm at "
              f"{V.fmt_pose(keys[i])}" + (f" [{V.geom_label(h, res[i][1][0])} / {V.geom_label(h, res[i][1][1])}]" if res[i][1] else "")
              + f"; grid poses touching: {len(hits)}/{len(res)}" + (f": {', '.join(V.fmt_pose(k) for k in hits[:9])}" if hits else ""))


PLATE = "r_hand_plate"


def plate_file():
    for line in (HAND / "robot.xml").read_text().splitlines():
        if 'name="r_hand_plate"' in line and "<mesh" in line:
            f = line.split('file="')[1].split('"')[0]
            q = HAND / "assets" / f
            return f"{f} ({q.stat().st_size} bytes)" if q.exists() else f
    return "?"


def im_pair(h, n):
    """index/middle on the same grid (verify_5f_model checks ring/pinky, middle/ring, thumb/*)."""
    poses, _ = V.settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n))
    keys = list(poses)
    table, who = V.clearance_table(h, poses, V.digit_geoms(h, "index"), V.digit_geoms(h, "middle"), 1e-3, 0.3e-3, 0.225e-3)
    V.report_pair(h, "index (A) vs middle (B)", keys, table, who, 0.225e-3)


def shells_check(h0, grid=13, shells_dir=None):
    """5-digit palm/back shells and pads (shells_dir, default cad/five_finger/stl) against this hand."""
    shells_dir = Path(shells_dir or REPO / "cad/five_finger/stl")
    shells = shell_files(shells_dir)
    own = shells_dir.resolve() == OWN_SHELLS.resolve()
    h = V.Hand(HAND, shells)
    poses, _ = V.settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, grid))
    keys = list(poses)
    coarse, fine = 1e-3, 0.3e-3
    touch = 0.75 * fine
    rel = shells_dir.resolve().relative_to(REPO) if shells_dir.resolve().is_relative_to(REPO) else shells_dir
    V.log(f"\n== 8. the 5-digit palm/back shells and pads ({rel}) vs every digit, {grid}x{grid} grid per digit "
          + ("(fitted by make_5f_shells.py to the merged sweeps of AH_Right_5F and AH_Right_5F_3S)" if own else ""))
    groups = {n: {sp: V.Group(h, [h.part_geom[n]], sp) for sp in (coarse, fine)} for n in shells}
    mujoco.mj_kinematics(h.m, h.d)
    spose = {n: g[fine].poses(h.d) for n, g in groups.items()}
    V.log("  palm-fixed parts vs the shells (bolted parts may touch by design; the plate is clamped in the lap joint):")
    fixed = {"plate": [h.part_geom["r_hand_plate"]], "wrist interface": [h.part_geom["r_wrist_interface"]]}
    for dig in h.digits:
        fixed[f"{dig} servos+frames"] = h.geoms[dig][1]
    for label_, geoms in fixed.items():
        gf = V.Group(h, geoms, fine)
        parts = []
        for n, gs in groups.items():
            dist, who = V.group_distance(gf, gf.poses(h.d), gs[fine], spose[n], 0.005)
            parts.append(f"{n} " + (f"{dist * 1e3:.2f} mm" + (" TOUCH" if dist < touch else "") if who else "> 5 mm"))
        V.log(f"    {label_:<22} " + ", ".join(parts))
    V.log(f"  moving parts vs the shells over the {grid * grid} grid poses:")
    for dig in h.digits:
        g = {sp: V.Group(h, h.geoms[dig][0], sp) for sp in (coarse, fine)}
        pa = V.posed(h, g[coarse], poses)
        for n, gs in groups.items():
            res = []
            for pp in pa:
                dist, who = V.group_distance(g[coarse], pp, gs[coarse], spose[n], 0.005)
                if touch <= dist < 0.003:
                    dist, who = V.group_distance(g[fine], pp, gs[fine], spose[n], dist + coarse)
                res.append((dist, who))
            i = int(np.argmin([r[0] for r in res]))
            hits = [keys[k] for k, r in enumerate(res) if r[0] < touch]
            V.log(f"  {dig:<6} vs {n:<10} min {res[i][0] * 1e3:5.2f} mm at {V.fmt_pose(keys[i])}"
                  + (f" [{V.geom_label(h, res[i][1][0])}]" if res[i][1] else "")
                  + f"; touching at {len(hits)}/{len(res)} poses" + (f": {', '.join(V.fmt_pose(k) for k in hits[:8])}" if hits else ""))


if __name__ == "__main__":
    main()
