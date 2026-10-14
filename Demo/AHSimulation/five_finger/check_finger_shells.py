#!/usr/bin/env python3
"""Check the styled finger shells and grip pads (cad/five_finger/fingers) in the 5-digit model.

A copy of AH_Right_5F/mjcf is made in --work with proximal_shell / distal_shell replaced by the
styled shells and the two grip pads added as extra visual geoms in the same bodies (same pos/quat:
all of them are in the stock finger part-studio frame, in mm, so the meshes get scale 0.001).
The dynamics are unchanged (every body has an explicit <inertial>; all geoms are visual).

Then, with the helpers of verify_5f_model.py, stock vs styled:
  1. palm side: the distal is flexed in the model and its motion is expressed in the proximal
     shell's CAD frame (confirms that -z is the flexion side, where the pads go);
  2. self-clearance over the joint range: every digit settled on an N x N grid over its two motors;
     what the styling added to each phalanx (the land / nail surface of the styled shell that is
     > 0.12 mm off the stock shell, and the pad) vs every other part of the same digit (other
     phalanx, link, gimbal, pins, screws, horns, servos, finger frames) and the palm plate / wrist
     interface. The rest of the styled shell is the stock shell, so this is where new contacts
     could come from;
  3. ring vs pinky (and --pairs) over every pair of grid poses (stock baseline: 887 / 6561 touching
     pairs for ring/pinky), with the same helpers as verify_5f_model.py;
  4. thumb vs index in the demo poses (CloseHand etc.): min distance and the volume by which the
     two digits' meshes overlap (manifold3d booleans);
  5. renders of the hand, open and closed (cad/five_finger/fingers/hand_styled_*.png).

Usage: python check_finger_shells.py [--grid 9] [--work DIR] [--stock-pairs] [--pairs ring:pinky,...]
                                     [--skip-pairs] [--skip-render]   (~3.5 min per digit pair and model)
"""

import argparse
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scipy.spatial import cKDTree  # noqa: E402
from verify_5f_model import (DEMO_POSES_5F, SIM_DIR, Group, Hand, clearance_table, fmt_pose,  # noqa: E402
                             geom_label, group_distance, log, pose_ctrl, posed, report_pair, sample_mesh, settle_grid,
                             BodyCloud)

REPO = HERE.parent.parent.parent
FINGERS = REPO / "cad" / "five_finger" / "fingers"
STYLED = {  # mesh name in the MJCF -> styled STL (mm, stock part-studio frame)
    "proximal_shell": FINGERS / "stl" / "Proximal_Shell_Styled.stl",
    "distal_shell": FINGERS / "stl" / "Distal_Shell_Styled.stl",
}
PADS = {"proximal_shell": ("proximal_pad", FINGERS / "stl" / "Proximal_Pad.stl"),
        "distal_shell": ("distal_pad", FINGERS / "stl" / "Distal_Pad.stl")}
PAD_RGBA = "0.30 0.31 0.33 1"
PROX_BODY, DIST_BODY = "parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b", "parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999"


def make_styled_model(src, dst):
    """Copy src (an mjcf folder) to dst with the styled shells and the pads."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    xml = (dst / "robot.xml").read_text()
    for mesh, path in STYLED.items():
        shutil.copy(path, dst / "assets" / f"{mesh}_styled.stl")
        xml, n = re.subn(rf'<mesh file="{mesh}.stl"/>',
                         f'<mesh name="{mesh}" file="{mesh}_styled.stl" scale="0.001 0.001 0.001"/>', xml)
        assert n == 1, mesh
        pad, pad_path = PADS[mesh]
        shutil.copy(pad_path, dst / "assets" / f"{pad}.stl")
        xml = xml.replace("  </asset>", f'    <mesh name="{pad}" file="{pad}.stl" scale="0.001 0.001 0.001"/>\n  </asset>', 1)
        # a pad geom next to every shell geom, same frame
        pat = re.compile(rf'^(\s*)<geom (type="mesh" class="visual" pos="[^"]+" quat="[^"]+") mesh="{mesh}" material="{mesh}_material"/>$', re.M)
        xml, n = pat.subn(lambda mo: mo.group(0) + f'\n{mo.group(1)}<!-- Grip pad {pad} (styled shells) -->\n'
                          f'{mo.group(1)}<geom {mo.group(2)} mesh="{pad}" material="grip_pad_material"/>', xml)
        assert n == 5, (mesh, n)  # one per digit
    xml = xml.replace("  </asset>", f'    <material name="grip_pad_material" rgba="{PAD_RGBA}"/>\n  </asset>', 1)
    (dst / "robot.xml").write_text(xml)
    return dst


def mesh_name(m, g):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        return ""
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or ""


def finger_bodies(h, dig):
    """(proximal body, distal body) ids of a digit."""
    moving = h.geoms[dig][0]
    pb = {int(h.m.geom_bodyid[g]) for g in moving if mesh_name(h.m, g) == "proximal_shell"}
    db = {int(h.m.geom_bodyid[g]) for g in moving if mesh_name(h.m, g) == "distal_shell"}
    assert len(pb) == 1 and len(db) == 1
    return pb.pop(), db.pop()


# ------------------------------------------------------------------ 1. palm side
def palm_side(h):
    """Distal tip (CAD point (0, 90, 0) mm) in the proximal shell's CAD frame, open vs closed."""
    spec = mujoco.MjSpec.from_file(str(h.dir / "scene.xml"))

    def cad_frame(body, mesh):
        g = [g for g in spec.body(body).geoms if g.meshname == mesh][0]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.array(g.quat))
        return np.array(g.pos), R.reshape(3, 3)

    out, tips = [], {}
    for pose in ("OpenHand", "CloseHand"):
        settle(h, DEMO_POSES_5F[pose])
        pp, pR = cad_frame(PROX_BODY, "proximal_shell")
        dp, dR = cad_frame(DIST_BODY, "distal_shell")
        bp, bd = h.d.body(PROX_BODY), h.d.body(DIST_BODY)
        RP, tP = bp.xmat.reshape(3, 3) @ pR, bp.xpos + bp.xmat.reshape(3, 3) @ pp
        RD, tD = bd.xmat.reshape(3, 3) @ dR, bd.xpos + bd.xmat.reshape(3, 3) @ dp
        tip = RP.T @ (tD + RD @ np.array([0, 0.090, 0]) - tP) * 1e3
        rel = RP.T @ RD
        tips[pose] = tip
        out.append(f"{pose}: index distal tip at {np.round(tip, 1)} mm in the proximal shell frame, "
                   f"distal rotated {np.degrees(np.arctan2(rel[2, 1], rel[1, 1])):+.1f} deg about the pin axis x")
    log("== 1. palm side (CAD frame: x = pin axis, y = along the finger)")
    for s in out:
        log("  " + s)
    side = "-z" if tips["CloseHand"][2] < tips["OpenHand"][2] - 10 else "+z" if tips["CloseHand"][2] > tips["OpenHand"][2] + 10 else "?"
    log(f"  -> flexing moves the distal tip {tips['CloseHand'][2] - tips['OpenHand'][2]:+.1f} mm along z of the shell frame: "
        f"the palm (flexion) side is {side}" + (" (where the pads are)" if side == "-z" else " - THE PADS ARE ON THE WRONG SIDE"))


def settle(h, pose, steps=600):
    h.reset()
    target = pose_ctrl(h, pose)
    start = h.d.ctrl.copy()
    for k in range(steps):
        h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / (steps // 2))
        mujoco.mj_step(h.m, h.d)


# ------------------------------------------------------------------ 2. self-clearance
def body_samples(h, geoms, spacing):
    """Surface samples of some geoms of one body, in that body's frame."""
    out = []
    for g in geoms:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, h.m.geom_quat[g])
        out.append(h.m.geom_pos[g] + sample_mesh(h.m, h.m.geom_dataid[g], spacing) @ R.reshape(3, 3).T)
    return np.concatenate(out)


def new_surface(stock, styled, body_name, tag, spacing):
    """Body-frame samples of what the styling added to one phalanx: the pad, and the part of the
    styled shell's surface that is not on the stock shell's surface (the land and the nail)."""
    def geoms(h, mesh):
        b = mujoco.mj_name2id(h.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return [g for g in range(h.m.ngeom) if h.m.geom_bodyid[g] == b and mesh_name(h.m, g) == mesh]
    old = cKDTree(body_samples(stock, geoms(stock, f"{tag}_shell"), 0.08e-3))
    shell = body_samples(styled, geoms(styled, f"{tag}_shell"), spacing)
    land = shell[old.query(shell)[0] > 0.12e-3]  # > 0.12 mm off the stock surface (samples every 0.08 mm)
    return {"land/nail": land, "pad": body_samples(styled, geoms(styled, f"{tag}_pad"), spacing)}


class PointGroup:
    """A Group (see verify_5f_model) made of given body-frame points of one body."""

    def __init__(self, pts, body, owner):
        self.bodies, self.clouds = [body], [BodyCloud(pts, np.full(len(pts), owner))]

    def poses(self, d):
        return [(d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in self.bodies]


def self_clearance(stock, styled, poses, coarse, fine, touch):
    """The styled shells equal the stock ones except for the added surface, so new contacts can only
    involve the added surface: check it (and the pads) against every other part of the same digit
    (other phalanx, link, gimbal, pins, screws, horns, servos, finger frames) and the palm."""
    h, m = styled, styled.m
    keys = list(poses)
    res = {}
    for dig in h.digits:
        moving, fixed = h.geoms[dig]
        for tag, body in zip(("proximal", "distal"), finger_bodies(h, dig)):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
            owner = {"land/nail": [g for g in moving if m.geom_bodyid[g] == body and mesh_name(m, g) == f"{tag}_shell"][0],
                     "pad": [g for g in moving if m.geom_bodyid[g] == body and mesh_name(m, g) == f"{tag}_pad"][0]}
            added = {sp: new_surface(stock, styled, name, tag, sp) for sp in (coarse, fine)}
            others = [g for g in moving + fixed + h.palm if m.geom_bodyid[g] != body]
            gb = {sp: Group(h, others, sp) for sp in (coarse, fine)}
            pb = {sp: posed(h, gb[sp], poses) for sp in (coarse, fine)}
            for what in ("land/nail", "pad"):
                ga = {sp: PointGroup(added[sp][what], body, owner[what]) for sp in (coarse, fine)}
                pa = posed(h, ga[coarse], poses)
                best = []
                for k in range(len(keys)):
                    dist, who = group_distance(ga[coarse], pa[k], gb[coarse], pb[coarse][k], 0.01)
                    if dist < 0.006:  # refine below 6 mm
                        dist, who = group_distance(ga[fine], pa[k], gb[fine], pb[fine][k], dist + coarse)
                    best.append((dist, who))
                i = int(np.argmin([b[0] for b in best]))
                hits = [k for k, b in enumerate(best) if b[0] < touch]
                res[(dig, tag, what)] = (best[i][0], len(hits))
                log(f"  {dig:<6} {tag:<8} {what:<9} ({len(added[fine][what]):5d} samples): min clearance "
                    + (f"{best[i][0] * 1e3:5.2f} mm at {fmt_pose(keys[i])} [{geom_label(h, best[i][1][1]).split(' (')[0]}]"
                       if best[i][1] else "> 10 mm")
                    + f"; touching at {len(hits)}/{len(keys)} poses")
    return res


# ------------------------------------------------------------------ 4. thumb vs index overlap
def world_mesh(h, g):
    m, d = h.m, h.d
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f


def overlap_volume(h, geoms_a, geoms_b):
    """Total volume (mm3) of pairwise mesh intersections, and the pairs involved."""
    import manifold3d as mf
    cache = {}

    def man(g):
        if g not in cache:
            v, f = world_mesh(h, g)
            mesh = mf.Mesh(vert_properties=(v * 1e3).astype(np.float32), tri_verts=f.astype(np.uint32))
            mesh.merge()  # MuJoCo keeps per-face duplicates of some vertices
            mm = mf.Manifold(mesh)
            cache[g] = mm if mm.status() == mf.Error.NoError and mm.volume() > 0 else None
        return cache[g]

    total, pairs, failed = 0.0, [], set()
    for a in geoms_a:
        for b in geoms_b:
            lo_a, hi_a = h.d.geom_xpos[a] - h.m.geom_rbound[a], h.d.geom_xpos[a] + h.m.geom_rbound[a]
            lo_b, hi_b = h.d.geom_xpos[b] - h.m.geom_rbound[b], h.d.geom_xpos[b] + h.m.geom_rbound[b]
            if np.any(hi_a < lo_b) or np.any(hi_b < lo_a):
                continue
            ma, mb = man(a), man(b)
            if ma is None or mb is None:
                failed |= {g for g, mm in ((a, ma), (b, mb)) if mm is None}
                continue
            vol = (ma ^ mb).volume()
            if vol > 1e-4:
                total += vol
                pairs.append((vol, geom_label(h, a).split(" (")[0], geom_label(h, b).split(" (")[0]))
    return total, sorted(pairs, reverse=True), failed


def thumb_index(hands, spacing):
    log("== 4. thumb vs index in the demo poses (thumb tucks under the index on purpose in some)")
    sp = spacing * 1e-3
    for name in ("CloseHand", "Perfect", "Pinched", "Fuck", "OpenHand"):
        out = []
        for label, h in hands.items():
            settle(h, DEMO_POSES_5F[name], steps=500)
            ga = Group(h, h.geoms["thumb"][0] + h.geoms["thumb"][1], sp)
            gb = Group(h, h.geoms["index"][0] + h.geoms["index"][1], sp)
            dist, who = group_distance(ga, ga.poses(h.d), gb, gb.poses(h.d), 0.005, skip_root_pair=True)
            vol, pairs, failed = overlap_volume(h, h.geoms["thumb"][0], h.geoms["index"][0])
            s = (f"{label}: min distance {dist * 1e3:.2f} mm" if np.isfinite(dist) else f"{label}: > 5 mm apart")
            if who is not None:
                s += f" [{geom_label(h, who[0]).split(' (')[0]} / {geom_label(h, who[1]).split(' (')[0]}]"
            s += f", overlap {vol:.2f} mm3" + (f" ({'; '.join(f'{a}/{b} {v:.2f}' for v, a, b in pairs[:4])})" if pairs else "")
            if failed:
                s += f" (non-manifold meshes skipped: {sorted({mesh_name(h.m, g) for g in failed})})"
            out.append(s)
        log(f"  {name:<10} " + " | ".join(out))


# ------------------------------------------------------------------ 5. renders
def render(h, out_dir, size=480):
    from PIL import Image, ImageDraw
    m = h.m
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, int(size * 4 / 3))
    m.vis.global_.offheight = max(m.vis.global_.offheight, size)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    m.geom_rgba[floor, 3] = 0
    for b in range(m.nbody):
        if m.body_mocapid[b] >= 0:
            m.geom_rgba[m.body_geomadr[b]:m.body_geomadr[b] + m.body_geomnum[b], 3] = 0
    # look: white shells, grey pads, light grey PLA, metallic pins / screws
    for g in range(m.ngeom):
        mn = mesh_name(m, g) or ""
        mat = m.geom_matid[g]
        if mat < 0:
            continue
        if mn.endswith("_shell"):
            m.mat_rgba[mat] = [0.96, 0.96, 0.95, 1]
        elif mn.endswith("_pad"):
            m.mat_rgba[mat] = [0.28, 0.29, 0.31, 1]
        elif mn.startswith(("parallel_pin", "std00447", "std00333", "plain_washer", "ph_pan_head", "m2_rod")):
            m.mat_rgba[mat] = [0.72, 0.74, 0.78, 1]
            m.mat_specular[mat], m.mat_shininess[mat], m.mat_reflectance[mat] = 0.9, 0.9, 0.2
    m.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    m.vis.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    m.vis.headlight.specular[:] = [0.2, 0.2, 0.2]
    for t in range(m.ntex):  # light background
        if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            adr, w, hh, nc = m.tex_adr[t], m.tex_width[t], m.tex_height[t], m.tex_nchannel[t]
            m.tex_data[adr:adr + w * hh * nc] = 235
    r = mujoco.Renderer(m, size, int(size * 4 / 3))
    hand = np.array([0.02, -0.022, 0.105])
    tips = np.array([0.025, -0.035, 0.14])
    fist = np.array([0.04, -0.02, 0.085])
    views = {  # name: (lookat, azimuth, elevation, distance)
        "OpenHand": {"palm": (hand, 180, -10, 0.28), "3/4 palm": (hand, 145, -22, 0.28),
                     "fingers, palm side": (tips, 172, -8, 0.16), "pinky side": (hand, 90, -8, 0.28),
                     "back": (hand, 0, -15, 0.28), "fingertips, 3/4": (tips + [0, 0, 0.01], 140, -30, 0.16)},
        "CloseHand": {"palm": (hand, 180, -10, 0.28), "3/4 palm": (hand, 145, -22, 0.28),
                      "from below (proximal pads)": (fist, 180, 55, 0.19), "pinky side": (hand, 90, -8, 0.28),
                      "back": (hand, 0, -15, 0.28), "fist, close-up": (fist, 150, 10, 0.17)},
    }
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    files = []
    for pose, vs in views.items():
        settle(h, DEMO_POSES_5F[pose])
        tiles = []
        for v, (look, az, el, dist) in vs.items():
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            r.update_scene(h.d, cam, opt)
            im = Image.fromarray(r.render())
            ImageDraw.Draw(im).text((8, 6), f"styled shells + grip pads - {pose} - {v}", fill=(0, 0, 0))
            tiles.append(np.asarray(im))
        sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:6], 1)], 0)
        fn = out_dir / f"hand_styled_{pose}.png"
        Image.fromarray(sheet).save(fn)
        files.append(fn)
    r.close()
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--coarse", type=float, default=1.0)
    ap.add_argument("--fine", type=float, default=0.3)
    ap.add_argument("--work", type=Path, default=None, help="where the modified MJCF copy goes (default: a temp dir)")
    ap.add_argument("--skip-pairs", action="store_true", help="skip the ring/pinky pose-pair sweep")
    ap.add_argument("--stock-pairs", action="store_true", help="also run the pose-pair sweeps on the stock model")
    ap.add_argument("--pairs", default="ring:pinky", help="digit pairs for the pose-pair sweep, e.g. ring:pinky,index:middle,thumb:index")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="ah_styled_"))
    work.mkdir(parents=True, exist_ok=True)
    dst = make_styled_model(SIM_DIR / "AH_Right_5F" / "mjcf", work / "AH_Right_5F_styled" / "mjcf")
    log(f"styled model: {dst} (mujoco {mujoco.__version__})")
    stock, styled = Hand(SIM_DIR / "AH_Right_5F" / "mjcf"), Hand(dst)
    nshell = sum(mesh_name(styled.m, g).endswith(("_shell", "_pad")) for g in range(styled.m.ngeom))
    log(f"  geoms: stock {stock.m.ngeom}, styled {styled.m.ngeom} (+{styled.m.ngeom - stock.m.ngeom} pads; {nshell} shell/pad geoms); "
        f"body masses identical: {bool(np.allclose(stock.m.body_mass, styled.m.body_mass))}")

    palm_side(styled)

    coarse, fine = args.coarse * 1e-3, args.fine * 1e-3
    touch = 0.75 * fine
    angles = np.linspace(-np.pi / 2, np.pi / 2, args.grid)
    t0 = time.time()
    poses = {k: settle_grid(h, angles)[0] for k, h in (("stock", stock), ("styled", styled))}
    log(f"== 2. self-clearance of the added surface over the {args.grid}x{args.grid} grid (all digits at the same command; "
        f"samples every {args.fine} mm, contact: < {touch * 1e3:.3f} mm) [settled in {time.time() - t0:.0f} s]")
    t0 = time.time()
    sc = self_clearance(stock, styled, poses["styled"], coarse, fine, touch)
    worst = min(sc.items(), key=lambda kv: kv[1][0])
    log(f"  -> added surface (lands, nail plates, pads) vs the rest of its digit and the palm: min clearance "
        f"{worst[1][0] * 1e3:.2f} mm ({' '.join(worst[0])}), touching poses {sum(v[1] for v in sc.values())} [{time.time() - t0:.0f} s]")

    if not args.skip_pairs:
        keys = list(poses["styled"])
        for a, b in [p.split(":") for p in args.pairs.split(",")]:
            log(f"== 3. {a} vs {b}, every pair of grid poses ({len(keys)}x{len(keys)}), stock vs styled shells:")
            for label in (["stock"] if args.stock_pairs else []) + ["styled"]:
                h = stock if label == "stock" else styled
                t0 = time.time()
                table, who = clearance_table(h, poses[label], h.geoms[a][0] + h.geoms[a][1], h.geoms[b][0] + h.geoms[b][1],
                                             coarse, fine, touch)
                report_pair(h, f"[{label}] {a} (A) vs {b} (B)", keys, table, who, touch)
                np.save(work / f"table_{label}_{a}_{b}.npy", table)
                log(f"    [{time.time() - t0:.0f} s]")
            if (work / f"table_stock_{a}_{b}.npy").exists():
                t_s, t_y = np.load(work / f"table_stock_{a}_{b}.npy"), np.load(work / f"table_styled_{a}_{b}.npy")
                sel = np.isfinite(t_s) & np.isfinite(t_y) & (t_s >= touch)
                log(f"  {a}/{b}: touching pose pairs stock {int((t_s < touch).sum())}, styled {int((t_y < touch).sum())}; "
                    f"same touching set: {bool(np.array_equal(t_s < touch, t_y < touch))}; touching only with the styled "
                    f"shells: {int(((t_y < touch) & ~(t_s < touch)).sum())}; min clearance change over the pairs that do not "
                    f"touch in stock: {np.min(t_y[sel] - t_s[sel], initial=np.inf) * 1e3:+.3f} mm")

    thumb_index({"stock": stock, "styled": styled}, args.fine)

    if not args.skip_render:
        files = render(Hand(dst), FINGERS)
        log("== 5. renders: " + ", ".join(str(f.relative_to(REPO)) for f in files))


if __name__ == "__main__":
    main()
