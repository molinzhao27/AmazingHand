#!/usr/bin/env python3
"""Palm and back shells for the 5-digit AmazingHand, generated around the mechanism.

The shells are built as implicit (signed-distance) solids on a voxel grid in the CAD part-studio
frame (mm; x from the index towards the pinky, y up the fingers, z towards the back):

1. Keep-out volumes from the MuJoCo model (AH_Right_5F), every digit settled on an N x N grid of
   its two motor commands:
     K_enc   parts the shell must enclose: palm plate, wrist interface, servos, finger frames and
             bushings, plus the servo horns and push rods over all poses;
     K_free  parts the shell must stay clear of: every finger's knuckle (gimbal), lever and both
             phalanges over all poses (this is what opens the finger and thumb openings).
2. Envelope E: a smooth, hand-like solid. Its outline and its back/palm surfaces follow the
   enclosed mechanism (heavily smoothed, so it reads as a dome on the back and a palm in front),
   with a thenar bulge around the thumb base, a hypothenar bulge on the pinky side and soft
   knuckle bumps on the back. E always contains K_enc grown by clearance + wall thickness.
3. Shell = E hollowed to WALL mm, minus K_free grown by FREE_CLEARANCE (closed so the webs between
   neighbouring finger openings disappear), cut at the wrist plane.
4. Split at the plate's back face into a rigid back shell (PLA) and a flexible palm shell
   (TPU), joined by a lap joint. The palm shell's flange is clamped between the plate and the
   back shell; the back shell is screwed through it with the stock screws into the plate and the
   wrist interface (plus one more into the widened plate's pinky-side hole).
5. Grip-pad recesses on the palm, with separate textured pad inserts (grey TPU).

Outputs (STL, mm, same frame as the stock parts) in cad/five_finger/stl/: R-Top_Shell_5F,
R-Palm_Shell_5F, R-Palm_Pad_Band_5F, R-Palm_Pad_Palm_5F and their mirrored L-* versions for the left
hand (the stock left shells are exactly the right ones mirrored about x = -50 mm).

Usage: python make_5f_shells.py [--vox 0.5] [--grid 19] [--cache keepout.npz]
"""

import argparse
import sys
import time
from pathlib import Path

import fast_simplification
import manifold3d as mf
import mujoco
import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "five_finger"))
import verify_5f_model as V  # noqa: E402
from build_5f_model import child_by_name, get_pose, parse  # noqa: E402

OUT_STL = HERE / "stl"
LO = np.array([-40.0, -104.0, -62.0])  # voxel grid bounds (plate frame, mm)
HI = np.array([128.0, 22.0, 42.0])
# plate pose in the MuJoCo root frame: part-studio frame (mm) <-> MJCF (m)
PLATE_POS = np.array([-0.0115, 0.0245, 0.1022])
PLATE_QUAT = np.array([0.5, 0.5, -0.5, -0.5])

WALL = 2.0  # shell wall thickness
CLR = 1.0  # clearance around enclosed parts
FREE = 1.2  # clearance around moving finger parts
Z_SPLIT = 15.5  # plate back face: palm shell below, back shell above
LAP_H, LAP_GAP = 1.5, 0.2  # lap joint height and clearance
FLANGE_Y = -85.0  # palm-shell floor flange on the wrist interface below this y
FLANGE_T = 1.3
Z_BACK, CROWN, R_EDGE = 25.0, 1.5, 8.0
CAP_BELOW_KNUCKLE = 3.0  # back shell top edge this far below the knuckle (pivot) line
Z_PALM_TOP = 0.0  # below this z the top edge follows the palm cap (under the finger sweeps)
PALM_MARGIN = 2.0  # extra depth of the palm surface below the enclosed parts
WRIST_Y = -97.5  # bottom cut, same as the stock shells
# back-shell screws (stock Top_Shell positions + the widened plate's pinky-side hole): (x, y, boss bottom z)
SCREWS = [(16.1, -11.7, Z_SPLIT), (47.5, -20.1, Z_SPLIT), (77.8, -32.9, Z_SPLIT),
          (4.0, -91.0, Z_SPLIT + FLANGE_T), (46.0, -91.0, Z_SPLIT + FLANGE_T)]
BOSS_R, HOLE_R, HEAD_R, HOLE_LEN = 3.6, 1.35, 2.5, 3.0  # stock: r1.35 hole 3 mm, r2.5 head counterbore
# grip pads (grey TPU inserts in recesses of the palm shell)
PAD_DEPTH, PAD_PROUD, PAD_GAP, PAD_TEX, PAD_PITCH = 1.0, 0.8, 0.15, 0.35, 3.0
PAD_EDGE = 2.5  # minimum distance from a pad to any edge/opening of the palm-facing surface
PAD_VOX = 0.2


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ 1. keep-out volumes
def keepout(n_grid, vox):
    """Voxelised K_enc (enclose), K_free (fingers' moving parts, stay clear) and K_thumb (the
    thumb's moving parts, stay clear) in the plate frame."""
    shape = np.ceil((HI - LO) / vox).astype(int)
    h = V.Hand(V.SIM_DIR / "AH_Right_5F" / "mjcf")
    m, d = h.m, h.d
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, PLATE_QUAT)
    R = R.reshape(3, 3)

    def depth(b):  # 0 = servo horn, 1 = push rod, >= 2 = finger-side parts
        k = 0
        while m.body_parentid[b] != V.ROOT_BODY:
            b, k = m.body_parentid[b], k + 1
        return k

    cache = {}

    def mark(grid, g):
        mid = m.geom_dataid[g]
        if mid not in cache:
            cache[mid] = V.sample_mesh(m, mid, vox * 0.7e-3)
        P = ((d.geom_xpos[g] + cache[mid] @ d.geom_xmat[g].reshape(3, 3).T) - PLATE_POS) @ R * 1000
        idx = np.floor((P - LO) / vox).astype(int)
        idx = idx[np.all((idx >= 0) & (idx < shape), axis=1)]
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    poses, _ = V.settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n_grid))
    K_enc, K_free, K_thumb = np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool)
    mujoco.mj_kinematics(m, d)
    for g in [g for dig in h.digits for g in h.geoms[dig][1]] + h.palm:
        mark(K_enc, g)
    for q in poses.values():
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for dig in h.digits:
            for g in h.geoms[dig][0]:
                mark(K_enc if depth(m.geom_bodyid[g]) <= 1 else K_thumb if dig == "thumb" else K_free, g)
    return K_enc, K_free, K_thumb


# ------------------------------------------------------------------ 2.-3. fields
class Grid:
    def __init__(self, shape, vox):
        self.vox = vox
        self.X, self.Y, self.Z = [LO[i] + (np.arange(shape[i]) + 0.5) * vox for i in range(3)]
        self.XX, self.YY = np.meshgrid(self.X, self.Y, indexing="ij")

    def sdf(self, mask):
        return (ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)) * self.vox

    def dist(self, mask):
        return ndi.distance_transform_edt(~mask) * self.vox


def smax(a, b, k):
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0, 1)
    return a * h + b * (1 - h) + k * h * (1 - h)


def smin(a, b, k):
    return -smax(-a, -b, k)


def vmax(*fields):
    """Intersection of signed fields (broadcasting)."""
    out = fields[0]
    for f in fields[1:]:
        out = np.maximum(out, f)
    return out


def knuckle_cap(G):
    """Top edge of the shells: a smooth curve CAP_BELOW_KNUCKLE under the four finger pivots."""
    root = parse(V.SIM_DIR / "AH_Right_5F" / "mjcf" / "robot.xml").getroot().find("worldbody/body")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, PLATE_QUAT)
    piv = []
    for n in (1, 2, 3, 5):
        p = get_pose(child_by_name(root, "site", f"closing_1 ({n})_2"))[0]
        piv.append(((p - PLATE_POS) @ R.reshape(3, 3) * 1000)[:2])
    piv = np.array(piv)
    return np.polyval(np.polyfit(piv[:, 0], piv[:, 1], 2), G.X) - CAP_BELOW_KNUCKLE


def envelope(G, K_enc):
    """Signed distance of the outer envelope (negative inside), not yet cut at the top, and the
    enclosed parts K (everything below the knuckle cap)."""
    ycap = knuckle_cap(G)
    K = K_enc & ~(G.YY >= ycap[:, None])[:, :, None]
    K[:, G.Y < WRIST_Y - 2.5, :] = False
    # outline: the enclosed parts' footprint, closed, grown and smoothed
    foot = ndi.binary_fill_holes(ndi.binary_closing(K.any(2), iterations=int(10 / G.vox)))
    d2 = ndi.gaussian_filter(G.sdf(foot), 4 / G.vox) - (CLR + WALL + 3.0)
    # palm surface: smooth lower envelope kept PALM_MARGIN below every enclosed part, so the
    # parts never push bumps through it
    zmin = np.where(K.any(2), G.Z[np.argmax(K, axis=2)], np.nan)
    idx = ndi.distance_transform_edt(np.isnan(zmin), return_distances=False, return_indices=True)
    req = zmin[tuple(idx)] - (CLR + WALL + PALM_MARGIN)
    zp = ndi.grey_erosion(req, size=int(14 / G.vox))
    for _ in range(8):
        zp = np.minimum(ndi.gaussian_filter(zp, 6 / G.vox), req)
    zp = ndi.gaussian_filter(zp, 3 / G.vox)
    excess = ndi.gaussian_filter(ndi.grey_dilation(np.maximum(zp - req, 0), size=int(6 / G.vox)), 3 / G.vox)
    zp -= 1.5 * excess
    # back surface: flat with a gentle crown, just clear of the servo backs
    zb = Z_BACK + CROWN * np.exp(-(((G.XX - 45.0) / 60) ** 2 + ((G.YY + 45.0) / 50) ** 2))
    Z = G.Z[None, None, :]
    s = smax(smax(d2[:, :, None], Z - zb[:, :, None], R_EDGE), zp[:, :, None] - Z, R_EDGE)
    s = np.maximum(s, (G.YY - ycap[:, None] - 12.0)[:, :, None])  # bounded above the cap
    dK = ndi.gaussian_filter(G.dist(K) - (CLR + WALL), 0.7)
    s = smin(s, dK, 2.0)
    return s, ycap, K


def top_cut(G, s_E, s_F, ycap):
    """Signed field of the region kept below the shells' top edge: the knuckle cap on the back,
    and on the palm side a smooth curve just under the lowest point where a finger sweeps through
    the palm wall (the thumb opening is cut separately)."""
    wall = (s_E < 0) & (s_E > -WALL - 1.0)
    hit = wall & (s_F < 0) & (G.Z[None, None, :] < Z_PALM_TOP)
    yhit = np.where(hit.any((1, 2)), G.Y[np.argmax(hit.any(2), axis=1)], ycap)
    yhit = np.minimum(yhit, ycap)
    ypalm = ndi.grey_erosion(yhit, size=int(12 / G.vox))
    for _ in range(6):
        ypalm = np.minimum(ndi.gaussian_filter(ypalm, 4 / G.vox), yhit - 1.5)
    ypalm = ndi.gaussian_filter(ypalm, 1.5 / G.vox)
    # blend from the palm cap (z <= Z_PALM_TOP - 6) to the knuckle cap (z >= Z_PALM_TOP + 6)
    w = np.clip((G.Z - (Z_PALM_TOP - 6.0)) / 12.0, 0, 1)
    w = w * w * (3 - 2 * w)
    y_top = ypalm[:, None] * (1 - w)[None, :] + ycap[:, None] * w[None, :]
    return G.YY[:, :, None] - y_top[:, None, :], ypalm


def openings(G, K_free, K_enc, ycap, closing=5.0):
    """Signed distance of the region the shell must leave free (negative inside): the moving finger
    parts, plus everything that passes through the shell's top (knuckle) edge - finger frames,
    bushings, the plate's top edge and the push rods' upper ends."""
    near_cap = (G.YY > (ycap[:, None] - 4.0))[:, :, None]
    O = G.dist(K_free | (K_enc & near_cap)) <= FREE
    O = ndi.binary_closing(O, structure=ndi.generate_binary_structure(3, 1), iterations=int(closing / G.vox))
    return ndi.gaussian_filter(G.sdf(O), 1.0)


# ------------------------------------------------------------------ 4. parts
def part_fields(G, s_E, s_O, s_top, K_enc_noplate):
    """Signed fields (negative inside) of the back shell and the palm shell."""
    Z, YY = G.Z[None, None, :], G.YY[:, :, None]
    s_in = ndi.gaussian_filter(G.sdf(ndi.binary_fill_holes(s_E < 0)), 0.6)
    wall = np.maximum(s_E, -(s_in + WALL))
    wall = smax(wall, s_top, 0.8)  # open top: the wall just ends at the cap curves
    wall = np.maximum(wall, -s_O)
    wall = np.maximum(wall, WRIST_Y - YY)
    half = -WALL / 2
    # back: everything above the lap, plus the outer half of the wall in the lap zone
    back_sel = np.minimum((Z_SPLIT + LAP_H) - Z, np.maximum(Z_SPLIT - Z, (half + LAP_GAP / 2) - s_E))
    back = np.maximum(wall, back_sel)
    # palm: everything below the split, plus the inner half of the wall in the lap zone
    top = Z_SPLIT + LAP_H - LAP_GAP
    palm_sel = np.minimum(Z - Z_SPLIT, np.maximum(Z - top, s_E - (half - LAP_GAP / 2)))
    palm = np.maximum(wall, palm_sel)
    # palm floor flange on the wrist interface's back face, clamped by the two bottom screws
    flange = vmax(s_E - (half - LAP_GAP / 2), Z_SPLIT - Z, Z - top, YY - FLANGE_Y, WRIST_Y - YY, -s_O,
                  0.3 - G.dist(K_enc_noplate))
    palm = np.minimum(palm, flange)
    # screw bosses on the back shell, from the plate / flange up into the back wall
    for x, y, z0 in SCREWS:
        r = np.sqrt((G.XX - x) ** 2 + (G.YY - y) ** 2)[:, :, None]
        boss = vmax(s_E, r - BOSS_R, z0 - Z)
        back = np.minimum(back, boss)
    return back, palm


# ------------------------------------------------------------------ 5. grip pads
def palm_face(G, s_E, palm):
    """2D mask of the palm-facing outer surface (normal within ~53 deg of -z) and its distance to
    the nearest edge or opening."""
    P = palm < 0
    has = P[:, :, G.Z < 0].any(2)
    k = np.argmax(P, axis=2)
    I, J = np.indices(has.shape)
    g = [np.gradient(s_E, axis=a)[I, J, k] for a in range(3)]
    nz = -g[2] / np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2 + 1e-12)
    facing = has & (nz > 0.6)
    return facing, ndi.distance_transform_edt(facing) * G.vox


def pad_outlines(G, facing, edge, ypalm):
    """2D signed distances (negative inside) of the pads: a band under the knuckles and a palm pad,
    kept PAD_EDGE from every edge or opening and 3 mm from each other. (The pinky side of the palm
    curves away too steeply for a hypothenar pad.)"""
    ok = edge > PAD_EDGE
    band = (G.YY < ypalm[:, None] - 5.0) & (G.YY > ypalm[:, None] - 13.5) & (G.XX > -4) & (G.XX < 100) & ok
    c, ang = np.array([60.0, -57.0]), np.radians(-14)
    u = (G.XX - c[0]) * np.cos(ang) + (G.YY - c[1]) * np.sin(ang)
    v = -(G.XX - c[0]) * np.sin(ang) + (G.YY - c[1]) * np.cos(ang)
    palm_pad = ((np.abs(u) / 25) ** 4 + (np.abs(v) / 12) ** 4 < 1) & ok
    pads = {}
    taken = np.zeros_like(ok)
    for name, mask in (("Band", band), ("Palm", palm_pad)):
        mask &= ~ndi.binary_dilation(taken, iterations=int(3 / G.vox))
        r = int(3 / G.vox)  # round the outline (opening with a 3 mm disc)
        mask = ndi.binary_opening(mask, structure=np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r)
        lab, n = ndi.label(mask)
        if n > 1:  # keep the biggest piece
            mask = lab == (1 + np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
        taken |= mask
        assert mask.sum() * G.vox ** 2 > 300, f"pad {name} too small"
        pads[name] = ndi.gaussian_filter(G.sdf(mask), 1.0)
    return pads


def pad_mesh(G, s_E, p2, name):
    """Insert for one recess, on a fine local grid: sits on the recess floor, PAD_GAP inside its
    walls, stands PAD_PROUD out of the palm with a hexagonal grip texture."""
    ii = np.argwhere(p2 < 1.0)
    lo2, hi2 = ii.min(0), ii.max(0) + 1
    zcol = np.argwhere((s_E[lo2[0]:hi2[0], lo2[1]:hi2[1]] < 1.5) & (s_E[lo2[0]:hi2[0], lo2[1]:hi2[1]] > -2.5)
                       & (G.Z < 0)[None, None, :])[:, 2]
    lo = np.array([G.X[lo2[0]], G.Y[lo2[1]], G.Z[zcol.min()]]) - 1.0
    hi = np.array([G.X[hi2[0] - 1], G.Y[hi2[1] - 1], G.Z[zcol.max()]]) + 1.0
    axes = [np.arange(lo[i], hi[i], PAD_VOX) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    coords = [(P[..., i] - G.X[0] if i == 0 else P[..., i] - (G.Y[0] if i == 1 else G.Z[0])) / G.vox for i in range(3)]
    sE = ndi.map_coordinates(s_E, coords, order=3, mode="nearest")
    q2 = ndi.map_coordinates(p2, coords[:2], order=1, mode="nearest")
    k = 2 * np.pi / PAD_PITCH
    x, y = P[..., 0], P[..., 1]
    hexa = (np.cos(k * x) + np.cos(k * (x / 2 + np.sqrt(3) * y / 2)) + np.cos(k * (x / 2 - np.sqrt(3) * y / 2))) / 3
    top = PAD_PROUD + PAD_TEX * (np.clip(hexa, -0.5, 1.0) - 1.0) / 1.5
    f = vmax(q2 + PAD_GAP, -PAD_DEPTH - sE)
    f = smax(f, sE - top, 0.4)
    verts, faces, _, _ = measure.marching_cubes(np.pad(f, 1, constant_values=1.0), 0.011, spacing=(PAD_VOX,) * 3)
    verts += lo - PAD_VOX
    mesh = trimesh.Trimesh(verts, faces)
    mesh = max(mesh.split(only_watertight=False), key=lambda p: len(p.faces))
    v, fc = mesh.vertices, mesh.faces
    if len(fc) > 50_000:
        v, fc = fast_simplification.simplify(v.astype(np.float32), fc, target_count=50_000)
    return manifold(v, fc)


def manifold(v, f):
    """Manifold from a triangle mesh (degenerate faces dropped)."""
    mesh = trimesh.Trimesh(np.asarray(v, np.float64), f)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    man = mf.Manifold(mf.Mesh(vert_properties=mesh.vertices.astype(np.float32), tri_verts=mesh.faces.astype(np.uint32)))
    assert man.status() == mf.Error.NoError and not man.is_empty(), man.status()
    return man


def to_mesh(G, field, min_faces=2000, target_faces=90_000):
    """Marching cubes -> drop specks -> decimate -> manifold."""
    pad = np.pad(field, 1, constant_values=1.0)
    verts, faces, _, _ = measure.marching_cubes(pad, 0.011, spacing=(G.vox,) * 3)
    verts += LO + 0.5 * G.vox - G.vox
    mesh = trimesh.Trimesh(verts, faces)
    parts = [p for p in mesh.split(only_watertight=False) if len(p.faces) >= min_faces]
    mesh = trimesh.util.concatenate(parts)
    v, f = mesh.vertices, mesh.faces
    if len(f) > target_faces:
        v, f = fast_simplification.simplify(v.astype(np.float32), f, target_count=target_faces)
    return manifold(v, f)


def cylinder(x, y, z0, z1, r):
    return mf.Manifold.cylinder(z1 - z0, r, r, 48).translate([x, y, z0])


def finish(back, palm):
    for x, y, z0 in SCREWS:
        back = back - cylinder(x, y, z0 - 1.0, z0 + HOLE_LEN, HOLE_R) - cylinder(x, y, z0 + HOLE_LEN, 40.0, HEAD_R)
        if z0 > Z_SPLIT:  # screws through the palm flange
            palm = palm - cylinder(x, y, Z_SPLIT - 1.0, z0 + 1.0, HOLE_R + 0.25)
    return back, palm


def save(man, path):
    m = man.to_mesh64()
    tm = trimesh.Trimesh(m.vert_properties[:, :3], m.tri_verts, process=False)
    assert tm.is_watertight and tm.is_volume, f"{path.name} is not a closed volume"
    tm.export(path)
    # a few marching-cubes sliver vertices may coincide once written as float32 (touching, not
    # open, edges; slicers handle them) - report them rather than hide them
    touching = len(tm.vertices) - len(np.unique(tm.vertices.astype(np.float32), axis=0))
    log(f"wrote {path.relative_to(REPO)}: {len(tm.faces)} faces, volume {tm.volume / 1000:.1f} cm3, "
        f"bbox {np.round(tm.bounds, 1).tolist()}" + (f", {touching} float32-coincident vertices" if touching else ""))
    return tm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vox", type=float, default=0.5, help="voxel size, mm")
    ap.add_argument("--grid", type=int, default=19, help="poses per motor for the swept volumes")
    ap.add_argument("--cache", type=Path, help="npz cache for the keep-out volumes")
    args = ap.parse_args()
    t0 = time.time()
    if args.cache and args.cache.exists():
        z = np.load(args.cache)
        K_enc, K_free, K_thumb, K_plate = z["K_enc"], z["K_free"], z["K_thumb"], z["K_plate"]
    else:
        K_enc, K_free, K_thumb = keepout(args.grid, args.vox)
        K_plate = keepout_palm(args.vox, K_enc.shape)
        if args.cache:
            np.savez_compressed(args.cache, K_enc=K_enc, K_free=K_free, K_thumb=K_thumb, K_plate=K_plate)
    log(f"keep-out volumes {K_enc.shape} ({time.time() - t0:.0f} s)")
    G = Grid(K_enc.shape, args.vox)
    s_E, ycap, _ = envelope(G, K_enc)
    s_F = openings(G, K_free, K_enc, ycap)  # the four fingers
    s_O = np.minimum(s_F, openings(G, K_thumb, np.zeros_like(K_enc), ycap, closing=3.0))  # + thumb
    s_top, ypalm = top_cut(G, s_E, s_F, ycap)
    back_f, palm_f = part_fields(G, s_E, s_O, s_top, K_enc & ~K_plate)
    facing, edge = palm_face(G, s_E, palm_f)
    pads = pad_outlines(G, facing, edge, ypalm)
    for p2 in pads.values():  # recesses
        palm_f = np.maximum(palm_f, -vmax(p2[:, :, None], -PAD_DEPTH - s_E, G.Z[None, None, :] - Z_SPLIT))
    log(f"fields ({time.time() - t0:.0f} s)")
    back, palm = finish(to_mesh(G, back_f), to_mesh(G, palm_f))
    OUT_STL.mkdir(exist_ok=True)
    parts = {"Top_Shell": back, "Palm_Shell": palm}
    parts.update({f"Palm_Pad_{name}": pad_mesh(G, s_E, p2, name) for name, p2 in pads.items()})
    for name, man in parts.items():
        save(man, OUT_STL / f"R-{name}_5F.stl")
        # the stock left shells are the right ones mirrored about x = -50 (like the plates)
        save(man.mirror([1, 0, 0]).translate([-100.0, 0, 0]), OUT_STL / f"L-{name}_5F.stl")
    log(f"done ({time.time() - t0:.0f} s)")


def keepout_palm(vox, shape):
    """Voxels of the palm plate and wrist interface alone (the palm flange may rest on them)."""
    h = V.Hand(V.SIM_DIR / "AH_Right_5F" / "mjcf")
    m, d = h.m, h.d
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, PLATE_QUAT)
    grid = np.zeros(shape, bool)
    mujoco.mj_kinematics(m, d)
    for g in h.palm:
        P = ((d.geom_xpos[g] + V.sample_mesh(m, m.geom_dataid[g], vox * 0.7e-3) @ d.geom_xmat[g].reshape(3, 3).T)
             - PLATE_POS) @ R.reshape(3, 3) * 1000
        idx = np.floor((P - LO) / vox).astype(int)
        idx = idx[np.all((idx >= 0) & (idx < shape), axis=1)]
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


if __name__ == "__main__":
    main()
