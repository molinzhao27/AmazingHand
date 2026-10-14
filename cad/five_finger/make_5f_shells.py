#!/usr/bin/env python3
"""Palm and back shells for the 5-digit AmazingHand, generated around the mechanism.

The shells are built as implicit (signed-distance) solids on a voxel grid in the CAD part-studio
frame (mm; x from the index towards the pinky, y up the fingers, z towards the back):

1. Keep-out volumes from the MuJoCo models (AH_Right_5F with the stock 2-phalanx fingers and AH_Right_5F_3S
   with the 3-phalanx fingers, merged), every digit settled on an N x N grid of its two motor commands:
     K_enc   parts the shell must enclose: palm plate, wrist interface, servos, finger frames and
             bushings, plus the servo horns and push rods over all poses;
     K_free  parts the shell must stay clear of: every finger's knuckle (gimbal), lever and both
             phalanges over all poses (this is what opens the finger and thumb openings).
2. Envelope E: a sculpted, hand-like solid laid out along the forearm axis (see envelope()): a palm
   slab with a domed back (transverse arch, knuckle bumps, rolled down to the knuckle line) and a
   wedge-shaped palm (thick at the heel, thin under the fingers, with a hollow), blended with a round
   wrist column, a thenar mound over the thumb's servos and a hypothenar mound. It is lifted smoothly
   wherever a part needs room, so E always contains K_enc grown by clearance + wall thickness.
3. Shell = E hollowed to WALL mm, minus K_free grown by FREE_CLEARANCE (closed so the webs between
   neighbouring finger openings disappear), cut at the wrist plane: square to the forearm axis of
   the powered wrist (cad/five_finger/wrist, which turns the hand by HAND_YAW so the forearm runs
   through the middle of the four fingers), and clear of the wrist's hand adapter. --no-wrist cuts
   at the stock plane (CAD y = -97.5) for a hand on the stock wrist interface alone.
4. Split at the plate's back face into a rigid back shell (PLA) and a flexible palm shell
   (TPU), joined by a lap joint. The palm shell's flange is clamped between the plate and the
   back shell; the back shell is screwed through it with the stock screws into the plate and the
   wrist interface (plus one more into the widened plate's pinky-side hole).
5. Panel lines (shallow grooves) across the back below the knuckles and around the wrist; one large
   grip-pad recess on the palm with a separate ribbed pad insert (black TPU).

Outputs (STL, mm, same frame as the stock parts) in cad/five_finger/stl/: R-Top_Shell_5F,
R-Palm_Shell_5F, R-Palm_Pad_Palm_5F and their mirrored L-* versions for the left
hand (the stock left shells are exactly the right ones mirrored about x = -50 mm).

Usage: python make_5f_shells.py [--vox 0.5] [--grid 19] [--cache keepout.npz] [--no-wrist]
       python make_5f_shells.py --model AH_Right_5F --out /tmp/shells   (fitted to one finger variant only)
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
from scipy.interpolate import PchipInterpolator
from skimage import measure

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "five_finger"))
sys.path.insert(0, str(HERE / "wrist"))
import layout as LAYOUT  # noqa: E402
import verify_5f_model as V  # noqa: E402
import wrist_design as wd  # noqa: E402
from build_5f_model import child_by_name, get_pose, parse  # noqa: E402

OUT_STL = HERE / "stl"
MODEL = "AH_Right_5F"  # hand model (folder under Demo/AHSimulation/AHSimulation) being swept
MODELS = ["AH_Right_5F", "AH_Right_5F_3S"]  # the shells clear the moving parts of all of these (2- and 3-phalanx fingers)
LO = np.array([-40.0, -142.0, -62.0])  # voxel grid bounds (plate frame, mm)
HI = np.array([128.0, 22.0, 42.0])
# plate pose in the MuJoCo root frame: part-studio frame (mm) <-> MJCF (m)
PLATE_POS = np.array([-0.0115, 0.0245, 0.1022])
PLATE_QUAT = np.array([0.5, 0.5, -0.5, -0.5])

WALL = 2.0  # shell wall thickness
CLR = 1.0  # clearance around enclosed parts
FREE = 1.2  # clearance around moving finger parts
THUMB_FREE = 2.2  # around the thumb: the left shells are the right ones mirrored, and the stock left thumb sits up to
#                   ~1 mm off the right one's mirror image
Z_SPLIT = 15.5  # plate back face: palm shell below, back shell above
LAP_H, LAP_GAP = 1.5, 0.2  # lap joint height and clearance
FLANGE_Y = -85.0  # palm-shell floor flange on the wrist interface below this y
FLANGE_T = 1.3
CAP_BELOW_KNUCKLE = 3.0  # back shell top edge this far below the knuckle (pivot) line
Z_PALM_TOP = 0.0  # below this z the top edge follows the palm cap (under the finger sweeps)
# Outer form, laid out in the wrist frame W (u = W x across the hand, + towards the pinky; v = W y up the forearm axis;
# heights in CAD z). A palm slab, blended (smooth union) with a round wrist column, a thenar and a hypothenar mound:
OUTLINE_RADIAL = [(0, -26), (42, -27), (52, -46), (62, -54), (80, -63), (94, -67), (170, -67)]  # (v, u): index / thumb edge;
OUTLINE_ULNAR = [(0, 20), (44, 22), (58, 46), (72, 62), (84, 67), (170, 67)]         # (v, u): pinky edge. Below
# v ~ 50 the slab stays inside the wrist column, so the palm waists into a round wrist neck
BACK_Z = (20.5, 30.0, (25.0, 100.0))  # back height at the wrist, at mid-hand, and the v range of the rise
ARCH = (6.0, 67.0)  # transverse arch of the back: it drops ARCH[0] mm at u = +-ARCH[1]
KNUCKLES = (1.0, 8.0, 3.5)  # knuckle bump height and radius (9 mm below each pivot); back rolled down 3.5 mm to the cap
PALM_DEPTH = (45.0, 22.0, (74.0, 118.0))  # palm surface below z = 0 at the heel, under the fingers, v range between
HOLLOW = ((26.0, 94.0), (20.0, 16.0), 3.0)  # palm hollow: centre (u, v), radii, depth
EDGE_R = ((45.0, 20.0, 22.0), (95.0, 11.0, 10.0))  # (v, back edge radius, palm edge radius): rounder at the wrist
COLUMN = ((-6.0, -11.5), 34.5, 2.4, 58.0)  # wrist column on the forearm axis: centre (u, z), radius, exponent, top v
THENAR = ((-33.0, 79.0, -24.0), 68.0, (17.0, 25.0, 38.0))  # centre, tilt in the u-z plane (along the thumb's
#                                                            base servo), half-axes (across, along v, along the tilt)
HYPOTHENAR = ((46.0, 76.0, -10.0), (16.0, 30.0, 16.0))  # centre, half-axes
BLEND = (16.0, 12.0, 14.0)  # smooth-union radii: column, thenar, hypothenar
# panel lines (grooves) on the outside, for a finished, product-like look
GROOVE_W, GROOVE_D = 1.4, 0.6
KNUCKLE_BAND = 17.0  # back: a line this far below the knuckle cap marks off a knuckle band
CUFF_LINE = 8.0  # a line around the wrist, this far above the wrist cut
WRIST_Y = -97.5  # stock bottom cut (CAD y), used with --no-wrist
# With the powered wrist the bottom cut is the plane W y = WRIST_W, square to the forearm axis. Without the palm
# extension it passes through the stock cut at the radial end of the wrist interface (CAD x = -9) and drops
# towards the ulnar side; wrist_design.PALM_EXT moves the wrist (and this cut) further from the hand.
WRIST = True
WRIST_W = float(wd.cad_to_w([-9.0, WRIST_Y, 0.0])[1]) - wd.PALM_EXT
# the cut rises towards the back of the hand and towards the palm (like the creases of a wrist), so the shells'
# bottom edges clear the fork at the hard stops: by BEVEL mm per mm of z above BEVEL_Z (extension), and by
# PALM_BEVEL mm per mm of z below PALM_BEVEL_Z (flexion; the palm's round wrist neck reaches down to z = -50)
BEVEL_Z, BEVEL = Z_SPLIT, 0.4
PALM_BEVEL_Z, PALM_BEVEL = -30.0, 0.4
ADAPTER_STL = HERE / "wrist" / "stl" / "R-Wrist_Hand_Adapter.stl"
# back-shell screws: holes in the palm plate (cad/five_finger/make_5f_plate.py) + the two into the wrist interface,
# (x, y, boss bottom z). Stock layout: the stock Top_Shell positions + the widened plate's pinky-side hole, bosses
# from the plate's back face. Parallel layout: one hole on each 5.5 mm rib between neighbouring servo pairs; the
# servos stand 5.8 mm proud of the plate's back face (to z = 21.3), so these bosses start just above them and a
# longer screw (2.5 x 16) runs through the gap between the servos (1.5 mm clear on each side) into the plate.
SERVO_BACK_Z = 21.3
SCREWS_BY_LAYOUT = {
    "stock": [(16.1, -11.7, Z_SPLIT), (47.5, -20.1, Z_SPLIT), (77.8, -32.9, Z_SPLIT)],
    "parallel": [(15.59, -26.60, SERVO_BACK_Z + 0.7), (44.84, -32.68, SERVO_BACK_Z + 0.7), (72.97, -45.86, SERVO_BACK_Z + 0.7)],
}
SCREWS = SCREWS_BY_LAYOUT[LAYOUT.LAYOUT] + [(4.0, -91.0, Z_SPLIT + FLANGE_T), (46.0, -91.0, Z_SPLIT + FLANGE_T)]
BOSS_R, HOLE_R, HEAD_R, HOLE_LEN = 3.6, 1.35, 2.5, 3.0  # stock: r1.35 hole 3 mm, r2.5 head counterbore
# grip pads (grey TPU inserts in recesses of the palm shell)
PAD_DEPTH, PAD_PROUD, PAD_GAP, PAD_TEX, PAD_PITCH = 1.0, 0.8, 0.15, 0.4, 1.8  # ribs across the palm, PAD_PITCH apart
PAD_BORDER = 5.0  # the pad keeps this even border to every edge and opening of the palm-facing surface
PAD_V_MIN = 50.0  # and ends above the wrist neck (W y)
PAD_VOX = 0.25


def log(*a):
    print(*a, flush=True)


def geom_points(m, g, h):
    """Surface samples (geom frame, m) of a mesh or primitive geom, spaced <= h (m)."""
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        return V.sample_mesh(m, m.geom_dataid[g], h)
    size = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        n = max(8, int(np.ceil(4 * np.pi * size[0] ** 2 / h ** 2)))
        p = np.random.default_rng(0).normal(size=(n, 3))
        return size[0] * p / np.linalg.norm(p, axis=1, keepdims=True)
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, hl = size[0], size[1]
        a = np.linspace(0, 2 * np.pi, max(8, int(np.ceil(2 * np.pi * r / h))), endpoint=False)
        z = np.linspace(-hl, hl, max(2, int(np.ceil(2 * hl / h)) + 1))
        side = np.stack(np.broadcast_arrays(r * np.cos(a)[:, None], r * np.sin(a)[:, None], z[None, :]), -1).reshape(-1, 3)
        rr = np.linspace(0, r, max(2, int(np.ceil(r / h)) + 1))
        disc = np.stack(np.broadcast_arrays(rr[:, None] * np.cos(a)[None, :], rr[:, None] * np.sin(a)[None, :], 0 * a[None, :]), -1).reshape(-1, 3)
        ends = [disc + [0, 0, hl], disc - [0, 0, hl]] if t == mujoco.mjtGeom.mjGEOM_CYLINDER else \
            [geom_points_sphere(r, h) + [0, 0, hl], geom_points_sphere(r, h) - [0, 0, hl]]
        return np.concatenate([side, *ends])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        axes = [np.linspace(-s_, s_, max(2, int(np.ceil(2 * s_ / h)) + 1)) for s_ in size]
        pts = []
        for i in range(3):
            j, k = [a_ for a_ in range(3) if a_ != i]
            for sgn in (-1, 1):
                grid = np.zeros((len(axes[j]), len(axes[k]), 3))
                grid[..., i] = sgn * size[i]
                grid[..., j] = axes[j][:, None]
                grid[..., k] = axes[k][None, :]
                pts.append(grid.reshape(-1, 3))
        return np.concatenate(pts)
    raise ValueError(f"geom type {t} not supported")


def geom_points_sphere(r, h):
    n = max(8, int(np.ceil(4 * np.pi * r ** 2 / h ** 2)))
    p = np.random.default_rng(0).normal(size=(n, 3))
    return r * p / np.linalg.norm(p, axis=1, keepdims=True)


# ------------------------------------------------------------------ 1. keep-out volumes
def keepout(n_grid, vox):
    """Voxelised K_enc (enclose), K_free (fingers' moving parts, stay clear) and K_thumb (the
    thumb's moving parts, stay clear) in the plate frame."""
    shape = np.ceil((HI - LO) / vox).astype(int)
    h = V.Hand(V.SIM_DIR / MODEL / "mjcf")
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
        key = (m.geom_dataid[g], m.geom_type[g], *np.round(m.geom_size[g], 6))
        if key not in cache:
            cache[key] = geom_points(m, g, vox * 0.7e-3)
        P = ((d.geom_xpos[g] + cache[key] @ d.geom_xmat[g].reshape(3, 3).T) - PLATE_POS) @ R * 1000
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


def mesh_voxels(mesh, vox, shape):
    """Solid voxels of a closed mesh (CAD mm) on the shell grid: one ray along +z per (x, y) column,
    filled between successive entry/exit hits, plus the surface voxels."""
    lo, hi = np.maximum(mesh.bounds[0], LO), np.minimum(mesh.bounds[1], HI)
    i0, i1 = np.floor((lo - LO) / vox).astype(int), np.ceil((hi - LO) / vox).astype(int)
    X, Y = [LO[k] + (np.arange(i0[k], i1[k]) + 0.5) * vox for k in (0, 1)]
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    orig = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, LO[2] - 1.0)], -1)
    locs, ray, _ = mesh.ray.intersects_location(orig, np.tile([0.0, 0.0, 1.0], (len(orig), 1)), multiple_hits=True)
    grid = np.zeros(shape, bool)
    for r in np.unique(ray):
        z = np.sort(locs[ray == r, 2])
        z = z[np.concatenate([[True], np.diff(z) > 1e-6])]  # hits on shared edges count once
        a, b = np.unravel_index(r, XX.shape)
        for z0, z1 in zip(z[0::2], z[1::2]):
            k0, k1 = int(np.floor((z0 - LO[2]) / vox)), int(np.ceil((z1 - LO[2]) / vox))
            grid[i0[0] + a, i0[1] + b, max(k0, 0):min(k1, shape[2])] = True
    return ndi.binary_dilation(grid)


# ------------------------------------------------------------------ 2.-3. fields
class Grid:
    def __init__(self, shape, vox):
        self.vox = vox
        self.X, self.Y, self.Z = [LO[i] + (np.arange(shape[i]) + 0.5) * vox for i in range(3)]
        self.XX, self.YY = np.meshgrid(self.X, self.Y, indexing="ij")
        w = wd.cad_to_w(np.stack([self.XX, self.YY, np.zeros_like(self.XX)], -1))
        self.WX, self.WY = w[..., 0], w[..., 1]  # across / along the forearm axis (powered wrist)

    def below_cut(self):
        """Signed 2D field, positive below the wrist cut (before the back bevel)."""
        return WRIST_W - self.WY if WRIST else WRIST_Y - self.YY

    def below_cut3(self):
        """Signed 3D field, positive below the wrist cut, with the back and palm bevels."""
        bevel = (np.maximum(self.Z - BEVEL_Z, 0.0) * BEVEL + np.maximum(PALM_BEVEL_Z - self.Z, 0.0) * PALM_BEVEL
                 if WRIST else 0.0 * self.Z)
        return self.below_cut()[:, :, None] + bevel[None, None, :]

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
    root = parse(V.SIM_DIR / MODEL / "mjcf" / "robot.xml").getroot().find("worldbody/body")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, PLATE_QUAT)
    piv = []
    for n in (1, 2, 3, 5):
        p = get_pose(child_by_name(root, "site", f"closing_1 ({n})_2"))[0]
        piv.append(((p - PLATE_POS) @ R.reshape(3, 3) * 1000)[:2])
    piv = np.array(piv)
    return np.polyval(np.polyfit(piv[:, 0], piv[:, 1], 2), G.X) - CAP_BELOW_KNUCKLE


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def push_out(deficit, vox):
    """Smooth field >= the positive part of deficit (8 mm wide, 3 mm soft): lifts a surface over the parts
    that stick through it without following their shapes."""
    return 1.3 * ndi.gaussian_filter(ndi.grey_dilation(np.maximum(deficit, 0), size=int(8 / vox)), 3 / vox)


def superellipsoid(d, r, p):
    """Approximate signed distance (mm) of a superellipsoid with half-axes r, at offsets d (tuple of arrays)."""
    return (sum((np.abs(di) / ri) ** p for di, ri in zip(d, r)) ** (1 / p) - 1) * min(r)


def envelope(G, K_enc):
    """Signed distance of the outer envelope (negative inside), not yet cut at the top, and the
    enclosed parts K (everything below the knuckle cap).

    The form is laid out in the wrist frame W (u across the hand, v up the forearm axis), like a hand:
    a palm slab (outline OUTLINE_RADIAL / OUTLINE_ULNAR; a domed back with a transverse arch, knuckle bumps
    and a roll-down to the knuckle line; a wedge-shaped palm, thick at the heel and thin under the fingers,
    with a hollow), blended with a round wrist column on the forearm axis, a thenar mound along the thumb's
    base servo and a hypothenar mound. It is fitted to the mechanism: the back, and the palm under the
    fingers, are lifted smoothly wherever a part needs it, and a last tight smooth union with the enclosed
    parts grown by CLR + WALL covers the few spots the primitives miss (logged)."""
    ycap = knuckle_cap(G)
    K = K_enc & ~(G.YY >= ycap[:, None])[:, :, None]
    K[G.below_cut() > 2.5] = False
    need = CLR + WALL
    U2, V2 = G.WX, G.WY
    U, V, Z = U2[:, :, None], V2[:, :, None], G.Z[None, None, :]
    has = K.any(2)
    zmax = np.where(has, G.Z[K.shape[2] - 1 - np.argmax(K[:, :, ::-1], axis=2)], -np.inf)
    zmin = np.where(has, G.Z[np.argmax(K, axis=2)], np.inf)
    # slab outline: distance to the radial / ulnar edge curves
    edge = [PchipInterpolator(*np.array(c, float).T) for c in (OUTLINE_RADIAL, OUTLINE_ULNAR)]
    d2 = smax((edge[0](V2) - U2) / np.sqrt(1 + edge[0].derivative()(V2) ** 2),
              (U2 - edge[1](V2)) / np.sqrt(1 + edge[1].derivative()(V2) ** 2), 2.0)
    # back
    zb = BACK_Z[0] + (BACK_Z[1] - BACK_Z[0]) * smoothstep(V2, *BACK_Z[2])
    zb -= ARCH[0] * np.clip(U2 / ARCH[1], -1.3, 1.3) ** 2
    for _, _, pivot in LAYOUT.transforms().values():
        w = wd.cad_to_w(pivot)
        zb += KNUCKLES[0] * np.exp(-((U2 - w[0]) ** 2 + (V2 - w[1] + 9.0) ** 2) / KNUCKLES[1] ** 2)
    zb -= KNUCKLES[2] * (1 - smoothstep(ycap[:, None] - G.YY, 0.0, 22.0))
    zb += push_out(zmax + need + 0.8 - zb, G.vox)
    # palm (depth below z = 0)
    dep = PALM_DEPTH[0] + (PALM_DEPTH[1] - PALM_DEPTH[0]) * smoothstep(V2, *PALM_DEPTH[2])
    (hu, hv), (hru, hrv), hd = HOLLOW
    dep -= hd * np.exp(-((U2 - hu) / hru) ** 2 - ((V2 - hv) / hrv) ** 2)
    dep += push_out(np.where(V2 > 95, need + 0.8 - zmin - dep, 0), G.vox)  # under the fingers
    r_edge = [(a + (b - a) * smoothstep(V2, EDGE_R[0][0], EDGE_R[1][0]))[:, :, None]
              for a, b in zip(EDGE_R[0][1:], EDGE_R[1][1:])]
    s = smax(smax(d2[:, :, None], Z - zb[:, :, None], r_edge[0]), -dep[:, :, None] - Z, r_edge[1])
    # wrist column, thenar, hypothenar
    (cu, cz), cr, cp, cv = COLUMN
    s = smin(s, smax(superellipsoid((U - cu, Z - cz), (cr, cr), cp), V - cv, 14.0), BLEND[0])
    (tu, tv, tz), tilt, tr = THENAR
    a = np.radians(tilt)
    along, across = (U - tu) * np.cos(a) + (Z - tz) * np.sin(a), -(U - tu) * np.sin(a) + (Z - tz) * np.cos(a)
    s = smin(s, superellipsoid((across, V - tv, along), tr, 2.4), BLEND[1])
    (hu, hv, hz), hr = HYPOTHENAR
    s = smin(s, superellipsoid((U - hu, V - hv, Z - hz), hr, 2.2), BLEND[2])
    s = np.maximum(s, (G.YY - ycap[:, None] - 12.0)[:, :, None])  # bounded above the cap
    dK = G.dist(K)
    short = (s - dK + need)[dK <= need].max()
    s = smin(s, ndi.gaussian_filter(dK - need - 0.8, 1.0 / G.vox), 3.0)
    log(f"envelope: the form misses the enclosed parts (+ {need} mm) by up to {short:.1f} mm before the last "
        f"blend, {max(0.0, (s - dK + need)[dK <= need].max()):.2f} mm after")
    g = np.gradient(s, G.vox)  # rescale to ~distance (|grad| = 1): wall, lap and pad offsets are in mm
    return s / np.maximum(np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2), 0.3), ycap, K


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


def openings(G, K_free, K_enc, ycap, closing=5.0, free=FREE):
    """Signed distance of the region the shell must leave free (negative inside): the moving finger
    parts, plus everything that passes through the shell's top (knuckle) edge - finger frames,
    bushings, the plate's top edge and the push rods' upper ends."""
    near_cap = (G.YY > (ycap[:, None] - 4.0))[:, :, None]
    O = G.dist(K_free | (K_enc & near_cap)) <= free
    O = ndi.binary_closing(O, structure=ndi.generate_binary_structure(3, 1), iterations=int(closing / G.vox))
    return ndi.gaussian_filter(G.sdf(O), 1.0)


# ------------------------------------------------------------------ 4. parts
def part_fields(G, s_E, s_O, s_top, K_enc_noplate):
    """Signed fields (negative inside) of the back shell and the palm shell."""
    Z, YY = G.Z[None, None, :], G.YY[:, :, None]
    s_in = ndi.gaussian_filter(G.sdf(ndi.binary_fill_holes(s_E < 0)), 0.6)
    wall = np.maximum(s_E, -(s_in + WALL))
    wall = smax(wall, s_top, 0.8)  # open top: the wall just ends at the cap curves
    wall = smax(wall, -s_O, 1.2)  # finger and thumb openings, with rounded edges
    wall = smax(wall, G.below_cut3(), 0.8)
    half = -WALL / 2
    # back: everything above the lap, plus the outer half of the wall in the lap zone
    back_sel = np.minimum((Z_SPLIT + LAP_H) - Z, np.maximum(Z_SPLIT - Z, (half + LAP_GAP / 2) - s_E))
    back = np.maximum(wall, back_sel)
    # palm: everything below the split, plus the inner half of the wall in the lap zone
    top = Z_SPLIT + LAP_H - LAP_GAP
    palm_sel = np.minimum(Z - Z_SPLIT, np.maximum(Z - top, s_E - (half - LAP_GAP / 2)))
    palm = np.maximum(wall, palm_sel)
    # palm floor flange on the wrist interface's back face, clamped by the two bottom screws
    flange = vmax(s_E - (half - LAP_GAP / 2), Z_SPLIT - Z, Z - top, YY - FLANGE_Y, G.below_cut3(), -s_O,
                  0.3 - G.dist(K_enc_noplate))
    palm = np.minimum(palm, flange)
    # screw bosses on the back shell, from the plate / flange (or just above the servos) up into the back wall,
    # kept clear of the enclosed parts
    clear = CLR - G.dist(K_enc_noplate)
    for x, y, z0 in SCREWS:
        r = np.sqrt((G.XX - x) ** 2 + (G.YY - y) ** 2)[:, :, None]
        boss = vmax(s_E, r - BOSS_R, z0 - Z, clear)
        back = np.minimum(back, boss)
    return back, palm


def panel_lines(G, s_E, ycap):
    """Signed field (negative inside) of the panel-line grooves, GROOVE_W wide and GROOVE_D deep (rounded):
    across the back KNUCKLE_BAND below the knuckle cap (down to the seam), and around the wrist CUFF_LINE
    above the wrist cut (parallel to it). They stop 2 mm short of the seam, which keeps the lap joint whole."""
    Z = G.Z[None, None, :]
    slope = np.gradient(ycap, G.vox)
    lines = [(((G.YY - ycap[:, None] + KNUCKLE_BAND) / np.sqrt(1 + slope[:, None] ** 2))[:, :, None], Z > Z_SPLIT + 2.0)]
    if WRIST:
        lines.append(((G.below_cut3() + CUFF_LINE) / np.sqrt(1 + BEVEL ** 2), np.abs(Z - Z_SPLIT) > 2.0))
    cut = np.full(s_E.shape, 1.0)
    for cd, where in lines:
        prof = GROOVE_D * np.sqrt(np.clip(1 - (2 * cd / GROOVE_W) ** 2, 0, 1)) * where
        cut = np.minimum(cut, np.maximum(np.abs(cd) - GROOVE_W / 2, -s_E - prof))
    return cut


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


def pad_outlines(G, facing, edge):
    """2D signed distance (negative inside) of the grip pad: one large pad over the palm-facing surface,
    PAD_BORDER from every edge and opening (so it runs under the fingers, down the pinky side and around the
    thumb opening with an even border) and above the wrist neck. (The thenar and hypothenar flanks face
    sideways, too steeply for a pad.)"""
    mask = facing & (edge > PAD_BORDER) & (G.WY > PAD_V_MIN)
    r = int(4 / G.vox)  # round the outline (opening with a 4 mm disc), then smooth it
    mask = ndi.binary_opening(mask, structure=np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r)
    mask = (ndi.gaussian_filter(G.sdf(mask), 2.5 / G.vox) < 0) & facing & (edge > PAD_BORDER - 1.5)
    lab, n = ndi.label(mask)
    if n > 1:  # keep the biggest piece
        mask = lab == (1 + np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
    assert mask.sum() * G.vox ** 2 > 1500, "grip pad too small"
    log(f"grip pad: {mask.sum() * G.vox ** 2 / 100:.0f} cm2 (projected)")
    return {"Palm": ndi.gaussian_filter(G.sdf(mask), 1.0)}


def pad_mesh(G, s_E, p2, name):
    """Insert for one recess, on a fine local grid: sits on the recess floor, PAD_GAP inside its
    walls, stands PAD_PROUD out of the palm with rounded grip ribs across the palm (PAD_PITCH apart)."""
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
    v = wd.cad_to_w(P)[..., 1]  # ribs along W x, across the palm
    rib = np.sqrt(np.clip(np.cos(2 * np.pi * v / PAD_PITCH), 0, 1))  # rounded crests, flat grooves
    top = PAD_PROUD + PAD_TEX * (rib - 1.0)
    f = vmax(q2 + PAD_GAP, -PAD_DEPTH - sE)
    f = smax(f, sE - top, 0.4)
    verts, faces, _, _ = measure.marching_cubes(np.pad(f, 1, constant_values=1.0), 0.011, spacing=(PAD_VOX,) * 3)
    verts += lo - PAD_VOX
    mesh = trimesh.Trimesh(verts, faces)
    mesh = max(mesh.split(only_watertight=False), key=lambda p: len(p.faces))
    v, fc = mesh.vertices, mesh.faces
    if len(fc) > 190_000:  # MuJoCo reads STLs of up to 200k faces
        v, fc = fast_simplification.simplify(v.astype(np.float32), fc, target_count=190_000)
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
        if abs(z0 - (Z_SPLIT + FLANGE_T)) < 1e-6:  # screws through the palm flange
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
    shown = path.relative_to(REPO) if path.resolve().is_relative_to(REPO) else path
    log(f"wrote {shown}: {len(tm.faces)} faces, volume {tm.volume / 1000:.1f} cm3, "
        f"bbox {np.round(tm.bounds, 1).tolist()}" + (f", {touching} float32-coincident vertices" if touching else ""))
    return tm


def main():
    global MODEL, OUT_STL, WRIST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vox", type=float, default=0.5, help="voxel size, mm")
    ap.add_argument("--grid", type=int, default=19, help="poses per motor for the swept volumes")
    ap.add_argument("--cache", type=Path, help="npz cache for the keep-out volumes (one file per model: <stem>_<model>.npz "
                    "when several models are given)")
    ap.add_argument("--model", nargs="+", default=MODELS, help="5-digit hand model(s) to fit the shells to; with several "
                    "(same palm, different fingers), the shells clear the moving parts of all of them")
    ap.add_argument("--out", type=Path, default=OUT_STL, help="output folder for the STLs")
    ap.add_argument("--no-wrist", action="store_true", help="stock wrist cut, no powered wrist (no adapter, no wrist section)")
    args = ap.parse_args()
    OUT_STL, WRIST = args.out, not args.no_wrist
    t0 = time.time()
    K_enc = K_free = K_thumb = K_plate = None
    for MODEL in args.model:
        cache = args.cache if args.cache is None or len(args.model) == 1 else \
            args.cache.with_name(f"{args.cache.stem}_{MODEL}{args.cache.suffix}")
        if cache and cache.exists():
            z = np.load(cache)
            k = z["K_enc"], z["K_free"], z["K_thumb"], z["K_plate"]
        else:
            k = (*keepout(args.grid, args.vox), None)
            k = (*k[:3], keepout_palm(args.vox, k[0].shape))
            if cache:
                np.savez_compressed(cache, K_enc=k[0], K_free=k[1], K_thumb=k[2], K_plate=k[3])
        K_enc, K_free, K_thumb, K_plate = k if K_enc is None else [a | b for a, b in zip((K_enc, K_free, K_thumb, K_plate), k)]
    MODEL = args.model[0]  # the knuckle line (the palms are the same)
    if WRIST:  # the wrist's hand adapter is rigid with the hand: enclose it / keep clear of it
        K_enc = K_enc | mesh_voxels(trimesh.load(ADAPTER_STL), args.vox, K_enc.shape)
    log(f"keep-out volumes {K_enc.shape} ({time.time() - t0:.0f} s)")
    G = Grid(K_enc.shape, args.vox)
    s_E, ycap, _ = envelope(G, K_enc)
    s_F = openings(G, K_free, K_enc, ycap)  # the four fingers
    s_O = np.minimum(s_F, openings(G, K_thumb, np.zeros_like(K_enc), ycap, closing=3.0, free=THUMB_FREE))  # + thumb
    s_top, ypalm = top_cut(G, s_E, s_F, ycap)
    back_f, palm_f = part_fields(G, s_E, s_O, s_top, K_enc & ~K_plate)
    facing, edge = palm_face(G, s_E, palm_f)
    pads = pad_outlines(G, facing, edge)
    for p2 in pads.values():  # recesses
        palm_f = np.maximum(palm_f, -vmax(p2[:, :, None], -PAD_DEPTH - s_E, G.Z[None, None, :] - Z_SPLIT))
    grooves = panel_lines(G, s_E, ycap)
    back_f, palm_f = np.maximum(back_f, -grooves), np.maximum(palm_f, -grooves)
    log(f"fields ({time.time() - t0:.0f} s)")
    back, palm = finish(to_mesh(G, back_f), to_mesh(G, palm_f))
    OUT_STL.mkdir(parents=True, exist_ok=True)
    parts = {"Top_Shell": back, "Palm_Shell": palm}
    parts.update({f"Palm_Pad_{name}": pad_mesh(G, s_E, p2, name) for name, p2 in pads.items()})
    for name, man in parts.items():
        save(man, OUT_STL / f"R-{name}_5F.stl")
        # the stock left shells are the right ones mirrored about x = -50 (like the plates)
        save(man.mirror([1, 0, 0]).translate([-100.0, 0, 0]), OUT_STL / f"L-{name}_5F.stl")
    log(f"done ({time.time() - t0:.0f} s)")


def keepout_palm(vox, shape):
    """Voxels of the palm plate and wrist interface alone (the palm flange may rest on them)."""
    h = V.Hand(V.SIM_DIR / MODEL / "mjcf")
    m, d = h.m, h.d
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, PLATE_QUAT)
    grid = np.zeros(shape, bool)
    mujoco.mj_kinematics(m, d)
    for g in h.palm:
        P = ((d.geom_xpos[g] + geom_points(m, g, vox * 0.7e-3) @ d.geom_xmat[g].reshape(3, 3).T)
             - PLATE_POS) @ R.reshape(3, 3) * 1000
        idx = np.floor((P - LO) / vox).astype(int)
        idx = idx[np.all((idx >= 0) & (idx < shape), axis=1)]
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


if __name__ == "__main__":
    main()
