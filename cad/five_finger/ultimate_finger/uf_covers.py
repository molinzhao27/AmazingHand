"""Implicit (voxel SDF) covers of the ultimate finger: rounded capsule segments, a glove-like black ribbed palm
half, a long pointed (ogive) fingertip.

Every cover is built in its phalanx frame (the straight finger; mm) on a voxel grid:

  1. Fields (cached per frame): distance to the phalanx's own core and pins (d_core), to everything that moves
     INSIDE the cover (the Link arm in P1, the coupler L2 in P2: d_mov, swept over the whole servo travel), and to
     everything else that moves past it (other phalanges, links, gimbal and hardware: d_ko, swept).
  2. Outer shape O: a barrel whose sections are superellipses (exponent 4 on the palm half, 3.2 on the back, rounder than a rounded
     rectangle; as round as the box-shaped cores allow within the width limits), fitted to contain the core + 0.25
     mm + the wall (and the inside movers + 0.6 mm + the wall), smoothed along the finger with a slightly convex
     (fuller mid-segment) back and palm line, and domed ends that converge on the joint pins (sections shrink
     towards the pin axis, so the ends read as knuckles and the two ends at a joint roll past each other).
     The fingertip is designed (ogive_tip): it tapers from the DIP into a round nose of ~4 mm radius in plan view,
     the nail side curves down and the gently convex pad side sweeps up to meet it.
  3. Material: a skin of WALL inside O, minus d_core < CLEAR, d_mov < CLEAR_MOV and the keep-out d_ko < KO (also
     the neighbouring covers, swept; widened across the finger to KO_W so no narrow tabs stand between cut-outs),
     all as signed fields combined with a smooth max, so the cut rims are rounded. The cut is applied to whole skin
     columns (column_cut: a rim is full-thickness, square to the surface, never a feathered wedge), and slivers
     thinner than 1 mm left between cuts are removed (sliver_cut).
  4. Split at a seam a little above the widest line: the back half is the white shell, the palm half (50-60 % of
     the circumference; on the fingertip it wraps over the end) is the black TPU glove, with transverse ribs
     (raised RIB_H, RIB_W wide, RIB_P pitch) on its palm-facing part; on the fingertip the ribs turn round the end.
  5. Marching cubes, decimation, manifold check.

STL only (no STEP) for these covers: they are TPU parts defined as implicit solids.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "slim_fingers"))

CLEAR = 0.25        # cover cavity to its own core
CLEAR_MOV = 0.6     # ... to parts moving inside it
WALL = 1.27         # skin (>= 1.2 after meshing)
KO = 0.72           # keep-out distance on the voxel field: >= 0.5 mm true clearance (sample + voxel quantisation)
RIB_H, RIB_W, RIB_P = 0.3, 0.55, 1.0
KO_W = 2.5          # keep-out cuts are widened across the finger to at least this (no narrow tabs between them)
COL_DEPTH = 1.2     # cut rims: the skin is removed in whole columns wherever the cut comes within this of the surface
H = 0.12            # voxel (mm)


class Grid:
    def __init__(self, lo, hi, h=H):
        self.h = float(h)
        self.lo = np.asarray(lo, float)
        self.shape = tuple(int(np.ceil((hi[i] - lo[i]) / h)) for i in range(3))
        self.ax = [self.lo[i] + (np.arange(self.shape[i]) + 0.5) * self.h for i in range(3)]

    def mark(self, pts, out):
        idx = np.floor((pts - self.lo) / self.h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
        idx = idx[ok]
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return int(ok.sum())

    def dist(self, mask):
        """Distance (mm) from every voxel centre to the nearest marked voxel centre."""
        if not mask.any():
            return np.full(self.shape, 1e3, np.float32)
        return (ndi.distance_transform_edt(~mask) * self.h).astype(np.float32)

    def sdf(self, inside):
        """Signed distance (mm, negative inside) of an occupancy mask."""
        return ((ndi.distance_transform_edt(~inside) - ndi.distance_transform_edt(inside)) * self.h).astype(np.float32)

    def mesh(self, field, level=0.0, target=150_000):
        """Largest connected component of {field < level} -> marching cubes -> decimation (falling back to lighter
        decimation until the mesh is a closed manifold)."""
        import fast_simplification
        import trimesh
        from skimage import measure
        lab, nlab = ndi.label(field < level)
        if nlab > 1:   # fragments left by the cuts: keep the main body
            sizes = ndi.sum_labels(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
            keep = 1 + int(np.argmax(sizes))
            field = np.where((lab > 0) & (lab != keep), np.maximum(field, level + 0.5 * self.h), field)
        field = ndi.gaussian_filter(field, 0.6)   # (voxels) no ambiguous marching-cubes saddles -> a manifold surface
        f = np.pad(field, 1, constant_values=max(1.0, float(field.max())))
        if not (f < level).any():
            return None
        v, fc, _, _ = measure.marching_cubes(f, level, spacing=(self.h,) * 3, allow_degenerate=False)
        v += self.lo + 0.5 * self.h - self.h
        full = trimesh.Trimesh(v, fc, process=True)
        parts = sorted(full.split(only_watertight=False), key=lambda p: -len(p.faces))
        full = parts[0]
        best = None
        for tgt in (target, int(0.8 * target), int(1.2 * target), int(0.65 * target), 190_000):   # MuJoCo: <= 200k faces
            tgt = min(tgt, 190_000)
            if len(full.faces) <= tgt:
                m = full.copy()
            else:
                vv, ff = fast_simplification.simplify(full.vertices.astype(np.float32), full.faces, target_count=tgt)
                m = trimesh.Trimesh(vv, ff, process=True)
            for attempt in range(2):
                m.update_faces(m.nondegenerate_faces())
                m.update_faces(m.unique_faces())
                m.remove_unreferenced_vertices()
                trimesh.repair.fix_normals(m)
                if m.is_watertight and m.is_volume:
                    return m
                trimesh.repair.fill_holes(m)
            if best is None:
                best = m
        return best


def smax(a, b, k):
    """Smooth maximum (rounded intersection, k mm): >= max(a, b)."""
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return a * h + b * (1 - h) + k * h * (1 - h)


def section_fit(grid, req, ys_fit, a, n, zc_fixed=None, n_top=None):
    """Per y slice: centre z and top / bottom half depths of the superellipse (half width a, exponent n) that
    contains the requirement mask (nan where the slice is empty); zc_fixed: given centre."""
    X, Z = grid.ax[0], grid.ax[2]
    zc = np.full(len(grid.ax[1]), np.nan)
    bt = np.full_like(zc, np.nan)
    bb = np.full_like(zc, np.nan)
    for j in np.flatnonzero(ys_fit):
        ii, kk = np.nonzero(req[:, j, :])
        if not len(ii):
            continue
        x, z = np.abs(X[ii]), Z[kk]
        u = np.clip(x / a, 0, 0.999) ** n
        den_b = (1 - u) ** (1 / n)
        nt = n if n_top is None else n_top
        den_t = (1 - np.clip(x / a, 0, 0.999) ** nt) ** (1 / nt)
        best = None
        for c in ([zc_fixed] if zc_fixed is not None else np.linspace(z.min(), z.max(), 41)):
            up, dn = z >= c, z < c
            t = float(np.max((z[up] - c) / den_t[up])) if up.any() else 0.0
            b = float(np.max((c - z[dn]) / den_b[dn])) if dn.any() else 0.0
            if best is None or t + b < best[0]:
                best = (t + b, c, t, b)
        zc[j], bt[j], bb[j] = best[1], best[2], best[3]
    return zc, bt, bb


def outer_field(grid, spec, zc, bt, bb, a_y=None):
    """Approximate signed distance (mm) of the styled outer solid O (smoothed). spec: a, n, body (y0, y1),
    domes: list of (y_start, y_pin, z_pin) - the sections shrink elliptically from y_start to y_pin, towards the pin
    height (the end converges on the pin); bulge: extra depth mid-segment (convex back / palm line)."""
    Y = grid.ax[1]
    y0, y1 = spec["body"]
    inb = (Y >= y0) & (Y <= y1)
    if a_y is None:
        a_y = spec["a"] * np.ones_like(Y)
    a, zc_y, bt_y, bb_y = (np.interp(Y, Y[inb], v[inb]) for v in (a_y, zc, bt, bb))   # constant beyond the body
    if "design" in spec:   # ... except beyond a designed end, which closes the cover
        a, bt_y, bb_y = (np.where(Y > y1, 0.0, v) for v in (a, bt_y, bb_y))
    # convex profile: fullest mid-segment
    t = np.clip((Y - y0) / max(y1 - y0, 1e-6), 0, 1)
    bump = np.sin(np.pi * t) * inb
    bt_y = bt_y + spec.get("bulge", 0.0) * bump
    bb_y = bb_y + spec.get("bulge_palm", spec.get("bulge", 0.0)) * bump
    s = np.ones_like(Y)
    zpin = zc_y.copy()
    for (ys, yp, zp) in spec["domes"]:
        L = yp - ys
        u = np.clip((Y - ys) / L, 0, None)
        sel = u > 0
        s[sel] = np.sqrt(np.clip(1 - u[sel] ** 2, 0, 1))
        zpin[sel] = zp
    X, Z = grid.ax[0][:, None], grid.ax[2][None, :]
    f = np.empty(grid.shape, np.float32)
    nn = np.full_like(Y, spec["n"]) if np.isscalar(spec["n"]) else spec["n"](Y)
    for j in range(len(Y)):
        n = nn[j]
        aa, tt, bbb = a[j] * s[j], bt_y[j] * s[j], bb_y[j] * s[j]
        if min(aa, tt, bbb) <= 1e-3:   # beyond a domed or designed end
            f[:, j, :] = 5.0
            continue
        zcs = zpin[j] + (zc_y[j] - zpin[j]) * s[j]
        dz = Z - zcs
        b = np.where(dz >= 0, tt, bbb)
        e = np.where(dz >= 0, spec.get("n_top", n), n) if "n_top" in spec else n
        r = ((np.abs(X) / aa) ** e + (np.abs(dz) / b) ** e) ** (1 / e)
        f[:, j, :] = (r - 1) * min(aa, tt, bbb)
    # analytic field -> distance estimate f / |grad f| (smooth: no voxel terracing), clipped far from the surface
    g = np.gradient(f, grid.h)
    gm = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    sd = np.clip(f / np.maximum(gm, 0.3), -4.0, 4.0).astype(np.float32)
    return ndi.gaussian_filter(sd, 0.15 / grid.h), dict(zc=zc_y, bt=bt_y, bb=bb_y, a=a, s=s, zpin=zpin)


def seam_field(grid, prof, spec):
    """Signed field, negative on the palm side of the seam (glove), positive on the back (shell)."""
    Z = grid.ax[2][None, None, :]
    Y = grid.ax[1]
    zs = ndi.gaussian_filter1d(prof["zc"] + spec.get("seam", 0.2) * prof["bt"], 4.0 / grid.h, mode="nearest")
    zs = (prof["zpin"] + (zs - prof["zpin"]) * prof["s"])[None, :, None]
    g = (Z - zs) * np.ones((grid.shape[0], 1, 1), np.float32)
    nail = spec.get("nail")
    if nail is not None:   # fingertip: the back shell is a nail that stops short of the end; the glove wraps the end
        X = grid.ax[0][:, None, None]
        Y = grid.ax[1][None, :, None]
        edge = nail["y"] + nail["curve"] * X ** 2
        g = np.minimum(g, (edge - Y).astype(np.float32))   # negative (palm) beyond the nail edge
    return g.astype(np.float32)


def rib_field(grid, spec, sd_O):
    """Signed field of the raised transverse ribs (negative inside a rib), before restriction to the glove."""
    Y = grid.ax[1][None, :, None]
    Z = grid.ax[2][None, None, :]
    s = np.broadcast_to(Y, grid.shape).astype(np.float32).copy()
    nose = spec.get("rib_nose")
    if nose is not None:   # around the rounded end: arc length on a circle about the nose centre
        yc, zc, R = nose
        th = np.arctan2(Y - yc, -(Z - zc))   # 0 straight down, + towards the end
        arc = (yc + R * np.clip(th, 0, None)).astype(np.float32)
        s = np.where(Y > yc, np.broadcast_to(arc, grid.shape), s)
    ph = np.mod(s - spec.get("rib_s0", 0.0), RIB_P)
    stripe = np.abs(ph - RIB_W / 2) - RIB_W / 2            # <0 inside the rib band
    shell = np.maximum(sd_O - RIB_H, -sd_O - 0.05)         # band from just inside O to RIB_H outside it
    gz = np.gradient(sd_O, grid.h, axis=2)
    facing = (gz + 0.5) * 2.0                              # palm-facing (outward normal down by > 30 deg)
    return np.maximum(np.maximum(shell, stripe), facing).astype(np.float32)


# ------------------------------------------------------------------ material and parts
def column_cut(grid, sd_O, cut, depth=COL_DEPTH, ns=11):
    """The cut extended to whole skin columns: a point of the skin (or of a rib on it) is removed when the cut reaches
    within `depth` of the outer surface above it, measured along the outer normal. Where the outer surface runs into
    the core or a keep-out at a shallow angle (the domes closing on the pins) the skin then ends in a full-thickness
    rim square to the surface instead of a feathered wedge."""
    sel = (sd_O < RIB_H + 0.15) & (sd_O > -(WALL + 0.3))
    idx = np.nonzero(sel)
    n = np.stack([np.gradient(sd_O, grid.h, axis=i)[idx] for i in range(3)], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    q = np.stack([grid.ax[i][idx[i]] for i in range(3)], 1) - sd_O[idx][:, None] * n   # foot on the outer surface
    best = cut[idx].astype(np.float32)
    for t in np.linspace(0.0, depth, ns):
        ijk = ((q - t * n - grid.lo) / grid.h - 0.5).T
        best = np.maximum(best, ndi.map_coordinates(cut, ijk, order=1, mode="nearest"))
    out = cut.copy()
    out[idx] = best
    return out


def sliver_cut(grid, f_base, r=0.5, keep=0.2):
    """Slivers left between cuts (thinner than 2 r, reaching more than `keep` beyond the solid that a ball of radius r
    can sweep): a positive field there, so they are removed. Sharp rim corners (< keep) stay."""
    M = f_base < 0
    er = ndi.distance_transform_edt(M) * grid.h > r
    opened = ndi.distance_transform_edt(~er) * grid.h <= r
    d = ndi.distance_transform_edt(~opened) * grid.h
    sl = M & (d > keep)
    if not sl.any():
        return np.full(grid.shape, -1.0, np.float32)
    return (0.15 - ndi.distance_transform_edt(~sl) * grid.h).astype(np.float32)


def cover_fields(grid, F, spec, extra_ko=None):
    """Fields of one cover: returns (field_back, field_palm, profile, sd_O); negative = material."""
    req = (F["d_core"] <= CLEAR + WALL + 0.06) | (F["d_mov"] <= CLEAR_MOV + WALL + 0.06)
    Y = grid.ax[1]
    y0, y1 = spec.get("fit", spec["body"])
    fit = (Y >= y0) & (Y <= y1)
    n_fit = spec["n"] if np.isscalar(spec["n"]) else float(spec["n"](np.array([0.5 * (y0 + y1)]))[0])
    zc, bt, bb = section_fit(grid, req, fit, spec["a"], n_fit, n_top=spec.get("n_top"))
    ok = fit & np.isfinite(zc)
    # one clean capsule section per segment: median centre, the largest extent over the fitted span
    z0 = float(np.median(zc[ok]))
    _, bt, bb = section_fit(grid, req, fit, spec["a"], n_fit, zc_fixed=z0, n_top=spec.get("n_top"))
    top = np.full_like(Y, z0 + np.nanmax(bt[ok]))
    bot = np.full_like(Y, z0 - np.nanmax(bb[ok]))
    zc = np.full_like(Y, z0)
    a_y = np.full_like(Y, spec["a"])
    if "design" in spec:   # designed fingertip beyond the neck: half width, centre, top, bottom per slice
        a_y, zc, top, bot = spec["design"](Y, a_y, zc, top, bot)
    bt, bb = top - zc, zc - bot
    sd_O, prof = outer_field(grid, spec, zc, bt, bb, a_y)
    # the outer surface is the smooth analytic one; the inner one (hidden) is cut from the exact distance transform,
    # so the skin is never thinner than WALL (the analytic distance estimate is only first order in the domes)
    d_in = (ndi.distance_transform_edt(sd_O < 0) * grid.h).astype(np.float32)
    f_inner = ndi.gaussian_filter(d_in - (WALL + 0.5 * grid.h), 0.5)   # < 0 within the wall of the outer surface
    f_skin = np.maximum(sd_O, f_inner)
    prof["f_inner"] = f_inner
    prof["d_in"] = d_in
    g = seam_field(grid, prof, spec)
    f_rib = np.maximum(rib_field(grid, spec, sd_O), g)
    f = np.minimum(f_skin, f_rib)
    cut = np.maximum(CLEAR - F["d_core"], CLEAR_MOV - F["d_mov"])
    ko = KO - F["d_ko"]
    if extra_ko is not None:
        ko = np.maximum(ko, KO - extra_ko)
    # what moves past the cover is cut out in bands at least KO_W wide across the finger: no narrow tabs of skin
    # left standing between the cut-outs of neighbouring parts (fork walls, pins, lugs)
    ko = ndi.maximum_filter1d(ko, max(1, int(round(KO_W / grid.h))), axis=0)
    cut = np.maximum(cut, ko)
    cut = column_cut(grid, sd_O, cut.astype(np.float32))
    cut = np.maximum(cut, sliver_cut(grid, smax(f_skin, cut, 0.3)))
    f = smax(f, cut, 0.3)
    prof["f_base"] = smax(f_skin, cut, 0.3)   # the skin without ribs (thin-feature check)
    return np.maximum(f, -g), np.maximum(f, g), prof, sd_O


def _hermite(t, p0, p1, m0, m1, L):
    h00, h01 = 2 * t ** 3 - 3 * t ** 2 + 1, -2 * t ** 3 + 3 * t ** 2
    h10, h11 = t ** 3 - 2 * t ** 2 + t, t ** 3 - t ** 2
    return p0 * h00 + p1 * h01 + L * (m0 * h10 + m1 * h11)


def _ogive(Y, y0, v0, m0, y_end, R, alpha, sgn=1.0, base=0.0):
    """Tangent ogive with a round nose: base + sgn * profile, the profile starting at v0 (slope m0) at y0, a cubic
    taper tangent to a nose circle of radius R that closes at y_end (junction where the circle slopes at alpha)."""
    yc = y_end - R
    yT, vT = yc + R * np.sin(alpha), R * np.cos(alpha)
    L = yT - y0
    t = np.clip((Y - y0) / L, 0, 1)
    v = _hermite(t, sgn * (v0 - base), vT, sgn * m0, -np.tan(alpha), L)
    nose = np.sqrt(np.clip(R ** 2 - (Y - yc) ** 2, 0, None))
    v = np.where(Y > yT, nose, v)
    return base + sgn * np.where(Y >= y_end, 0.0, v)


def ogive_tip(neck, y_end, z_end, R=4.0, alpha=np.radians(12), Rs=4.0, alpha_s=np.radians(18), m_w=-0.15,
              m_top=-0.10, m_bot=-0.08):
    """Designed fingertip beyond the neck, closing at y_end: in plan view the width tapers continuously (a tangent
    ogive) into a round nose of radius R; in side view the nail side curves down and the pad side, gently convex
    (it first dips by m_bot: the pulp), sweeps up to meet it at z_end in a nose of radius Rs."""
    def design(Y, a_y, zc, top, bot):
        from scipy.ndimage import gaussian_filter1d
        on = Y > neck
        a = np.where(on, _ogive(Y, neck, a_y, m_w, y_end, R, alpha), a_y)
        top_n = np.where(on, _ogive(Y, neck, top, m_top, y_end, Rs, alpha_s, 1.0, z_end), top)
        bot_n = np.where(on, _ogive(Y, neck, bot, m_bot, y_end, Rs, alpha_s, -1.0, z_end), bot)
        # round the slope change at the neck (the taper starts with a slope; the nose is left as designed)
        w = np.clip((neck + 4.0 - Y) / 2.0, 0, 1)
        sig = 1.2 / max(float(np.median(np.diff(Y))), 1e-6)
        a, top_n, bot_n = (w * gaussian_filter1d(v, sig, mode="nearest") + (1 - w) * v for v in (a, top_n, bot_n))
        return a, 0.5 * (top_n + bot_n), top_n, bot_n
    return design


def finger_specs(U):
    """Cover style per phalanx (mm, finger design frame; scale-aware through the joint positions)."""
    pip, dip, tip = float(U.PIP[0]), float(U.DIP[0]), float(U.TIP_END)
    neck, y_end, z_end = dip + 5.0, tip + 1.5, -1.8

    def n_tip(Y):   # squarer around the DIP core, rounder towards the point
        return 4.0 - 1.7 * np.clip((Y - neck) / (y_end - neck), 0, 1)

    return dict(
        P1=dict(a=8.45, n=4.0, n_top=3.2, body=(14.0, pip - 6.5), fit=(22.0, pip - 9.0), domes=[(14.0, 2.0, 0.0), (pip - 6.5, pip, 0.0)],
                bulge=0.5, bulge_palm=0.3, seam=0.15, bounds=((-9.6, 3.0, -12.8), (9.6, pip + 4.0, 15.0))),
        P2=dict(a=7.95, n=4.0, n_top=3.2, body=(pip + 6.5, dip - 6.5), fit=(pip + 7.5, dip - 12.0),
                domes=[(pip + 6.5, pip, 0.0), (dip - 6.5, dip, 0.0)], bulge=0.4, bulge_palm=0.3, seam=0.15,
                bounds=((-9.1, pip - 3.0, -11.5), (9.1, dip + 3.0, 12.0))),
        P3=dict(a=6.75, n=n_tip, body=(neck, y_end), fit=(dip + 2.0, neck), domes=[(neck, dip, 0.0)],
                seam=0.15, design=ogive_tip(neck, y_end, z_end), nail=dict(y=y_end - 5.5, curve=0.06),
                rib_nose=(y_end - 4.0, z_end, 4.0), trim_from=neck,
                bounds=((-8.6, dip - 3.0, -11.5), (8.6, y_end + 2.0, 10.5))),
    )


def skin_thickness(grid, sd_O, f_inner, f_mat):
    """Skin thickness where the cover exists: the inner (cavity) surface to the outer surface."""
    from scipy.spatial import cKDTree
    from skimage import measure
    vo, _, _, _ = measure.marching_cubes(sd_O, 0.0, spacing=(grid.h,) * 3)
    vi, _, _, _ = measure.marching_cubes(np.clip(f_inner, -2.0, 2.0), 0.0, spacing=(grid.h,) * 3)
    idx = np.clip(np.round(vi / grid.h).astype(int), 0, np.array(grid.shape) - 1)
    near = ndi.binary_dilation(f_mat < 0, iterations=2)
    vi = vi[near[idx[:, 0], idx[:, 1], idx[:, 2]]]
    if not len(vi):
        return float("nan"), float("nan"), float("nan")
    d, _ = cKDTree(vo).query(vi, workers=4)
    return float(d.min()), float(np.percentile(d, 1)), float(np.median(d))


def thin_features(grid, f_base, r=0.5):
    """Material of the skin (ribs left out) thinner than 2 r: what a morphological opening with a ball of radius r
    removes. Returns (volume mm3, largest distance of such material from the opened solid, its y range): a sharp rim
    corner gives < 0.25 mm, a feathered wedge (a cover surface meeting a cut at a shallow angle) much more."""
    M = f_base < 0
    er = ndi.distance_transform_edt(M) * grid.h > r
    opened = ndi.distance_transform_edt(~er) * grid.h <= r
    thin = M & ~opened
    if not thin.any():
        return 0.0, 0.0, None
    d = ndi.distance_transform_edt(~opened) * grid.h
    js = np.nonzero(thin.any(axis=(0, 2)))[0]
    return (float(thin.sum() * grid.h ** 3), float(d[thin].max()),
            (float(grid.ax[1][js.min()]), float(grid.ax[1][js.max()])))


def build_covers(frames, pts_of, movers, ko_of, specs, pose_rel, ks, log, names):
    """Generic cover builder. frames: order of the phalanges; pts_of[f]: own surface samples; movers[f]: list of
    (src, pts) moving inside f's cover; ko_of[f]: list of (src, pts) moving past it; pose_rel(dst, src, k) ->
    3x3 2-D transform of src into dst at pose k; the previous phalanx's cover is swept into the next one's
    keep-out. Returns {name: trimesh}, info."""
    import make_slim_fingers as SF
    out, info = {}, {}
    prev = None
    for fr in frames:
        sp = specs[fr]
        g = Grid(*sp["bounds"])
        own = np.zeros(g.shape, bool)
        g.mark(pts_of[fr], own)
        mov = np.zeros(g.shape, bool)
        ko = np.zeros(g.shape, bool)
        lo, hi = g.lo - 1.0, g.lo + np.array(g.shape) * g.h + 1.0
        for target, lst in ((mov, movers.get(fr, [])), (ko, ko_of[fr])):
            for src, p in lst:
                for k in ks:
                    q = p.copy()
                    T = pose_rel(fr, src, k)
                    q[:, 1:] = p[:, 1:] @ T[:2, :2].T + T[:2, 2]
                    q = q[np.all((q >= lo) & (q <= hi), axis=1)]
                    if len(q):
                        g.mark(q, target)
        F = dict(d_core=g.dist(own), d_mov=g.dist(mov), d_ko=g.dist(ko))
        del own, mov, ko
        if "trim_from" in sp:   # this core's tip body is trimmed to the cover's cavity: no requirement beyond
            F["d_core"] = np.where((g.ax[1] > sp["trim_from"])[None, :, None], 1e3, F["d_core"]).astype(np.float32)
        extra = None
        if prev is not None:   # the neighbouring cover, swept
            pc = np.concatenate([SF.sample_surface(out[n], 0.2) for n in names[prev]])
            m = np.zeros(g.shape, bool)
            for k in ks:
                T = pose_rel(fr, prev, k)
                q = pc.copy()
                q[:, 1:] = pc[:, 1:] @ T[:2, :2].T + T[:2, 2]
                q = q[np.all((q >= lo) & (q <= hi), axis=1)]
                if len(q):
                    g.mark(q, m)
            extra = g.dist(m)
        fb, fp, prof, sd_O = cover_fields(g, F, sp, extra)
        skin = skin_thickness(g, sd_O, prof.pop("f_inner"), np.minimum(fb, fp))
        thin = thin_features(g, prof.pop("f_base"))
        for nm, f in zip(names[fr], (fb, fp)):
            out[nm] = g.mesh(f)
        d_in = prof.pop("d_in")
        info[fr] = dict(prof=prof, skin=skin, thin=thin, grid=(g.lo.tolist(), g.h, list(g.shape)))
        if "trim_from" in sp:   # the cavity shrunk by CLEAR (+0.05): the core's tip body is trimmed to it (mesh boolean)
            info[fr]["trim_cavity"] = g.mesh(ndi.gaussian_filter(WALL + CLEAR + 0.05 + 0.5 * g.h - d_in, 0.5), target=200_000)
        del d_in
        log(f"  {fr} cover: back {len(out[names[fr][0]].faces)} / glove {len(out[names[fr][1]].faces)} faces; skin "
            f"(inner offset to outer surface) min {skin[0]:.2f}, 1st percentile {skin[1]:.2f}, median {skin[2]:.2f} mm; "
            f"thinner than 1.0 mm (opening r 0.5): {thin[0]:.1f} mm3, up to {thin[1]:.2f} mm deep")
        prev = fr
        del F, fb, fp, sd_O
    return out, info
