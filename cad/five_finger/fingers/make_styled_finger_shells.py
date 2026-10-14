#!/usr/bin/env python3
"""Styled ("more realistic") finger shells for the AmazingHand, with separate grip pads.

Every digit (index, middle, ring, pinky, thumb) uses the same finger module: a PLA Proximal and
Distal phalanx, each covered by a Filaflex 82A TPU shell held by two thermoplastic M2.5x6 screws
(one per side, through the shell's counterbored hole into the phalanx). This script restyles the
two shells without touching any surface that meets the finger kit:

  * the stock shell solids are read from cad/step and material is only ADDED, on the outer side of
    the palm (flexion) face and on the fingertip's back; nothing is cut out of the stock solid, so
    the inner surface, the screw holes and counterbores, the internal guide ribs and the open ends
    are kept (checked: every one of those faces is found unchanged in the result; on the distal
    shell the open-end ring face is the same plane with its outer edge re-split, < 1 um);
  * palm side: a raised land (a soft pad bed, LAND_H proud of the stock surface, blending back into
    it with a cosine ramp) with a pocket in it whose floor is the untouched stock outer surface:
    the recess for a grip pad. The stock wall is 1.0 mm everywhere; it is never thinned;
  * the grip pads are separate glue-in parts (grey TPU): their back is the stock outer surface,
    they are PAD_T thick (= their protrusion over the stock shell) with a 0.1 mm glue gap, and carry
    a grip texture: a fingerprint-like whorl on the fingertip pad, chevron treads on the proximal
    pad. The proximal pad and its land are tapered / kept off the palm-side corners where a
    neighbouring finger's side comes within ~0.5 mm at extreme abduction;
  * the distal shell gets a slightly raised nail plate on its back.

Frame (mm): the finger part-studio frame of the stock STEP files, x along the joint pins
(lateral), y along the finger, z towards the back of the finger. The palm (flexion) side is -z:
Demo/AHSimulation/five_finger/check_finger_shells.py flexes the finger in the MuJoCo model and
finds the distal tip moving to -z in the proximal shell's frame. Additions stay within
|x| <= 5.8 mm (proximal) / 7.7 mm (distal) of shells 23.8 / 21.5 mm wide: the sides, where
neighbouring fingers touch at extreme abduction, do not grow.

Print: shells as the stock ones (Filaflex 82A, no supports, standing); pads standing on their
flat proximal end (stl/print/*_Pad_print.stl, z = up), no supports, with a ~5 mm brim since the
footprint is small.

Writes (mm, stock part-studio frame, so they drop into the same assembly):
  step/{Proximal_Shell_Styled,Distal_Shell_Styled,Proximal_Pad,Distal_Pad}.step
  stl/ (same names) .stl, watertight, and stl/print/*_Pad_print.stl
  styled_vs_stock.png, pads.png (previews), report_cad.txt (checks)

Usage: .venv/bin/python cad/five_finger/fingers/make_styled_finger_shells.py [--no-preview] [--cache DIR]
(~5 min; --cache keeps the sampled surface patches, the slow part, for re-runs)
"""

import argparse
import time
from types import SimpleNamespace
from pathlib import Path

import cadquery as cq
import numpy as np
from matplotlib.path import Path as MPath
from OCP.Approx import Approx_ParametrizationType
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepFill import BRepFill
from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.GeomAbs import GeomAbs_C2
from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
STOCK_DIR = REPO / "cad" / "step"

# ------------------------------------------------------------------ design parameters (mm)
LAND_H = 0.45      # pad bed (land) height over the stock outer surface = recess depth
SKIN = 0.55        # land slab thickness: its inner face stays inside the 1 mm stock wall (merged)
BURY = -0.30       # land field beyond the ramp, x LAND_H: dips 0.14 mm into the stock wall
PAD_T = 0.80       # pad thickness = max protrusion over the stock outer surface
PAD_GAP = 0.10     # pad outline inset w.r.t. the recess, per side (glue clearance)
GROOVE_D = 0.30    # texture groove depth (pad keeps >= PAD_T - GROOVE_D = 0.5 mm under a groove)
GROOVE_W = 0.45    # texture groove width
TEX_BORDER = 0.70  # plain border of the pad around the texture
NAIL_H = 0.30      # nail plate height on the back of the distal shell
FIT_TOL = 2e-3     # B-spline surface fitting tolerance

PRINT_OVERHANG = 45.0  # deg from vertical, for the print-orientation check


# ------------------------------------------------------------------ 2D outlines (finger x-y plane)
def _clean(pts):
    pts = np.array(pts, float)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
    pts = pts[keep]
    return pts[:-1] if np.linalg.norm(pts[0] - pts[-1]) < 1e-6 else pts


def _arc(cx, cy, rx, ry, a0, a1, n):
    t = np.linspace(a0, a1, n)
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


def rounded_rect(x0, x1, y0, y1, r0, r1, inset=0.0, n=20):
    """Rectangle with corner radius r0 at the y0 end and r1 at the y1 end, offset inwards by inset."""
    x0, x1, y0, y1 = x0 + inset, x1 - inset, y0 + inset, y1 - inset
    r0, r1 = max(r0 - inset, 0.05), max(r1 - inset, 0.05)
    p = (_arc(x1 - r0, y0 + r0, r0, r0, -np.pi / 2, 0, n) + _arc(x1 - r1, y1 - r1, r1, r1, 0, np.pi / 2, n)
         + _arc(x0 + r1, y1 - r1, r1, r1, np.pi / 2, np.pi, n) + _arc(x0 + r0, y0 + r0, r0, r0, np.pi, 1.5 * np.pi, n))
    return _clean(p)


def rounded_polygon(corners, radii, inset=0.0, n=20):
    """Convex polygon (corners counter-clockwise) with filleted corners, offset inwards by inset."""
    P = np.array(corners, float)
    m = len(P)
    lines = []
    for i in range(m):
        d = P[(i + 1) % m] - P[i]
        d /= np.linalg.norm(d)
        lines.append((P[i] + np.array([-d[1], d[0]]) * inset, d))  # left normal = inwards
    pts = []
    for i in range(m):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        t, _ = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        c = p1 + t * d1  # (offset) corner
        r = max(radii[i] - inset, 0.05)
        phi = np.arccos(np.clip(np.dot(-d1, d2), -1, 1))  # interior angle
        tl = r / np.tan(phi / 2)
        t1, t2 = c - d1 * tl, c + d2 * tl
        ctr = t1 + np.array([-d1[1], d1[0]]) * r
        a1, a2 = np.arctan2(*(t1 - ctr)[::-1]), np.arctan2(*(t2 - ctr)[::-1])
        a2 = a2 + 2 * np.pi if a2 < a1 else a2
        pts += _arc(ctr[0], ctr[1], r, r, a1, a2, n)
    return _clean(pts)


def u_shape(w, y0, yc, ye, rc, inset=0.0, n=20):
    """Fingertip pad: straight edge at y0 (corner radius rc), parallel sides at +-w up to yc,
    half ellipse to the tip at ye; offset inwards by inset."""
    w, y0, ye, rc = w - inset, y0 + inset, ye - inset, max(rc - inset, 0.05)
    p = (_arc(w - rc, y0 + rc, rc, rc, -np.pi / 2, 0, n) + [(w, yc)]
         + _arc(0, yc, w, ye - yc, 0, np.pi, 4 * n) + _arc(-w + rc, y0 + rc, rc, rc, np.pi, 1.5 * np.pi, n))
    return _clean(p)


def sdf(poly, q):
    """Signed distance (negative inside) from points q (N, 2) to the closed polygon poly."""
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0, 1)
    d = np.linalg.norm(q[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)
    return np.where(MPath(poly).contains_points(q), -d, d)


def _resample(poly, h=0.35):
    q = np.vstack([poly, poly[:1]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    t = np.linspace(0, s[-1], int(np.ceil(s[-1] / h)), endpoint=False)
    return np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]


def prism(poly, z0=-30.0, z1=5.0, smooth=True):
    """Prism along z over a closed outline (a periodic spline through it, or the polygon itself);
    z1 = 5 keeps it clear of the back of the shells."""
    if smooth:
        wire = cq.Wire.assembleEdges([cq.Edge.makeSpline([cq.Vector(x, y, z1) for x, y in _resample(poly)], periodic=True)])
    else:
        wire = cq.Wire.makePolygon([cq.Vector(x, y, z1) for x, y in poly], close=True)
    sol = cq.Solid.extrudeLinear(cq.Face.makeFromWires(wire), cq.Vector(0, 0, z0 - z1))
    assert sol.isValid() and abs(sol.Volume()) > 0
    return sol


def ellipse_prism(ax, ay, cy, z0=-30.0, z1=5.0):
    return cq.Workplane("XY").workplane(offset=z1).center(0, cy).ellipse(ax, ay).extrude(z0 - z1).val()


# ------------------------------------------------------------------ B-rep helpers
def outer_faces(solid):
    """The outer skin of a stock shell: its B-spline faces reaching the full width of the part."""
    xmax = solid.BoundingBox().xmax

    def half_width(f):
        bb = f.BoundingBox()
        return max(abs(bb.xmin), abs(bb.xmax))
    return [f for f in solid.Faces() if f.geomType() == "BSPLINE" and half_width(f) > xmax - 0.2]


def divided(solid):
    """The same solid with its closed (periodic) faces split in two (ShapeUpgrade_ShapeDivideClosed).
    Geometrically identical; OCCT's booleans between the distal shell's periodic outer skin and the
    land take ~10 min, on the split faces ~2 s. UnifySameDomain (clean) merges the halves again."""
    sd = ShapeUpgrade_ShapeDivideClosed(solid.wrapped)
    sd.SetNbSplitPoints(1)
    sd.Perform()
    return cq.Shape.cast(sd.Result()).Solids()[0]


def mesh_volume(shape):
    """Volume from a fine tessellation (OCCT's integration is off by ~1 % on these B-spline shells)."""
    v, t = tess(shape, 0.005, 0.05)
    tri = v[t]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def ray_hit(shape, x, y, side):
    """Outermost hit of the line through (x, y) parallel to z on the palm (side=-1) or back (+1);
    returns (point, outward unit normal) or None."""
    it = BRepIntCurveSurface_Inter()
    it.Init(shape.wrapped, gp_Lin(gp_Pnt(float(x), float(y), 0.0), gp_Dir(0, 0, 1)), 1e-7)
    best = None
    while it.More():
        p = it.Pnt()
        if best is None or side * p.Z() > side * best[0][2]:
            f = it.Face()
            n = BRepLProp_SLProps(BRepAdaptor_Surface(f), it.U(), it.V(), 1, 1e-7).Normal()
            n = np.array([n.X(), n.Y(), n.Z()]) * (-1.0 if f.Orientation() == TopAbs_REVERSED else 1.0)
            best = (np.array([p.X(), p.Y(), p.Z()]), n)
        it.Next()
    return best


CACHE = None  # optional directory for the sampled patches (--cache), they take a few minutes


def sample_patch(skin, side, y0, y1, ny, nu, xcap, min_nz=0.5):
    if CACHE is not None:
        key = Path(CACHE) / f"patch_{abs(hash((round(skin.Area(), 6), side, y0, y1, ny, nu, xcap, min_nz)))}.npz"
        if key.exists():
            z = np.load(key)
            return z["P"], z["N"]
        P, N = _sample_patch(skin, side, y0, y1, ny, nu, xcap, min_nz)
        np.savez(key, P=P, N=N)
        return P, N
    return _sample_patch(skin, side, y0, y1, ny, nu, xcap, min_nz)


def _sample_patch(skin, side, y0, y1, ny, nu, xcap, min_nz):
    """Grid of points/normals on one side of the outer skin: ny rows of constant y, nu columns
    x = u * X(y), u in [-1, 1]; X(y) stays where the surface faces that side (|n_z| > min_nz)."""
    ys = np.linspace(y0, y1, ny)
    xs = []
    for y in ys:  # bisection for the edge of the usable (side-facing) surface, x >= 0 (shells are symmetric)
        ok = lambda x: (h := ray_hit(skin, x, y, side)) is not None and side * h[1][2] > min_nz  # noqa: E731
        lo, hi = 0.0, xcap + 0.5
        assert ok(lo), f"no {side} surface at y={y}"
        if ok(hi):
            lo = hi
        else:
            for _ in range(14):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        xs.append(lo)
    xs = np.minimum(np.array(xs), xcap + 0.3)
    xh = gaussian_filter1d(minimum_filter1d(xs, 9, mode="nearest"), 2.5, mode="nearest") - 0.3  # smooth grid lines
    assert np.all(xh <= xs - 0.05), "patch half-width"
    P = np.zeros((nu, ny, 3))
    N = np.zeros_like(P)
    for j, (y, xhj) in enumerate(zip(ys, xh)):
        for i, u in enumerate(np.linspace(-1, 1, nu)):
            h = ray_hit(skin, u * xhj, y, side)
            assert h is not None and side * h[1][2] > 0.2
            P[i, j], N[i, j] = h
    return P, N


def fit_face(grid):
    """Cubic B-spline face approximating a (nu, nv, 3) point grid (higher degrees ring)."""
    a = TColgp_Array2OfPnt(1, grid.shape[0], 1, grid.shape[1])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            a.SetValue(i + 1, j + 1, gp_Pnt(*map(float, grid[i, j])))
    s = GeomAPI_PointsToBSplineSurface(a, Approx_ParametrizationType.Approx_ChordLength, 3, 3, GeomAbs_C2, FIT_TOL).Surface()
    return cq.Face(BRepBuilderAPI_MakeFace(s, 1e-6).Face())


def solid_between(outer, inner):
    """Closed solid bounded by two fitted grid surfaces (same grid topology) and 4 ruled sides."""
    fo, fi = fit_face(outer), fit_face(inner)
    faces, ei = [fo, fi], fi.Edges()
    for e in fo.Edges():
        ej = min(ei, key=lambda k: (k.Center() - e.Center()).Length)
        if (e.startPoint() - ej.startPoint()).Length > (e.startPoint() - ej.endPoint()).Length:
            ej = cq.Edge(ej.wrapped.Reversed())
        faces.append(cq.Face(BRepFill.Face_s(e.wrapped, ej.wrapped)))
    sew = BRepBuilderAPI_Sewing(1e-4)
    for f in faces:
        sew.Add(f.wrapped)
    sew.Perform()
    so = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(sew.SewedShape())).Solid()
    sol = cq.Solid(so)
    if sol.Volume() < 0:
        sol = cq.Solid(so.Reversed())
    assert sol.isValid() and sol.Volume() > 0
    return sol


def ramp_field(P, outline, rim, ramp, xtaper=None):
    """1 on the outline grown by rim, cosine down to BURY over `ramp`, BURY beyond; optionally
    also faded to BURY between |x| = xtaper[0] and xtaper[1] (keeps the land off the sides)."""
    s = sdf(outline, P[..., :2].reshape(-1, 2)).reshape(P.shape[:2])
    t = np.clip((s - rim) / ramp, 0, 1)
    w = np.where(s <= rim, 1.0, BURY + (1 - BURY) * 0.5 * (1 + np.cos(np.pi * t)))
    if xtaper is not None:
        tx = np.clip((np.abs(P[..., 0]) - xtaper[0]) / (xtaper[1] - xtaper[0]), 0, 1)
        w = BURY + (w - BURY) * 0.5 * (1 + np.cos(np.pi * tx))
    border = np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]])
    assert np.allclose(border, BURY), "land must be buried along the patch border"
    return w


def land_solid(P, N, outline, rim, ramp, height, wall, xtaper=None):
    """Raised land: outer surface = stock + height * w along the stock normal, inner surface SKIN
    below it; the inner surface must stay inside the stock wall (it merges with the shell)."""
    d = height * ramp_field(P, outline, rim, ramp, xtaper)
    inner = d - SKIN
    assert inner.max() < -0.05 and (inner > -(wall - 0.15)).all(), "land slab must stay inside the wall"
    return solid_between(P + N * d[..., None], P + N * inner[..., None])


def tess(shape, tol=0.01, ang=0.1):
    """Triangulate the whole shape at once (old per-face triangulations dropped first, otherwise
    faces meshed at different times do not share their edge nodes and the mesh has cracks)."""
    BRepTools.Clean_s(shape.wrapped)
    vs, ts = shape.tessellate(tol, ang)
    return np.array([v.toTuple() for v in vs]), np.array(ts)


def area_samples(shapes, h):
    """Surface samples roughly every h mm on a list of faces."""
    out = []
    for f in shapes:
        v, t = tess(f, 0.005, 0.05)
        BRepTools.Clean_s(f.wrapped)
        if len(t) == 0:
            continue
        tri = v[t]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = np.maximum(1, np.ceil(area / (h * h) * 2).astype(int))
        rng = np.random.default_rng(0)
        idx = np.repeat(np.arange(len(tri)), n)
        r = rng.random((len(idx), 2))
        r = np.where(r.sum(1, keepdims=True) > 1, 1 - r, r)
        out.append(tri[idx, 0] + r[:, :1] * (tri[idx, 1] - tri[idx, 0]) + r[:, 1:] * (tri[idx, 2] - tri[idx, 0]))
        out.append(v)
    return np.concatenate(out)


def export_stl(shape, path, tol=0.01, ang=0.1):
    import trimesh
    v, t = tess(shape, tol, ang)
    m = trimesh.Trimesh(v, t)
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0, f"{path}: mesh not watertight"
    m.export(str(path))
    BRepTools.Clean_s(shape.wrapped)
    return m


def same_face(f, g, tol=1e-5):
    return (f.geomType() == g.geomType() and abs(f.Area() - g.Area()) < tol * max(1.0, f.Area())
            and (f.Center() - g.Center()).Length < tol * 10)


# ------------------------------------------------------------------ part definitions
PARTS = {
    "Proximal": dict(
        stock="Amazing Hand Parts - Proximal_Shell.step",
        patch=dict(y0=16.4, y1=45.0, ny=96, nu=63, xcap=9.2),
        # tapered: narrower at its proximal end, where the neighbouring finger's side comes within
        # ~0.5 mm when two fingers are abducted towards each other (see check_finger_shells.py)
        pad=lambda inset=0.0: rounded_polygon([(-3.9, 20.0), (3.9, 20.0), (5.2, 41.5), (-5.2, 41.5)], [1.0, 1.0, 2.8, 2.8], inset),
        rim=0.8, ramp=1.6, xtaper=(4.8, 6.3),
        texture="chevron",
        tex=dict(y_first=21.6, y_last=40.6, pitch=1.45, slope=0.30),
    ),
    "Distal": dict(
        stock="Amazing Hand Parts - Distal_Shell.step",
        divide=True,
        patch=dict(y0=62.6, y1=91.2, ny=97, nu=49, xcap=9.0),
        pad=lambda inset=0.0: u_shape(5.6, 66.0, 76.5, 87.5, 1.0, inset),
        rim=0.9, ramp=1.8,
        texture="whorl",
        tex=dict(cy=78.6, ratio=0.62, r0=0.75, pitch=1.2, n=10),
        nail=dict(patch=dict(y0=77.6, y1=91.0, ny=68, nu=41, xcap=6.5),
                  outline=rounded_rect(-3.9, 3.9, 80.2, 89.0, 1.9, 2.7), ramp=1.2),
    ),
}


def texture_cutter(part, outline_inset):
    """Union of the groove prisms, clipped to the textured area of the pad."""
    t = part["tex"]
    zone = prism(part["pad"](outline_inset))
    cutters = []
    if part["texture"] == "whorl":  # concentric ovals around the fingertip's 'core', like a fingerprint
        for k in range(t["n"]):
            ay = t["r0"] + k * t["pitch"]
            ring = ellipse_prism(t["ratio"] * ay + GROOVE_W, ay + GROOVE_W, t["cy"]).cut(ellipse_prism(t["ratio"] * ay, ay, t["cy"]))
            cutters.append(ring)
    else:  # chevron treads, apex towards the fingertip
        X = 6.0
        for y in np.arange(t["y_first"], t["y_last"] + 1e-6, t["pitch"]):
            c = t["slope"] * X
            pts = [(-X, y - c), (0, y), (X, y - c), (X, y - c + GROOVE_W), (0, y + GROOVE_W), (-X, y - c + GROOVE_W)]
            cutters.append(prism(np.array(pts), smooth=False))
    cut = cutters[0]
    for c in cutters[1:]:
        cut = cut.fuse(c)
    return cut.intersect(zone)


def build(name, part, log):
    t0 = time.time()
    stock = cq.importers.importStep(str(STOCK_DIR / part["stock"])).solids().vals()[0]
    skin_faces = outer_faces(stock)
    skin = cq.Compound.makeCompound(skin_faces)
    inner_faces = [f for f in stock.Faces() if not any(f.isSame(g) for g in skin_faces)]
    skin_edges = [e for f in skin_faces for e in f.Edges()]
    # inner surface for wall-thickness checks: exclude the end faces / counterbores that meet the skin
    wall_faces = [f for f in inner_faces if not any(e.isSame(s) for e in f.Edges() for s in skin_edges)]
    inner_tree = cKDTree(area_samples(wall_faces, 0.08))
    log(f"== {name} shell: stock {part['stock']}  valid {stock.isValid()}  volume {stock.Volume():.1f} mm3, "
        f"{len(stock.Faces())} faces ({len(skin_faces)} outer skin)")

    # palm-side patch and its local stock wall thickness
    P, N = sample_patch(skin, -1, **part["patch"])
    wall = inner_tree.query(P.reshape(-1, 3))[0].reshape(P.shape[:2])
    log(f"  palm patch: {P.shape[0]}x{P.shape[1]} points, y {P[..., 1].min():.1f}..{P[..., 1].max():.1f}, "
        f"|x| <= {np.abs(P[..., 0]).max():.1f}; stock wall there {wall.min():.2f}..{wall.max():.2f} mm "
        f"(median {np.median(wall):.2f}) [{time.time() - t0:.0f} s]")

    outline = part["pad"]()
    land = land_solid(P, N, outline, part["rim"], part["ramp"], LAND_H, wall, part.get("xtaper"))
    land = land.cut(prism(outline))  # the pad recess: its floor is the untouched stock surface
    work = divided(stock) if part.get("divide") else stock
    styled = work.fuse(land)

    if "nail" in part:
        nl = part["nail"]
        Pn, Nn = sample_patch(skin, +1, **nl["patch"])
        walln = inner_tree.query(Pn.reshape(-1, 3))[0].reshape(Pn.shape[:2])
        styled = styled.fuse(land_solid(Pn, Nn, nl["outline"], 0.0, nl["ramp"], NAIL_H, walln))
        log(f"  nail patch: back side y {Pn[..., 1].min():.1f}..{Pn[..., 1].max():.1f}, stock wall {walln.min():.2f}..{walln.max():.2f} mm")
    styled = styled.clean()
    if styled.ShapeType() != "Solid":
        styled = styled.Solids()[0] if len(styled.Solids()) == 1 else styled
    log(f"  styled shell: valid {styled.isValid()}  solids {len(styled.Solids())}  volume {mesh_volume(styled):.1f} mm3 "
        f"(stock {mesh_volume(stock):.1f})  faces {len(styled.Faces())} [{time.time() - t0:.0f} s]")

    # grip pad: back = stock outer surface, PAD_T thick, glue gap all round, textured top
    pad = solid_between(P + N * PAD_T, P).intersect(prism(part["pad"](PAD_GAP)))
    pad = pad.Solids()[0]
    top = max(pad.Faces(), key=lambda f: -f.Center().z if f.geomType() == "BSPLINE" else -1e9)
    try:
        pad = pad.fillet(0.25, top.Edges())
    except Exception as e:  # noqa: BLE001
        log(f"  (pad edge fillet skipped: {e})")
    groove_slab = solid_between(P + N * (PAD_T + 0.5), P + N * (PAD_T - GROOVE_D))
    pad = pad.cut(groove_slab.intersect(texture_cutter(part, PAD_GAP + TEX_BORDER))).clean()
    pad = pad.Solids()[0] if len(pad.Solids()) == 1 else pad
    log(f"  {name} pad: valid {pad.isValid()}  solids {len(pad.Solids())}  volume {mesh_volume(pad):.1f} mm3  "
        f"faces {len(pad.Faces())} [{time.time() - t0:.0f} s]")
    return dict(stock=stock, work=work, styled=styled, pad=pad, skin=skin_faces, inner=inner_faces, wall_faces=wall_faces,
                inner_tree=inner_tree, P=P, N=N)


# ------------------------------------------------------------------ checks
def check(name, r, part, log):
    stock, styled, pad = r["stock"], r["styled"], r["pad"]
    log(f"== checks: {name}")
    # 1. nothing removed from the stock solid, every non-skin face kept identically
    work = r["work"]  # = stock, or stock with its closed faces split (same geometry, fast booleans)
    if work is not stock:
        same = sum(any(same_face(f, g) for g in work.Faces()) for f in stock.Faces())
        log(f"  boolean copy of the stock with closed faces split: {len(work.Faces())} faces, {same}/{len(stock.Faces())} "
            f"stock faces unchanged, volume {mesh_volume(work):.3f} vs {mesh_volume(stock):.3f} mm3")
    removed = work.cut(styled)
    removed = mesh_volume(removed) if removed.Solids() else 0.0
    sf = styled.Faces()
    kept = [any(same_face(f, g) for g in sf) for f in r["inner"]]
    log(f"  material removed from the stock shell: {removed:.2e} mm3; stock faces other than the outer skin "
        f"found unchanged in the styled shell: {sum(kept)}/{len(r['inner'])}")
    for f, k in zip(r["inner"], kept):
        if not k:
            near = min((g for g in sf if g.geomType() == f.geomType()), key=lambda g: (g.Center() - f.Center()).Length)
            log(f"    not bit-identical: {f.geomType()} face at {tuple(round(c, 2) for c in f.Center().toTuple())}, area "
                f"{f.Area():.4f} -> {near.Area():.4f} mm2, centre moved {(near.Center() - f.Center()).Length * 1e3:.2f} um "
                f"(same surface, boundary edge re-split by the closed-face split)")
    assert removed < 1e-3 and sum(kept) >= len(r["inner"]) - 1
    # 2. what was added, and where
    added = styled.cut(work)
    va, vs0 = tess(added, 0.005, 0.05)[0], tess(stock, 0.01, 0.1)[0]
    bb = SimpleNamespace(xmin=va[:, 0].min(), xmax=va[:, 0].max(), ymin=va[:, 1].min(), ymax=va[:, 1].max(),
                         zmin=va[:, 2].min(), zmax=va[:, 2].max())
    sb = SimpleNamespace(xmin=vs0[:, 0].min(), xmax=vs0[:, 0].max())
    log(f"  added material: {mesh_volume(added):.1f} mm3 in x {bb.xmin:.2f}..{bb.xmax:.2f} (stock shell {sb.xmin:.2f}..{sb.xmax:.2f}), "
        f"y {bb.ymin:.2f}..{bb.ymax:.2f}, z {bb.zmin:.2f}..{bb.zmax:.2f}")
    vs, vy = tess(stock, 0.005, 0.05)[0], tess(styled, 0.005, 0.05)[0]
    log(f"  extent (fine mesh): stock x {vs[:, 0].min():.3f}..{vs[:, 0].max():.3f}, z {vs[:, 2].min():.3f}..{vs[:, 2].max():.3f}; "
        f"styled x {vy[:, 0].min():.3f}..{vy[:, 0].max():.3f}, z {vy[:, 2].min():.3f}..{vy[:, 2].max():.3f}")
    # 3. protrusion over the stock outer surface (normal distance), shell and pad
    skin_tree = cKDTree(area_samples(r["skin"], 0.05))
    skin_comp = cq.Compound.makeCompound(r["skin"])

    def exact_max(pts, d, k=40):  # refine the largest sampled distances on the B-rep
        top = np.argsort(d)[-k:]
        return max(BRepExtrema_DistShapeShape(cq.Vertex.makeVertex(*p).wrapped, skin_comp.wrapped).Value() for p in pts[top])
    new_faces = [f for f in sf if not any(same_face(f, g, tol=1e-3) for g in r["inner"])]
    pts_shell = area_samples(new_faces, 0.1)
    d_shell = skin_tree.query(pts_shell)[0]
    pts_pad = area_samples(pad.Faces(), 0.1)
    d_pad = skin_tree.query(pts_pad)[0]
    prot_shell, prot_pad = exact_max(pts_shell, d_shell), exact_max(pts_pad, d_pad)
    log(f"  max distance from the stock outer surface: styled shell {prot_shell:.3f} mm, pad {prot_pad:.3f} mm "
        f"(pad back on the stock surface: {np.percentile(d_pad, 1):.3f} mm at the 1st percentile)")
    assert prot_shell < max(LAND_H, NAIL_H) + 0.03 and prot_pad < PAD_T + 0.03, "surface fit overshoots"
    # 4. wall thickness (distance to the stock inner surface) where the outer surface changed
    th = r["inner_tree"].query(pts_shell)[0]
    stock_th = r["inner_tree"].query(area_samples(r["skin"], 0.25))[0]
    grown = d_shell > 0.10
    in_patch = (pts_shell[:, 1] > r["P"][..., 1].min()) & (pts_shell[:, 1] < r["P"][..., 1].max()) & (pts_shell[:, 2] < 0)
    rec = MPath(part["pad"]()).contains_points(pts_shell[:, :2]) & in_patch & ~grown
    log(f"  wall thickness (to the stock inner surface): stock skin min {stock_th.min():.2f} / median {np.median(stock_th):.2f} mm; "
        f"styled, where >= 0.1 mm was added: min {th[grown].min():.2f} mm; recess floor (untouched stock wall) "
        f"min {th[rec].min():.2f} / median {np.median(th[rec]):.2f} mm; whole styled skin min {th.min():.2f} mm")
    # 5. pad thickness
    log(f"  pad thickness: {PAD_T:.2f} mm, {PAD_T - GROOVE_D:.2f} mm under the texture grooves; pad proud of the land by "
        f"{PAD_T - LAND_H:.2f} mm; recess depth {LAND_H:.2f} mm, glue gap {PAD_GAP:.2f} mm per side")
    return dict(prot_shell=prot_shell, prot_pad=prot_pad, wall_min=th[grown].min(), wall_rec=th[rec].min(),
                stock_wall=stock_th.min(), added_x=max(abs(bb.xmin), abs(bb.xmax)))


def print_orientation(name, pad, log):
    """Pads print standing on their straight proximal end (finger +y -> printer +z), like the stock
    shells, which also print standing: no supports, the texture is in the layer outlines."""
    v, t = tess(pad, 0.01, 0.1)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # finger y -> print z, finger z -> print -y
    vp = v @ R.T
    vp[:, 2] -= vp[:, 2].min()
    tri = vp[t]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2 * area[:, None], 1e-12)
    bed = (tri[..., 2].max(1) < 0.02)
    down = (nrm[:, 2] < -np.sin(np.radians(90 - PRINT_OVERHANG))) & ~bed
    # overhangs are only a problem if they are wide: report the steepest downward strip heights
    log(f"  {name} pad print orientation: standing on its proximal end ({area[bed].sum():.1f} mm2 on the bed), "
        f"{vp[:, 2].max():.1f} mm tall; down-facing area steeper than {PRINT_OVERHANG:.0f} deg from vertical: "
        f"{area[down].sum():.2f} mm2 of {area.sum():.0f} mm2 (groove ceilings, {GROOVE_D:.2f} mm deep ledges)")
    return vp, t


def preview(results, out_png, pads_png):
    import mujoco
    import trimesh
    from PIL import Image, ImageDraw

    def scene(meshes, size=(420, 560)):
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth, spec.visual.global_.offheight = size[1], size[0]
        spec.visual.headlight.ambient = [0.38, 0.38, 0.38]
        spec.visual.headlight.diffuse = [0.42, 0.42, 0.42]
        spec.visual.headlight.specular = [0.15, 0.15, 0.15]
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[1, 1, 1], rgb2=[0.86, 0.88, 0.9], width=64, height=64)
        light = spec.worldbody.add_light()
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.dir, light.diffuse, light.specular = [0.5, 0.6, -0.6], [0.25, 0.25, 0.25], [0.15, 0.15, 0.15]
        for k, (v, f, rgba, off) in enumerate(meshes):
            spec.add_mesh(name=f"m{k}", uservert=(v + off).ravel().tolist(), userface=f.ravel().tolist())
            g = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{k}", rgba=list(rgba))
            g.pos = [0, 0, 0]
        m = spec.compile()
        for k in range(m.nmat):
            pass
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d, mujoco.Renderer(m, size[0], size[1])

    def shot(m, d, r, lookat, az, el, dist):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = lookat, az, el, dist
        r.update_scene(d, cam)
        return r.render().copy()

    WHITE, GREY = (0.93, 0.93, 0.92, 1), (0.30, 0.31, 0.33, 1)
    views = [("palm side", 90, 75), ("side", 0, 8), ("back", 90, -75), ("3/4 palm", 35, 35)]
    rows = []
    for name, r in results.items():
        bb = r["stock"].BoundingBox()
        look = np.array([0, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2])
        for variant in ("stock", "styled"):
            if variant == "stock":
                mesh = [(*tess(r["stock"], 0.01, 0.1), WHITE, np.zeros(3))]
            else:
                mesh = [(*tess(r["styled"], 0.01, 0.1), WHITE, np.zeros(3)), (*tess(r["pad"], 0.01, 0.1), GREY, np.zeros(3))]
            m, d, rend = scene(mesh, size=(360, 480))
            tiles = []
            for label, az, el in views:
                img = Image.fromarray(shot(m, d, rend, look, az, el, 75))
                ImageDraw.Draw(img).text((8, 6), f"{name} shell, {variant}{' + grip pad' if variant == 'styled' else ''}: {label}", fill=(0, 0, 0))
                tiles.append(np.asarray(img))
            rend.close()
            rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out_png)

    # pads alone + exploded view of the pad above its recess
    tiles = []
    for name, r in results.items():
        vy, fy = tess(r["styled"], 0.01, 0.1)
        vp, fp = tess(r["pad"], 0.01, 0.1)
        m, d, rend = scene([(vy, fy, WHITE, np.zeros(3)), (vp, fp, GREY, np.array([0, 0, -9.0]))])
        bb = r["pad"].BoundingBox()
        look = np.array([0, (bb.ymin + bb.ymax) / 2, bb.zmin - 4])
        for label, az, el, dist in [("exploded", 30, 30, 70), ("pad close-up", 90, 89, 40)]:
            img = Image.fromarray(shot(m, d, rend, look if label == "exploded" else np.array([0, look[1], bb.zmin - 9]), az, el, dist))
            ImageDraw.Draw(img).text((8, 6), f"{name} pad: {label}", fill=(0, 0, 0))
            tiles.append(np.asarray(img))
        rend.close()
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)).save(pads_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--cache", type=Path, default=None, help="directory to cache the sampled surface patches in")
    args = ap.parse_args()
    global CACHE
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
        CACHE = args.cache
    (HERE / "step").mkdir(exist_ok=True)
    (HERE / "stl" / "print").mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"design: land {LAND_H} mm, pad {PAD_T} mm (glue gap {PAD_GAP}), texture grooves {GROOVE_W} x {GROOVE_D} mm, "
        f"nail plate {NAIL_H} mm; palm side = -z (found by flexing the finger in the MuJoCo model)")
    results, summary = {}, {}
    for name, part in PARTS.items():
        r = build(name, part, log)
        results[name] = r
        summary[name] = check(name, r, part, log)
        for label, shape in ((f"{name}_Shell_Styled", r["styled"]), (f"{name}_Pad", r["pad"])):
            cq.exporters.export(cq.Workplane().add(shape), str(HERE / "step" / f"{label}.step"))
            m = export_stl(shape, HERE / "stl" / f"{label}.stl")
            log(f"  wrote {label}: STEP + STL ({len(m.faces)} triangles, watertight)")
        vp, t = print_orientation(name, r["pad"], log)
        import trimesh
        trimesh.Trimesh(vp, t).export(str(HERE / "stl" / "print" / f"{name}_Pad_print.stl"))
    log("== summary")
    for name, s in summary.items():
        log(f"  {name}: pad protrudes {s['prot_pad']:.2f} mm, land {s['prot_shell']:.2f} mm over the stock surface; "
            f"additions within |x| <= {s['added_x']:.2f} mm; min wall where material was added {s['wall_min']:.2f} mm "
            f"(recess floor = untouched stock wall, min {s['wall_rec']:.2f}; stock skin min {s['stock_wall']:.2f})")
    (HERE / "report_cad.txt").write_text("\n".join(lines) + "\n")
    if not args.no_preview:
        preview(results, HERE / "styled_vs_stock.png", HERE / "pads.png")
        print("wrote previews")


if __name__ == "__main__":
    main()
