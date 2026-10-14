#!/usr/bin/env python3
"""Slim finger parts for the AmazingHand (5-digit variant, every digit uses the same finger module).

The stock finger is 23.8 mm wide over the proximal shell and 21.5 mm over the distal shell, around
PLA cores that are only 16.2 mm and 10.0 mm wide: the Filaflex shells stand off the cores by 2.5 mm
(proximal) and 4.5 mm (distal) on each side. This script keeps the whole kinematic kit (pins, hinge
axes, hole positions, link, gimbal, rods, servo drive, screw-on shell attachment) and

  * (a) replaces both TPU shells by slim lofted shells that hug the cores: 0.25 mm sliding clearance
        to the core, 1.2 mm wall; the soft cushion on the palm side is kept;
  * (b) trims non-functional PLA from the cores: a shallow spot face around each shell screw hole
        (the shell's screw cup sits in it so the screw head ends flush with the slim surface) and the
        square corners of the distal core's tip (rounded in top view so the fingertip can be round);
  * shapes the joint ends of both shells from the actual 4-bar kinematics: the proximal shell ends
        and the distal shell begins at planes (flat print faces) that clear everything the other
        phalanx, the link and the gimbal sweep through over the whole joint range (+-4 deg margin),
        so the PIP and MCP knuckles are exposed, Shadow-hand style;
  * adds a grip pad on the palm side of the fingertip: a separate 1 mm textured TPU insert glued onto the
        plain 1.2 mm shell (and a one-piece shell + pad body for single-material printers). A recessed
        (flush) pad was tried and dropped: on the curved palm wall a recess + inner thickening leaves
        either a 0.4-0.6 mm joggle or knife-edge lips.

Screws: the slim shells need M2.5 x 5 thermoplastic screws instead of M2.5 x 6 (an M2.5 x 6 cannot be
recessed in a 1.2 mm wall: its tip would hit the link in the proximal and the opposite screw in the
distal). Same holes, same head (d 4.2 x 1.55 mm).

Frame (mm): the finger part-studio frame of the stock STEP files: x along the pins (lateral),
y along the finger (MCP pin at y = 0, PIP pin at y = 52), z towards the back of the finger
(the palm/flexion side is -z). All outputs are in this frame, so they drop into the stock assembly
and into the MuJoCo model (Demo/AHSimulation/slim_fingers/build_slim_model.py).

Writes to cad/five_finger/slim_fingers/:
  step/{Proximal_Slim,Distal_Slim,Proximal_Shell_Slim,Distal_Shell_Slim,Distal_Pad_Slim_untextured}.step
  stl/  {Proximal_Slim,...,Distal_Shell_Slim,Distal_Pad_Slim (textured)}.stl + Distal_Shell_Slim_OnePiece.stl
  report_cad.txt (checks), design.json (planes, parameters, for the sim/verification scripts)

Usage: .venv/bin/python cad/five_finger/slim_fingers/make_slim_fingers.py
"""

import json
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import shapely
import trimesh
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.gp import gp_Pnt
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Polygon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STOCK_STEP = REPO / "cad" / "step"
STOCK_STL = REPO / "cad" / "stl"
# vendor hardware (pins, screws, washer, ball joints) as exported in the MuJoCo model, metres
HW_DIR = REPO / "Demo" / "AHSimulation" / "AHSimulation" / "AH_Right_5F" / "mjcf" / "assets"

# ------------------------------------------------------------------ design parameters (mm)
CLEAR = 0.25          # TPU shell inner surface to PLA core (sliding fit)
ROUND_R, ROUND_EXTRA = 0.8, 0.08   # inner section corners rounded to 0.8 mm (see round_corners)
WALL = 1.2            # TPU wall (minimum)
WALL_LOFT = WALL + 0.04   # outer offset: the lofts deviate from the exact sections by < 0.03 between stations
CORE_HALF_P = 8.1     # proximal core half width (= 16 mm pins + 0.1 per side, unchanged)
CORE_HALF_D = 5.0     # distal core half width (unchanged)
PROX_HALF = CORE_HALF_P + CLEAR + 0.08 + WALL   # 9.63: slim proximal shell half width
DIST_HALF = 7.35      # slim distal shell half width (set by the recessed M2.5x5 head, see SCREW)

PROX_ZM = 3.0         # proximal section: height of the widest point
PROX_CROWN = 0.35     # extra dorsal rise of the inner surface above the core top (rounder back)
PROX_IN_BOTTOM = -7.6  # palm cushion: inner bottom of the proximal shell (stock -8.8)
DIST_ZM = 1.14        # distal section: widest at the screw axis height
DIST_CROWN = 0.2
DIST_CUSHION = 3.4    # palm cushion: air gap under the distal core (stock 4.6)
TIP_Y0, TIP_Y = 82.0, 93.0  # distal outer: sections shrink elliptically from TIP_Y0 to the apex at TIP_Y
TIP_YIN = 91.75       # inner cavity apex (1.25 mm nose wall)
TIP_Z = -0.6          # apex height
CORE_TIP_Y = 84.0    # beyond this the distal core tip is trimmed to the shell cavity shrunk by CLEAR (its square corners)

# screw: M2.5 x 5 thermoplastic, head d 4.2 x 1.55 (as the stock M2.5x6 in the model)
SCREW = dict(head_d=4.2, head_h=1.55, shank=5.05, recess=0.15)
CUP_IN_D, CUP_OUT_D, POCKET_D = 4.6, 7.0, 7.2   # TPU cup (head recess; outer: 1.2 mm cup wall), PLA spot face
CUP_FLOOR = 1.2       # TPU under the screw head
HOLE_D = 2.6          # screw clearance hole in the cup floor (stock 2.5)
PROX_SCREW = (30.0, 4.0)        # (y, z) of the shell screw axis, proximal (stock hole)
DIST_SCREW = (74.442, 1.139)    # distal

# joints
JOINT_CLEAR = 0.5     # min clearance of a shell to anything sweeping past it (3-D, whole range)
PHI_RANGE = (-30.2 - 4.0, 75.4 + 4.0)   # MCP flexion range measured in the sim (+-90 deg motors) + margin
# MCP end of the proximal shell: the stock cylindrical cut (r 36.257 about (y 44.26, z 12.5), keeps the
# palm side clear of the link and rods) and a dorsal plane y >= MCP_Y0 + MCP_S * (z - 9.5) that clears the
# gimbal washer/screw, the finger frame and the plate at full hyperextension (from the swept point clouds
# of the sim, Demo/AHSimulation/slim_fingers/verify_slim.py checks it with 0.6 mm margin).
MCP_CYL = (44.26, 12.5, 36.257)
MCP_Y0, MCP_S = 12.6, 0.30

# pad (fingertip grip insert, glued on: back = the shell's outer surface)
PAD_T = 1.0           # pad thickness (stands this much proud of the shell)
PAD_GROOVE_D, PAD_GROOVE_W = 0.35, 0.5
PAD_OUTLINE = dict(y0=63.5, y1=89.5, half=3.6)   # top-view outline (rounded towards the tip); |x| <= 3.6 keeps the
# recess rim on the flatter part of the palm curve (wider pads leave a knife-edge lip at the rim)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ small geometry helpers
def superellipse(a, zm, bt, bb, nt, nb, n=240):
    """Closed curve in the (x, z) plane: half width a at height zm, top zm+bt, bottom zm-bb."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = np.where(s >= 0, nt, nb)
    x = a * np.sign(c) * np.abs(c) ** (2 / e)
    z = zm + np.where(s >= 0, bt, bb) * np.sign(s) * np.abs(s) ** (2 / e)
    return Polygon(np.c_[x, z])


def ray_sample(poly, center, n=96):
    """n boundary points of a star-shaped polygon, on rays at fixed angles from center (consistent
    parametrisation from section to section, so the loft is smooth)."""
    out = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        d = np.array([np.cos(th), np.sin(th)])
        hit = LineString([center, center + 80 * d]).intersection(poly.exterior)
        pts = [hit] if hit.geom_type == "Point" else list(hit.geoms)
        far = max(pts, key=lambda q: np.hypot(q.x - center[0], q.y - center[1]))
        out.append((far.x, far.y))
    return np.array(out)


def section_wire(y, pts):
    return cq.Wire.assembleEdges([cq.Edge.makeSpline([cq.Vector(x, y, z) for x, z in pts], periodic=True)])


def loft(wires, apex=None):
    b = BRepOffsetAPI_ThruSections(True, False, 1e-6)
    for w in wires:
        b.AddWire(w.wrapped)
    if apex is not None:
        b.AddVertex(BRepBuilderAPI_MakeVertex(gp_Pnt(*apex)).Vertex())
    b.CheckCompatibility(False)
    b.Build()
    s = cq.Solid(b.Shape())
    assert s.isValid(), "loft failed"
    return s


def x_cylinder(y, z, r, x0, x1):
    """Cylinder along x from x0 to x1 through (y, z)."""
    lo, hi = min(x0, x1), max(x0, x1)
    return cq.Solid.makeCylinder(r, hi - lo, cq.Vector(lo, y, z), cq.Vector(1, 0, 0))


def yz_prism(poly_yz, half=30.0):
    """Extrude a polygon given in the (y, z) plane along x over [-half, half]."""
    pts = [tuple(p) for p in np.asarray(poly_yz.exterior.coords)[:-1]]
    return cq.Workplane("YZ").workplane(offset=-half).polyline(pts).close().extrude(2 * half).val()


def half_plane_yz(a, b, keep_below, lo=(-40, -40), hi=(140, 40)):
    """Region y <= a + b z (keep_below) or y >= a + b z, clipped to a big box, as a (y, z) polygon."""
    z0, z1 = lo[1], hi[1]
    y_at = lambda z: a + b * z  # noqa: E731
    if keep_below:
        pts = [(lo[0], z0), (y_at(z0), z0), (y_at(z1), z1), (lo[0], z1)]
    else:
        pts = [(y_at(z0), z0), (hi[0], z0), (hi[0], z1), (y_at(z1), z1)]
    return Polygon(pts)


def solid_of(shape):
    s = shape.val() if isinstance(shape, cq.Workplane) else shape
    sols = s.Solids() if hasattr(s, "Solids") else [s]
    assert len(sols) == 1, f"expected one solid, got {len(sols)}"
    return sols[0]


def to_trimesh(shape, tol=0.02):
    v, f = shape.tessellate(tol, 0.15)
    m = trimesh.Trimesh(np.array([(p.x, p.y, p.z) for p in v]), np.array(f), process=True)
    m.merge_vertices(digits_vertex=5)
    trimesh.repair.fix_normals(m)
    return m


def sample_surface(mesh, h):
    """Deterministic surface samples (vertices, edges every h, triangle interiors)."""
    v, faces = mesh.vertices, mesh.faces
    tri = v[faces]
    pts = [v]
    e = mesh.edges_unique
    a, b = v[e[:, 0]], v[e[:, 1]]
    n = np.ceil(np.linalg.norm(b - a, axis=1) / h).astype(int)
    for k in np.unique(n[n > 1]):
        s = np.arange(1, k)[None, :, None] / k
        sel = n == k
        pts.append((a[sel][:, None] + s * (b[sel] - a[sel])[:, None]).reshape(-1, 3))
    lengths = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    area = mesh.area_faces
    inr = 2 * area / np.maximum(lengths.sum(1), 1e-12)
    kk = np.where(inr > h / 2, np.ceil(lengths.max(1) / h), 0).astype(int)
    for k in np.unique(kk[kk > 2]):
        ij = np.array([(i, j) for i in range(1, k) for j in range(1, k - i)], float) / k
        w = np.column_stack([1 - ij.sum(1), ij])
        pts.append(np.einsum("bk,tkd->tbd", w, tri[kk == k]).reshape(-1, 3))
    return np.concatenate(pts)


def section_props(poly):
    """Area, centroid z, second moments about the centroid (Ixx: bending in the flexion plane,
    Izz: lateral bending) of a (x, z) polygon with holes."""
    def ring(c):
        c = np.asarray(c)
        x, z = c[:-1, 0], c[:-1, 1]
        x1, z1 = np.roll(x, -1), np.roll(z, -1)
        cr = x * z1 - x1 * z
        A = cr.sum() / 2
        Sx = ((x + x1) * cr).sum() / 6
        Sz = ((z + z1) * cr).sum() / 6
        Ixx = ((z * z + z * z1 + z1 * z1) * cr).sum() / 12   # int z^2
        Izz = ((x * x + x * x1 + x1 * x1) * cr).sum() / 12   # int x^2
        return A, Sx, Sz, Ixx, Izz
    tot = np.zeros(5)
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    for p in polys:
        p = shapely.geometry.polygon.orient(p, 1.0)
        tot += ring(p.exterior.coords)
        for h in p.interiors:
            tot += ring(h.coords)  # oriented clockwise -> negative
    A, Sx, Sz, Ixx, Izz = tot
    xc, zc = Sx / A, Sz / A
    return dict(area=A, zc=zc, Ixx=Ixx - A * zc * zc, Izz=Izz - A * xc * xc)


# ------------------------------------------------------------------ 4-bar kinematics (planar, y-z)
# ground = gimbal: MCP pin O = (0, 0), link pin A = (0, -6); proximal O->P (P0 = (52, 0)); distal pin P,
# link-distal pin B (B0 = (49.216, 5.315)). Rotation sign: + = flexion (+y towards -z).
O2, A2, P0, B0 = np.zeros(2), np.array([0.0, -6.0]), np.array([52.0, 0.0]), np.array([49.216, 5.315])
L_PB, L_AB = np.linalg.norm(B0 - P0), np.linalg.norm(B0 - A2)


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


def fourbar(phis_deg):
    """Body poses (R, t) in the gimbal frame for each MCP angle: dict name -> list of (R2, t2)."""
    out = {}
    for seq in (sorted(p for p in phis_deg if p >= 0), sorted((p for p in phis_deg if p < 0), reverse=True)):
        B = B0
        for phi in seq:
            P = rot2(np.radians(phi)) @ P0
            dvec = A2 - P
            dd = np.linalg.norm(dvec)
            a = (L_PB ** 2 - L_AB ** 2 + dd ** 2) / (2 * dd)
            hh = np.sqrt(max(L_PB ** 2 - a ** 2, 0.0))
            m = P + a * dvec / dd
            perp = np.array([-dvec[1], dvec[0]]) / dd
            B = min((m + hh * perp, m - hh * perp), key=lambda c: np.linalg.norm(c - B))
            ang = lambda v: np.arctan2(v[1], v[0])  # noqa: E731
            dist = -(ang(B - P) - ang(B0 - P0))
            link = -(ang(B - A2) - ang(B0 - A2))
            Rp, Rd, Rl = rot2(np.radians(phi)), rot2(dist), rot2(link)
            out[phi] = {"gimbal": (np.eye(2), np.zeros(2)), "prox": (Rp, np.zeros(2)),
                        "dist": (Rd, P - Rd @ P0), "link": (Rl, A2 - Rl @ A2)}
    return [out[k] for k in sorted(out)]


def rel_pose(Ta, Tb):
    """Pose of body b in the frame of body a."""
    (Ra, ta), (Rb, tb) = Ta, Tb
    return Ra.T @ Rb, Ra.T @ (tb - ta)


def apply2(pts, R2, t2):
    out = pts.copy()
    out[:, 1:] = pts[:, 1:] @ R2.T + t2
    return out


# ------------------------------------------------------------------ stock geometry
def load_step(name):
    return solid_of(cq.importers.importStep(str(STOCK_STEP / f"Amazing Hand Parts - {name}.step")))


def load_stl(name):
    return trimesh.load(STOCK_STL / f"Amazing Hand Parts - {name}.stl")


def hw_mesh(name, origin, zaxis):
    """Vendor hardware mesh (MJCF asset, metres) placed in a part frame: mesh z along zaxis."""
    m = trimesh.load(HW_DIR / f"{name}.stl")
    m.apply_scale(1000.0)
    z = np.array(zaxis, float)
    ref = np.array([0, 1, 0]) if abs(z[1]) < 0.9 else np.array([1, 0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = origin
    m.apply_transform(T)
    return m


PIN16, PIN10 = "parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b", "parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999"
SCREW25 = "std00447_thermoplastique_m2_5x6__configuration_default"


def screw_mesh(origin, zaxis):
    """M2.5x5: the model's M2.5x6 with the shank shortened by 1 mm (head kept)."""
    m = trimesh.load(HW_DIR / f"{SCREW25}.stl")
    m.apply_scale(1000.0)
    v = m.vertices.copy()
    shank = v[:, 2] > SCREW["head_h"] + 0.05
    v[shank, 2] = SCREW["head_h"] + (v[shank, 2] - SCREW["head_h"]) * (SCREW["shank"] / (7.6 - SCREW["head_h"]))
    m = trimesh.Trimesh(v, m.faces, process=False)
    z = np.array(zaxis, float)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = origin
    m.apply_transform(T)
    return m


def hardware(prox_screw_x, dist_screw_x):
    """Hardware meshes per moving body, in that body's part frame (positions from the stock MJCF)."""
    return {
        "prox": [hw_mesh(PIN16, (0.0, 0.0, 0.0), (-1, 0, 0)), hw_mesh(PIN16, (0.0, 52.0, 0.0), (-1, 0, 0)),
                 screw_mesh((prox_screw_x, *PROX_SCREW), (-1, 0, 0)), screw_mesh((-prox_screw_x, *PROX_SCREW), (1, 0, 0))],
        "dist": [hw_mesh(PIN10, (0.0, 49.216, 5.315), (-1, 0, 0)),
                 screw_mesh((dist_screw_x, *DIST_SCREW), (-1, 0, 0)), screw_mesh((-dist_screw_x, *DIST_SCREW), (1, 0, 0))],
        "gimbal": [hw_mesh("std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default", (0, 0, 15.7), (0, 0, -1)),
                   hw_mesh("plain_washer_large_grade_a_m2_5__9a369f0dc77bf9c598cdf3fb468977e5", (0, 0, 13.8), (0, 0, 1)),
                   hw_mesh(PIN10, (0.0, 0.0, -6.0), (-1, 0, 0))],
        "link": [hw_mesh("rotule_ball", (s * 8.05, 3.596, -11.051), (-s, 0, 0)) for s in (-1, 1)]
        + [hw_mesh("ph_pan_head_screw_m2x0_40_x_10__2803432263e518bbd16bccbbef8784ed", (s * 12.55, 3.596, -11.051), (s, 0, 0))
           for s in (-1, 1)],
    }


# ------------------------------------------------------------------ cores
class Cores:
    def __init__(self):
        self.prox_stl, self.dist_stl = load_stl("Proximal"), load_stl("Distal")

    def hull(self, which, y):
        m = self.prox_stl if which == "prox" else self.dist_stl
        s = m.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
        return MultiPoint(s.vertices[:, [0, 2]]).convex_hull


def make_cores(cores):
    prox, dist = load_step("Proximal"), load_step("Distal")
    # spot faces for the shell screw cups, both sides
    pfloor = PROX_HALF - SCREW["recess"] - SCREW["head_h"] - CUP_FLOOR
    dfloor = DIST_HALF - SCREW["recess"] - SCREW["head_h"] - CUP_FLOOR
    pr = cq.Workplane().add(prox)
    for s in (-1, 1):
        pr = pr.cut(x_cylinder(*PROX_SCREW, POCKET_D / 2, s * pfloor, s * (CORE_HALF_P + 2)))
    dr = cq.Workplane().add(dist)
    for s in (-1, 1):
        dr = dr.cut(x_cylinder(*DIST_SCREW, POCKET_D / 2, s * dfloor, s * (CORE_HALF_D + 2)))
    # trim the square tip corners of the distal core to the slim shell's nose cavity shrunk by CLEAR
    tipbox = cq.Solid.makeBox(20, 20, 30, cq.Vector(-10, CORE_TIP_Y, -15))
    cavity = dist_body(cores, -CLEAR, TIP_YIN - CLEAR)
    dr = dr.cut(cq.Workplane().add(tipbox).cut(cq.Workplane().add(cavity)))
    return solid_of(pr), solid_of(dr), pfloor, dfloor


# ------------------------------------------------------------------ shell sections
def round_corners(poly):
    """Round the convex corners of a section to radius ROUND_R (morphological opening). The core's
    corners are 135 deg chamfers, so this cuts at most ROUND_R * 0.082 into the offset core, which
    ROUND_EXTRA covers: the clearance to the core stays >= CLEAR. Smooth inner corners keep the lofted
    wall uniform (sharp corners make the section splines overshoot) and round the outside."""
    return poly.buffer(-ROUND_R, quad_segs=16).buffer(ROUND_R, quad_segs=16)


def prox_inner(cores, y):
    yy = float(np.clip(y, 8.0, 46.0))
    h = cores.hull("prox", yy)
    top = h.bounds[3] + CLEAR + PROX_CROWN
    se = superellipse(CORE_HALF_P + CLEAR + ROUND_EXTRA, PROX_ZM, top - PROX_ZM, PROX_ZM - PROX_IN_BOTTOM, 2.6, 2.1)
    return round_corners(shapely.union_all([h.buffer(CLEAR + ROUND_EXTRA, quad_segs=16), se]).convex_hull)


def dist_inner(cores, y):
    yy = float(np.clip(y, 56.0, TIP_Y0))
    h = cores.hull("dist", yy)
    top = h.bounds[3] + CLEAR + DIST_CROWN
    bot = h.bounds[1] - DIST_CUSHION
    se = superellipse(DIST_HALF - WALL, DIST_ZM, top - DIST_ZM, DIST_ZM - bot, 2.8, 2.1)
    return round_corners(shapely.union_all([h.buffer(CLEAR + ROUND_EXTRA, quad_segs=16), se]).convex_hull)


PROX_YS = [4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
DIST_YS = [53.0, 55.5, 58.0, 61.0, 64.0, 67.0, 70.0, 73.0, 76.0, 79.0, TIP_Y0]


def prox_body(cores, off):
    """Proximal shell 'body' bounded by the inner surface offset by off (off=0: cavity, WALL: outer)."""
    ws = [section_wire(y, ray_sample(prox_inner(cores, y).buffer(off + 0.03, quad_segs=16), np.array([0.0, PROX_ZM])))
          for y in PROX_YS]
    return loft(ws)


def dist_body(cores, off, apex_y):
    """Distal shell body bounded by the inner surface offset by off, closed by an elliptic nose with
    its apex at (0, apex_y, TIP_Z)."""
    ws = [section_wire(y, ray_sample(dist_inner(cores, y).buffer(off + 0.03, quad_segs=16), np.array([0.0, DIST_ZM])))
          for y in DIST_YS]
    base = ray_sample(dist_inner(cores, TIP_Y0).buffer(off + 0.03, quad_segs=16), np.array([0.0, DIST_ZM]))
    L = apex_y - TIP_Y0
    for f in (0.25, 0.5, 0.7, 0.85, 0.95):
        y = TIP_Y0 + f * L
        s = np.sqrt(1 - f * f)
        p = np.c_[base[:, 0] * s, TIP_Z + (base[:, 1] - TIP_Z) * s]
        ws.append(section_wire(y, p))
    return loft(ws, apex=(0.0, apex_y, TIP_Z))


# ------------------------------------------------------------------ sweeps (3-D, planar relative motion)
class Sweeper:
    """Surface samples of each moving body of the finger in its part frame; moves them through the
    4-bar range into the frame of another body."""

    def __init__(self, bodies, spacing=0.3, step=0.5):
        self.pts = {k: np.concatenate([sample_surface(m, spacing) for m in ms]) if ms else np.zeros((0, 3)) for k, ms in bodies.items()}
        phis = np.arange(PHI_RANGE[0], PHI_RANGE[1] + 1e-9, step)
        self.poses = fourbar(phis)

    def swept(self, frame, body):
        out = []
        for T in self.poses:
            R2, t2 = rel_pose(T[frame], T[body])
            out.append(apply2(self.pts[body], R2, t2))
        return np.concatenate(out)


def near_points(shell_mesh, pts, dist, spacing=0.2):
    """Points within dist of the shell surface (inside the 1.2 mm wall counts too: <= 0.6 from a surface)."""
    tree = cKDTree(sample_surface(shell_mesh, spacing))
    lo, hi = shell_mesh.bounds[0] - dist - 1, shell_mesh.bounds[1] + dist + 1
    sel = np.all((pts >= lo) & (pts <= hi), axis=1)
    q = pts[sel]
    d, _ = tree.query(q, distance_upper_bound=dist, workers=-1)
    return q[np.isfinite(d)]


def wall_thickness(mesh, spacing=0.12, reach=2.5, n_query=15000):
    """Local wall thickness at n_query surface points: distance to the nearest surface sample (every
    `spacing` mm) whose normal faces the opposite way (n . n' < -0.85: across a wall, not a rim corner)."""
    m = mesh.copy()
    trimesh.repair.fix_normals(m)
    pts, fid = trimesh.sample.sample_surface_even(m, int(m.area / spacing ** 2), seed=1)
    nrm = m.face_normals[fid]
    tree = cKDTree(pts)
    q = np.random.default_rng(2).choice(len(pts), min(n_query, len(pts)), replace=False)
    th = np.full(len(q), np.inf)
    for k, nb in enumerate(tree.query_ball_point(pts[q], reach, workers=-1)):
        nb = np.asarray(nb)
        # opposite-facing AND behind the point (inside the material), not across an air gap
        opp = nb[(nrm[nb] @ nrm[q[k]] < -0.85) & ((pts[nb] - pts[q[k]]) @ nrm[q[k]] < 0)]
        if len(opp):
            th[k] = np.linalg.norm(pts[opp] - pts[q[k]], axis=1).min()
    return th, pts[q]


def brep_section_wall(solid, y, n_in=2500, n_out=6000):
    """Exact in-plane wall thickness of a tube-like solid at y: section the B-rep (OCC), sample the two
    closed section curves densely, min distance inner -> outer. None unless the section is two loops."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.gp import gp_Dir, gp_Pln
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    sec = BRepAlgoAPI_Section(solid.wrapped, gp_Pln(gp_Pnt(0, y, 0), gp_Dir(0, 1, 0)), False)
    sec.Build()
    ex, loops = TopExp_Explorer(sec.Shape(), TopAbs_EDGE), []
    while ex.More():
        d = GCPnts_QuasiUniformDeflection(BRepAdaptor_Curve(TopoDS.Edge_s(ex.Current())), 0.001)
        pts = np.array([(d.Value(k).X(), d.Value(k).Z()) for k in range(1, d.NbPoints() + 1)])
        loops.append(pts)
        ex.Next()
    if len(loops) != 2 or any(np.linalg.norm(lp[0] - lp[-1]) > 1e-3 for lp in loops):
        return None
    loops.sort(key=lambda lp: -Polygon(lp).area)
    def dense(lp, n):
        seg = np.linalg.norm(np.diff(lp, axis=0), axis=1)
        t = np.r_[0, np.cumsum(seg)]
        u = np.linspace(0, t[-1], n)
        return np.c_[np.interp(u, t, lp[:, 0]), np.interp(u, t, lp[:, 1])]
    d, _ = cKDTree(dense(loops[0], n_out)).query(dense(loops[1], n_in))
    return float(d.min())


def min_distance(shell_mesh, pts, cutoff=3.0):
    """Smallest distance (capped at cutoff) between shell surface samples and pts."""
    lo, hi = shell_mesh.bounds[0] - cutoff, shell_mesh.bounds[1] + cutoff
    q = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if not len(q):
        return np.inf
    tree = cKDTree(sample_surface(shell_mesh, 0.2))
    d, _ = tree.query(q, distance_upper_bound=cutoff, workers=-1)
    return float(d.min())


# ------------------------------------------------------------------ main build
def build():
    t0 = time.time()
    rep = []
    cores = Cores()
    prox_core, dist_core, pfloor, dfloor = make_cores(cores)
    log(f"cores done ({time.time() - t0:.1f} s): pocket floors at |x| = {pfloor:.2f} (proximal), {dfloor:.2f} (distal)")
    prox_screw_x = PROX_HALF - SCREW["recess"]    # head top (origin of the screw mesh)
    dist_screw_x = DIST_HALF - SCREW["recess"]

    # ---------------- proximal shell
    p_out, p_in = prox_body(cores, WALL_LOFT), prox_body(cores, 0.0)
    shell = cq.Workplane().add(p_out).cut(cq.Workplane().add(p_in))
    for s in (-1, 1):
        cup = cq.Workplane().add(x_cylinder(*PROX_SCREW, CUP_OUT_D / 2, s * pfloor, s * (PROX_HALF + 2))).intersect(cq.Workplane().add(p_out))
        shell = shell.union(cup)
    for s in (-1, 1):
        shell = shell.cut(x_cylinder(*PROX_SCREW, CUP_IN_D / 2, s * (pfloor + CUP_FLOOR), s * (PROX_HALF + 3)))
        shell = shell.cut(x_cylinder(*PROX_SCREW, HOLE_D / 2, s * (pfloor - 1), s * (PROX_HALF + 3)))
    # MCP end: stock cylinder + dorsal plane
    cy, cz, cr = MCP_CYL
    shell = shell.intersect(x_cylinder(cy, cz, cr, -30, 30))
    shell = shell.intersect(yz_prism(half_plane_yz(MCP_Y0 - MCP_S * 9.5, MCP_S, keep_below=False)))
    p_shell_uncut = solid_of(shell)
    log(f"proximal shell body done ({time.time() - t0:.1f} s)")

    # ---------------- distal shell + pad (uncut base)
    d_out = dist_body(cores, WALL_LOFT, TIP_Y)
    d_in = dist_body(cores, 0.0, TIP_YIN)
    shell = cq.Workplane().add(d_out).cut(cq.Workplane().add(d_in))
    for s in (-1, 1):
        cup = cq.Workplane().add(x_cylinder(*DIST_SCREW, CUP_OUT_D / 2, s * dfloor, s * (DIST_HALF + 2))).intersect(cq.Workplane().add(d_out))
        shell = shell.union(cup)
    for s in (-1, 1):
        shell = shell.cut(x_cylinder(*DIST_SCREW, CUP_IN_D / 2, s * (dfloor + CUP_FLOOR), s * (DIST_HALF + 3)))
        shell = shell.cut(x_cylinder(*DIST_SCREW, HOLE_D / 2, s * (dfloor - 1), s * (DIST_HALF + 3)))
    # pad: a proud, textured insert glued onto the palm side of the fingertip (the shell itself stays a plain
    # 1.2 mm wall); footprint in top view, palm side only
    z_cap = -4.6   # footprint prism reaches up to here (below the core): palm side only
    prism = cq.Workplane("XY").workplane(offset=-20).polyline([tuple(p) for p in np.asarray(pad_outline(0.0).exterior.coords)[:-1]]).close().extrude(20 + z_cap).val()
    d_pad = dist_body(cores, WALL_LOFT + PAD_T, TIP_Y + PAD_T)     # pad top surface
    pad_smooth = cq.Workplane().add(d_pad).cut(cq.Workplane().add(d_out)).intersect(prism)
    pad = pad_texture(pad_smooth, cores)
    d_onepiece_uncut = solid_of(shell.union(pad))
    log(f"distal shell body + pad done ({time.time() - t0:.1f} s)")

    # ---------------- joint ends: two planes (flat print faces), chosen together
    # proximal shell keeps y <= a + b z, distal shell keeps y >= c + e z. Constraints, over the whole
    # MCP range (+-4 deg), clearance JOINT_CLEAR: proximal shell vs swept distal core/pin/screws, link, gimbal;
    # distal shell (+pad) vs swept proximal core/pins/screws, link, gimbal; and the two shells vs each
    # other (a proximal-shell point only counts if it survives the proximal plane).
    hw = hardware(prox_screw_x, dist_screw_x)
    gim, lnk = load_stl("Gimbal"), load_stl("Link")
    sw = Sweeper({"dist": [to_trimesh(dist_core, 0.02)] + hw["dist"], "link": [lnk] + hw["link"],
                  "gimbal": [gim] + hw["gimbal"], "prox": [to_trimesh(prox_core, 0.02)] + hw["prox"]})
    log(f"sweeper ready ({time.time() - t0:.1f} s): {len(sw.poses)} poses over MCP {PHI_RANGE[0]:.1f}..{PHI_RANGE[1]:.1f} deg")
    pm, dm = to_trimesh(p_shell_uncut, 0.02), to_trimesh(d_onepiece_uncut, 0.02)
    Qp = np.concatenate([near_points(pm, sw.swept("prox", b), JOINT_CLEAR) for b in ("dist", "link", "gimbal")])
    Qd = np.concatenate([near_points(dm, sw.swept("dist", b), JOINT_CLEAR) for b in ("prox", "link", "gimbal")])
    rep.append(f"swept parts near the uncut shells away from the PIP (must be 0): proximal shell y < 30: {int((Qp[:, 1] <= 30).sum())}, "
               f"distal shell y > 68: {int((Qd[:, 1] >= 68).sum())}")
    Qp, Qd = Qp[Qp[:, 1] > 30], Qd[Qd[:, 1] < 68]
    ps = sample_surface(pm, 0.3)
    ps = ps[ps[:, 1] > 35]
    tree = cKDTree(sample_surface(dm, 0.2))
    tagged = []
    for T in sw.poses:
        q = apply2(ps, *rel_pose(T["dist"], T["prox"]))
        d, _ = tree.query(q, distance_upper_bound=JOINT_CLEAR)
        hit = np.isfinite(d)
        tagged.append(np.c_[q[hit], ps[hit]])
    tagged = np.concatenate(tagged)
    best = None
    for b in np.arange(-0.2, 0.201, 0.05):          # end faces close to perpendicular to the finger
        # every swept point near the uncut shell must stay JOINT_CLEAR away from the cut face
        amax = float(np.min(Qp[:, 1] - b * Qp[:, 2])) - JOINT_CLEAR * np.hypot(1, b)
        for a in np.arange(amax, amax - 6.0, -0.25):
            alive = tagged[:, 4] <= a + b * tagged[:, 5]
            yz = np.r_[Qd[:, 1:3], tagged[alive][:, 1:3]]
            for e in np.arange(-0.3, 0.301, 0.05):
                c = float(np.max(yz[:, 0] - e * yz[:, 1])) + JOINT_CLEAR * np.hypot(1, e)
                gap = (c + 2 * e) - (a + 2 * b)   # knuckle gap at mid height (z = 2)
                key = (round(gap, 3), -a)
                if best is None or key < best[0]:
                    best = (key, a, b, c, e)
    _, pa, pb, da, db = best
    rep.append(f"proximal shell PIP end plane: y = {pa:.3f} + {pb:.3f} z  (y = {pa + pb * 12:.2f} at z 12, "
               f"{pa:.2f} at z 0, {pa - pb * 8:.2f} at z -8; stock: y = 49.33 + 0.155 z)")
    rep.append(f"distal shell base plane:      y = {da:.3f} + {db:.3f} z  (y = {da + db * 8:.2f} at z 8, {da:.2f} at z 0, "
               f"{da - db * 8:.2f} at z -8; stock: y = 56.48 - 0.247 z)")
    prox_shell = solid_of(cq.Workplane().add(p_shell_uncut).intersect(yz_prism(half_plane_yz(pa, pb, keep_below=True))))
    keep = yz_prism(half_plane_yz(da, db, keep_below=False))
    dist_shell = solid_of(shell.intersect(keep))
    dist_onepiece = solid_of(cq.Workplane().add(d_onepiece_uncut).intersect(keep))
    pad = solid_of(pad.intersect(keep))
    pad_smooth = solid_of(pad_smooth.intersect(keep))
    log(f"shells cut ({time.time() - t0:.1f} s)")

    design = dict(prox_end_plane=[pa, pb], dist_base_plane=[da, db], prox_screw_head_x=prox_screw_x,
                  dist_screw_head_x=dist_screw_x, screw="M2.5x5 thermoplastic", clear=CLEAR, wall=WALL,
                  prox_half=PROX_HALF, dist_half=DIST_HALF, joint_clear=JOINT_CLEAR, phi_range=PHI_RANGE,
                  mcp_cyl=MCP_CYL, mcp_dorsal_plane=[MCP_Y0, MCP_S])
    parts = dict(Proximal_Slim=prox_core, Distal_Slim=dist_core, Proximal_Shell_Slim=prox_shell,
                 Distal_Shell_Slim=dist_shell, Distal_Pad_Slim=pad, Distal_Shell_Slim_OnePiece=dist_onepiece,
                 Distal_Pad_Slim_untextured=pad_smooth)
    return parts, design, rep, sw, hw


def pad_outline(inset):
    """Top-view (x, y) outline of the fingertip pad: straight sides, elliptic end towards the tip."""
    y0, y1, w = PAD_OUTLINE["y0"], PAD_OUTLINE["y1"], PAD_OUTLINE["half"]
    yc = y1 - 9.0
    t = np.linspace(0, np.pi, 60)
    arc = np.c_[w * np.cos(t), yc + (y1 - yc) * np.sin(t)]
    pts = np.r_[[[w, y0]], arc, [[-w, y0]]]
    poly = Polygon(pts)
    r = 1.5
    poly = poly.buffer(-r).buffer(r)  # round the base corners
    return poly.buffer(-inset, quad_segs=16) if inset else poly


def pad_texture(pad, cores):
    """Fingerprint-like arcs: concentric grooves around a centre beyond the tip, cut PAD_GROOVE_D deep."""
    top = dist_body(cores, WALL_LOFT + PAD_T + 1.0, TIP_Y + PAD_T + 1.0)
    floor = dist_body(cores, WALL_LOFT + PAD_T - PAD_GROOVE_D, TIP_Y + PAD_T - PAD_GROOVE_D)
    band = cq.Workplane().add(top).cut(cq.Workplane().add(floor))
    cx, cy = 0.0, 97.0
    grooves = None
    for r in np.arange(10.0, 36.0, 1.6):
        ring = cq.Workplane("XY").workplane(offset=-20).center(cx, cy).circle(r + PAD_GROOVE_W / 2).circle(r - PAD_GROOVE_W / 2).extrude(16)
        grooves = ring if grooves is None else grooves.union(ring)
    return pad.cut(band.intersect(grooves))


# ------------------------------------------------------------------ checks
def checks(parts, design, rep, sw, hw):
    stock = {n: load_step(n) for n in ("Proximal", "Distal", "Proximal_Shell", "Distal_Shell")}
    meshes = {k: to_trimesh(v, 0.02) for k, v in parts.items()}
    rep.append("")
    rep.append("== B-rep validity, volume, bounding box (mm)")
    for k, v in list(parts.items()) + [(f"stock {k}", v) for k, v in stock.items()]:
        bb = v.BoundingBox()
        rep.append(f"  {k:<28} valid {v.isValid()}  solids {len(v.Solids())}  volume {v.Volume():8.1f} mm3  "
                   f"x [{bb.xmin:6.2f}, {bb.xmax:6.2f}] y [{bb.ymin:6.2f}, {bb.ymax:6.2f}] z [{bb.zmin:6.2f}, {bb.zmax:6.2f}]")
    rep.append("")
    rep.append("== widths (lateral, along the pins), widest point and per station")
    sm = {k: to_trimesh(v, 0.02) for k, v in stock.items()}
    rep.append("  (from the tessellated parts; OCC bounding boxes of B-spline faces are not tight)")
    for a, b in (("Proximal_Shell", "Proximal_Shell_Slim"), ("Distal_Shell", "Distal_Shell_Slim")):
        wa, wb = np.ptp(sm[a].vertices[:, 0]), np.ptp(meshes[b].vertices[:, 0])
        ya = sm[a].vertices[np.argmax(np.abs(sm[a].vertices[:, 0])), 1]
        yb = meshes[b].vertices[np.argmax(np.abs(meshes[b].vertices[:, 0])), 1]
        rep.append(f"  {a:<16} widest: stock {wa:6.2f} (at y {ya:5.1f})  slim {wb:6.2f} (at y {yb:5.1f})  "
                   f"-{wa - wb:5.2f} mm ({100 * (wa - wb) / wa:4.1f} %)")
    for y in (20, 30, 40, 60, 66, 74.442, 80, 86):
        row = []
        for st, sl in (("Proximal_Shell", "Proximal_Shell_Slim"), ("Distal_Shell", "Distal_Shell_Slim")):
            s1 = sm[st].section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
            s2 = meshes[sl].section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
            if s1 is not None and s2 is not None:
                w1, w2 = np.ptp(s1.vertices[:, 0]), np.ptp(s2.vertices[:, 0])
                h1, h2 = np.ptp(s1.vertices[:, 2]), np.ptp(s2.vertices[:, 2])
                row.append(f"{sl.split('_')[0].lower()} width {w1:5.2f} -> {w2:5.2f} ({100 * (w1 - w2) / w1:4.1f} % less), "
                           f"depth {h1:5.2f} -> {h2:5.2f}")
        if row:
            rep.append(f"  y = {y:6.2f}: " + "; ".join(row))

    rep.append("")
    rep.append("== wall thickness of the slim shells (distance to the nearest opposite-facing surface; TPU min 1.2;"
               " 15000 points)")
    for k in ("Proximal_Shell_Slim", "Distal_Shell_Slim", "Distal_Pad_Slim"):
        th, pts = wall_thickness(meshes[k])
        t = th[np.isfinite(th)]
        thin = np.argsort(th)[:3]
        rep.append(f"  {k:<22} min {t.min():.2f}  1st percentile {np.percentile(t, 1):.2f}  5th {np.percentile(t, 5):.2f}  "
                   f"median {np.median(t):.2f} mm; thinnest at " + ", ".join(f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) {th[i]:.2f}"
                                                                         for i, p in zip(thin, pts[thin])))
    rep.append(f"  (pad: {PAD_T:.2f} mm thick, {PAD_T - PAD_GROOVE_D:.2f} mm under a groove by construction; "
               "its groove walls are reported as thin)")
    rep.append("  exact B-rep sections (in-plane wall, every 0.5 mm along y, sections that are two closed loops,"
               " i.e. away from the cut ends, the screw cups and the nose):")
    for k, ys in (("Proximal_Shell_Slim", np.arange(14.0, 50.0, 0.5)), ("Distal_Shell_Slim", np.arange(60.0, 90.0, 0.5))):
        w = [(y, brep_section_wall(parts[k], y)) for y in ys]
        w = [(y, v) for y, v in w if v is not None]
        ymin, vmin = min(w, key=lambda t: t[1])
        rep.append(f"    {k:<22} {len(w)} sections: min {vmin:.3f} at y {ymin:.1f}, median {np.median([v for _, v in w]):.3f} mm")
    rep.append("")
    rep.append("== static fit: shell vs its core (assembled, straight finger)")
    for sh, co, cup_y in (("Proximal_Shell_Slim", "Proximal_Slim", PROX_SCREW[0]), ("Distal_Shell_Slim", "Distal_Slim", DIST_SCREW[0])):
        inter = cq.Workplane().add(parts[sh]).intersect(cq.Workplane().add(parts[co]))
        vol = sum(s.Volume() for s in inter.solids().vals()) if inter.solids().vals() else 0.0
        cs = sample_surface(meshes[co], 0.2)
        far = np.abs(cs[:, 1] - cup_y) > POCKET_D / 2 + 0.5   # away from the screw cups (they seat in the spot faces)
        d = min_distance(meshes[sh], cs[far])
        rep.append(f"  {sh:<22} overlap with {co}: {vol:.3f} mm3; min gap away from the screw cups {d:.2f} mm")
    # cups seat on the spot-face floors
    rep.append("")
    rep.append("== joint sweeps (MCP " + f"{PHI_RANGE[0]:.1f}..{PHI_RANGE[1]:.1f} deg, 0.5 deg steps; distances between surface samples)")
    pm, dm = meshes["Proximal_Shell_Slim"], meshes["Distal_Shell_Slim"]
    sw.pts["prox"] = np.concatenate([sample_surface(to_trimesh(parts["Proximal_Slim"], 0.02), 0.3)]
                                    + [sample_surface(m, 0.3) for m in hw["prox"]] + [sample_surface(pm, 0.3)])
    sw.pts["dist"] = np.concatenate([sample_surface(to_trimesh(parts["Distal_Slim"], 0.02), 0.3)]
                                    + [sample_surface(m, 0.3) for m in hw["dist"]] + [sample_surface(dm, 0.3)]
                                    + [sample_surface(meshes["Distal_Pad_Slim"], 0.3)])
    for frame, shm, name in (("prox", pm, "Proximal_Shell_Slim"), ("dist", dm, "Distal_Shell_Slim"),
                             ("dist", meshes["Distal_Pad_Slim"], "Distal_Pad_Slim")):
        for body in ("prox", "dist", "link", "gimbal"):
            if body == frame:
                continue
            d = min_distance(shm, sw.swept(frame, body))
            rep.append(f"  {name:<22} vs swept {body:<7}: min {d:5.2f} mm" + ("  (> 3 mm)" if not np.isfinite(d) else ""))
    # same for the stock shells (for reference: they touch)
    stock_hw = hardware(11.7, 9.6)
    stock_sw = Sweeper({"dist": [load_stl("Distal"), load_stl("Distal_Shell")] + stock_hw["dist"],
                        "prox": [load_stl("Proximal"), load_stl("Proximal_Shell")] + stock_hw["prox"],
                        "link": [load_stl("Link")] + hw["link"], "gimbal": [load_stl("Gimbal")] + hw["gimbal"]}, spacing=0.5, step=1.0)
    for frame, st in (("prox", "Proximal_Shell"), ("dist", "Distal_Shell")):
        for body in ("prox", "dist", "link", "gimbal"):
            if body == frame:
                continue
            d = min_distance(sm[st], stock_sw.swept(frame, body))
            rep.append(f"  stock {st:<16} vs swept {body:<7} (stock parts): min {d:5.2f} mm")

    rep.append("")
    rep.append("== cores: material removed and section properties at the shell-screw section")
    for st, sl, y in (("Proximal", "Proximal_Slim", PROX_SCREW[0]), ("Distal", "Distal_Slim", DIST_SCREW[0])):
        v0, v1 = stock[st].Volume(), parts[sl].Volume()
        rep.append(f"  {sl:<14} volume {v0:7.1f} -> {v1:7.1f} mm3 (-{v0 - v1:5.1f}, {100 * (v0 - v1) / v0:4.1f} %)")
        for tag, mesh in (("stock", sm[st]), ("slim ", meshes[sl])):
            for yy in (y, y - 2.5):
                p = section_props(section_poly(mesh, yy))
                rep.append(f"    {tag} y {yy:6.2f}: area {p['area']:6.1f} mm2  Ixx (flexion) {p['Ixx']:8.1f} mm4  Izz (lateral) {p['Izz']:8.1f} mm4")
    # tip containment
    rep.append("")
    rep.append("== screws")
    rep.append(f"  proximal: head top at |x| {design['prox_screw_head_x']:.2f}, head bottom {design['prox_screw_head_x'] - SCREW['head_h']:.2f}, "
               f"tip at |x| {design['prox_screw_head_x'] - SCREW['head_h'] - SCREW['shank']:.2f} (link channel wall 3.25, link +-1.9; "
               f"thread engagement {design['prox_screw_head_x'] - SCREW['head_h'] - CUP_FLOOR - 3.25:.2f} mm in PLA)")
    rep.append(f"  distal:   head top at |x| {design['dist_screw_head_x']:.2f}, tip at |x| {design['dist_screw_head_x'] - SCREW['head_h'] - SCREW['shank']:.2f} "
               f"(opposite screw tip at the mirror position), thread engagement "
               f"{design['dist_screw_head_x'] - SCREW['head_h'] - CUP_FLOOR - (design['dist_screw_head_x'] - SCREW['head_h'] - SCREW['shank']):.2f} mm")
    return meshes, sm


def section_poly(mesh, y):
    """Cross-section of a closed mesh at y as a (x, z) polygon with holes (even-odd rule)."""
    s = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    poly = Polygon()
    for loop in s.discrete:
        if len(loop) >= 4:
            poly = poly.symmetric_difference(Polygon(loop[:, [0, 2]]).buffer(0))
    return poly


def export(parts):
    (HERE / "step").mkdir(exist_ok=True)
    (HERE / "stl").mkdir(exist_ok=True)
    # STEP: every part except the textured pad and the one-piece shell: the groove texture's intersection
    # curves with the lofted offset surfaces make a 40-60 MB STEP, so the pad's STEP is the untextured
    # pad (editable); the STLs carry the texture.
    for k, v in parts.items():
        if k == "Distal_Pad_Slim_untextured":
            cq.exporters.export(cq.Workplane().add(v), str(HERE / "step" / f"{k}.step"))
            continue
        if k not in ("Distal_Shell_Slim_OnePiece", "Distal_Pad_Slim"):
            cq.exporters.export(cq.Workplane().add(v), str(HERE / "step" / f"{k}.step"))
        if k == "Distal_Pad_Slim_untextured":
            continue
        f = HERE / "stl" / f"{k}.stl"
        cq.exporters.export(cq.Workplane().add(v), str(f), tolerance=0.01, angularTolerance=0.1)
        m = trimesh.load(f)                 # drop degenerate pole triangles (loft apex), merge, close, fix winding
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        m.merge_vertices(digits_vertex=5)
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        m.export(f)
        if not m.is_watertight:
            log(f"WARNING: {f.name} is not watertight")


# ------------------------------------------------------------------ previews (offscreen MuJoCo)
def render_scene(items, views, size=(560, 420), dist=150.0, lookat=(0, 46, 2)):
    """items: list of (trimesh mesh in mm, rgba, x offset mm). views: list of (azimuth, elevation, title)."""
    import mujoco
    from PIL import Image, ImageDraw
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = size
    spec.visual.headlight.ambient = [0.38] * 3
    spec.visual.headlight.diffuse = [0.62] * 3
    spec.visual.headlight.specular = [0.08] * 3
    tmp = HERE / "_preview_tmp"
    tmp.mkdir(exist_ok=True)
    for i, (mesh, rgba, dx) in enumerate(items):
        f = tmp / f"m{i}.stl"
        mesh.export(f)
        spec.add_mesh(name=f"m{i}", file=str(f), scale=[1e-3] * 3)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{i}", rgba=rgba, pos=[dx * 1e-3, 0, 0])
    model = spec.compile()
    for f in tmp.glob("*.stl"):
        f.unlink()
    tmp.rmdir()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.array(lookat) * 1e-3
    cam.distance = dist * 1e-3
    tiles = []
    for az, el, title in views:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(data, cam)
        im = Image.fromarray(r.render())
        d = ImageDraw.Draw(im)
        d.rectangle([4, 3, 10 + 6 * len(title), 17], fill=(20, 20, 20))
        d.text((7, 4), title, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    return tiles


def previews():
    """preview_finger.png: stock (left) and slim (right) finger, straight, with gimbal, link and hardware;
    preview_parts.png: the slim parts."""
    from PIL import Image
    design = json.loads((HERE / "design.json").read_text())
    stl = {k: trimesh.load(HERE / "stl" / f"{k}.stl") for k in
           ("Proximal_Slim", "Distal_Slim", "Proximal_Shell_Slim", "Distal_Shell_Slim", "Distal_Pad_Slim")}
    hw_slim = hardware(design["prox_screw_head_x"], design["dist_screw_head_x"])
    hw_stock = hardware(11.7, 9.6)
    PLA, PLA2, TPU, PAD, MET = [0.55, 0.65, 0.9, 1], [0.55, 0.85, 0.65, 1], [0.95, 0.95, 0.93, 1], [0.24, 0.25, 0.27, 1], [0.6, 0.62, 0.66, 1]
    common = [(load_stl("Gimbal"), [0.75, 0.55, 0.8, 1]), (load_stl("Link"), [0.85, 0.6, 0.4, 1])]
    stock_items = common + [(load_stl("Proximal"), PLA), (load_stl("Distal"), PLA2), (load_stl("Proximal_Shell"), TPU),
                            (load_stl("Distal_Shell"), TPU)] + [(m, MET) for k in hw_stock for m in hw_stock[k]]
    slim_items = common + [(stl["Proximal_Slim"], PLA), (stl["Distal_Slim"], PLA2), (stl["Proximal_Shell_Slim"], TPU),
                           (stl["Distal_Shell_Slim"], TPU), (stl["Distal_Pad_Slim"], PAD)] + [(m, MET) for k in hw_slim for m in hw_slim[k]]
    items = [(m, c, -16.0) for m, c in stock_items] + [(m, c, 16.0) for m, c in slim_items]
    views = [(90, -89, "back (dorsal): stock left, slim right"), (90, 89, "palm side: stock left, slim right"),
             (60, -35, "3/4 back"), (120, 30, "3/4 palm")]
    tiles = render_scene(items, views, dist=175.0)
    img = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    Image.fromarray(img).save(HERE / "preview_finger.png")
    # side view, one finger each (stacked)
    side = [render_scene([(m, c, 0.0) for m, c in it], [(0, 0, f"{tag}: side view")], size=(900, 300), dist=120.0, lookat=(0, 42, 0))[0]
            for tag, it in (("stock", stock_items), ("slim", slim_items))]
    Image.fromarray(np.concatenate(side, 0)).save(HERE / "preview_side.png")
    # slim parts, separated
    parts = [(stl["Proximal_Slim"], PLA, -36.0), (stl["Proximal_Shell_Slim"], TPU, -12.0), (stl["Distal_Slim"], PLA2, 12.0),
             (stl["Distal_Shell_Slim"], TPU, 30.0), (stl["Distal_Pad_Slim"], PAD, 46.0)]
    tiles = render_scene(parts, [(60, -30, "slim parts: Proximal, Proximal_Shell, Distal, Distal_Shell, Distal_Pad"),
                                 (120, 35, "the same from the palm side")], size=(760, 460), dist=190.0, lookat=(5, 50, 0))
    Image.fromarray(np.concatenate(tiles, 1)).save(HERE / "preview_parts.png")
    log("wrote preview_finger.png, preview_side.png, preview_parts.png")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks-only", action="store_true", help="re-run the checks on the exported STEP files")
    ap.add_argument("--previews-only", action="store_true", help="only render the preview images from the exported STLs")
    args = ap.parse_args()
    t0 = time.time()
    if args.previews_only:
        previews()
        return
    if args.checks_only:
        design = json.loads((HERE / "design.json").read_text())
        parts = {k: solid_of(cq.importers.importStep(str(HERE / "step" / f"{k}.step")))
                 for k in ("Proximal_Slim", "Distal_Slim", "Proximal_Shell_Slim", "Distal_Shell_Slim")}
        parts["Distal_Pad_Slim"] = solid_of(cq.importers.importStep(str(HERE / "step" / "Distal_Pad_Slim_untextured.step")))
        rep = json.loads((HERE / "build_log.json").read_text())
        hw = hardware(design["prox_screw_head_x"], design["dist_screw_head_x"])
        sw = Sweeper({"dist": [], "link": [load_stl("Link")] + hw["link"], "gimbal": [load_stl("Gimbal")] + hw["gimbal"], "prox": []})
    else:
        parts, design, rep, sw, hw = build()
        export(parts)
        (HERE / "design.json").write_text(json.dumps(design, indent=2))
        (HERE / "build_log.json").write_text(json.dumps(rep, indent=1))
        log("exported STEP/STL, design.json")
    checks(parts, design, rep, sw, hw)
    txt = "\n".join(["make_slim_fingers.py report (all mm)", ""] + rep)
    (HERE / "report_cad.txt").write_text(txt + "\n")
    log(txt)
    previews()
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
