#!/usr/bin/env python3
"""CadQuery source of the AmazingHand 5F powered wrist and forearm base (right hand).

Writes, for every printed part, a STEP and an STL in mm in the hand's CAD part-studio frame at the
neutral wrist pose (x index->pinky, y along the fingers, z towards the back; same frame as
cad/step and cad/five_finger), plus envelopes of the bought parts (servos, horns, bearings, pins,
ball studs, ball links, rods) in stl/ref/, a parts table (wrist_parts.json) and printability /
interference reports (previews/*.txt).

All dimensions come from wrist_design.py (single source of truth shared with the MuJoCo model).

Usage:  python make_wrist.py            (everything)
        python make_wrist.py --no-checks
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wrist_design as wd  # noqa: E402

V = cq.Vector
STEP_DIR, STL_DIR, REF_DIR, PREV_DIR = HERE / "step", HERE / "stl", HERE / "stl" / "ref", HERE / "previews"
PLA_DENSITY = 1.00            # g/cm3, wrist parts printed with 3-4 walls and ~25 % infill
STL_TOL, STL_ANG = 0.04, 0.15


def log(*a):
    print(*a, flush=True)


# ============================================================================ primitives (W frame, mm)
def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, pnt=V(x0, y0, z0))


def cyl(p, d, r, h):
    d = np.asarray(d, float)
    return cq.Solid.makeCylinder(r, h, pnt=V(*p), dir=V(*(d / np.linalg.norm(d))))


def sphere(p, r):
    return cq.Solid.makeSphere(r, pnt=V(*p), angleDegrees1=-90, angleDegrees2=90)


def fuse(*shapes):
    s = shapes[0]
    return s.fuse(*shapes[1:]).clean() if len(shapes) > 1 else s


def cut(s, *tools):
    return s.cut(*tools).clean()


def prism(pts3d, vec):
    """Extrude the planar polygon pts3d (list of 3D points) by vec."""
    w = cq.Wire.makePolygon([V(*p) for p in pts3d], close=True)
    return cq.Solid.extrudeLinear(cq.Face.makeFromWires(w), V(*vec))


def prism_yz(pts_yz, x0, x1):
    return prism([(x0, y, z) for y, z in pts_yz], (x1 - x0, 0, 0))


def prism_xz(pts_xz, y0, y1):
    return prism([(x, y0, z) for x, z in pts_xz], (0, y1 - y0, 0))


def prism_xy(pts_xy, z0, z1):
    return prism([(x, y, z0) for x, y in pts_xy], (0, 0, z1 - z0))


def teardrop(p, axis, r, h, up, flat=None, n=64):
    """Horizontal-hole helper: a circle (polygon circumscribing radius r) plus a 45 deg roof towards
    `up` (the print's up direction), optionally truncated at distance `flat` from the axis."""
    p, a, u = (np.asarray(v, float) for v in (p, axis, up))
    a, u = a / np.linalg.norm(a), u / np.linalg.norm(u)
    w = np.cross(a, u)
    rc = r / np.cos(np.pi / n)
    pts = [(rc * np.cos(t), rc * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)] + [(r * np.sqrt(2), 0.0)]
    from scipy.spatial import ConvexHull
    P = np.array(pts)
    P = P[ConvexHull(P).vertices]                          # (u, w) coordinates, convex, ccw
    if flat is not None:                                   # clip the convex polygon by u <= flat
        out = []
        for k in range(len(P)):
            A, B = P[k], P[(k + 1) % len(P)]
            if A[0] <= flat:
                out.append(A)
            if (A[0] - flat) * (B[0] - flat) < 0:
                t = (flat - A[0]) / (B[0] - A[0])
                out.append(A + t * (B - A))
        P = np.array(out)
    return prism([p + q[0] * u + q[1] * w for q in P], a * h)


def superellipse(a, zf, zb, zm, nf, nb, m=1440):
    """Forearm section outline (x, z), counter-clockwise: |x / a|^n + |(z - zm) / b|^n = 1 with b = zm - zf, n = nf on
    the palm side and b = zb - zm, n = nb on the back (the halves meet with vertical tangents at x = +-a)."""
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    n = np.where(s > 0, nb, nf)
    return np.column_stack([a * np.sign(c) * np.abs(c) ** (2 / n),
                            zm + np.where(s > 0, zb - zm, zm - zf) * np.sign(s) * np.abs(s) ** (2 / n)])


def offset_curve(C, d):
    """Closed counter-clockwise polyline C offset outwards by d (a convex curve stays exact to the sampling)."""
    t = np.roll(C, -1, 0) - np.roll(C, 1, 0)
    nrm = np.column_stack([t[:, 1], -t[:, 0]])
    return C + d * nrm / np.linalg.norm(nrm, axis=1)[:, None]


_PROFILE = None


def forearm_profile(y):
    """Section parameters (a, zf, zb, zm, nf, nb) at height y: monotone cubic (PCHIP) through FOREARM_KEYS, so the taper is
    smooth and has no flat bands at the keys."""
    global _PROFILE
    if _PROFILE is None:
        from scipy.interpolate import PchipInterpolator
        K = np.array(sorted(wd.FOREARM_KEYS))
        _PROFILE = (PchipInterpolator(K[:, 0], K[:, 1:], axis=0), K[0, 0], K[-1, 0])
    f, y_lo, y_hi = _PROFILE
    return tuple(f(float(np.clip(y, y_lo, y_hi))))


def cuff_offset(y):
    """How far the outer surface stands proud of the base section at height y: the wrist cuff."""
    y0, y1 = wd.CUFF_Y
    t = float(np.clip((y - y1) / (y0 - y1), 0.0, 1.0))
    return wd.CUFF_PROUD * t * t * (3 - 2 * t)


def section_pts(y, d=0.0, m=1440, normal=False):
    """Points (x, z) of the base section at height y offset outwards by d (d < 0: inwards). normal: d is measured
    normal to the lofted surface, not in the section plane (an inset grows by sqrt(1 + slope^2) where the surface
    flares, so the wall keeps its thickness through the wrist flare)."""
    C = superellipse(*forearm_profile(y), m=m)
    if not d:
        return C
    if not normal:
        return offset_curve(C, d)
    h = 0.25
    t = np.roll(C, -1, 0) - np.roll(C, 1, 0)
    n = np.column_stack([t[:, 1], -t[:, 0]]) / np.linalg.norm(t, axis=1)[:, None]
    slope = np.sum((superellipse(*forearm_profile(y - h), m=m) - superellipse(*forearm_profile(y + h), m=m)) * n, 1) / (2 * h)
    return C + (d * np.sqrt(1 + slope ** 2))[:, None] * n


SECTION_PTS = 72           # spline points per section (equal arc length)


def section(y, d=0.0, normal=False):
    """Closed spline wire in the plane W y = const: the base section offset by d (outer skin: d = cuff_offset(y),
    inner wall: d = -WALL, normal=True)."""
    C = section_pts(y, d, normal=normal)
    L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.vstack([C, C[:1]]), axis=0), axis=1))]
    s = np.linspace(0.0, L[-1], SECTION_PTS, endpoint=False)
    Cc = np.vstack([C, C[:1]])
    pts = [V(x, y, z) for x, z in zip(np.interp(s, L, Cc[:, 0]), np.interp(s, L, Cc[:, 1]))]
    return cq.Wire.assembleEdges([cq.Edge.makeSpline(pts, periodic=True)])


def outer_section(y):
    return section(y, cuff_offset(y))


def sector(r0, r1, a0, a1, y0, y1, n=24):
    """Annular sector around the W y axis; angles (deg) measured as atan2(z, x)."""
    ang = np.radians(np.linspace(a0, a1, n))
    outer = [(r1 * np.cos(t), r1 * np.sin(t)) for t in ang]
    inner = [(r0 * np.cos(t), r0 * np.sin(t)) for t in ang[::-1]]
    return prism_xz(outer + inner, y0, y1)


def rot_z180(s, y0):
    """Rotate 180 deg about the W axis (x=0, y=y0) parallel to z: maps the ulnar servo onto the radial one."""
    return s.rotate(V(0, y0, 0), V(0, y0, 1), 180)


Y_PAIR = 0.5 * (wd.Y_SHAFT[1] + wd.Y_SHAFT[2])    # servo 2 = servo 1 rotated 180 deg about (0, Y_PAIR) || z
FLEX_STOP_DEG, DEV_STOP_DEG = 66.0, 31.0
_f = np.radians(FLEX_STOP_DEG)
# lowest point of the adapter flange's bottom edge (y = H_FLANGE - FLANGE_T, r = FLANGE_R) after flexing by the stop angle
FLEX_STOP_Y = (wd.H_FLANGE - wd.FLANGE_T) * np.cos(_f) - wd.FLANGE_R * np.sin(_f)
_d = np.radians(DEV_STOP_DEG)
# height reached by the fork boss's top outer corner (x = LUG_OUT, y = LUG_R) when the hand deviates by the stop angle
DEV_STOP_Y = wd.LUG_OUT * np.sin(_d) + wd.LUG_R * np.cos(_d)


# ============================================================================ bearing seats, pins
def bearing_seat(axis, up, through=0.0):
    """Cut tool for an F683ZZ seat in a lug whose inner face is at r = LUG_IN along `axis`: flange
    counterbore, bearing seat (teardrop roof towards the print's up), d4 hole for pressing the pin in."""
    a = np.asarray(axis, float)
    o = np.zeros(3)
    cb = cyl(o + a * (wd.LUG_IN - 0.05), a, (wd.BRG_FD + 0.4) / 2, wd.BRG_FT + 0.1)
    seat = teardrop(o + a * (wd.LUG_IN + wd.BRG_FT + 0.05 - 0.01), a, wd.BRG_D / 2, wd.BRG_B + 0.01, up, flat=wd.LUG_R - 1.2)
    hole = cyl(o + a * (wd.LUG_IN + wd.BRG_FT + wd.BRG_B), a, (wd.PIN_D + 1.0) / 2, wd.LUG_OUT - wd.LUG_IN + through)
    return fuse(cb, seat, hole)


AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


# ============================================================================ printed parts
def make_ring():
    h = wd.RING_H
    ring = cq.Workplane("XZ", origin=(0, h / 2, 0)).circle(wd.RING_R_OUT).circle(wd.RING_R_IN).extrude(h)
    ring = ring.edges().chamfer(0.5).val()
    for a in AXES.values():
        a = np.array(a, float)
        t = np.cross(a, [0, 1, 0])
        c = a * (wd.RING_R_OUT - 0.3)          # spot face for the shim washer (DIN 988 3x6x0.5)
        spot = box(*sorted([c[0] - 3.5 * abs(t[0]), c[0] + 3.5 * abs(t[0]) + 3 * a[0]]), -3.5, 3.5,
                   *sorted([c[2] - 3.5 * abs(t[2]), c[2] + 3.5 * abs(t[2]) + 3 * a[2]]))
        ring = cut(ring, spot, cyl(a * (wd.RING_R_IN - 1), a, (wd.PIN_D - 0.1) / 2, wd.RING_R_OUT - wd.RING_R_IN + 2))
    return ring


def interface_to_w(shape):
    """Interface frame (CAD axes, origin on the spigot axis in the plane of the interface's recessed pads,
    CAD y = INTERFACE_PADS_Y) -> W."""
    return shape.rotate(V(0, 0, 0), V(0, 0, 1), np.degrees(wd.HAND_YAW)).translate(V(*wd.INTERFACE_TOP_W))


def make_hand_adapter():
    yb = wd.H_FLANGE - wd.FLANGE_T                     # 18.5: flange underside (W)
    # Body: a d56 cylinder on the interface's spigot axis, from the pad plane down to the W plane y = yb (the
    # gimbal side, as at zero yaw): a wedge, FLANGE_MIN + PALM_EXT tall at the radial rim, taller towards the ulnar
    # side. It is hollow: a 6 mm top flange (screw seats, pad pockets), a 4 mm wall, and a 45 deg conical roof
    # down to the cable bore (prints flange-down without support), >= 4 mm above the gimbal-side face.
    h = 40.0 + wd.PALM_EXT
    flange = interface_to_w(cyl((0, -h, 0), (0, 1, 0), wd.FLANGE_R, h)).cut(box(-80, 80, yb - h, yb, -80, 80))
    top_t, wall, bot_t, r_in, r_bore = 6.0, 4.0, 4.0, wd.FLANGE_R - 4.0, 11.0
    sg, cg = np.sin(wd.HAND_YAW), np.cos(wd.HAND_YAW)
    plane_y = lambda r: (yb - wd.INTERFACE_TOP_W[1] + sg * r) / cg      # highest point of the W plane y = yb within radius r (I frame)
    y1 = max(plane_y(r_in) + bot_t, plane_y(r_bore) + bot_t + (r_in - r_bore))
    if y1 < -top_t - 2.0:
        roof = cq.Solid.makeCone(r_bore, r_in, r_in - r_bore, pnt=V(0, y1 - (r_in - r_bore), 0), dir=V(0, 1, 0))
        cavity = fuse(cyl((0, y1 - 0.01, 0), (0, 1, 0), r_in, -top_t - y1 + 0.01), roof)
        flange = flange.cut(interface_to_w(cavity))
    # pockets for the interface's three protruding pads (CAD y -102.2 .. -101; sectors 31-89, 151-209, 271-329 deg)
    pockets = [interface_to_w(sector(16.5, 30, c - 30.2, c + 30.2, -1.45, 1)) for c in (60, 180, 300)]
    holes = [interface_to_w(cyl((0, -h, 0), (0, 1, 0), 11.0, h + 2))]   # cable bore d22, along the spigot axis
    for ang in (20, -20, 140, -140):  # 4 x 3.0x12 PT / M3x12 screw along the spigot axis, head from below
        x, z = 21 * np.cos(np.radians(ang)), 21 * np.sin(np.radians(ang))
        # the head seats 4.7 mm below the pad plane (as at zero yaw); the counterbore runs out through the cavity and
        # the underside, so a driver reaches it from below
        holes += [interface_to_w(cyl((x, -h, z), (0, 1, 0), 1.7, h + 3)), interface_to_w(cyl((x, -h, z), (0, 1, 0), 3.25, h - 4.7))]
    body = cut(flange, *pockets, *holes)
    lugs = []
    for s in (+1, -1):          # front (-z) / back (+z) lugs: deviation axis = W z; the back one is a block out to the bar
        z0, z1 = sorted([s * wd.LUG_IN, s * (wd.LUG_OUT if s < 0 else wd.BAR_Z[1])])
        lugs += [box(-wd.LUG_R, wd.LUG_R, 0, yb + 0.2, z0, z1), cyl((0, 0, z0), (0, 0, 1), wd.LUG_R, z1 - z0)]
    bar = box(-wd.BAR_X, wd.BAR_X, -wd.BAR_Y, wd.BAR_Y, wd.BAR_Z[0], wd.BAR_Z[1])
    gussets = [prism_xy([(s * wd.LUG_R, wd.BAR_Y), (s * wd.BAR_X, wd.BAR_Y), (s * wd.LUG_R, wd.BAR_Y + wd.BAR_X - wd.LUG_R)],
                        wd.BAR_Z[0], wd.BAR_Z[1]) for s in (1, -1)]
    # deviation hard stops: pads under the flange that land on the fork's bearing bosses at ~31 deg
    pads = [box(*sorted([s * wd.LUG_IN, s * 16.0]), DEV_STOP_Y, yb + 0.2, -4.0, 4.0) for s in (1, -1)]
    body = fuse(body, *lugs, bar, *gussets, *pads)
    tools = [bearing_seat(AXES["+z"], (0, -1, 0), through=wd.BAR_Z[1] - wd.LUG_OUT + 1),   # pin access through the block
             bearing_seat(AXES["-z"], (0, -1, 0))]
    for s in (1, -1):           # M3 heat-set inserts for the ball studs at the bar ends
        tools.append(cyl((s * (wd.BAR_X + 0.01), 0, wd.E_LEVER), (-s, 0, 0), 2.0, 6.0))
    return cut(body, *tools)


def make_fork(slots):
    y0, y1 = wd.CAP_Y
    body = cq.Solid.extrudeLinear(cq.Face.makeFromWires(outer_section(y0)), V(0, y1 - y0, 0))
    body = cq.Workplane().add(body).faces(">Y").edges().fillet(wd.CAP_FILLET).val()   # rounded top of the cuff ring
    lugs = []
    for s in (1, -1):           # stems lean forward below the boss, clear of the hand's lever bar in extension
        x0, x1 = sorted([s * wd.LUG_IN, s * wd.LUG_OUT])
        lugs += [prism_yz([(y1 - 0.5, -24.0), (y1 - 0.5, -12.0), (-6.0, 2.0), (0.0, 2.0), (0.0, -wd.LUG_R)], x0, x1),
                 cyl((x0, 0, 0), (1, 0, 0), wd.LUG_R, x1 - x0)]
    bridge = box(-wd.LUG_OUT, wd.LUG_OUT, y1 - 0.5, -18.0, -24.0, -17.0)        # ties the two stems together
    # flexion hard stops: the adapter flange's edge lands on these pads at ~66 deg (palm-ward and extension)
    # (the back pad stays narrow: the back cap screws sit at x = +-7)
    stops = [box(-w_, w_, y1 - 0.5, FLEX_STOP_Y, *sorted([s * 25.0, s * 31.0])) for s, w_ in ((1, 3.5), (-1, 9.0))]
    body = fuse(body, *lugs, bridge, *stops)
    tools = [bearing_seat(AXES["+x"], (0, 1, 0)), bearing_seat(AXES["-x"], (0, 1, 0)),
             cyl((0, y0 - 5, 0), (0, 1, 0), 6.0, 12)]                 # cable hole d12 under the ring bore
    for poly in slots:                                                  # push-rod slots
        tools.append(prism_xz(poly, y0 - 5, y1 + 1))
    for x, z in CAP_SCREWS:                                             # 4 x M3x10 into the shells' top bosses
        tools += [cyl((x, y0 - 5, z), (0, 1, 0), 1.7, 12), cyl((x, y1 - 2.5, z), (0, 1, 0), 3.25, 3)]
    return cut(body, *tools)


def shell_solid():
    """Forearm shell: one smooth loft of superellipse sections from the wrist cuff to the elbow, WALL thick (WALL +
    CUFF_PROUD in the cuff band). The panel line is a ruled V in the skin, GROOVE_W wide and GROOVE_D deep; its upper
    flank drops more than it is deep (<= 45 deg overhang standing on the elbow end). Returns (shell, mid): the solid
    bounded by the middle of the wall, which clips the inner pads, ribs and webs (they end inside the wall, so nothing
    pokes out of the skin and no faces coincide with it)."""
    y0, y3 = wd.CAP_Y[0], wd.FOREARM_BOTTOM
    yt, w = wd.GROOVE_Y, wd.GROOVE_W
    yv = yt - 0.55 * w                                  # bottom of the V
    # sections 1 mm apart through the cuff and its lower edge, then ~6 mm apart; each loft evenly spaced (unevenly
    # spaced loft sections mesh with holes)
    y2 = wd.CUFF_Y[1] - 8.0
    above = cq.Solid.makeLoft([outer_section(y) for y in np.linspace(y0, yt, int(round(y0 - yt)) + 1)], False)
    groove = cq.Solid.makeLoft([outer_section(yt), section(yv, cuff_offset(yv) - wd.GROOVE_D), outer_section(yt - w)], True)
    edge = cq.Solid.makeLoft([outer_section(y) for y in np.linspace(yt - w, y2, int(round(yt - w - y2)) + 1)], False)
    body = cq.Solid.makeLoft([outer_section(y) for y in np.linspace(y2, y3, 28)], False)
    outer = fuse(above, groove, edge, body)
    # cavity and mid-wall solid: the same two-piece spacing (the wrist flare changes fast)
    ya, yb, t = np.linspace(y0, y2, int(round(y0 - y2)) + 1), np.linspace(y2, y3, 28), wd.WALL + 0.05
    loft2 = lambda d: fuse(*[cq.Solid.makeLoft([section(y, d, True) for y in yy], False) for yy in (ya, yb)])
    cavity = fuse(loft2(-t),            # (+0.05: the cavity and skin lofts interpolate a little differently)
                  cq.Solid.extrudeLinear(cq.Face.makeFromWires(section(y0, -t, True)), V(0, 3, 0)),   # through both ends
                  cq.Solid.extrudeLinear(cq.Face.makeFromWires(section(y3, -t, True)), V(0, -3, 0)))
    return cut(outer, cavity), loft2(-0.5 * wd.WALL)


def wall_boss(y, ang, r=3.5, overlap=0.5):
    """Centre (x, z) of a boss of radius r that overlaps the inner wall at height y by `overlap`, on the ray from the
    section centre (0, zm) at angle ang (deg, atan2(z - zm, x)), and the wall's outward normal there."""
    C = section_pts(y, -wd.WALL)
    zm = forearm_profile(y)[3]
    k = int(np.argmin(np.abs((np.arctan2(C[:, 1] - zm, C[:, 0]) - np.radians(ang) + np.pi) % (2 * np.pi) - np.pi)))
    t = C[(k + 1) % len(C)] - C[k - 1]
    nrm = np.array([t[1], -t[0]]) / np.linalg.norm(t)
    c = C[k] - nrm * (r - overlap)
    return (float(c[0]), float(c[1])), nrm


def boss_with_web(x, z, y0, y1, web, mid, insert_from_top=True, cone=True):
    """Vertical screw boss d7 (M3 heat-set insert) tied to the wall by a 3 mm web running in the direction web (x, z)
    into the wall, clipped by `mid` (web None: no web)."""
    parts = [cyl((x, y0, z), (0, 1, 0), 3.5, y1 - y0)]
    if web is not None:
        u = np.asarray(web, float) / np.linalg.norm(web)
        c, p = np.array([x, z]), 1.5 * np.array([-u[1], u[0]])
        parts.append(prism_xz([c - p, c + p, c + p + 30 * u, c - p + 30 * u], y0, y1).intersect(mid))
    if cone:   # 45 deg support cone under a boss that hangs from the top (standing print)
        if insert_from_top:
            parts.append(cq.Solid.makeCone(0.01, 3.5, 3.5, pnt=V(x, y0 - 3.5, z), dir=V(0, 1, 0)))
        else:
            parts.append(cq.Solid.makeCone(3.5, 0.01, 3.5, pnt=V(x, y1, z), dir=V(0, 1, 0)))
    s = fuse(*parts)
    yi = y1 - 6 if insert_from_top else y0
    return s, cyl((x, yi, z), (0, 1, 0), 2.0, 6.01)


_CAP_FRONT, _CAP_NRM = zip(*[wall_boss(wd.CAP_Y[0], a) for a in (-40.0, -140.0)])
CAP_SCREWS = list(_CAP_FRONT) + [(s * 7.0, 26.0) for s in (1, -1)]       # back pair between the back stop pad and the rods
_FL, _FL_NRM = zip(*[wall_boss(wd.FOREARM_BOTTOM, a) for a in (-40.0, -140.0, 40.0, 140.0)])
FLANGE_SCREWS = list(_FL)
# servo frame: its base lies 0.2 mm above seat ribs / pads at z = FRAME_Z0 - 0.2, its side blocks face bosses at
# |x| = FRAME_SIDE_X (just outside the horn arms); both pairs of countersunk screws seat on spot faces in the skin
FRAME_Z0, FRAME_SIDE_X = -42.3, 31.0
FRAME_TABS = [(s * 15.0, wd.Y_SHAFT[2] - wd.SERVO_W / 2 - 3.3 - 4.5) for s in (1, -1)]     # palm-side screws
FRAME_SIDE = [(wd.SIDE[i] * FRAME_SIDE_X, wd.Y_SHAFT[i], -24.5) for i in (1, 2)]          # side-wall screws
TAB_SEAT_Z, SIDE_SEAT_X = -45.0, 35.0        # head seats of the four M3x8 countersunk, below the curved skin
SPOT_R = 3.75                                # d7.5 spot face from the skin down to the head seat


def countersunk(p, axis, seat, length):
    """Cut tools for a countersunk M3 screw of `length` along the horizontal `axis` (pointing into the part), its head
    seated at p + seat * axis, with a teardrop spot face from there out through the skin (prints standing)."""
    a = np.asarray(axis, float)
    p = np.asarray(p, float)
    o = p + a * seat
    return [teardrop(o - a * 20, a, SPOT_R, 20.01, (0, 1, 0)), cq.Solid.makeCone(3.4, 1.6, 1.8, pnt=V(*o), dir=V(*a)),
            cyl(o, a, 1.7, length + 2)]


def make_forearm_front(shell, mid):
    front = shell.intersect(box(-80, 80, -400, 0, -100, wd.SPLIT_Z))
    adds, tools = [], []
    for (x, z), n in zip(CAP_SCREWS[:2], _CAP_NRM):
        s, t = boss_with_web(x, z, wd.CAP_Y[0] - 12, wd.CAP_Y[0], n, mid, True)
        adds.append(s)
        tools.append(t)
    for (x, z), n in zip(FLANGE_SCREWS[:2], _FL_NRM[:2]):
        s, t = boss_with_web(x, z, wd.FOREARM_BOTTOM, wd.FOREARM_BOTTOM + 12, n, mid, False)
        adds.append(s)
        tools.append(t)
    zs = FRAME_Z0 - 0.2                              # seat plane of the frame base
    y_top = wd.servo_box(1)[1][1] + 0.3 + 3.0        # frame base: y_bot - 9 .. y_top (make_servo_frame)
    y_bot = wd.servo_box(2)[1][0] - 0.3 - 3.0 - 9.0
    for sx in (1, -1):                               # two seat ribs under the frame base (undersides 50 deg)
        adds.append(prism_yz([(y_top, zs), (y_bot, zs), (y_bot - 24, zs - 20), (y_top, zs - 20)],
                             sx * 12.0 - 1.5, sx * 12.0 + 1.5).intersect(mid))
    for (x, y) in FRAME_TABS:      # pads behind the palm-side screws, countersunk M3x8 into the frame's inserts
        adds.append(prism_yz([(y + 5, zs), (y - 5, zs), (y - 29, zs - 20), (y + 5, zs - 20)], x - 5.5, x + 5.5).intersect(mid))
        tools += countersunk((x, y, zs), (0, 0, 1), TAB_SEAT_Z - zs, 8)
    from scipy.spatial import ConvexHull
    keel = np.array([(6.0 * np.cos(t), 6.0 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 48, endpoint=False)] + [(-9.6, 0.0)])
    keel = keel[ConvexHull(keel).vertices]          # d12 boss section (y, z) with a 51 deg keel underneath
    for (x, y, z) in FRAME_SIDE:   # bosses behind the side-wall screws, countersunk M3x8 into the frame's side blocks
        sx = np.sign(x)
        adds.append(prism_yz([(y + a, z + b) for a, b in keel], x, x + sx * 20).intersect(mid))
        tools += countersunk((x, y, z), (-sx, 0, 0), FRAME_SIDE_X - SIDE_SEAT_X, 8)
    front = fuse(front, *adds)
    return cut(front, *tools)


def make_forearm_back(shell, mid):
    back = shell.intersect(box(-80, 80, -400, 0, wd.SPLIT_Z, 100))
    adds, tools = [], []
    for (x, z) in CAP_SCREWS[2:]:
        s, t = boss_with_web(x, z, wd.CAP_Y[0] - 12, wd.CAP_Y[0], (0, 1), mid, True)
        adds.append(s)
        tools.append(t)
    for (x, z), n in zip(FLANGE_SCREWS[2:], _FL_NRM[2:]):
        s, t = boss_with_web(x, z, wd.FOREARM_BOTTOM, wd.FOREARM_BOTTOM + 12, n, mid, False)
        adds.append(s)
        tools.append(t)
    # ribs that keep each servo seated in the frame (press on the gear-end face, z = 5.25): struts down to the back
    # wall, 50 deg from the horizontal
    z_end, z_far = wd.servo_box(1)[2][1] + 0.3, 50.0
    dy = 1.2 * (z_far - z_end)
    for i in (1, 2):
        ys = wd.Y_SHAFT[i]
        for xr in (-10.0, 10.0):
            pts = [(ys + 6, z_end), (ys - 6, z_end), (ys - 6 - dy, z_far), (ys + 6 - dy, z_far)]
            adds.append(prism_yz(pts, xr - 1.0, xr + 1.0).intersect(mid))
    # cable exit notch at the elbow end, back centre
    tools.append(cyl((0, wd.FOREARM_BOTTOM, 0), (0, 0, 1), 6.0, 60))
    back = fuse(back, *adds)
    return cut(back, *tools)


def make_servo_frame():
    """One printed frame holding both servos (base on the forearm's front wall, plates around the bodies)."""
    b1, b2 = wd.servo_box(1), wd.servo_box(2)
    c = 0.3                      # clearance around the servo bodies
    t = 3.0
    z0, ztop = FRAME_Z0, -12.0
    y_top = b1[1][1] + c + t
    y_bot = b2[1][0] - c - t
    parts = [box(-20.5, 20.5, y_bot - 9, y_top, z0, z0 + 2.25)]                 # base
    parts.append(box(-25.8, 25.8, b1[1][1] + c, y_top, z0 + 2.25, ztop))         # top plate
    parts.append(box(-25.8, 25.8, b2[1][1] + c, b1[1][0] - c, z0 + 2.25, ztop))  # middle plate (shared)
    parts.append(box(-25.8, 25.8, y_bot, b2[1][0] - c, z0 + 2.25, ztop))         # bottom plate
    for i, b in ((1, b1), (2, b2)):
        s = wd.SIDE[i]
        yb0, yb1 = b[1][0] - c, b[1][1] + c
        xb = b[0][0] if s > 0 else b[0][1]                                         # bottom face (x = -+12.5)
        xt = b[0][1] if s > 0 else b[0][0]                                         # top (horn) face (x = +-22.5)
        rail_b = box(*sorted([xb - s * c, xb - s * (c + t)]), yb0, yb1, z0 + 2.25, ztop)
        rail_b = rail_b.cut(box(*sorted([xb - s * 10, xb + s * 1]), b[1][0] + 3, b[1][1] - 3, b[2][0] + 1.5, -20.0))  # connector window
        rail_t = box(*sorted([xt + s * c, xt + s * (c + t)]), yb0, yb1, z0 + 2.25, -18.0)
        parts += [rail_b, rail_t]
    for (x, y) in FRAME_TABS:
        parts.append(box(x - 5, x + 5, y - 4, y + 4, z0, z0 + 5))
    xin = FRAME_SIDE_X - 0.2                                                       # side pads' face - 0.2
    for (x, y, z) in FRAME_SIDE:                                                   # blocks on the horn-side rails
        sx = np.sign(x)
        xr = sx * (wd.X_TOP_FACE + 0.3)
        zt, zr0 = z + 6, FRAME_Z0 + 2.25  # block on the rail with a ~45 deg underside inside the front corner radius
        xo = sx * (xin - 0.3)
        parts.append(prism_xz([(xr, zr0), (xr, zt), (xo, zt), (xo, zt - 14.0), (xr + sx * 0.5, zr0)], y - 7, y + 7))
    frame = fuse(*parts)
    # palm-side corners chamfered at 51 deg (clear of the servo bodies), so the forearm's oval section can hug them
    corner = [(19.8, z0 - 0.7), (40.0, z0 - 0.7), (40.0, z0 - 0.7 + 1.25 * 20.2)]
    tools = [prism_xz(corner, -140, -30), prism_xz([(-x, z) for x, z in corner], -140, -30)]
    tools += [cyl((x, y, z0 - 0.01), (0, 0, 1), 2.0, 5.0) for (x, y) in FRAME_TABS]  # M3 inserts from the front
    tools += [cyl((sx * xin + 0.01 * sx, y, z), (-sx, 0, 0), 2.0, 5.0) for (x, y, z) in FRAME_SIDE for sx in [np.sign(x)]]
    return cut(frame, *tools)


def make_horn_arm():
    """Printed arm (servo 1 / ulnar side, in the plane x in [X_TOP_FACE+3.5, +7.5]); servo 2 uses the same part."""
    x0 = wd.X_TOP_FACE + wd.HORN_GAP + wd.HORN_T
    x1 = x0 + wd.ARM_T
    ys, zs = wd.Y_SHAFT[1], wd.Z_SHAFT
    zt = zs + wd.R_HORN
    rt = wd.ARM_TIP_R
    arm = fuse(cyl((x0, ys, zs), (1, 0, 0), 12.5, wd.ARM_T), cyl((x0, ys, zt), (1, 0, 0), rt, wd.ARM_T),
               prism_yz([(ys - 9.5, zs), (ys + 9.5, zs), (ys + rt, zt), (ys - rt, zt)], x0, x1))
    tools = [cyl((x0 - 1, ys, zs), (1, 0, 0), 3.6, 8)]                                 # horn centre screw d7
    for k in range(4):                                                                 # 4 x M2.5 into the disc horn
        a = np.radians(45 + 90 * k)
        tools.append(cyl((x0 - 1, ys + 7 * np.cos(a), zs + 7 * np.sin(a)), (1, 0, 0), 1.35, 8))
    tools.append(cyl((x0 - 1, ys, zt), (1, 0, 0), 1.6, 8))                             # ball stud M3
    hexagon = [(ys + 3.23 * np.cos(np.radians(60 * k)), zt + 3.23 * np.sin(np.radians(60 * k))) for k in range(6)]
    tools.append(prism_yz(hexagon, x1 - 2.5, x1 + 0.1))                                # M3 nut, outer face
    return cut(arm, *tools)


STAND_SQUARE = 80.0                            # 4 x M5 stand screws on this square, 10 mm outside the shells' skin
FLANGE_A, FLANGE_B, FLANGE_N = 58.0, 58.0, 4.0   # elbow plate half-sizes (x, z) and exponent: M5 counterbores 7.5 mm from the edge


def make_elbow_flange():
    """Elbow plate: a superellipse centred under the forearm's bottom section, just large enough for the 4 x M5 stand
    screws on the STAND_SQUARE, outside the shells (heads reachable from above)."""
    y0, y1 = wd.ELBOW_Y, wd.FOREARM_BOTTOM
    _, zf, zb = forearm_profile(wd.FOREARM_BOTTOM)[:3]
    zc = 0.5 * (zf + zb)
    C = superellipse(FLANGE_A, zc - FLANGE_B, zc + FLANGE_B, zc, FLANGE_N, FLANGE_N, m=144)
    w = cq.Wire.assembleEdges([cq.Edge.makeSpline([V(x, y0, z) for x, z in C], periodic=True)])
    plate = cq.Solid.extrudeLinear(cq.Face.makeFromWires(w), V(0, y1 - y0, 0))
    plate = cq.Workplane().add(plate).edges("not |Y").chamfer(1.0).val()
    tools = [cyl((0, y0 - 1, zc), (0, 1, 0), 10.0, 10)]                                  # cable hole d20
    h = STAND_SQUARE / 2
    for sx in (1, -1):
        for sz in (1, -1):                                                              # 4 x M5, counterbored from the top
            tools += [cyl((sx * h, y0 - 1, zc + sz * h), (0, 1, 0), 2.75, 10),
                      cyl((sx * h, y1 - 3.5, zc + sz * h), (0, 1, 0), 4.9, 5)]
    for (x, z) in FLANGE_SCREWS:                                                        # M3x10 up into the shells
        tools += [cyl((x, y0 - 1, z), (0, 1, 0), 1.7, 10), cyl((x, y0 - 1, z), (0, 1, 0), 3.25, 1 + 3.0)]
    # weight-saving pocket in the top face, hidden inside the shells: 3 mm floor, islands under the shells' bosses
    pocket = cq.Solid.extrudeLinear(cq.Face.makeFromWires(section(y1 - 3.0, -wd.WALL - 3.0)), V(0, 4, 0))
    tools.append(pocket.cut(*[cyl((x, y1 - 4, z), (0, 1, 0), 6.0, 6) for (x, z) in FLANGE_SCREWS]))
    return cut(plate, *tools)


# ============================================================================ rod slots in the fork cap
def rod_slots(clear=4.5):
    """Convex slots (x, z) that the push rods (and the ball links' necks) sweep through the cap
    plane, over the joint-limit rectangle, grown by `clear` (neck radius 3.25 + 1.25 mm)."""
    ylo, yhi = wd.CAP_Y[0] - 3.0, wd.CAP_Y[1]
    pts = {1: [], 2: []}
    for phi in np.radians(np.linspace(-65, 65, 27)):
        for psi in np.radians(np.linspace(-30, 30, 13)):
            th = wd.servo_ik(phi, psi)
            if not np.all(np.isfinite(th)):
                continue
            for k, i in enumerate((1, 2)):
                a, b = wd.ball_bottom(i, th[k]), wd.ball_top(i, phi, psi)
                for t in np.linspace(0, 1, 200):
                    p = a + t * (b - a)
                    if ylo - 1 <= p[1] <= yhi + 1:
                        pts[i].append((p[0], p[2]))
    out = []
    circ = [(clear * np.cos(t), clear * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    for i in (1, 2):
        P = np.array(pts[i])
        P = np.array([p + c for p in P for c in circ])
        from scipy.spatial import ConvexHull
        h = ConvexHull(P)
        out.append([tuple(P[v]) for v in h.vertices])
    return out


# ============================================================================ bought parts (envelopes, W)
def servo_envelope():
    """STS3215 (servo 1): body box + output spline + 25T disc horn envelope."""
    b = wd.servo_box(1)
    xt = wd.X_TOP_FACE
    body = box(b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1])
    sh = (xt, wd.Y_SHAFT[1], wd.Z_SHAFT)
    return fuse(body, cyl(sh, (1, 0, 0), 3.0, wd.HORN_GAP + 0.01),
                cyl((xt + wd.HORN_GAP, sh[1], sh[2]), (1, 0, 0), wd.HORN_D / 2, wd.HORN_T))


def bearing_env(axis):
    a = np.asarray(axis, float)
    return fuse(cyl(a * wd.LUG_IN, a, wd.BRG_FD / 2, wd.BRG_FT),
                cyl(a * (wd.LUG_IN + wd.BRG_FT), a, wd.BRG_D / 2, wd.BRG_B)).cut(cyl(a * (wd.LUG_IN - 1), a, wd.PIN_D / 2, 10))


def pin_env(axis):
    a = np.asarray(axis, float)
    return cyl(a * (wd.LUG_OUT - wd.PIN_L), a, wd.PIN_D / 2, wd.PIN_L)


def stud_env(seat, ball, flange_d=6.0):
    seat, ball = np.asarray(seat, float), np.asarray(ball, float)
    d = ball - seat
    L = np.linalg.norm(d)
    return fuse(cyl(seat, d, flange_d / 2, 1.5), cyl(seat, d, 1.75, L), sphere(ball, 5.8 / 2))


def link_env(ball, toward, stud_axis):
    """M3 ball link: socket (d9.5 x 7 along the stud axis) around the ball + threaded neck d6.5 towards the rod."""
    ball, toward, s = (np.asarray(v, float) for v in (ball, toward, stud_axis))
    u = (toward - ball) / np.linalg.norm(toward - ball)
    s = s / np.linalg.norm(s)
    return fuse(cyl(ball - 3.5 * s, s, 4.75, 7.0), cyl(ball + 2 * u, u, 3.25, 9.0))


def bought_parts_neutral():
    """Envelopes of the bought parts at the neutral pose (W)."""
    out = {}
    s1 = servo_envelope()
    out["STS3215_motor1"] = s1
    out["STS3215_motor2"] = rot_z180(s1, Y_PAIR)
    for name, ax in AXES.items():
        out[f"F683ZZ_{name}"] = bearing_env(ax)
        out[f"pin_d3x10_{name}"] = pin_env(ax)
    for i in (1, 2):
        s = wd.SIDE[i]
        top = wd.P_TOP_HAND[i]
        bot = wd.ball_bottom(i, 0.0)
        out[f"stud_top_{i}"] = stud_env((s * wd.BAR_X, 0, wd.E_LEVER), top)
        arm_in = wd.X_TOP_FACE + wd.HORN_GAP + wd.HORN_T
        out[f"stud_arm_{i}"] = stud_env((s * arm_in, bot[1], bot[2]), bot)
        out[f"link_top_{i}"] = link_env(top, bot, (1, 0, 0))
        out[f"link_bot_{i}"] = link_env(bot, top, (1, 0, 0))
        a, b = bot + np.array([0, 11, 0]), top - np.array([0, 11, 0])
        out[f"rod_{i}"] = cyl(a, b - a, 1.5, np.linalg.norm(b - a))
    return out


# ============================================================================ checks
PRINT_UP = {"R-Wrist_Hand_Adapter": (0, -1, 0), "Wrist_Gimbal_Ring": (0, 1, 0), "R-Wrist_Fork": (0, 1, 0),
            "R-Forearm_Front": (0, 1, 0), "R-Forearm_Back": (0, 1, 0), "R-Wrist_Servo_Frame": (0, 0, 1),
            "Wrist_Horn_Arm": (1, 0, 0), "Elbow_Flange": (0, 1, 0)}


def print_up_cad(name):
    """Print 'up' in the CAD frame of the exported STL (PRINT_UP is in W; the adapter prints on its flange top,
    which is the CAD plane of the wrist interface)."""
    return np.array([0.0, -1.0, 0.0]) if name == "R-Wrist_Hand_Adapter" else wd.R_W2CAD @ np.asarray(PRINT_UP[name], float)


PRINT_NOTE = {
    "R-Wrist_Hand_Adapter": "flange face down (lugs up); the three 1.4 mm pad pockets bridge, gussets carry the lever bar",
    "Wrist_Gimbal_Ring": "flat",
    "R-Wrist_Fork": "cap face down, lugs up; bearing seats are teardrops",
    "R-Forearm_Front": "standing on the elbow end",
    "R-Forearm_Back": "standing on the elbow end",
    "R-Wrist_Servo_Frame": "base down (the face that lies on the forearm's front wall)",
    "Wrist_Horn_Arm": "flat, stud face down (nut pocket up)",
    "Elbow_Flange": "flat, underside down",
}


def wall_thickness(mesh, n=6000, seed=0):
    """Local wall thickness at random surface points: distance along -normal to the opposite face."""
    rng = np.random.default_rng(seed)
    pts, fi = trimesh.sample.sample_surface_even(mesh, n, seed=int(rng.integers(1 << 31)))
    nrm = mesh.face_normals[fi]
    origins = pts - nrm * 1e-3
    locs, idx_ray, _ = mesh.ray.intersects_location(origins, -nrm, multiple_hits=True)
    th = np.full(len(pts), np.inf)
    d = np.linalg.norm(locs - origins[idx_ray], axis=1)
    np.minimum.at(th, idx_ray, np.where(d > 1e-4, d, np.inf))
    return th[np.isfinite(th)]


def overhang_report(mesh, up):
    """Area of faces pointing down more than 45 deg from vertical in the print orientation, excluding the bed."""
    up = np.asarray(up, float)
    h = mesh.vertices @ up
    hmin = h.min()
    nz = mesh.face_normals @ up
    fh = mesh.triangles_center @ up
    a = mesh.area_faces
    bed = fh < hmin + 0.05
    flat_down = (nz < -0.985) & ~bed            # horizontal ceilings: bridges / roofs
    steep = (nz < -np.cos(np.radians(45))) & (nz >= -0.985) & ~bed
    return a[steep].sum(), a[flat_down].sum(), a.sum()


def check_parts(parts, bought):
    lines = []
    log("== printability")
    lines.append("Printability (STL, mm). Thickness = distance along -normal from random surface points to the opposite face.")
    for name, shp in parts.items():
        m = trimesh.load(STL_DIR / f"{name}.stl")
        th = wall_thickness(m)
        steep, flat, tot = overhang_report(m, print_up_cad(name))
        row = (f"{name:22s} watertight={m.is_watertight} faces={len(m.faces):6d} vol={m.volume / 1000:6.1f} cm3  "
               f"thickness p1={np.percentile(th, 1):4.2f} p5={np.percentile(th, 5):4.2f} median={np.median(th):4.2f} mm, "
               f"samples < 1.2 mm: {100 * np.mean(th < 1.2):4.1f} %  | print {PRINT_NOTE[name]}: overhang > 45 deg "
               f"{steep:6.0f} mm2, horizontal ceilings {flat:6.0f} mm2 (of {tot:6.0f})")
        lines.append(row)
        log("  " + row)
    log("== interference at the neutral pose (boolean intersection volume)")
    lines.append("")
    lines.append("Interference at the neutral pose: volume of the boolean intersection of every pair (mm3).")
    allp = {**parts, **bought}
    names = list(allp)
    # the arm/frame/servo2/horn2 instances placed for servo 2
    bbs = {n: allp[n].BoundingBox() for n in names}
    expected = {("Wrist_Gimbal_Ring", n) for n in names if n.startswith("pin_")}      # d4 pins pressed in d3.9 holes
    expected |= {(f"stud_{w}_{i}", f"link_{v}_{i}") for i in (1, 2) for w, v in (("top", "top"), ("arm", "bot"))}  # ball in socket
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            A, B = bbs[a], bbs[b]
            if A.xmin > B.xmax or B.xmin > A.xmax or A.ymin > B.ymax or B.ymin > A.ymax or A.zmin > B.zmax or B.zmin > A.zmax:
                continue
            vol = allp[a].intersect(allp[b]).Volume()
            if vol > 1e-3:
                tag = "expected (press fit / ball in its socket)" if (a, b) in expected or (b, a) in expected else "INTERFERENCE"
                if tag == "INTERFERENCE" and ((a.startswith("pin_") and b.startswith("F683")) or (b.startswith("pin_") and a.startswith("F683"))):
                    tag = "expected (pin in bearing bore, same d4)"
                row = f"  {a} / {b}: {vol:.3f} mm3  {tag}"
                lines.append(row)
                log(row)
    lines.append("  (pairs not listed: no intersection)")
    (PREV_DIR / "cad_checks.txt").write_text("\n".join(lines) + "\n")


# ============================================================================ main
def w_to_cad(shape):
    return shape.rotate(V(0, 0, 0), V(0, 0, 1), -np.degrees(wd.HAND_YAW)).translate(V(*wd.C_CAD))


def export(shape, name, folder_step=STEP_DIR, folder_stl=STL_DIR):
    s = w_to_cad(shape)
    if folder_step is not None:
        cq.exporters.export(s, str(folder_step / f"{name}.step"))
    cq.exporters.export(s, str(folder_stl / f"{name}.stl"), tolerance=STL_TOL, angularTolerance=STL_ANG)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-checks", action="store_true")
    args = ap.parse_args()
    for d in (STEP_DIR, STL_DIR, REF_DIR, PREV_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    slots = rod_slots()
    shell, mid = shell_solid()
    parts = {
        "R-Wrist_Hand_Adapter": make_hand_adapter(),
        "Wrist_Gimbal_Ring": make_ring(),
        "R-Wrist_Fork": make_fork(slots),
        "R-Forearm_Front": make_forearm_front(shell, mid),
        "R-Forearm_Back": make_forearm_back(shell, mid),
        "R-Wrist_Servo_Frame": make_servo_frame(),
        "Wrist_Horn_Arm": make_horn_arm(),
        "Elbow_Flange": make_elbow_flange(),
    }
    log(f"built {len(parts)} parts in {time.time() - t0:.1f} s")
    table = {}
    for name, shp in parts.items():
        n_sol = len(shp.Solids())
        export(shp, name)
        vol = trimesh.load(STL_DIR / f"{name}.stl").volume   # mesh volume: OCC's estimate is ~1-2 % off on the lofted skins
        bb = w_to_cad(shp).BoundingBox()
        table[name] = dict(volume_cm3=round(vol / 1000, 2), mass_g=round(vol / 1000 * PLA_DENSITY, 1), solids=n_sol,
                           valid=bool(shp.isValid()), bbox_cad_mm=[round(v, 2) for v in (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)],
                           print=PRINT_NOTE[name])
        log(f"  {name:22s} solids={n_sol} valid={shp.isValid()} volume {vol / 1000:6.2f} cm3")
    # left hand: every hand-specific part mirrored about the CAD plane x = -50 mm, the same rule that maps the
    # stock R-* parts onto the L-* parts (ring, horn arm and elbow flange are their own mirror images)
    for name in ("R-Wrist_Hand_Adapter", "R-Wrist_Fork", "R-Forearm_Front", "R-Forearm_Back", "R-Wrist_Servo_Frame"):
        lname = "L-" + name[2:]
        m = w_to_cad(parts[name]).mirror("YZ", (-50.0, 0, 0))
        cq.exporters.export(m, str(STEP_DIR / f"{lname}.step"))
        cq.exporters.export(m, str(STL_DIR / f"{lname}.stl"), tolerance=STL_TOL, angularTolerance=STL_ANG)
        table[lname] = dict(volume_cm3=table[name]["volume_cm3"], mass_g=table[name]["mass_g"], solids=len(m.Solids()),
                            valid=bool(m.isValid()), mirror_of=name, print=PRINT_NOTE[name])
    # instance of the arm on servo 2 (same part, rotated) for the checks / sim
    arm2 = rot_z180(parts["Wrist_Horn_Arm"], Y_PAIR)
    export(arm2, "Wrist_Horn_Arm_motor2_placed", None, REF_DIR)
    bought = bought_parts_neutral()
    for name, shp in bought.items():
        export(shp, name, None, REF_DIR)
    meta = dict(frame="CAD part-studio frame of the AmazingHand (mm), neutral wrist pose",
                wrist_centre_cad_mm=wd.C_CAD.tolist(), hand_yaw_deg=float(np.degrees(wd.HAND_YAW)), parts=table,
                rod_lengths_mm={f"motor{i}": wd.ROD_L[i] for i in (1, 2)},
                note="Wrist_Horn_Arm: print 2 (one per servo); stl/ref: bought-part envelopes and the placed second arm")
    (HERE / "wrist_parts.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not args.no_checks:
        bought["Wrist_Horn_Arm_motor2"] = arm2
        check_parts(parts, bought)
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
