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


def section_wire(y, a, zf, zb, rf, rb, inset=0.0):
    """Rounded rectangle in the plane W y = const: |x| <= a, zf <= z <= zb, front / back corner radii."""
    a, zf, zb = a - inset, zf + inset, zb - inset
    rf, rb = max(rf - inset, 0.5), max(rb - inset, 0.5)
    P = lambda x, z: V(x, y, z)
    def arc(cx, cz, r, a0, a1):
        am = 0.5 * (a0 + a1)
        return cq.Edge.makeThreePointArc(P(cx + r * np.cos(a0), cz + r * np.sin(a0)), P(cx + r * np.cos(am), cz + r * np.sin(am)),
                                         P(cx + r * np.cos(a1), cz + r * np.sin(a1)))
    e = [cq.Edge.makeLine(P(-a + rf, zf), P(a - rf, zf)), arc(a - rf, zf + rf, rf, -np.pi / 2, 0),
         cq.Edge.makeLine(P(a, zf + rf), P(a, zb - rb)), arc(a - rb, zb - rb, rb, 0, np.pi / 2),
         cq.Edge.makeLine(P(a - rb, zb), P(-a + rb, zb)), arc(-a + rb, zb - rb, rb, np.pi / 2, np.pi),
         cq.Edge.makeLine(P(-a, zb - rb), P(-a, zf + rf)), arc(-a + rf, zf + rf, rf, np.pi, 1.5 * np.pi)]
    return cq.Wire.assembleEdges(e)


def forearm_profile(y):
    """Outer section parameters (a, zf, zb, rf, rb) at height y: smoothstep blend between FOREARM_KEYS."""
    K = wd.FOREARM_KEYS
    y = float(np.clip(y, K[-1][0], K[0][0]))
    for k0, k1 in zip(K, K[1:]):
        if k1[0] <= y <= k0[0]:
            t = (k0[0] - y) / (k0[0] - k1[0])
            t = t * t * (3 - 2 * t)
            return tuple((1 - t) * np.array(k0[1:]) + t * np.array(k1[1:]))
    return K[0][1:]


def section(y, inset=0.0, at=None):
    return section_wire(y, *forearm_profile(y if at is None else at), inset=inset)


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


def make_hand_adapter():
    yb = wd.H_FLANGE - wd.FLANGE_T                     # 18.5: flange underside
    yt = wd.H_FLANGE + 1.2                             # 26.2: mates the interface's recessed pads (CAD y -101)
    flange = cyl((0, yb, 0), (0, 1, 0), wd.FLANGE_R, yt - yb)
    # pockets for the interface's three protruding pads (CAD y -102.2 .. -101; sectors 31-89, 151-209, 271-329 deg)
    pockets = [sector(16.5, 30, c - 30.2, c + 30.2, yt - 1.45, yt + 1) for c in (60, 180, 300)]
    holes = [cyl((0, yb - 1, 0), (0, 1, 0), 11.0, 12)]            # cable bore d22
    for ang in (20, -20, 140, -140):                               # 4 x 3.0x12 PT / M3x12 screw, head from below
        x, z = 21 * np.cos(np.radians(ang)), 21 * np.sin(np.radians(ang))
        holes += [cyl((x, yb - 1, z), (0, 1, 0), 1.7, 12), cyl((x, yb - 1, z), (0, 1, 0), 3.25, 1 + 3.0)]
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
    body = cq.Solid.extrudeLinear(cq.Face.makeFromWires(section(y0, 0.0)), V(0, y1 - y0, 0))
    body = cq.Workplane().add(body).faces(">Y").edges().fillet(2.0).val()      # soft top edge
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
    """Forearm shell: smooth loft (wrist taper) + exact straight prism (servo region) + loft (elbow flare)."""
    K = wd.FOREARM_KEYS
    y0, y1, y2, y3 = K[0][0], K[1][0], K[2][0], K[3][0]

    def piece(inset, ext):
        top = cq.Solid.makeLoft([section(y, inset) for y in np.linspace(y0, y1, 11)], False)
        mid = cq.Solid.extrudeLinear(cq.Face.makeFromWires(section(y1, inset)), V(0, y2 - y1, 0))
        bot = cq.Solid.makeLoft([section(y, inset) for y in np.linspace(y2, y3, 13)], False)
        parts = [top, mid, bot]
        if ext:   # extend the cavity through both ends
            parts += [cq.Solid.extrudeLinear(cq.Face.makeFromWires(section(y0, inset)), V(0, 3, 0)),
                      cq.Solid.extrudeLinear(cq.Face.makeFromWires(section(y3, inset)), V(0, -3, 0))]
        return fuse(*parts)

    return piece(0.0, False).cut(piece(wd.WALL, True)).clean()


def boss_with_web(x, z, y0, y1, side_x, insert_from_top=True, cone=True):
    """Vertical screw boss d7 (M3 heat-set insert) tied to the side wall at x = side_x by a 3 mm web
    (side_x None: no web)."""
    parts = [cyl((x, y0, z), (0, 1, 0), 3.5, y1 - y0)]
    if side_x is not None:
        xa, xb = sorted([x, side_x])
        parts.append(box(xa, xb, y0, y1, z - 1.5, z + 1.5))
    if cone:   # 45 deg support cone under a boss that hangs from the top (standing print)
        if insert_from_top:
            parts.append(cq.Solid.makeCone(0.01, 3.5, 3.5, pnt=V(x, y0 - 3.5, z), dir=V(0, 1, 0)))
        else:
            parts.append(cq.Solid.makeCone(3.5, 0.01, 3.5, pnt=V(x, y1, z), dir=V(0, 1, 0)))
    s = fuse(*parts)
    yi = y1 - 6 if insert_from_top else y0
    return s, cyl((x, yi, z), (0, 1, 0), 2.0, 6.01)


def corner_boss_xy(y, sign_x, front):
    """Boss centre in a corner of the interior outline at height y, on the corner bisector."""
    a, zf, zb, rf, rb = forearm_profile(y)
    ri = (rf if front else rb) - wd.WALL
    cx = sign_x * (a - wd.WALL - ri)
    cz = (zf + wd.WALL + ri) if front else (zb - wd.WALL - ri)
    d = ri - 3.5 + 0.5
    return cx + sign_x * d / np.sqrt(2), cz + (-1 if front else 1) * d / np.sqrt(2)


A_MID, ZF_MID, ZB_MID, _, _ = forearm_profile(-90.0)          # servo region
CAP_SCREWS = [corner_boss_xy(wd.CAP_Y[0], s, True) for s in (1, -1)] + [(s * 7.0, ZB_MID - wd.WALL - 3.0) for s in (1, -1)]
FLANGE_SCREWS = [corner_boss_xy(wd.FOREARM_BOTTOM, s, True) for s in (1, -1)] + [corner_boss_xy(wd.FOREARM_BOTTOM, s, False) for s in (1, -1)]
FRAME_TABS = [(s * 15.0, wd.Y_SHAFT[2] - wd.SERVO_W / 2 - 3.3 - 4.5) for s in (1, -1)]     # front-wall screws
FRAME_SIDE = [(wd.SIDE[i] * (A_MID - wd.WALL), wd.Y_SHAFT[i], -24.5) for i in (1, 2)]      # side-wall screws
FRAME_Z0 = ZF_MID + wd.WALL + 0.2    # front wall inner face in the servo region (+0.2 clearance)


def make_forearm_front(shell):
    front = shell.intersect(box(-80, 80, -400, 0, -100, wd.SPLIT_Z))
    adds, tools = [], []
    for (x, z) in CAP_SCREWS[:2]:
        s, t = boss_with_web(x, z, wd.CAP_Y[0] - 12, wd.CAP_Y[0], np.sign(x) * (A_MID - 1.0), True)
        adds.append(s)
        tools.append(t)
    a_bot = forearm_profile(wd.FOREARM_BOTTOM + 12)[0]
    for (x, z) in FLANGE_SCREWS[:2]:
        s, t = boss_with_web(x, z, wd.FOREARM_BOTTOM, wd.FOREARM_BOTTOM + 12, np.sign(x) * (a_bot - 1.0), False)
        adds.append(s)
        tools.append(t)
    for (x, y) in FRAME_TABS:      # countersunk M3x10 from the palm side into the servo frame's inserts
        tools += [cyl((x, y, -60), (0, 0, 1), 1.7, 30),
                  cq.Solid.makeCone(3.4, 1.6, 1.8, pnt=V(x, y, -45.01), dir=V(0, 0, 1))]
    for (x, y, z) in FRAME_SIDE:   # countersunk M3x8 through the side walls into the frame's side blocks
        sx = np.sign(x)
        tools += [cyl((x - sx * 1, y, z), (sx, 0, 0), 1.7, 10),
                  cq.Solid.makeCone(3.4, 1.6, 1.8, pnt=V(x + sx * (wd.WALL + 0.01), y, z), dir=V(-sx, 0, 0))]
    front = fuse(front, *adds)
    return cut(front, *tools)


def make_forearm_back(shell):
    back = shell.intersect(box(-80, 80, -400, 0, wd.SPLIT_Z, 100))
    adds, tools = [], []
    zb_in = ZB_MID - wd.WALL
    for (x, z) in CAP_SCREWS[2:]:
        s, t = boss_with_web(x, z, wd.CAP_Y[0] - 12, wd.CAP_Y[0], None, True)
        adds.append(s.fuse(box(x - 1.5, x + 1.5, wd.CAP_Y[0] - 12, wd.CAP_Y[0], z, zb_in + 1)))
        tools.append(t)
    a_bot = forearm_profile(wd.FOREARM_BOTTOM + 12)[0]
    for (x, z) in FLANGE_SCREWS[2:]:
        s, t = boss_with_web(x, z, wd.FOREARM_BOTTOM, wd.FOREARM_BOTTOM + 12, np.sign(x) * (a_bot - 1.0), False)
        adds.append(s)
        tools.append(t)
    # ribs that keep each servo seated in the frame (press on the gear-end face, z = 5.25)
    z_end = wd.servo_box(1)[2][1] + 0.3
    for i in (1, 2):
        ys = wd.Y_SHAFT[i]
        for xr in (-10.0, 10.0):
            pts = [(ys + 6, z_end), (ys - 6, z_end), (ys - 6 - (zb_in + 0.5 - z_end), zb_in + 0.5), (ys + 6, zb_in + 0.5)]
            adds.append(prism_yz(pts, xr - 1.0, xr + 1.0))
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
    parts = [box(-21.5, 21.5, y_bot - 9, y_top, z0, z0 + 2.25)]                 # base (inside the front corner radii)
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
    xin = A_MID - wd.WALL - 0.2                                                    # side-wall inner face - 0.2
    for (x, y, z) in FRAME_SIDE:                                                   # blocks on the horn-side rails
        sx = np.sign(x)
        xr = sx * (wd.X_TOP_FACE + 0.3)
        zt, zr0 = z + 6, FRAME_Z0 + 2.25  # block on the rail with a ~45 deg underside inside the front corner radius
        xo = sx * (xin - 0.3)
        parts.append(prism_xz([(xr, zr0), (xr, zt), (xo, zt), (xo, zt - 14.0), (xr + sx * 0.5, zr0)], y - 7, y + 7))
    frame = fuse(*parts)
    tools = [cyl((x, y, z0 - 0.01), (0, 0, 1), 2.0, 5.0) for (x, y) in FRAME_TABS]   # M3 inserts from the front
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


def make_elbow_flange():
    y0, y1 = wd.ELBOW_Y, wd.FOREARM_BOTTOM
    _, zf, zb, _, _ = forearm_profile(wd.FOREARM_BOTTOM)
    zc = 0.5 * (zf + zb)
    half = 60.0
    w = cq.Wire.makePolygon([V(-half, y0, zc - half), V(half, y0, zc - half), V(half, y0, zc + half), V(-half, y0, zc + half)], close=True)
    plate = cq.Solid.extrudeLinear(cq.Face.makeFromWires(w.fillet2D(16.0, w.Vertices())), V(0, y1 - y0, 0))
    plate = cq.Workplane().add(plate).edges("not |Y").chamfer(1.0).val()
    tools = [cyl((0, y0 - 1, zc), (0, 1, 0), 10.0, 10)]                                  # cable hole d20
    for sx in (1, -1):
        for sz in (1, -1):                                                              # 4 x M5 on a 100 mm square
            tools += [cyl((sx * 50, y0 - 1, zc + sz * 50), (0, 1, 0), 2.75, 10),
                      cyl((sx * 50, y1 - 3.5, zc + sz * 50), (0, 1, 0), 4.9, 5)]
    for (x, z) in FLANGE_SCREWS:                                                        # M3x10 up into the shells
        tools += [cyl((x, y0 - 1, z), (0, 1, 0), 1.7, 10), cyl((x, y0 - 1, z), (0, 1, 0), 3.25, 1 + 3.0)]
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
        steep, flat, tot = overhang_report(m, PRINT_UP[name])
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
def export(shape, name, folder_step=STEP_DIR, folder_stl=STL_DIR):
    s = shape.translate(V(*wd.C_CAD))       # W -> CAD frame
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
    shell = shell_solid()
    parts = {
        "R-Wrist_Hand_Adapter": make_hand_adapter(),
        "Wrist_Gimbal_Ring": make_ring(),
        "R-Wrist_Fork": make_fork(slots),
        "R-Forearm_Front": make_forearm_front(shell),
        "R-Forearm_Back": make_forearm_back(shell),
        "R-Wrist_Servo_Frame": make_servo_frame(),
        "Wrist_Horn_Arm": make_horn_arm(),
        "Elbow_Flange": make_elbow_flange(),
    }
    log(f"built {len(parts)} parts in {time.time() - t0:.1f} s")
    table = {}
    for name, shp in parts.items():
        n_sol = len(shp.Solids())
        export(shp, name)
        vol = shp.Volume()
        bb = shp.translate(V(*wd.C_CAD)).BoundingBox()
        table[name] = dict(volume_cm3=round(vol / 1000, 2), mass_g=round(vol / 1000 * PLA_DENSITY, 1), solids=n_sol,
                           valid=bool(shp.isValid()), bbox_cad_mm=[round(v, 2) for v in (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)],
                           print=PRINT_NOTE[name])
        log(f"  {name:22s} solids={n_sol} valid={shp.isValid()} volume {vol / 1000:6.2f} cm3")
    # left hand: every hand-specific part mirrored about the CAD plane x = -50 mm, the same rule that maps the
    # stock R-* parts onto the L-* parts (ring, horn arm and elbow flange are their own mirror images)
    for name in ("R-Wrist_Hand_Adapter", "R-Wrist_Fork", "R-Forearm_Front", "R-Forearm_Back", "R-Wrist_Servo_Frame"):
        lname = "L-" + name[2:]
        m = parts[name].translate(V(*wd.C_CAD)).mirror("YZ", (-50.0, 0, 0))
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
                wrist_centre_cad_mm=wd.C_CAD.tolist(), parts=table,
                rod_lengths_mm={f"motor{i}": wd.ROD_L[i] for i in (1, 2)},
                note="Wrist_Horn_Arm: print 2 (one per servo); stl/ref: bought-part envelopes and the placed second arm")
    (HERE / "wrist_parts.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not args.no_checks:
        bought["Wrist_Horn_Arm_motor2"] = arm2
        check_parts(parts, bought)
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
