#!/usr/bin/env python3
"""Concept CAD of the 3-phalanx AmazingHand finger (CadQuery 2.x + shapely). Not production-ready.

Reads design.json (written by Demo/AHSimulation/three_segment/synth_linkage.py) and builds, in the
finger design frame of the stock CAD (mm; x lateral = pin axes, y distal, z dorsal; every part is
drawn in place in the straight pose, exactly like the stock STEP files):

  Link_3S          the stock Link's base (pivot, ball-stud boss; cut from the stock STEP) with a new
                   arm to P4. Drawn in the stock Link's own frame, so the rods/balls are unchanged.
  P1_Proximal_3S   stock Proximal MCP end (cut from the stock STEP) + new body to the PIP, with the
                   palmar lug that carries pin A.
  P2_Middle_3S     middle phalanx: PIP, P4 (Link) and DIP pins; the old Distal's role (coupler).
  P3_Distal_3S     distal phalanx: fork round P2, DIP and B pins, fingertip core.
  L2_Coupler_3S    DIP coupler A-B (two-force member).
  P1_Shell_3S, P2_Pad_3S, P3_Shell_3S   TPU covers (concept).

Layering (x, mm): centre +-1.9 (Link arm, L2, as thick as the stock Link arm), P2 walls 2.15..5.0,
outer forks 5.25..8.1 (P1 walls, P3 cheeks, as wide as the stock Proximal). Pins are the stock
2 mm parallel pins (2x16 through the 16.2 mm forks, 2x10 at P4). Every part is the blank of its
layer minus the swept envelope (over the whole servo travel, sampled every 0.25 deg of Link angle)
of every other part and pin that shares that layer, grown by a 0.6 mm clearance; so the parts are
collision-free in the flexion plane by construction (the sim re-checks it in 3D).

Outputs: step/*.step, stl/*.stl, step/Finger_3S_assembly.step (straight pose), figures/cad_*.png,
and the "cad" section of design.json (file names, which sim body carries which mesh).

Usage: python make_three_segment.py
"""

import json
import sys
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM = REPO / "Demo" / "AHSimulation" / "three_segment"
sys.path.insert(0, str(SIM))
from linkage import O_L, O_M, Design  # noqa: E402

STOCK_STEP = REPO / "cad" / "step"
STOCK_STL = REPO / "cad" / "stl"
DESIGN = HERE / "design.json"
OUT_STEP, OUT_STL, FIG = HERE / "step", HERE / "stl", HERE / "figures"

# ---------------------------------------------------------------- parameters (mm)
CEN = 1.9  # centre layer half width (stock Link arm is 3.8 mm thick)
P2W = (2.15, 5.0)  # P2 walls
OUT = (5.25, 8.1)  # P1 walls, P3 cheeks (stock Proximal is 16.2 mm wide)
CLR = 0.6  # in-plane clearance between parts sharing a layer
# Lateral (x) offsets of the stock CAD frames in the stock assembly, relative to the gimbal (read from
# the stock MJCF by analyze_stock_finger.py / checked by build_finger_models.py): the Link sits 0.35 mm
# off the gimbal mid-plane, the Proximal -0.25 mm. Every new part is drawn on the Link's mid-plane
# (x = 0 here = Link frame), so the reused stock MCP end of P1 is moved by PROX_DX - LINK_DX.
LINK_DX, PROX_DX = 0.35, -0.25
PIN_R = 1.0
HOLE_PRESS, HOLE_RUN = 1.95, 2.15  # pin held / pin turning (diameters)
SCREW_PILOT = 2.0  # M2.5 thermoplastic screw (stock STD00447) pilot
SCREW_CLEAR = 2.8
ARM_H, NECK_H, END_R = 2.75, 2.25, 2.6  # Link arm half height, neck half height near P4, end boss
L2_R, L2_H = 2.75, 2.0  # L2 end bosses, bar half height
SHELL_T = 1.6  # TPU wall
RES = 32  # shapely circle segments per quarter


BALL = np.array([3.596, -11.051])  # rod-ball stud centre on the stock Link (Link frame)


def circle(c, r):
    return Point(*c).buffer(r, quad_segs=RES)


def hull(*geoms):
    return unary_union(geoms).convex_hull


# ---------------------------------------------------------------- design + motion
design = json.loads(DESIGN.read_text())
D = Design.from_json(design)
g = design["geometry"]
PIP, DIP, P4, A, B = D.PIP0, D.DIP0, D.P4_0, D.A0, D.B0
P4L = np.array(g["P4_in_link_frame"])
TIP_END = g["L1"] + g["L2"] + g["L3_end"]

drive = np.load(SIM / "data" / "stock_flex_line.npz")
PSI = np.radians(np.arange(drive["psi"].min(), drive["psi"].max() + 1e-9, 0.25))
SOL = D.solve(PSI)
assert SOL["ok"].all(), "design does not assemble over the whole servo travel"


def M(theta, pivot, now):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    T = np.eye(3)
    T[:2, :2], T[:2, 2] = R, now - R @ pivot
    return T


def pose(frame, k):
    """2D rigid motion: part frame (straight-pose coords; Link: its own frame) -> gimbal frame."""
    s = SOL
    if frame == "gimbal":
        return np.eye(3)
    if frame == "link":
        return M(PSI[k], O_L, O_L)
    if frame == "P1":
        return M(s["mcp"][k], O_M, O_M)
    if frame == "P2":
        return M(s["mcp"][k] + s["pip"][k], PIP, s["PIP"][k])
    if frame == "P3":
        return M(s["mcp"][k] + s["pip"][k] + s["dip"][k], DIP, s["DIP"][k])
    if frame == "L2":
        return M(s["mcp"][k] + s["l2"][k], A, s["A"][k])
    raise KeyError(frame)


def apply(T, geom):
    return affinity.affine_transform(geom, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 2], T[1, 2]])


def sweep(geom, src, dst, step=1):
    """Union of geom (in frame src) over the motion, expressed in frame dst."""
    if geom.is_empty:
        return geom
    if src == dst:
        return geom
    geom = geom.simplify(0.02)
    out = []
    for k in range(0, len(PSI), step):
        T = np.linalg.inv(pose(dst, k)) @ pose(src, k)
        out.append(apply(T, geom))
    return unary_union(out)


# ---------------------------------------------------------------- slabs: (x0, x1, frame, 2D geometry)
LAT_CLR = 0.25  # lateral (x) clearance between neighbouring layers of different parts


class Part:
    def __init__(self, name, frame, lat=LAT_CLR):
        self.name, self.frame, self.slabs, self.lat = name, frame, [], lat

    def add(self, x0, x1, geom, sym=True):
        self.slabs.append((x0, x1, geom))
        if sym and x0 >= 0:
            self.slabs.append((-x1, -x0, geom))


def stl_slabs(path, bands, ymax=None, n=5):
    """Projection of a stock STL into x-bands: union of its sections at n planes inside each band."""
    mesh = trimesh.load(path)
    out = []
    for x0, x1 in bands:
        polys = []
        for xs in np.linspace(x0 + 0.03, x1 - 0.03, n):
            sec = mesh.section(plane_origin=[xs, 0, 0], plane_normal=[1, 0, 0])
            if sec is None:
                continue
            loops = sorted((Polygon(l[:, [1, 2]]) for l in sec.discrete if len(l) >= 4), key=lambda p: -p.area)
            geom = Polygon()
            for p in loops:  # even-odd: holes are the loops inside others
                geom = geom.symmetric_difference(p.buffer(0))
            polys.append(geom)
        geom = unary_union(polys)
        if ymax is not None:
            geom = geom.intersection(box(-50, -50, ymax, 50))
        out.append((x0, x1, geom))
    return out


def obstacles_in(x0, x1, parts, exclude=()):
    """Union of the swept envelopes of every slab overlapping (x0, x1), per source frame."""
    return [(p.frame, geom) for p in parts if p.name not in exclude
            for (a, b, geom) in p.slabs if min(b + p.lat, x1) - max(a - p.lat, x0) > 1e-6 and not geom.is_empty]


def carve(blank, x0, x1, frame, parts, exclude=(), clr=CLR):
    obs = [sweep(geom, src, frame).buffer(clr, quad_segs=8) for src, geom in obstacles_in(x0, x1, parts, exclude)]
    return blank.difference(unary_union(obs)) if obs else blank


def add_carved(part, x0, x1, blank, others, min_area=1.0, keep=None, both=True):
    """Carve blank for the band (x0, x1) (and, if both, separately for the mirrored band: the gimbal
    is not centred on the Link plane, so the two sides differ) and add the slabs to part."""
    bands = [(x0, x1)] + ([(-x1, -x0)] if both and x0 >= 0 else [])
    for a, b in bands:
        geom = clean(carve(blank, a, b, part.frame, others), min_area)
        if keep is not None:
            geom = keep(geom)
        if not geom.is_empty:
            part.add(a, b, geom, sym=False)


def clean(geom, min_area=1.0):
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    keep = [p for p in geoms if isinstance(p, Polygon) and p.area >= min_area]
    return unary_union(keep) if keep else Polygon()


# ---------------------------------------------------------------- CadQuery helpers
def extrude(geom, x0, x1):
    """Extrude a (multi)polygon given in (y, z) along x from x0 to x1."""
    solid = None
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    for poly in geoms:
        if poly.is_empty or poly.area < 1e-3:
            continue
        poly = shapely.geometry.polygon.orient(poly.simplify(0.01), 1.0)
        wp = cq.Workplane("YZ", origin=(x0, 0, 0)).polyline(list(poly.exterior.coords)[:-1]).close().extrude(x1 - x0)
        for ring in poly.interiors:
            hole = cq.Workplane("YZ", origin=(x0 - 0.1, 0, 0)).polyline(list(ring.coords)[:-1]).close().extrude(
                x1 - x0 + 0.2)
            wp = wp.cut(hole)
        solid = wp if solid is None else solid.union(wp)
    return solid


def build_solid(part):
    solid = None
    for x0, x1, geom in part.slabs:
        s = extrude(geom, x0, x1)
        if s is not None:
            solid = s if solid is None else solid.union(s, tol=1e-4)
    return solid


def pin_hole(solid, c, d, x0=-9.0, x1=9.0):
    cyl = cq.Workplane("YZ", origin=(x0, c[0], c[1])).circle(d / 2).extrude(x1 - x0)
    return solid.cut(cyl)


def cut_box_y(solid, ymin=None, ymax=None):
    b = cq.Workplane("XY").box(100, (ymax if ymax is not None else 200) - (ymin if ymin is not None else -200), 100,
                               centered=(True, False, True)).translate((0, ymin if ymin is not None else -200, 0))
    return solid.intersect(b)


# ---------------------------------------------------------------- the parts
def main():
    t0 = time.time()
    for p in (OUT_STEP, OUT_STL, FIG):
        p.mkdir(parents=True, exist_ok=True)
    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        log.append(s)
        print(s, flush=True)

    say(f"design: P1 {g['L1']}, P2 {g['L2']}, P3 to tip {g['L3_tip']} mm; P4 {P4}, A {A}, B {B}; "
        f"motion sampled at {len(PSI)} Link angles {np.degrees(PSI[0]):.1f}..{np.degrees(PSI[-1]):.1f} deg")

    # ---- stock obstacles (2D slabs from the stock STLs)
    gimbal = Part("gimbal", "gimbal", lat=0.0)  # P1's MCP end bears on the gimbal face (as in the stock)
    for x0, x1, geom in stl_slabs(STOCK_STL / "Amazing Hand Parts - Gimbal.stl",
                                  [(-5.0, -2.15), (-2.15, 2.15), (2.15, 5.0)]):
        gimbal.add(x0 - LINK_DX, x1 - LINK_DX, geom, sym=False)  # gimbal frame -> Link mid-plane frame
    # Link base kept from the stock part: only the pivot boss and the ball-stud boss (the stock arm's
    # root rises towards the MCP and would hit the gimbal once the straight pose moves to f ~ -56)
    LINK_CUT = 7.5
    base_mask = hull(circle(O_L, 3.25), circle(BALL, 3.6))
    link_base = Part("link_base", "link")
    for x0, x1, geom in stl_slabs(STOCK_STL / "Amazing Hand Parts - Link.stl",
                                  [(-5.15, -1.9), (-1.9, 1.9), (1.9, 5.15)], ymax=LINK_CUT):
        link_base.add(x0, x1, geom.intersection(base_mask), sym=False)
    ball = Part("ball_studs", "link")  # ball studs + rod sockets on the Link base, both sides
    ball.add(5.15, 13.0, circle(BALL, 4.2))
    pins = {}
    for name, frame, c, half in (("pin_MCP", "P1", O_M, 8.0), ("pin_PIP", "P1", PIP, 8.0), ("pin_A", "P1", A, 8.0),
                                 ("pin_P4", "P2", P4, 5.0), ("pin_DIP", "P2", DIP, 8.0), ("pin_B", "P3", B, 8.0),
                                 ("pin_LinkPivot", "gimbal", O_L, 5.0)):
        p = Part(name, frame)
        p.add(-half, half, circle(c, PIN_R), sym=False)
        pins[name] = p

    # ---- Link arm (centre layer, Link frame)
    u = (P4L - O_L) / np.linalg.norm(P4L - O_L)
    root = O_L + 5.6 * u
    neck = P4L - 9.0 * u
    arm2d = unary_union([hull(circle(root, ARM_H), circle(neck, ARM_H)), hull(circle(neck, NECK_H), circle(P4L, NECK_H)),
                         circle(P4L, END_R)])
    link_arm = Part("link_arm", "link")
    link_arm.add(-CEN, CEN, arm2d, sym=False)

    # ---- L2 coupler (centre layer, L2 frame = straight pose)
    l2_2d = unary_union([circle(A, L2_R), circle(B, L2_R), _bar(A, B, L2_H)])
    l2 = Part("L2", "L2")
    l2.add(-CEN, CEN, l2_2d, sym=False)

    # ---- P2 middle phalanx
    p2_blank = unary_union([hull(circle(PIP, 3.2), circle(P4, 3.0)), box(PIP[0], -4.6, DIP[0], 4.0),
                            circle(PIP, 3.2), circle(DIP, 3.2), hull(circle(PIP, 3.2), circle(P4, 3.0), Point(PIP[0], 4.0))])
    p2 = Part("P2", "P2")
    others = [gimbal, link_base, link_arm, ball, l2] + [pins[k] for k in ("pin_MCP", "pin_A", "pin_B", "pin_LinkPivot")]
    add_carved(p2, *P2W, p2_blank, others)
    add_carved(p2, -P2W[0], P2W[0], p2_blank, others, min_area=4.0, both=False)

    # ---- P1 proximal: stock MCP end (y < 7.5) + new body
    P1_JOIN = 7.0
    p1_blank_w = unary_union([box(P1_JOIN, -3.3, 42.0, 9.5),
                              hull(box(38.0, -3.3, 42.0, 9.5), circle(PIP, 3.3), circle(A, 3.0)),
                              hull(circle(PIP, 3.3), circle(A, 3.0))])
    p1 = Part("P1", "P1")
    others1 = [gimbal, link_base, link_arm, ball, l2, p2] + [pins[k] for k in ("pin_P4", "pin_DIP", "pin_B",
                                                                               "pin_LinkPivot")]
    add_carved(p1, *OUT, p1_blank_w, others1)
    # inner region: only the dorsal part (open towards the palm, like the stock Proximal: the Link
    # runs in it and must be insertable), split at every layer boundary of the obstacles
    inner_blank = box(P1_JOIN, -3.3, 44.5, 9.5).intersection(p1_blank_w)
    edges = [0.0, CEN, P2W[0], P2W[1], 5.15, OUT[0]]
    def dorsal_only(geom):
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        return unary_union([q for q in geoms if q.intersects(box(-50, 8.6, 100, 9.6)) and q.area > 2])

    for x0, x1 in zip(edges[:-1], edges[1:]):
        add_carved(p1, x0, x1, inner_blank, others1, keep=dorsal_only)

    # ---- P3 distal phalanx
    tip_c = np.array([TIP_END - 5.3, -0.6])
    p3 = Part("P3", "P3")
    others3 = [link_arm, l2, p2, p1] + [pins[k] for k in ("pin_PIP", "pin_A", "pin_P4")]
    cheeks = unary_union([hull(circle(DIP, 3.3), circle(B, 3.0)), hull(circle(DIP, 3.3), box(DIP[0], -4.8, 78.0, 4.6)),
                          hull(circle(B, 3.0), box(DIP[0], -4.8, 78.0, 4.6))])
    add_carved(p3, *OUT, cheeks, others3)
    body = hull(box(DIP[0] - 4.0, -5.2, 80.0, 5.0), circle(tip_c, 5.3))
    def tip_piece(geom):  # keep the piece that holds the fingertip
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        return unary_union([q for q in geoms if q.intersects(circle(tip_c, 1.0))])

    for x0, x1 in ((0.0, CEN), (CEN, P2W[0]), (P2W[0], P2W[1]), (P2W[1], OUT[0])):
        add_carved(p3, x0, x1, body, others3, min_area=2.0, keep=tip_piece)

    # ---- shells / pads (TPU)
    p1_shell = Part("P1_shell", "P1")
    outer = box(9.5, -9.2, 40.5, 11.9).buffer(-3.0).buffer(3.0, quad_segs=RES)
    cavity = box(9.5 - 1, -7.6, 40.5 + 1, 10.0)
    others_s1 = [gimbal, link_base, link_arm, ball, l2, p2, p3] + list(pins.values())
    add_carved(p1_shell, 8.35, 10.0, outer, others_s1)
    p1_shell.add(-8.35, 8.35, clean(carve(outer.difference(cavity), -8.35, 8.35, "P1", others_s1), 2.0), sym=False)
    p2_pad = Part("P2_pad", "P2")
    pad = box(53.5, -7.2, 67.5, -4.6).buffer(-1.2).buffer(1.2, quad_segs=RES)
    p2_pad.add(-P2W[1], P2W[1], clean(carve(pad, -P2W[1], P2W[1], "P2", [p1, p3, l2] + list(pins.values())), 2.0),
               sym=False)
    p3_shell = Part("P3_shell", "P3")
    souter = hull(box(78.5, -9.0, 80.5, 7.0), circle(tip_c - [0, 1.2], 7.3)).intersection(box(78.5, -20, 120, 20))
    scav = body.buffer(0.25)
    others_s3 = [l2, p2, p1, link_arm] + list(pins.values())
    add_carved(p3_shell, 5.5, 9.0, souter, others_s3)
    p3_shell.add(-5.5, 5.5, clean(carve(souter.difference(scav), -5.5, 5.5, "P3", others_s3), 2.0), sym=False)

    say(f"2D layouts carved [{time.time() - t0:.0f} s]")
    for part in (link_arm, l2, p1, p2, p3, p1_shell, p2_pad, p3_shell):
        say(f"  {part.name:<9} " + "  ".join(f"[{a:+.2f},{b:+.2f}] {gm.area:5.0f}" for a, b, gm in part.slabs))

    # ---------------------------------------------------------------- solids
    solids = {}
    # Link: stock base (cut) + new arm, in the stock Link frame
    stock_link = cq.importers.importStep(str(STOCK_STEP / "Amazing Hand Parts - Link.step"))
    base = stock_link.intersect(extrude(base_mask, -6.0, 6.0))
    arm = extrude(arm2d, -CEN, CEN)
    link = base.union(arm, tol=1e-4)
    solids["link3s"] = pin_hole(link, P4L, HOLE_RUN, -CEN - 1, CEN + 1)

    # P1: stock MCP end + new body
    stock_p1 = cq.importers.importStep(str(STOCK_STEP / "Amazing Hand Parts - Proximal.step"))
    p1_end = cut_box_y(stock_p1, ymax=P1_JOIN + 0.5).translate((PROX_DX - LINK_DX, 0, 0))
    # the reused MCP end must now clear the new Link and a 13 deg larger MCP range: cut the swept
    # envelopes of the gimbal, the Link (base + new arm) and the rod-ball studs, band by band
    obst = [gimbal, link_base, link_arm, ball]
    xs = {-OUT[0], OUT[0]}
    for p in obst:
        for a, b, _ in p.slabs:
            xs |= {float(np.clip(a - p.lat, -OUT[0], OUT[0])), float(np.clip(b + p.lat, -OUT[0], OUT[0]))}
    xs = sorted(xs)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        if x1 - x0 < 1e-3:
            continue
        cut = unary_union([sweep(gm, src, "P1").buffer(CLR, quad_segs=8) for src, gm in
                           obstacles_in(x0, x1, obst)]).intersection(box(-20, -30, P1_JOIN + 1.0, 30))
        if not cut.is_empty:
            p1_end = p1_end.cut(extrude(cut, x0 - 1e-3, x1 + 1e-3))
    s = p1_end.union(build_solid(p1), tol=1e-4)
    for c in (PIP, A):
        s = pin_hole(s, c, HOLE_PRESS)
    for ys in (20.0, 34.0):  # shell screws into the walls
        for sgn in (1, -1):
            s = s.cut(cq.Workplane("YZ", origin=(sgn * 4.0 if sgn > 0 else -9.0, ys, 4.5)).circle(SCREW_PILOT / 2)
                      .extrude(5.0))
    solids["p1"] = s

    s = build_solid(p2)
    s = pin_hole(s, PIP, HOLE_RUN)
    s = pin_hole(s, P4, HOLE_PRESS)
    s = pin_hole(s, DIP, HOLE_RUN)
    solids["p2"] = s

    s = build_solid(p3)
    s = pin_hole(s, DIP, HOLE_PRESS)
    s = pin_hole(s, B, HOLE_PRESS)
    s = s.cut(cq.Workplane("YZ", origin=(-9.0, 86.0, 0.0)).circle(SCREW_PILOT / 2).extrude(18.0))
    solids["p3"] = s

    s = build_solid(l2)
    s = pin_hole(s, A, HOLE_RUN)
    s = pin_hole(s, B, HOLE_RUN)
    solids["l2"] = s

    s = build_solid(p1_shell)
    for ys in (20.0, 34.0):
        s = s.cut(cq.Workplane("YZ", origin=(-11.0, ys, 4.5)).circle(SCREW_CLEAR / 2).extrude(22.0))
    solids["p1_shell"] = s
    solids["p2_pad"] = build_solid(p2_pad)
    s = build_solid(p3_shell)
    s = s.cut(cq.Workplane("YZ", origin=(-11.0, 86.0, 0.0)).circle(SCREW_CLEAR / 2).extrude(22.0))
    solids["p3_shell"] = s

    names = dict(link3s="Link_3S", p1="P1_Proximal_3S", p2="P2_Middle_3S", p3="P3_Distal_3S", l2="L2_Coupler_3S",
                 p1_shell="P1_Shell_3S", p2_pad="P2_Pad_3S", p3_shell="P3_Shell_3S")
    material = dict(link3s="PLA", p1="PLA", p2="PLA", p3="PLA", l2="PLA", p1_shell="TPU (Filaflex 82A)",
                    p2_pad="TPU (Filaflex 82A)", p3_shell="TPU (Filaflex 82A)")
    stl_names = {}
    say("\npart              material   volume cm3   mass g   bbox mm (x, y, z)              solids  watertight")
    for key, sol in solids.items():
        shape = sol.val() if len(sol.vals()) == 1 else cq.Compound.makeCompound(sol.vals())
        n_solids = len(sol.solids().vals())
        cq.exporters.export(sol, str(OUT_STEP / f"{names[key]}.step"))
        stl = OUT_STL / f"{names[key]}.stl"
        cq.exporters.export(sol, str(stl), tolerance=0.02, angularTolerance=0.15)
        tm = trimesh.load(stl)
        bb = shape.BoundingBox()
        rho = 1.24 if material[key] == "PLA" else 1.21
        say(f"{names[key]:<17} {material[key][:4]:<6} {shape.Volume() / 1e3:10.2f} {shape.Volume() * rho / 1e3:8.2f}   "
            f"{bb.xlen:5.1f} x {bb.ylen:5.1f} x {bb.zlen:5.1f}            {n_solids:3d}     {tm.is_watertight} "
            f"({len(tm.faces)} faces)")
        assert n_solids == 1, f"{key} is not one solid"
        stl_names[key] = stl.name

    # assembly in the straight pose (Link: rotate its frame by psi_s about the Link pivot)
    psi_s = np.radians(design["drive"]["psi_straight_deg"])
    asm = cq.Assembly(name="Finger_3S_straight")
    colors = dict(link3s="orange", p1="gray80", p2="gray70", p3="gray80", l2="steelblue", p1_shell="white",
                  p2_pad="gray30", p3_shell="white")
    for key, sol in solids.items():
        loc = cq.Location()
        if key == "link3s":  # rotation about the x axis through O_L by -psi_s (flexion sign: -angle about +x)
            loc = cq.Location(cq.Vector(0, O_L[0], O_L[1])) * cq.Location(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0),
                                                                          -np.degrees(psi_s)) * \
                cq.Location(cq.Vector(0, -O_L[0], -O_L[1]))
        asm.add(sol, name=names[key], loc=loc, color=cq.Color(colors[key]))
    asm.save(str(OUT_STEP / "Finger_3S_assembly.step"))

    design["cad"] = dict(
        stl=stl_names, step={k: f"{v}.step" for k, v in names.items()},
        bodies=dict(p1=["p1", "p1_shell"], p2=["p2", "p2_pad"], p3=["p3", "p3_shell"], l2=["l2"]),
        layers_mm=dict(centre=[-CEN, CEN], p2_walls=list(P2W), outer_forks=list(OUT)),
        holes_mm=dict(press=HOLE_PRESS, running=HOLE_RUN, screw_pilot=SCREW_PILOT, screw_clear=SCREW_CLEAR),
        clearance_mm=CLR, link_cut_y=LINK_CUT, lateral_offset_mm=dict(link_and_new_parts=LINK_DX, stock_proximal=PROX_DX,
                                                                        p1_mcp_end_shift=PROX_DX - LINK_DX), p1_join_y=P1_JOIN, tip_end_y=TIP_END,
    )
    DESIGN.write_text(json.dumps(design, indent=2) + "\n")
    figures(dict(link_arm=link_arm, link_base=link_base, l2=l2, p1=p1, p2=p2, p3=p3, p1_shell=p1_shell,
                 p2_pad=p2_pad, p3_shell=p3_shell, gimbal=gimbal), pins, say)
    say(f"\nwrote step/, stl/, figures/ and design.json['cad']  [{time.time() - t0:.0f} s]")
    (HERE / "make_three_segment.log").write_text("\n".join(log) + "\n")


def _perp(a, b):
    d = (b - a) / np.linalg.norm(b - a)
    return np.array([-d[1], d[0]])


def _bar(a, b, h):
    n = _perp(a, b) * h
    return Polygon([a + n, b + n, b - n, a - n])


# ---------------------------------------------------------------- figures
def figures(parts, pins, say):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    sys.path.insert(0, str(SIM))
    import plotstyle as ps

    ps.setup()

    def draw(ax, geom, T, color, alpha=0.55, lw=0.6):
        geom = apply(T, geom)
        for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            if poly.is_empty or not isinstance(poly, Polygon):
                continue
            verts, codes = [], []
            for ring in [poly.exterior, *poly.interiors]:
                c = np.asarray(ring.coords)
                verts += c.tolist()
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec=ps.INK, lw=lw, alpha=alpha))

    col = dict(gimbal="#b9b8b3", link_base=ps.C[0], link_arm=ps.C[0], p1=ps.C[1], p2=ps.C[2], p3=ps.C[3],
               l2=ps.C[4], p1_shell="#dddcd8", p2_pad="#9a9994", p3_shell="#dddcd8")
    frame = dict(gimbal="gimbal", link_base="link", link_arm="link", p1="P1", p2="P2", p3="P3", l2="L2",
                 p1_shell="P1", p2_pad="P2", p3_shell="P3")
    kk = [int(np.argmin(np.abs(PSI - D.psi_s))), int(np.argmin(np.abs(PSI - np.radians(31.3)))), len(PSI) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, k, title in zip(axes, kk, ("straight", "f = 0", "full servo travel")):
        for name in ("p1_shell", "p3_shell", "p2_pad", "gimbal", "p1", "p3", "p2", "link_base", "link_arm", "l2"):
            part = parts[name]
            # draw the widest slab of each part (side view silhouette = union of all slabs)
            geom = unary_union([s[2] for s in part.slabs])
            draw(ax, geom, pose(frame[name], k), col[name], alpha=0.35 if "shell" in name or "pad" in name else 0.6)
        for p in pins.values():
            draw(ax, p.slabs[0][2], pose(p.frame, k), "#ffffff", alpha=1.0, lw=0.8)
        s = SOL
        ax.set_title(f"{title}: MCP {np.degrees(s['mcp'][k]):.1f}, PIP {np.degrees(s['pip'][k]):.1f}, "
                     f"DIP {np.degrees(s['dip'][k]):.1f} deg", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-45, 100)
        ax.set_ylim(-80, 25)
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm, dorsal +)")
    fig.suptitle("Concept CAD, side silhouettes (all layers) at three poses - blue Link, orange P1, aqua P2, "
                 "yellow P3, pink L2, grey shells", fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(FIG / "cad_side_poses.png", dpi=130)
    plt.close(fig)

    # per-part layer drawings
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, name in zip(axes.ravel(), ("p1", "p2", "p3", "l2", "link_arm", "p1_shell")):
        part = parts[name]
        shown = [sl for sl in part.slabs if sl[0] >= -1e-9 or sl[0] == -sl[1]]
        for i, (x0, x1, geom) in enumerate(shown):
            draw(ax, geom, np.eye(3), ps.C[i % 5], alpha=0.35)
            ax.plot([], [], color=ps.C[i % 5], lw=6, alpha=0.5, label=f"|x| {max(x0, 0):.2f}..{x1:.2f}")
        for lab, c in (("MCP", O_M), ("PIP", PIP), ("DIP", DIP), ("P4", P4), ("A", A), ("B", B)):
            if name == "link_arm" and lab not in ("P4",):
                continue
            ax.plot(*c, "+", color=ps.INK, ms=6)
            ax.annotate(lab, c, xytext=(3, 3), textcoords="offset points", fontsize=7, color=ps.INK2)
        if name == "link_arm":
            ax.plot(*P4L, "+", color=ps.INK)
            ax.annotate("P4 (Link frame)", P4L, xytext=(3, 3), textcoords="offset points", fontsize=7, color=ps.INK2)
            draw(ax, unary_union([s[2] for s in parts["link_base"].slabs]), np.eye(3), ps.C[0], alpha=0.2)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.legend(loc="lower left", fontsize=6.5, ncol=2)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Layer profiles (y, z) of the new parts (one colour per x-layer)", color=ps.INK)
    fig.tight_layout()
    fig.savefig(FIG / "cad_layer_profiles.png", dpi=120)
    plt.close(fig)
    say("figures: cad_side_poses.png, cad_layer_profiles.png")


if __name__ == "__main__":
    main()
