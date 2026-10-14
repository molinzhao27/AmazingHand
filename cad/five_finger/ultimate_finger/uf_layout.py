"""2D layer layout of the "ultimate" finger cores (thin 3-phalanx finger for the 5-digit AmazingHand).

Shared by make_ultimate_finger.py (CAD) and the figures. Everything is in the finger design frame of the
stock CAD (mm): x lateral (pin axes), y distal along the straight finger, z dorsal; MCP at the origin. Every
part is drawn in place in the straight pose; the Link in the stock Link's own frame (like the stock STEP).

The linkage (segment lengths, P4, A, B, Link arm angle) is the recommended design of the 3-phalanx study
(cad/five_finger/three_segment/design.json, kept unchanged: it is verified and well conditioned). What is new
here is the lateral stacking, which sets the finger width:

    band   |x| (mm)      parts
    C      0    .. 1.5   Link arm (3.0 thick, was 3.8), DIP coupler L2 (3.0 thick)
    g1     1.5  .. 1.75  running gap (filled only where a part bridges its own bands)
    M      1.75 .. 4.0   P2 at the PIP (inside P1's fork), P3 at the DIP (inside P2's fork)
    g2     4.0  .. 4.25  running gap
    O      4.25 .. 6.0   P1 fork at the PIP, P2 fork at the DIP; P3's tip body only to 4.75

so both inner knuckles are 12.0 mm wide (stock proximal core 16.2). Pins: 2 x 12 (PIP, DIP, A: flush with
the 12 mm forks), 2 x 8 (P4, B: flush with the 8.0 mm inner parts, retained by the outer fork 0.25 mm away);
the MCP (2 x 16) and the Link pivot (2 x 10) are stock.

Each part is the union of slabs (x0, x1, 2D (y, z) profile). A slab is carved by the swept envelope (over the
whole servo travel) of every slab of every other part overlapping it laterally (with the 0.25 mm running
gap), grown by CLR in-plane; parts are added in priority order (knuckle "essentials" first, then the
"fillers"), so every pair of parts is collision-free in the flexion plane by construction.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM3 = REPO / "Demo" / "AHSimulation" / "three_segment"
sys.path.insert(0, str(SIM3))
from linkage import O_L, O_M, Design  # noqa: E402

STOCK_STEP = REPO / "cad" / "step"
STOCK_STL = REPO / "cad" / "stl"
DESIGN3 = REPO / "cad" / "five_finger" / "three_segment" / "design.json"

# ---------------------------------------------------------------- lateral layers (mm, Link mid-plane = 0)
C_HALF = 1.5
LAT = 0.25                       # running gap between parts in neighbouring bands
M_BAND = (1.75, 4.0)
O_BAND = (4.25, 6.0)
O_P2 = (4.25, 5.85)              # P2's outer walls (the middle sleeve sits on them); its DIP-pin bosses reach 6.0
TIP_O = (4.25, 4.75)             # P3 tip body beyond the DIP fork: 9.5 mm wide core
BANDS = dict(C=(-C_HALF, C_HALF), g1=(C_HALF, M_BAND[0]), M=M_BAND, g2=(M_BAND[1], O_BAND[0]), O=O_BAND)
CLR = 0.6                        # in-plane clearance between parts sharing a band
OPEN_R = 0.45                    # carved profiles are opened by this radius: no slivers thinner than 0.9 mm
LINK_DX, PROX_DX = 0.35, -0.25   # stock assembly: Link 0.35 mm off the gimbal mid-plane, Proximal -0.25
HOLE_PRESS, HOLE_RUN = 1.95, 2.15
PIN_R = 1.0
ARM_ROOT_H, NECK_H, END_R = 4.0, 2.25, 2.6   # Link arm: 8 mm tall at the root tapering to 4.5 at the neck (its bending
#                                              moment falls linearly towards P4), P4 eye (running: 1.53 mm wall)
L2_R, L2_H = 2.6, 1.75                   # L2 eyes (1.53 mm wall), bar half height (3.5 mm)
R_PIP_P1, R_A = 3.0, 2.6                 # P1 fork bosses (press fit)
R_PIP_P2, R_P4 = 2.7, 2.5                # P2 bosses at the PIP (running) and P4 (press)
R_DIP_P2 = 3.0                           # P2 fork boss at the DIP (press)
R_DIP_P3, R_B = 2.7, 2.5                 # P3 bosses at the DIP (running) and B (press)
P1_Z = (-3.3, 9.0)                       # P1 shaft profile (z), as tall as the stock Proximal's body
P2_Z = (-4.4, 4.4)                       # P2 shaft profile
P1_JOIN = 7.0                            # stock MCP end of P1 (cut from the stock STEP) up to here
BALL = np.array([3.596, -11.051])        # rod-ball stud centre on the stock Link (Link frame)
RES = 32

# pins: name -> (frame of the part that holds it (press fit), centre key, half length, parts allowed to touch it)
PINS = {
    "MCP": ("P1", "O_M", 8.0, ("P1", "gimbal")),          # stock 2 x 16
    "LinkPivot": ("gimbal", "O_L", 5.0, ("gimbal", "link_base", "link_arm")),  # stock 2 x 10
    "PIP": ("P1", "PIP", 6.0, ("P1", "P2")),              # 2 x 12
    "A": ("P1", "A", 6.0, ("P1", "L2")),                  # 2 x 12, crosses P2's band in a slot
    "P4": ("P2", "P4", 4.0, ("P2", "link_arm")),          # 2 x 8
    "DIP": ("P2", "DIP", 6.0, ("P2", "P3")),              # 2 x 12
    "B": ("P3", "B", 4.0, ("P3", "L2")),                  # 2 x 8
}
PIN_SPEC = {"MCP": "2x16 (stock)", "LinkPivot": "2x10 (stock)", "PIP": "2x12", "A": "2x12", "P4": "2x8", "DIP": "2x12",
            "B": "2x8"}


def circle(c, r):
    return Point(*c).buffer(r, quad_segs=RES)


def hull(*geoms):
    return unary_union(geoms).convex_hull


def bar(a, b, h):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]]) * h
    return Polygon([a + n, b + n, b - n, a - n])


def clean(geom, min_area=1.0):
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    keep = [p for p in geoms if isinstance(p, Polygon) and p.area >= min_area]
    return unary_union(keep) if keep else Polygon()


# ---------------------------------------------------------------- design + motion
# UF_SCALE (environment) < 1 builds a shorter variant (the pinky): the three phalanx lengths are scaled, the pin
# offsets at the PIP and DIP (the cranks P4, A, B) and the whole drive are kept, so the four-bars keep their shape.
SCALE = float(os.environ.get("UF_SCALE", "1.0"))
SUFFIX = "" if SCALE == 1.0 else f"_S{int(round(SCALE * 100)):03d}"
design3 = json.loads(DESIGN3.read_text())
_g = design3["geometry"]
_L1, _L2 = _g["L1"] * SCALE, _g["L2"] * SCALE
_P, _Dp = np.array([_L1, 0.0]), np.array([_L1 + _L2, 0.0])
D = Design(_L1, _L2, _g["L3_tip"] * SCALE, _P + np.array(_g["P4"]) - [_g["L1"], 0], _P + np.array(_g["A"]) - [_g["L1"], 0],
           _Dp + np.array(_g["B"]) - [_g["L1"] + _g["L2"], 0], np.radians(design3["drive"]["psi_straight_deg"]), _g["tip"][1])
PIP, DIP, P4, A, B = D.PIP0, D.DIP0, D.P4_0, D.A0, D.B0
P4L = D.link_arm_in_link_frame()
TIP_END = D.L1 + D.L2 + _g["L3_end"] * SCALE            # 91.5 (full size): fingertip core end (the TPU cap adds ~1.5)
g3 = dict(_g, L1=D.L1, L2=D.L2, L3_tip=round(D.L3t, 3), L3_end=round(_g["L3_end"] * SCALE, 3), PIP=PIP.tolist(),
          DIP=DIP.tolist(), P4=np.round(P4, 3).tolist(), A=np.round(A, 3).tolist(), B=np.round(B, 3).tolist(),
          tip=np.round(D.TIP0, 3).tolist(), P4_in_link_frame=np.round(P4L, 4).tolist(), link_length=round(D.L_link, 3),
          crank_P2=round(D.c, 3), crank_A=round(D.rA, 3), crank_B=round(D.rB, 3), coupler_L2=round(D.dAB, 3), scale=SCALE)
POINTS = dict(O_M=O_M, O_L=O_L, PIP=PIP, DIP=DIP, P4=P4, A=A, B=B)

drive = np.load(SIM3 / "data" / "stock_flex_line.npz")
PSI = np.radians(np.arange(drive["psi"].min(), drive["psi"].max() + 1e-9, 0.25))
SOL = D.solve(PSI)
assert SOL["ok"].all(), "design does not assemble over the whole servo travel"


def M2(theta, pivot, now):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    T = np.eye(3)
    T[:2, :2], T[:2, 2] = R, now - R @ pivot
    return T


def pose(frame, k, sol=None, psi=None):
    """2D rigid motion (3x3) of a part frame (straight-pose coordinates; Link: its own frame) -> gimbal frame."""
    s = SOL if sol is None else sol
    psi = PSI if psi is None else psi
    if frame == "gimbal":
        return np.eye(3)
    if frame == "link":
        return M2(psi[k], O_L, O_L)
    if frame == "P1":
        return M2(s["mcp"][k], O_M, O_M)
    if frame == "P2":
        return M2(s["mcp"][k] + s["pip"][k], PIP, s["PIP"][k])
    if frame == "P3":
        return M2(s["mcp"][k] + s["pip"][k] + s["dip"][k], DIP, s["DIP"][k])
    if frame == "L2":
        return M2(s["mcp"][k] + s["l2"][k], A, s["A"][k])
    raise KeyError(frame)


def apply(T, geom):
    return affinity.affine_transform(geom, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 2], T[1, 2]])


_sweep_cache = {}


def sweep(geom, src, dst, step=1):
    """Union of geom (in frame src) over the motion, expressed in frame dst."""
    if geom.is_empty or src == dst:
        return geom
    key = (geom.wkb, src, dst, step)
    if key in _sweep_cache:
        return _sweep_cache[key]
    gs = geom.simplify(0.02)
    out = [apply(np.linalg.inv(pose(dst, k)) @ pose(src, k), gs) for k in range(0, len(PSI), step)]
    res = unary_union(out)
    _sweep_cache[key] = res
    return res


# ---------------------------------------------------------------- parts made of slabs
class Part:
    def __init__(self, name, frame, lat=LAT, own_pins=()):
        self.name, self.frame, self.slabs, self.lat, self.own_pins = name, frame, [], lat, set(own_pins)

    def add(self, x0, x1, geom):
        if not geom.is_empty:
            self.slabs.append((x0, x1, geom))


REG = []  # every part, in the order it was laid out


def register(part):
    REG.append(part)
    return part


def obstacles(part, x0, x1):
    out = []
    for p in REG:
        if p is part or p.name in part.own_pins or p.name.removeprefix("pin_") in part.own_pins:
            continue
        for a, b, geom in p.slabs:
            if min(b + p.lat, x1) - max(a - p.lat, x0) > 1e-6 and not geom.is_empty:
                out.append((p.frame, geom))
    return out


def carve(part, x0, x1, blank, clr=CLR):
    obs = [sweep(geom, src, part.frame).buffer(clr, quad_segs=8) for src, geom in obstacles(part, x0, x1)]
    return blank.difference(unary_union(obs)) if obs else blank


def add_band(part, band, blank, min_area=1.0, keep=None, both=True):
    """Carve blank for the band (x0, x1) and its mirror (separately: the gimbal is 0.35 mm off-centre), add."""
    x0, x1 = band
    bands = [(x0, x1)] + ([(-x1, -x0)] if both and x0 >= 0 else [])
    for a, b in bands:
        geom = carve(part, a, b, blank)
        geom = geom.buffer(-OPEN_R, quad_segs=8).buffer(OPEN_R, quad_segs=8)   # drop slivers < 0.9 mm (opening: only removes)
        geom = clean(geom.intersection(blank), min_area)
        if keep is not None:
            geom = keep(geom)
        part.add(a, b, geom)


def stl_slabs(path, bands, ymax=None, n=5, dx=0.0):
    """Projection of a stock STL into x-bands (band edges in the STL's own x + dx)."""
    mesh = trimesh.load(path)
    out = []
    for x0, x1 in bands:
        polys = []
        for xs in np.linspace(x0 + 0.03, x1 - 0.03, n):
            sec = mesh.section(plane_origin=[xs + dx, 0, 0], plane_normal=[1, 0, 0])
            if sec is None:
                continue
            loops = sorted((Polygon(lp[:, [1, 2]]) for lp in sec.discrete if len(lp) >= 4), key=lambda p: -p.area)
            geom = Polygon()
            for p in loops:
                geom = geom.symmetric_difference(p.buffer(0))
            polys.append(geom)
        geom = unary_union(polys)
        if ymax is not None:
            geom = geom.intersection(box(-50, -50, ymax, 50))
        out.append((x0, x1, geom))
    return out


def dorsal_only(zmin):
    def keep(geom):
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        return unary_union([q for q in geoms if q.bounds[3] >= zmin and q.area > 2])
    return keep


def biggest(geom):
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    geoms = [g for g in geoms if not g.is_empty]
    return max(geoms, key=lambda g: g.area) if geoms else Polygon()


def tip_body():
    tip_c = np.array([TIP_END - 4.9, -0.7])
    return hull(box(DIP[0] - 3.0, -4.9, DIP[0] + 7.0, 4.3), circle(tip_c, 4.9)), tip_c


def layout():
    """Lay out every part; returns dict name -> Part (cores, stock obstacles, pins)."""
    REG.clear()
    _sweep_cache.clear()
    P = {}
    # ---- stock obstacles
    gim = register(Part("gimbal", "gimbal", lat=0.0, own_pins=("MCP", "LinkPivot")))
    edges = [-5.35, -4.25, -1.5, 1.5, 4.25, 4.65]
    for x0, x1, geom in stl_slabs(STOCK_STL / "Amazing Hand Parts - Gimbal.stl", list(zip(edges[:-1], edges[1:])),
                                  dx=LINK_DX):
        gim.add(x0, x1, geom)
    base_mask = hull(circle(O_L, 3.25), circle(BALL, 3.6))
    base = register(Part("link_base", "link", own_pins=("LinkPivot", "P4")))
    for x0, x1, geom in stl_slabs(STOCK_STL / "Amazing Hand Parts - Link.stl", [(-5.15, -1.5), (-1.5, 1.5), (1.5, 5.15)],
                                  ymax=7.5):
        base.add(x0, x1, geom.intersection(base_mask))
    ball = register(Part("ball_studs", "link", own_pins=("LinkPivot", "P4")))
    for s in (1, -1):
        ball.add(*sorted((s * 5.15, s * 13.0)), circle(BALL, 4.2))
    pins = {}
    for name, (frame, key, half, _) in PINS.items():
        p = register(Part(f"pin_{name}", frame))
        p.add(-half, half, circle(POINTS[key], PIN_R))
        pins[name] = p
    # ---- Link arm and L2 (fixed shapes; clearances to the pins come from the synthesis)
    u = (P4L - O_L) / np.linalg.norm(P4L - O_L)
    root, neck = O_L + 5.6 * u, P4L - 9.0 * u
    arm2d = unary_union([hull(circle(root, ARM_ROOT_H), circle(neck, NECK_H)), hull(circle(neck, NECK_H), circle(P4L, NECK_H)),
                         circle(P4L, END_R)]).intersection(box(root[0], -40, 200, 40)).union(circle(root, NECK_H + 0.4))
    arm = register(Part("link_arm", "link", own_pins=("P4", "LinkPivot")))
    arm.add(-C_HALF, C_HALF, arm2d)
    l2 = register(Part("L2", "L2", own_pins=("A", "B")))
    l2.add(-C_HALF, C_HALF, unary_union([circle(A, L2_R), circle(B, L2_R), bar(A, B, L2_H)]))

    p1 = register(Part("P1", "P1", own_pins=("MCP", "PIP", "A")))
    p2 = register(Part("P2", "P2", own_pins=("PIP", "P4", "DIP")))
    p3 = register(Part("P3", "P3", own_pins=("DIP", "B")))
    body3, tip_c = tip_body()

    # ---- essentials (knuckles), in priority order
    yp = PIP[0]
    p1_fork = unary_union([hull(circle(PIP, R_PIP_P1), circle(A, R_A)),
                           hull(box(yp - 9.0, P1_Z[0], yp - 3.0, P1_Z[1]), circle(PIP, R_PIP_P1)),
                           hull(box(yp - 9.0, P1_Z[0], yp - 3.0, P1_Z[0] + 1), circle(A, R_A))])
    add_band(p1, O_BAND, p1_fork, keep=biggest)
    add_band(p2, M_BAND, hull(circle(PIP, R_PIP_P2), circle(P4, R_P4)), keep=biggest)
    p2_fork = unary_union([circle(DIP, R_DIP_P2), box(DIP[0] - 7.0, P2_Z[0], DIP[0], P2_Z[1])])
    add_band(p2, O_P2, p2_fork, keep=biggest)
    add_band(p2, (O_P2[1], O_BAND[1]), circle(DIP, R_DIP_P2), keep=biggest)   # pin bosses flush with the 2 x 12 pin
    p3_m = unary_union([hull(circle(DIP, R_DIP_P3), circle(B, R_B)), body3])
    add_band(p3, M_BAND, p3_m, keep=biggest)

    # ---- fillers: P2 shaft, P1 shaft, P3 tip
    p2_shaft = unary_union([box(PIP[0], P2_Z[0], DIP[0] - 2.0, P2_Z[1]), hull(circle(PIP, R_PIP_P2), circle(P4, R_P4))])
    add_band(p2, M_BAND, p2_shaft, min_area=2.0)
    add_band(p2, BANDS["g2"], unary_union([p2_shaft, p2_fork]), min_area=2.0)
    add_band(p2, O_P2, unary_union([box(PIP[0] + 4.0, P2_Z[0], DIP[0], P2_Z[1]), p2_fork]), min_area=2.0)
    add_band(p2, BANDS["g1"], box(PIP[0] + 1.5, P2_Z[0], DIP[0] - 4.0, P2_Z[1]), min_area=2.0)
    add_band(p2, BANDS["C"], box(PIP[0] + 2.0, P2_Z[0], DIP[0] - 4.5, P2_Z[1]), min_area=4.0, both=False)

    p1_shaft = unary_union([box(P1_JOIN, P1_Z[0], yp - 2.5, P1_Z[1]), hull(box(yp - 7.0, P1_Z[0], yp - 2.5, P1_Z[1]),
                                                                            circle(PIP, R_PIP_P1))])
    add_band(p1, O_BAND, p1_shaft, min_area=2.0)
    add_band(p1, BANDS["g2"], p1_shaft, min_area=2.0)
    add_band(p1, M_BAND, p1_shaft, min_area=2.0)
    add_band(p1, BANDS["g1"], p1_shaft, min_area=2.0)
    add_band(p1, BANDS["C"], box(P1_JOIN, P1_Z[0], yp - 2.5, P1_Z[1]), min_area=4.0, both=False,
             keep=dorsal_only(P1_Z[1] - 0.3))

    add_band(p3, BANDS["g2"], body3, min_area=2.0, keep=biggest)
    add_band(p3, TIP_O, body3, min_area=2.0, keep=biggest)
    add_band(p3, BANDS["g1"], body3, min_area=2.0, keep=biggest)
    add_band(p3, BANDS["C"], body3, min_area=4.0, both=False, keep=biggest)

    P.update(gimbal=gim, link_base=base, ball=ball, link_arm=arm, L2=l2, P1=p1, P2=p2, P3=p3, pins=pins,
             base_mask=base_mask, arm2d=arm2d, tip_c=tip_c, body3=body3)
    return P
