#!/usr/bin/env python3
"""The "ultimate" AmazingHand finger: thin, styled, 3 phalanges, on the unchanged 2-servo drive.

Merges three earlier studies of the 5-digit hand:
  * cad/five_finger/three_segment   3 phalanges with the stock drive (linkage kept unchanged, see uf_layout.py);
  * cad/five_finger/slim_fingers    TPU shells hugging the cores (0.25 mm clearance, >= 1.2 mm wall), flat shell
                                    ends chosen from a joint-range sweep (>= 0.5 mm), exposed knuckles;
  * cad/five_finger/fingers         TPU grip-pad inserts in a 0.45 mm land with a 0.1 mm glue gap, rounded top edge.

Styling (after the user's reference hand): white TPU covers with rounded-rectangle sections (generous corner radii,
solid TPU where the outline rounds beyond the core), black TPU pads with fine transverse ribs (1 mm pitch) over most
of the palm face of every phalanx, and a fingertip cap that is a little fuller than its neck, with a domed, nail-like
back that rolls over the end (soft "D" side profile) and the ribbed pad wrapping around the tip.

New here: the cores are re-stacked for slimness (uf_layout.py: 3.0 mm centre links, 2 x 12 / 2 x 8 pins, 12.0 mm
knuckles instead of 16.2), the shells are sleeves/caps lofted around each core plus everything that moves inside
it, and the thumb gets a matching pad set on the slim-finger parts (it stays 2-phalanx).

Frame (mm): the finger design frame of the stock CAD: x lateral (pin axes), y distal, z dorsal (palm = -z), MCP at
the origin, every part drawn in place in the straight pose; the Link in the stock Link's own frame. The thumb parts
are in the stock 2-phalanx finger frame (as cad/five_finger/slim_fingers).

Writes step/*.step, stl/*.stl (incl. pins and the textured pads), figures/*.png, design.json and report_cad.txt.

Usage: .venv/bin/python cad/five_finger/ultimate_finger/make_ultimate_finger.py [--no-thumb] [--no-figures]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import shapely
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
if "--scale" in sys.argv:   # must be known before uf_layout is imported (it builds the geometry at import)
    import os
    os.environ["UF_SCALE"] = sys.argv[sys.argv.index("--scale") + 1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "slim_fingers"))
import uf_layout as U  # noqa: E402
import make_slim_fingers as SF  # noqa: E402  (loft / sampling helpers and the slim thumb parts)

OUT_STEP, OUT_STL, FIG = HERE / "step", HERE / "stl", HERE / "figures"
DESIGN = HERE / f"design{U.SUFFIX}.json"
HW_DIR = U.REPO / "Demo" / "AHSimulation" / "AHSimulation" / "AH_Right_5F" / "mjcf" / "assets"

# ------------------------------------------------------------------ shell / pad parameters (mm)
CLEAR = 0.25          # TPU inner surface to its own PLA core (slim rule)
CLEAR_MOV = 0.6       # ... to parts that move inside the shell (Link arm in P1, coupler L2 in P2)
ROUND_EXTRA = 0.08    # margin for the rounding of the inner section corners
WALL, WALL_LOFT = 1.2, 1.27   # TPU wall (minimum), loft offset (the lofts deviate up to ~0.07 between stations)
JOINT_CLEAR = 0.5     # shell ends: >= this from everything that sweeps past (whole servo travel)
EDGE_MARGIN = 0.4     # plane pairs: the proximal shell's surface samples count up to this far past its cut
LAND_H = 0.45         # pad bed: raised land with a pocket whose floor is the plain shell surface
RIM = 0.8             # land rim width around the pocket
PAD_GAP = 0.10        # glue gap pad <-> pocket wall
PAD_T = 0.80          # pad thickness (0.35 proud of the land)
GROOVE_D, GROOVE_W, TEX_BORDER = 0.30, 0.45, 0.6
GROOVE_PITCH = 1.0    # fine transverse ribs: 0.55 mm rib, 0.45 mm groove
NAIL_H = 0.30         # optional raised nail plate (NAIL = False: the cap's dome is the nail)
NAIL = False
PAD_EDGE = 1.5        # pad half width = cover half width - PAD_EDGE (the pad wraps a little onto the palm-side corners)
PAD_END = 1.8         # pad end to the cover's end plane, on the palm face
PAD_END_TIP = 2.6     # fingertip pad start to the cap's base plane
SLEEVE_STYLE = dict(P1_Shell=dict(gx=0.25, gz=0.6, r_max=3.6), P2_Shell=dict(gx=0.0, gz=0.6, r_max=3.6))
TIP = dict(a_max=6.70,    # fingertip half width at its fullest (13.4 mm, target <= 13.5)
           dome=0.35,     # nail side rises this much above the neck
           pulp=0.6,      # pad side bulges this much below the neck
           rt=3.6, rb=3.4,   # outer corner radii, nail side / pad side (3.4: the pad and its land stay inside the width)
           apex_frac=0.35,   # tip end at 35 % of the height from the pad side: flat pad, domed nail, rounded end
           wrap=3.2,      # the pad wraps the tip up to this height above the end point (lower half of the end)
           pad_h=0.6,     # on the fuller part the pad covers the palm face up to this height above its lowest line
           neck=4.5,      # the designed tip starts this far beyond the DIP (the P3 core is trimmed to it beyond)
           nose=6.0)      # length of the rounded end
PAD_FILLET = 0.25
LAND_IN = 0.5         # lands reach this far into the wall (robust fusion where the offset lofts do not nest exactly)
NOSE_F = (0.25, 0.5, 0.7, 0.85, 0.95)   # nose stations (fractions of its length) before the apex
SLOPES = np.arange(-0.25, 0.2501, 0.05)   # end-plane slopes dy/dz (end faces within ~14 deg of square)
SLOPES_WIDE = np.round(np.arange(-0.8, 0.801, 0.1), 3)   # middle sleeve: the knuckles are free on one side only


def log(*a):
    s = " ".join(str(x) for x in a)
    REPORT.append(s)
    import resource
    print(s + f"   (peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB)", flush=True)


REPORT = []


# ------------------------------------------------------------------ CadQuery helpers (cores)
def extrude(geom, x0, x1):
    solid = None
    for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        if poly.is_empty or poly.area < 1e-3:
            continue
        poly = shapely.geometry.polygon.orient(poly.simplify(0.01), 1.0)
        wp = cq.Workplane("YZ", origin=(x0, 0, 0)).polyline(list(poly.exterior.coords)[:-1]).close().extrude(x1 - x0)
        for ring in poly.interiors:
            hole = cq.Workplane("YZ", origin=(x0 - 0.1, 0, 0)).polyline(list(ring.coords)[:-1]).close().extrude(x1 - x0 + 0.2)
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
    return solid.cut(cq.Workplane("YZ", origin=(x0, c[0], c[1])).circle(d / 2).extrude(x1 - x0))


def cut_box_y(solid, ymin=-200.0, ymax=200.0):
    b = cq.Workplane("XY").box(100, ymax - ymin, 100, centered=(True, False, True)).translate((0, ymin, 0))
    return solid.intersect(b)


def pin_solid(c, length, xc=0.0, d=2.0, ch=0.2):
    s = cq.Workplane("YZ", origin=(xc - length / 2, c[0], c[1])).circle(d / 2).extrude(length)
    return s.faces("<X or >X").chamfer(ch)


def one_solid(wp):
    sols = wp.solids().vals() if isinstance(wp, cq.Workplane) else wp.Solids()
    assert len(sols) == 1, f"expected one solid, got {len(sols)}"
    return sols[0]


def make_cores(P):
    t0 = time.time()
    S = {}
    # Link: stock base (pivot + ball-stud bosses, cut from the stock STEP) + new 3.0 mm arm, stock Link frame
    stock_link = cq.importers.importStep(str(U.STOCK_STEP / "Amazing Hand Parts - Link.step"))
    base = stock_link.intersect(extrude(P["base_mask"], -6.0, 6.0))
    link = base.union(extrude(P["arm2d"], -U.C_HALF, U.C_HALF), tol=1e-4)
    S["link"] = pin_hole(link, U.P4L, U.HOLE_RUN, -U.C_HALF - 1, U.C_HALF + 1)
    # P1: stock MCP end (re-cut against the gimbal, Link and rod-ball sweeps) + new body
    stock_p1 = cq.importers.importStep(str(U.STOCK_STEP / "Amazing Hand Parts - Proximal.step"))
    p1_end = cut_box_y(stock_p1, ymax=U.P1_JOIN + 0.5).translate((U.PROX_DX - U.LINK_DX, 0, 0))
    obst = [P["gimbal"], P["link_base"], P["link_arm"], P["ball"]]
    xs = {-8.75, 8.15}
    for p in obst:
        for a, b, _ in p.slabs:
            xs |= {float(np.clip(a - p.lat, -8.75, 8.15)), float(np.clip(b + p.lat, -8.75, 8.15))}
    xs = sorted(xs)
    probe = U.Part("P1_end_probe", "P1")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        if x1 - x0 < 1e-3:
            continue
        geoms = [U.sweep(gm, p.frame, "P1").buffer(U.CLR, quad_segs=8) for p in obst for a, b, gm in p.slabs
                 if min(b + p.lat, x1) - max(a - p.lat, x0) > 1e-6]
        cut = unary_union(geoms).intersection(box(-20, -30, U.P1_JOIN + 1.0, 30)) if geoms else Polygon()
        if not cut.is_empty:
            p1_end = p1_end.cut(extrude(cut, x0 - 1e-3, x1 + 1e-3))
    del probe
    s = p1_end.union(build_solid(P["P1"]), tol=1e-4)
    for c in (U.PIP, U.A):
        s = pin_hole(s, c, U.HOLE_PRESS)
    S["P1"] = s
    s = build_solid(P["P2"])
    s = pin_hole(s, U.PIP, U.HOLE_RUN)
    s = pin_hole(s, U.P4, U.HOLE_PRESS)
    s = pin_hole(s, U.DIP, U.HOLE_PRESS)
    S["P2"] = s
    s = build_solid(P["P3"])
    s = pin_hole(s, U.DIP, U.HOLE_RUN)
    s = pin_hole(s, U.B, U.HOLE_PRESS)
    S["P3"] = s
    s = build_solid(P["L2"])
    s = pin_hole(s, U.A, U.HOLE_RUN)
    S["L2"] = pin_hole(s, U.B, U.HOLE_RUN)
    # pins (bought: hardened steel dowels ISO 8734 / DIN 6325, d2 m6), centred on the parts that hold them
    S["pin_PIP"] = pin_solid(U.PIP, 12.0)
    S["pin_A"] = pin_solid(U.A, 12.0)
    S["pin_P4"] = pin_solid(U.P4, 8.0)
    S["pin_DIP"] = pin_solid(U.DIP, 12.0)
    S["pin_B"] = pin_solid(U.B, 8.0)
    S["pin_MCP"] = pin_solid(U.O_M, 16.0, xc=U.PROX_DX - U.LINK_DX)
    for k in S:
        one_solid(S[k])
    log(f"cores and pins: {len(S)} solids [{time.time() - t0:.0f} s]")
    return S


# ------------------------------------------------------------------ meshes, motion, sampling
def to_mesh(wp, tol=0.02):
    return SF.to_trimesh(one_solid(wp), tol)


def hw(name, origin, zaxis):
    return SF.hw_mesh(name, origin, zaxis)


def body_meshes(S):
    """Meshes (mm) of everything in each moving frame of the 3-phalanx finger (frame coordinates)."""
    dx = -U.LINK_DX
    gim = trimesh.load(U.STOCK_STL / "Amazing Hand Parts - Gimbal.stl")
    gim.apply_translation([dx, 0, 0])
    frame1 = trimesh.load(U.STOCK_STL / "Amazing Hand Parts - Finger_Frame-1.stl")   # palm-fixed (abduction 0)
    frame1.apply_translation([dx, 0, 0])
    return {
        "gimbal": [gim, frame1, hw("std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default", (dx, 0, 15.7), (0, 0, -1)),
                   hw("plain_washer_large_grade_a_m2_5__9a369f0dc77bf9c598cdf3fb468977e5", (dx, 0, 13.8), (0, 0, 1)),
                   hw(SF.PIN10, (dx, 0.0, -6.0), (-1, 0, 0))],
        "link": [to_mesh(S["link"])] + [hw("rotule_ball", (s * 8.05, 3.596, -11.051), (-s, 0, 0)) for s in (-1, 1)]
        + [hw("ph_pan_head_screw_m2x0_40_x_10__2803432263e518bbd16bccbbef8784ed", (s * 12.55, 3.596, -11.051), (s, 0, 0))
           for s in (-1, 1)]
        + [trimesh.creation.icosphere(subdivisions=3, radius=4.2).apply_translation((s * 8.05, 3.596, -11.051))
           for s in (-1, 1)],   # rod-end housings (rotule_lever): they tilt with the rods, a sphere bounds them
        "P1": [to_mesh(S["P1"]), to_mesh(S["pin_MCP"]), to_mesh(S["pin_PIP"]), to_mesh(S["pin_A"])],
        "P2": [to_mesh(S["P2"]), to_mesh(S["pin_P4"]), to_mesh(S["pin_DIP"])],
        "P3": [to_mesh(S["P3"]), to_mesh(S["pin_B"])],
        "L2": [to_mesh(S["L2"])],
    }


def xform(pts, T):
    out = pts.copy()
    out[:, 1:] = pts[:, 1:] @ T[:2, :2].T + T[:2, 2]
    return out


class Sweeps:
    def __init__(self, meshes, spacing=0.3, step=2):
        self.pts = {k: (np.concatenate([SF.sample_surface(m, spacing) for m in ms]) if ms else np.zeros((0, 3)))
                    for k, ms in meshes.items()}
        self.ks = list(range(0, len(U.PSI), step))

    def add(self, frame, mesh, spacing=0.3):
        self.pts[frame] = np.concatenate([self.pts[frame], SF.sample_surface(mesh, spacing)])

    def swept(self, dst, src, pts=None):
        pts = self.pts[src] if pts is None else pts
        if src == dst:
            return pts.copy()
        return np.concatenate([xform(pts, np.linalg.inv(U.pose(dst, k)) @ U.pose(src, k)) for k in self.ks])

    def swept_chunks(self, dst, src, pts=None, chunk=12):
        """The same points as swept(), a few poses at a time (bounded memory)."""
        pts = self.pts[src] if pts is None else pts
        if src == dst:
            yield pts.copy()
            return
        for i in range(0, len(self.ks), chunk):
            yield np.concatenate([xform(pts, np.linalg.inv(U.pose(dst, k)) @ U.pose(src, k)) for k in self.ks[i:i + chunk]])

    def near(self, dst, src, test, pts=None):
        """Points of src swept into dst that pass test(points) -> subset, chunk by chunk, de-duplicated (0.02 mm)."""
        hits = [test(c) for c in self.swept_chunks(dst, src, pts)]
        hits = [h for h in hits if len(h)]
        if not hits:
            return np.zeros((0, 3))
        h = np.concatenate(hits)
        return np.unique(np.round(h / 0.02).astype(np.int64), axis=0) * 0.02


# ------------------------------------------------------------------ lofted shells
class Shell:
    """A TPU sleeve (open both ends) or cap (closed nose) around one phalanx core.

    Inner section at each station: convex hull of the core's surface points in a y-window (CLEAR) and of the
    points of parts moving inside it (CLEAR_MOV), united with a superellipse (smoothly varying along the finger,
    so the outside reads as a rounded segment), corners rounded. Offsets of it are lofted (B-spline)."""

    def __init__(self, name, frame, stations, core_pts, mover_pts, y_lo, y_hi, crown, cushion, nt=2.6, nb=2.1,
                 nose=None, sym_half=None, style=None):
        self.name, self.frame, self.stations = name, frame, list(stations)
        self.nose = nose
        win = max(np.diff(self.stations).max() / 2 + 0.4, 1.2)
        req = []
        for y in self.stations:
            yy = float(np.clip(y, y_lo, y_hi))
            c = core_pts[np.abs(core_pts[:, 1] - yy) <= win][:, [0, 2]]
            m = mover_pts[np.abs(mover_pts[:, 1] - yy) <= win][:, [0, 2]] if len(mover_pts) else np.zeros((0, 2))
            hc = MultiPoint(c).convex_hull
            hm = MultiPoint(m).convex_hull if len(m) >= 3 else Polygon()
            ring = unary_union([hc.buffer(CLEAR + ROUND_EXTRA, quad_segs=16)] +
                               ([hm.buffer(CLEAR_MOV, quad_segs=16)] if not hm.is_empty else []))
            xmin, zmin, xmax, zmax = ring.bounds
            wide = c[np.abs(c[:, 0]) >= 0.9 * np.abs(c[:, 0]).max()]
            zm = float(np.mean([wide[:, 1].min(), wide[:, 1].max()]))
            a = max(abs(xmin), abs(xmax)) if sym_half is None else sym_half(yy)
            req.append(dict(ring=ring, a=a, top=zmax + crown, bot=zmin - cushion, zm=zm))
        self.req = req
        smooth = {}
        for key, sgn in (("a", 1), ("top", 1), ("bot", -1), ("zm", 0)):
            v = np.array([r[key] for r in req])
            if sgn == 0:
                smooth[key] = gaussian_filter1d(v, 1.0, mode="nearest")
                continue
            s = v.copy()
            for _ in range(6):   # smooth envelope that never falls inside the requirement
                s = gaussian_filter1d(s, 1.2, mode="nearest")
                s = np.maximum(s, v) if sgn > 0 else np.minimum(s, v)
            smooth[key] = gaussian_filter1d(s, 0.8, mode="nearest")
            smooth[key] = np.maximum(smooth[key], v) if sgn > 0 else np.minimum(smooth[key], v)
        self.sections = []
        for i, y in enumerate(self.stations):
            a, top, bot, zm = smooth["a"][i], smooth["top"][i], smooth["bot"][i], smooth["zm"][i]
            se = SF.superellipse(a, zm, top - zm, zm - bot, nt, nb)
            sec = unary_union([req[i]["ring"], se]).convex_hull
            sec = sec.buffer(-0.8, quad_segs=16).buffer(0.8, quad_segs=16)
            self.sections.append(sec)
        # outer skin: the cavity grown by the wall, unless a style shapes it (it must contain that)
        self.outer = [sec.buffer(WALL_LOFT, quad_segs=16) for sec in self.sections]
        self.centres = [np.array(sec.centroid.coords[0]) for sec in self.sections]
        if style is not None:
            style(self)

    def sec_at(self, i, off):
        """Section i at offset off: the cavity (off = 0) and anything inside the wall follow the cavity section,
        the outer skin (off = WALL_LOFT) and everything outside it (lands, pads, texture bands) the outer section."""
        if off > 0.6:
            return self.outer[i].buffer(off - WALL_LOFT + 0.03, quad_segs=16)
        return self.sections[i].buffer(off + 0.03, quad_segs=16)

    def body(self, off, apex_shift=0.0):
        ws = []
        for i, y in enumerate(self.stations):
            ws.append(SF.section_wire(y, SF.ray_sample(self.sec_at(i, off), self.centres[i])))
        apex = None
        if self.nose is not None:   # elliptic nose to an apex (fingertip)
            y0, apex_y, apex_z = self.nose
            base_sec = self.sec_at(len(self.stations) - 1, off)
            c = self.centres[-1]
            base = SF.ray_sample(base_sec, c)
            L = apex_y + off - y0
            for f in NOSE_F:
                s = np.sqrt(1 - f * f)
                p = np.c_[base[:, 0] * s, apex_z + (base[:, 1] - apex_z) * s]
                ws.append(SF.section_wire(y0 + f * L, p))
            apex = (0.0, apex_y + off, apex_z)
        return SF.loft(ws, apex=apex)


def rrect2(a, top, bot, rt, rb):
    """Rounded rectangle |x| <= a, bot <= z <= top in the (x, z) plane, corner radius rt on top and rb at the bottom."""
    rt = min(rt, a - 1e-3, (top - bot) / 2 - 1e-3)
    rb = min(rb, a - 1e-3, (top - bot) / 2 - 1e-3)
    up = box(-a + rt, bot - 100.0, a - rt, top - rt).buffer(rt, quad_segs=24)
    dn = box(-a + rb, bot + rb, a - rb, top + 100.0).buffer(rb, quad_segs=24)
    return up.intersection(dn)


def fit_rrect(poly, gx, gz, r_max, r_min):
    """Largest corner radius (r_max .. r_min) of a rounded rectangle that contains poly when it may grow by at most gx
    sideways and gz up and down. Returns (half width, top, bottom, radius)."""
    x0, z0, x1, z1 = poly.bounds
    a0 = max(-x0, x1)
    for r in np.arange(r_max, r_min - 1e-9, -0.1):
        for dz in np.arange(0.0, gz + 1e-9, 0.1):
            for dx in np.arange(0.0, gx + 1e-9, 0.05):
                if poly.difference(rrect2(a0 + dx, z1 + dz, z0 - dz, r, r)).area < 1e-4:
                    return a0 + dx, z1 + dz, z0 - dz, float(r)
    return a0, z1, z0, float(r_min)


def envelope(v, sgn):
    """Smooth curve along the stations that never falls inside the requirement v (sgn +1: >= v, -1: <= v)."""
    s = v.copy()
    for _ in range(6):
        s = gaussian_filter1d(s, 1.2, mode="nearest")
        s = np.maximum(s, v) if sgn > 0 else np.minimum(s, v)
    s = gaussian_filter1d(s, 0.8, mode="nearest")
    return np.maximum(s, v) if sgn > 0 else np.minimum(s, v)


def sleeve_style(gx, gz, r_max):
    """Sleeve outline: a rounded rectangle around the walled cavity with the largest corner radius the growth limits
    allow (gx sideways: the widths are at their minimum; gz up/down), smoothed along the finger. The TPU is solid
    between the cavity and the rounder outline (no voids under the skin)."""
    def apply(sh):
        fits = np.array([fit_rrect(o, gx, gz, r_max, WALL_LOFT + 0.5) for o in sh.outer])
        a, top = envelope(fits[:, 0], 1), envelope(fits[:, 1], 1)
        bot, r = envelope(fits[:, 2], -1), envelope(fits[:, 3], -1)
        sh.outer = [unary_union([rrect2(a[i], top[i], bot[i], r[i], r[i]), o]).convex_hull for i, o in enumerate(sh.outer)]
        sh.style_info = dict(corner_radius=[round(float(r.min()), 2), round(float(r.max()), 2)])
    return apply


def tip_style(p, y_neck, y_nose, apex_y):
    """Fingertip cap: up to y_neck the walled P3 core (rounded as a sleeve); beyond it a designed fingertip: half width
    growing to p['a_max'], the nail side rising by p['dome'] and the pad side bulging by p['pulp'] towards the start of
    the rounded end (y_nose), corner radii blending to p['rt'] (nail) / p['rb'] (pad). There the cavity is the outline
    shrunk by the wall, and the P3 core's tip body is trimmed to it (finish_shells). The end is an elliptic nose to an
    apex at p['apex_frac'] of the height from the pad side: a soft "D" seen from the side."""
    def apply(sh):
        ys = np.asarray(sh.stations)
        n = int(np.argmax(ys >= y_neck))
        fits = [fit_rrect(o, 0.0, 0.6, p["rb"], WALL_LOFT + 0.5) for o in sh.outer]
        a_n, t_n, b_n, r_n = fits[n]
        outer, inner = [], []
        for i, y in enumerate(ys):
            if i < n:
                a, t, b, r = fits[i]
                outer.append(unary_union([rrect2(a, t, b, r, r), sh.outer[i]]).convex_hull)
                inner.append(sh.sections[i])
                continue
            u = float(np.clip((y - y_neck) / max(y_nose - y_neck, 1e-6), 0.0, 1.0))
            u = u * u * (3 - 2 * u)
            o = rrect2(max(a_n, a_n + (p["a_max"] - a_n) * u), t_n + p["dome"] * u, b_n - p["pulp"] * u,
                       r_n + (p["rt"] - r_n) * u, r_n + (p["rb"] - r_n) * u)
            outer.append(o)
            inner.append(o.buffer(-(WALL_LOFT + 0.04), quad_segs=16))   # 0.04: margin for the loft between stations
        sh.outer, sh.sections = outer, inner
        sh.centres = [np.array(sec.centroid.coords[0]) for sec in inner]
        z0, z1 = inner[-1].bounds[1], inner[-1].bounds[3]
        sh.nose = (y_nose, apex_y, z0 + p["apex_frac"] * (z1 - z0))
        # the cavity blends into the designed tip from the station before it: the P3 core's tip body is trimmed to the
        # cavity from there (never closer than 3.0 mm to the DIP: its bosses, R 2.7, stay whole)
        sh.trim_y = float(max(ys[n - 1], U.DIP[0] + 3.0))
        sh.style_info = dict(neck_y=round(float(ys[n]), 2), nose_y=round(float(y_nose), 2), apex_z=round(sh.nose[2], 2),
                             core_trim_from_y=round(sh.trim_y, 2),
                             width=round(2 * max(max(-o.bounds[0], o.bounds[2]) for o in outer), 2))
    return apply


def prism_xy(outline_xy, z0, z1, spacing=0.5):
    """Vertical prism over a (smooth, convex-ish) top-view outline, as ONE periodic B-spline side face: far fewer faces
    than a polyline prism, so the booleans with the lofted shells are much faster (and the pad edges can be filleted)."""
    ring = np.asarray(outline_xy.exterior.coords)[:-1]
    seg = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1)
    t = np.r_[0, np.cumsum(seg)]
    n = max(24, int(t[-1] / spacing))
    u = np.linspace(0, t[-1], n, endpoint=False)
    closed = np.vstack([ring, ring[:1]])
    pts = np.c_[np.interp(u, t, closed[:, 0]), np.interp(u, t, closed[:, 1])]
    edge = cq.Edge.makeSpline([cq.Vector(x, y, z0) for x, y in pts], periodic=True)
    face = cq.Face.makeFromWires(cq.Wire.assembleEdges([edge]))
    return cq.Solid.extrudeLinear(face, cq.Vector(0, 0, z1 - z0))


def prism_yz(outline_yz, half=20.0, spacing=0.5):
    """Prism along x (|x| <= half) over a (y, z) outline, one periodic B-spline side face (as prism_xy)."""
    ring = np.asarray(outline_yz.exterior.coords)[:-1]
    seg = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1)
    t = np.r_[0, np.cumsum(seg)]
    n = max(24, int(t[-1] / spacing))
    u = np.linspace(0, t[-1], n, endpoint=False)
    closed = np.vstack([ring, ring[:1]])
    pts = np.c_[np.interp(u, t, closed[:, 0]), np.interp(u, t, closed[:, 1])]
    edge = cq.Edge.makeSpline([cq.Vector(-half, y, z) for y, z in pts], periodic=True)
    face = cq.Face.makeFromWires(cq.Wire.assembleEdges([edge]))
    return cq.Solid.extrudeLinear(face, cq.Vector(2 * half, 0, 0))


def keep_plane(a, b, below):
    """Half space y <= a + b z (below) or y >= a + b z, as a prism along x."""
    return SF.yz_prism(SF.half_plane_yz(a, b, keep_below=below, lo=(-60, -60), hi=(160, 60)), half=40.0)


def rounded_rect(x0, x1, y0, y1, r):
    return box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def u_outline(w, y0, ye, rc=1.2):
    """Fingertip pad (top view): straight base at y0, parallel sides +-w, half ellipse to the tip at ye."""
    yc = ye - 1.6 * w
    t = np.linspace(0, np.pi, 60)
    pts = np.r_[[[w, y0]], np.c_[w * np.cos(t), yc + (ye - yc) * np.sin(t)], [[-w, y0]]]
    return Polygon(pts).buffer(-rc).buffer(rc, quad_segs=16)


def taper_outline(w0, w1, y0, y1, r0=1.0, r1=2.0):
    """Proximal/middle pad: trapezoid, narrower at y0, rounded corners."""
    poly = Polygon([(-w0, y0), (w0, y0), (w1, y1), (-w1, y1)])
    return poly.buffer(-min(r0, r1)).buffer(min(r0, r1), quad_segs=16)


def texture_cutters(kind, outline, tex):
    zone = outline.buffer(-(PAD_GAP + TEX_BORDER))
    geoms = []
    if kind == "ribs":   # transverse ribs (strips at constant y); on a fingertip only up to the start of the wrap
        y_end = tex["wrap"]["yc"] - GROOVE_W if "wrap" in tex else zone.bounds[3]
        for y in np.arange(zone.bounds[1] + 0.3, y_end + 1e-6, tex["pitch"]):
            geoms.append(box(-10.0, y, 10.0, y + GROOVE_W))
    elif kind == "whorl":
        cy, ratio = tex["cy"], tex["ratio"]
        for k in range(tex["n"]):
            ay = tex["r0"] + k * tex["pitch"]
            outer = shapely.affinity.scale(Point(0, cy).buffer(1.0, quad_segs=48), ratio * ay + GROOVE_W, ay + GROOVE_W)
            inner = shapely.affinity.scale(Point(0, cy).buffer(1.0, quad_segs=48), ratio * ay, ay)
            geoms.append(outer.difference(inner))
    else:
        X = 8.0
        for y in np.arange(tex["y_first"], tex["y_last"] + 1e-6, tex["pitch"]):
            c = tex["slope"] * X
            geoms.append(Polygon([(-X, y - c), (0, y), (X, y - c), (X, y - c + GROOVE_W), (0, y + GROOVE_W),
                                  (-X, y - c + GROOVE_W)]))
    cut = unary_union(geoms).intersection(zone)
    return [g for g in (cut.geoms if hasattr(cut, "geoms") else [cut]) if g.area > 0.02]


def yz_grooves(ps):
    """Rib grooves of a pad given by a side-profile region (ps['tex']['yz']), as (y, z) polygons to extrude across x:
    strips at constant y up to the start of the rounded end (yc), then wedges around the nose centre (yc, zc) spaced
    GROOVE_PITCH along an ellipse through the pad surface (radii ly along y, hz down), GROOVE_W wide there, so the
    ribs follow the end. All clipped to the region shrunk by TEX_BORDER (a plain border along every pad edge)."""
    t, w = ps["tex"], ps["tex"]["wrap"]
    zone = t["yz"].buffer(-TEX_BORDER)
    yc, zc, ly, hz = w["yc"], w["zc"], w["ly"], w["hz"]
    out, y_last = [], None
    for y in np.arange(zone.bounds[0] + 0.3, yc - GROOVE_W + 1e-6, t["pitch"]):
        out.append(box(y, -40.0, y + GROOVE_W, 40.0).intersection(zone))
        y_last = y
    tt = np.linspace(0.0, np.pi, 2001)
    pt = np.c_[yc + ly * np.sin(tt), zc - hz * np.cos(tt)]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pt, axis=0), axis=1))]
    R = 4.0 * max(ly, hz)
    clip = zone.intersection(box(yc - 0.05, -40.0, yc + 3 * R, 40.0))
    s0 = (y_last + t["pitch"] - yc) if y_last is not None else 0.3
    for sk in np.arange(s0, s[-1], t["pitch"]):
        tk = float(np.interp(sk, s, tt))
        dt = 0.5 * GROOVE_W / max(np.hypot(ly * np.cos(tk), hz * np.sin(tk)), 1e-6)
        rays = []
        for t_ in (tk - dt, tk + dt):
            d = np.array([ly * np.sin(t_), -hz * np.cos(t_)])
            rays.append(np.array([yc, zc]) + R * d / np.linalg.norm(d))
        out.append(Polygon([(yc, zc), tuple(rays[0]), tuple(rays[1])]).intersection(clip))
    polys = []
    for g in out:
        for q in (g.geoms if hasattr(g, "geoms") else [g]):
            if q.geom_type == "Polygon" and q.area > 0.01:
                polys.append(q)
    return polys


def to_manifold(tm):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))


def from_manifold(man):
    m = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)
    trimesh.repair.fix_normals(tm)
    return tm


def texture_mesh(pad_solid, bodies, ps):
    """Grip texture as a mesh boolean (manifold3d): the pad minus (groove band between two offset lofts) x (groove
    prisms). The STEP keeps the smooth pad (a grooved B-rep of lofted faces is huge and slow)."""
    import manifold3d as mf
    pad = to_manifold(SF.to_trimesh(pad_solid, 0.01))
    band = to_manifold(SF.to_trimesh(bodies(ps["off"] + PAD_T + 0.6), 0.02)) - \
        to_manifold(SF.to_trimesh(bodies(ps["off"] + PAD_T - GROOVE_D), 0.02))
    cut = None
    for g in ([] if "yz" in ps["tex"] else texture_cutters(ps["kind"], ps["outline"], ps["tex"])):
        g = shapely.geometry.polygon.orient(g, 1.0)   # counter-clockwise outer contour (manifold's fill rule)
        ring = [np.asarray(g.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in g.interiors]
        pr = mf.Manifold.extrude(mf.CrossSection(ring), ps["z1"] - ps["z0"] + 2).translate([0, 0, ps["z0"] - 1])
        assert pr.status() == mf.Error.NoError and pr.volume() > 0, "groove prism"
        cut = pr if cut is None else cut + pr
    if "yz" in ps["tex"]:   # side-profile pad (fingertip): grooves across x, following the rounded end
        X = 10.0
        for g in yz_grooves(ps):
            g = shapely.geometry.polygon.orient(g, 1.0)
            pr = mf.Manifold.extrude(mf.CrossSection([np.asarray(g.exterior.coords)[:-1]]), 2 * X).transform(
                np.array([[0.0, 0.0, 1.0, -X], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            assert pr.status() == mf.Error.NoError and pr.volume() > 0, "groove prism (yz)"
            cut = pr if cut is None else cut + pr
    out = pad - (band ^ cut) if cut is not None else pad
    assert out.status() == mf.Error.NoError and out.volume() > 0.5 * pad.volume(), "textured pad"
    return from_manifold(out)


def styled_shell(spec_bodies, pad_specs, nail=None, off=None, with_shell=True):
    """spec_bodies(off) -> lofted body at offset off (0 = cavity, WALL_LOFT = plain outer).
    pad_specs: list of dict(outline (x,y), z0, z1 (palm-side prism), kind, tex). Returns shell, pads (smooth solids)."""
    off = WALL_LOFT if off is None else off
    plain = spec_bodies(off)
    cav = spec_bodies(0.0) if with_shell else None
    shell = cq.Workplane().add(plain).cut(cq.Workplane().add(cav)) if with_shell else None
    pads = []
    under = cq.Workplane().add(spec_bodies(off - LAND_IN))   # lands/nail reach LAND_IN into the wall: one fused solid
    if pad_specs:
        land_body = cq.Workplane().add(spec_bodies(off + LAND_H))
        land_layer = land_body.cut(under)   # (cutting the thin layer first is the robust order for these lofts)
        top_body = cq.Workplane().add(spec_bodies(off + PAD_T))
        for ps in pad_specs:
            o = ps["outline"]
            yz = ps["tex"].get("yz")   # fingertip: the pad region is a side profile (edges cross the surface steeply)
            if yz is not None:
                region = lambda g: prism_yz(yz.buffer(g, quad_segs=16) if g else yz)  # noqa: E731
            else:
                region = lambda g: prism_xy(o.buffer(g, quad_segs=16) if g else o, ps["z0"], ps["z1"])  # noqa: E731
            land = land_layer.intersect(cq.Workplane().add(region(PAD_GAP + RIM)))
            assert land.solids().vals(), f"empty land for {ps['name']}"
            shell = land if shell is None else shell.union(land)
            pocket = cq.Workplane().add(region(PAD_GAP)).cut(cq.Workplane().add(plain))
            shell = shell.cut(pocket)
            pad = top_body.cut(cq.Workplane().add(plain)).intersect(cq.Workplane().add(region(0.0)))
            pad_s = one_solid(pad)
            try:   # rounded top edge: the B-spline top face is the one farthest down (palm side)
                top = min((f for f in pad_s.Faces() if f.geomType() == "BSPLINE"), key=lambda f: f.Center().z)
                pad_s = pad_s.fillet(PAD_FILLET, top.Edges())
                ok_fillet = True
            except Exception:  # noqa: BLE001
                ok_fillet = False
            pads.append(dict(smooth=pad_s, fillet=ok_fillet, name=ps["name"], spec=dict(ps, off=off)))
    if nail is not None:
        nb = cq.Workplane().add(spec_bodies(off + NAIL_H)).cut(under).intersect(
            cq.Workplane().add(prism_xy(nail["outline"], nail["z0"], nail["z1"])))
        shell = nb if shell is None else shell.union(nb)
    return shell, pads, plain, cav


# ------------------------------------------------------------------ end planes from the joint sweeps
def near(mesh, pts, dist):
    return SF.near_points(mesh, pts, dist)


_TREES = {}


def _tree(mesh):
    if id(mesh) not in _TREES:
        _TREES[id(mesh)] = (mesh, cKDTree(SF.sample_surface(mesh, 0.2)))
    return _TREES[id(mesh)][1]


def near_material(outer, cav, pts, dist):
    """Points within dist of the shell MATERIAL (between the outer skin and the cavity skin). A point inside the
    1.24 mm wall is at most 0.62 mm from one skin and has d_out + d_in ~ the wall thickness, which flags it without a
    (memory-hungry) ray-cast inside test."""
    lo, hi = outer.bounds[0] - dist - 1, outer.bounds[1] + dist + 1
    q = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if not len(q):
        return q
    reach = WALL_LOFT + dist
    d_out, _ = _tree(outer).query(q, distance_upper_bound=reach, workers=4)
    d_in, _ = _tree(cav).query(q, distance_upper_bound=reach, workers=4)
    hit = (d_out < dist) | (d_in < dist) | (d_out + d_in < WALL_LOFT + 0.15)
    return q[hit]


def hull_yz(Q):
    """Only the (y, z) convex-hull points matter for a plane y = a + b z bounding them (the planes are x-independent)."""
    if len(Q) < 4:
        return Q
    from scipy.spatial import ConvexHull
    yz = np.unique(np.round(Q[:, 1:3] / 0.01).astype(np.int64), axis=0) * 0.01
    if len(yz) < 4:
        return np.c_[np.zeros(len(yz)), yz]
    return np.c_[np.zeros(len(ConvexHull(yz).vertices)), yz[ConvexHull(yz).vertices]]


def plane_single(Q, slopes, keep_below, clr=JOINT_CLEAR, prefer=0.0):
    """One end plane against near points Q (frame of the shell): returns (a, b) with the most coverage."""
    best = None
    for b in slopes:
        if keep_below:
            a = float(np.min(Q[:, 1] - b * Q[:, 2])) - clr * np.hypot(1, b)
            key = -(a + b * prefer)
        else:
            a = float(np.max(Q[:, 1] - b * Q[:, 2])) + clr * np.hypot(1, b)
            key = a + b * prefer
        if best is None or key < best[0]:
            best = (key, a, b)
    return best[1], best[2]


def plane_pair(QA, shellA_pts, frameA, shellB_mesh, QB, frameB, sweeps, zmid, clr=JOINT_CLEAR, slopes_a=SLOPES,
               slopes_b=SLOPES):
    """Distal end of the proximal shell A (keeps y <= a + b z, frame A) and base of the distal shell B (keeps
    y >= c + e z, frame B), chosen together (slim_fingers approach): clear the sweeps by clr, the surviving part
    of A clear of B, smallest knuckle gap at z = zmid, then the longest A."""
    tagged = []
    tree = cKDTree(SF.sample_surface(shellB_mesh, 0.2))
    for k in sweeps.ks:
        T = np.linalg.inv(U.pose(frameB, k)) @ U.pose(frameA, k)
        q = xform(shellA_pts, T)
        d, _ = tree.query(q, distance_upper_bound=clr)
        hit = np.isfinite(d)
        tagged.append(np.c_[q[hit], shellA_pts[hit]])
    tagged = np.concatenate(tagged) if tagged else np.zeros((0, 6))
    if len(tagged):   # (y, z) in B and in A is all that matters: de-duplicate on a 0.05 mm grid
        tagged = np.unique(np.round(tagged[:, [1, 2, 4, 5]] / 0.05).astype(np.int64), axis=0) * 0.05
        tagged = np.c_[np.zeros(len(tagged)), tagged[:, :2], np.zeros(len(tagged)), tagged[:, 2:]]
    QA, QB = hull_yz(QA), hull_yz(QB)
    best = None
    for b in slopes_a:
        amax = float(np.min(QA[:, 1] - b * QA[:, 2])) - clr * np.hypot(1, b)
        for a in np.arange(amax, amax - 8.0, -0.25):
            # A's points up to EDGE_MARGIN past its end plane count as alive: the samples stop short of the cut edge,
            # and the closest approach of two cut shells is edge to edge
            alive = tagged[:, 4] <= a + b * tagged[:, 5] + EDGE_MARGIN * np.hypot(1, b)
            yz = np.r_[QB[:, 1:3], tagged[alive][:, 1:3]]
            for e in slopes_b:
                c = float(np.max(yz[:, 0] - e * yz[:, 1])) + clr * np.hypot(1, e)
                gap = (c + e * zmid) - (a + b * zmid)
                key = (round(gap, 2), -a)
                if best is None or key < best[0]:
                    best = (key, a, b, c, e)
    return best[1:]


# ------------------------------------------------------------------ build
def build_shells(S, P, sw):
    t0 = time.time()
    core = {k: SF.sample_surface(to_mesh(S[k]), 0.25) for k in ("P1", "P2", "P3")}
    out = {}
    # ---------------- P1 sleeve: the Link arm runs inside it
    arm_own = SF.sample_surface(to_mesh(S["link"]), 0.3)
    u = (U.P4L - U.O_L) / np.linalg.norm(U.P4L - U.O_L)
    arm_own = arm_own[((arm_own[:, 1:] - U.O_L) @ u >= 7.5) & (np.abs(arm_own[:, 0]) <= U.C_HALF + 0.05)]  # the arm only
    arm_pts = sw.swept("P1", "link", arm_own)
    yp, yd = U.PIP[0], U.DIP[0]
    st1 = np.arange(6.0, yp + 3.01, 2.0)
    sh1 = Shell("P1_Shell", "P1", st1, core["P1"], arm_pts, y_lo=10.5, y_hi=yp - 6.0, crown=0.35, cushion=0.0,
                style=sleeve_style(**SLEEVE_STYLE["P1_Shell"]))
    # ---------------- P2 sleeve: the coupler L2 runs inside it
    l2_own = SF.sample_surface(to_mesh(S["L2"]), 0.3)
    l2_own = l2_own[np.linalg.norm(l2_own[:, 1:] - U.B, axis=1) > U.L2_R + 0.9]   # not its B eye (rises out at the DIP)
    l2_pts = sw.swept("P2", "L2", l2_own)
    st2 = np.arange(yp + 1.0, yd + 3.01, 2.0)
    sh2 = Shell("P2_Shell", "P2", st2, core["P2"], l2_pts, y_lo=yp + 9.0, y_hi=yd - 10.0, crown=0.3, cushion=0.6,
                style=sleeve_style(**SLEEVE_STYLE["P2_Shell"]))
    # ---------------- P3 cap: designed fingertip, rounded end to an apex
    y_nose = U.TIP_END + CLEAR - TIP["nose"]
    st3 = np.r_[np.arange(yd - 1.0, y_nose - 0.5, 2.0), y_nose]
    nose = (y_nose, U.TIP_END + CLEAR, -0.7)
    sh3 = Shell("P3_Shell", "P3", st3, core["P3"], np.zeros((0, 3)), y_lo=yd + 4.0, y_hi=y_nose, crown=0.25, cushion=0.9,
                nose=nose, style=tip_style(TIP, yd + TIP["neck"], y_nose, U.TIP_END + CLEAR))
    for sh in (sh1, sh2, sh3):
        log(f"  {sh.name} style: {sh.style_info}")
    log(f"shell sections [{time.time() - t0:.0f} s]")

    # ---------------- end planes (uncut plain bodies and their cavities as meshes)
    m1, m2, m3 = (SF.to_trimesh(sh.body(WALL_LOFT), 0.05) for sh in (sh1, sh2, sh3))
    k1, k2, k3 = (SF.to_trimesh(sh.body(0.0), 0.05) for sh in (sh1, sh2, sh3))
    # points of the parts that run INSIDE a sleeve (Link arm in P1, L2's bar in P2) are only tested against the outer
    # skin (they are in the cavity by construction); everything else against the shell material (both skins + wall)
    lp = sw.pts["link"]
    u = (U.P4L - U.O_L) / np.linalg.norm(U.P4L - U.O_L)
    arm = ((lp[:, 1:] - U.O_L) @ u >= 7.5) & (np.abs(lp[:, 0]) <= U.C_HALF + 0.05)
    l2p = sw.pts["L2"]
    beye = np.linalg.norm(l2p[:, 1:] - U.B, axis=1) <= U.L2_R + 0.9

    def nm(outer, cav, frame, src, pts=None):
        return sw.near(frame, src, lambda q: near_material(outer, cav, q, JOINT_CLEAR), pts)

    def nr(outer, frame, src, pts=None):
        return sw.near(frame, src, lambda q: near(outer, q, JOINT_CLEAR), pts)

    # PIP knuckle: P1 sleeve end vs P2 sleeve start
    QA = np.concatenate([nm(m1, k1, "P1", f) for f in ("P2", "L2", "P3")]
                        + [nr(m1, "P1", "link", lp[arm]), nm(m1, k1, "P1", "link", lp[~arm])])
    c1p = core["P1"]
    knuckle = c1p[(np.linalg.norm(c1p[:, 1:] - U.PIP, axis=1) <= U.R_PIP_P1 + 0.05)
                  | (np.linalg.norm(c1p[:, 1:] - U.A, axis=1) <= U.R_A + 0.05)]   # keep P1's PIP fork bosses exposed
    QA = np.concatenate([QA, knuckle])
    QA = QA[QA[:, 1] > yp - 22]
    QB = np.concatenate([nm(m2, k2, "P2", f) for f in ("P1", "link")]
                        + [nr(m2, "P2", "L2", l2p[~beye]), nm(m2, k2, "P2", "L2", l2p[beye])])
    QB = QB[QB[:, 1] < yd - 11]
    sA = SF.sample_surface(m1, 0.3)
    a1, b1, c2, e2 = plane_pair(QA, sA[sA[:, 1] > yp - 17], "P1", m2, QB, "P2", sw, zmid=0.0, slopes_b=SLOPES_WIDE)
    # DIP knuckle: P2 sleeve end vs P3 cap base
    QA2 = np.concatenate([nm(m2, k2, "P2", "P3"), nr(m2, "P2", "L2", l2p[~beye]),
                          nm(m2, k2, "P2", "L2", l2p[beye])])
    QA2 = QA2[QA2[:, 1] > c2 + 3]
    QB2 = np.concatenate([nm(m3, k3, "P3", f) for f in ("P2", "L2", "P1")])
    c2p = core["P2"]
    QA2 = np.concatenate([QA2, c2p[np.linalg.norm(c2p[:, 1:] - U.DIP, axis=1) <= U.R_DIP_P2 + 0.05]])  # DIP fork exposed
    QB2 = QB2[QB2[:, 1] < yd + 11]
    sA2 = SF.sample_surface(m2, 0.3)
    a2, b2, c3, e3 = plane_pair(QA2, sA2[sA2[:, 1] > yd - 13], "P2", m3, QB2, "P3", sw, zmid=0.0, slopes_a=SLOPES_WIDE)
    # MCP end of the P1 sleeve: gimbal (+ screw, washer, finger frame at abduction 0), Link base + ball joints and rod-end
    # housings, P1's own MCP end
    QM = np.concatenate([nm(m1, k1, "P1", "gimbal"), nm(m1, k1, "P1", "link", lp[~arm]),
                         near_material(m1, k1, core["P1"][core["P1"][:, 1] < 9.0], JOINT_CLEAR)])
    QM = QM[QM[:, 1] < 25]
    c1, e1 = plane_single(hull_yz(QM), SLOPES, keep_below=False, prefer=1.0)
    planes = dict(P1_Shell=dict(start=[c1, e1], end=[a1, b1]), P2_Shell=dict(start=[c2, e2], end=[a2, b2]),
                  P3_Shell=dict(start=[c3, e3]))
    for k, v in planes.items():
        log(f"  {k:<9} " + "  ".join(f"{side} plane y = {p[0]:.2f} {p[1]:+.2f} z" for side, p in v.items()))
    log(f"end planes [{time.time() - t0:.0f} s]")
    return dict(specs=dict(P1_Shell=sh1, P2_Shell=sh2, P3_Shell=sh3), planes=planes, meshes=dict(P1_Shell=m1, P2_Shell=m2,
                                                                                               P3_Shell=m3))


def pad_plan(planes, specs):
    """Pads over most of the palm face of every cover: half width = the cover's outer half width - PAD_EDGE along the
    finger (so they wrap a little onto the rounded palm-side corners), ends PAD_END inside the end planes, rounded
    corners; transverse ribs. The fingertip pad runs past the tip and is cut by the plane z = z1 (TIP['wrap'] above
    the end point), so it wraps around the rounded end; its ribs follow the end (yz_grooves)."""
    def at(pl, z):
        return pl[0] + pl[1] * z
    pads = []
    for name, pname, z1 in (("P1_Shell", "P1_Pad", -1.0), ("P2_Shell", "P2_Pad", -1.5), ("P3_Shell", "P3_Pad", None)):
        sh = specs[name]
        ys = np.asarray(sh.stations)
        A = np.array([max(-o.bounds[0], o.bounds[2]) for o in sh.outer])
        zb = float(min(o.bounds[1] for o in sh.outer)) + 0.4   # the palm face
        # ends PAD_END inside the end planes over the pad's whole height (the planes slant; the pad climbs the corners);
        # the fingertip pad starts PAD_END_TIP from the cap's base (its land must clear the middle sleeve's end as the
        # DIP flexes)
        y0 = max(at(planes[name]["start"], zb), at(planes[name]["start"], zb + 1.5)) + \
            (PAD_END_TIP if name == "P3_Shell" else PAD_END)
        y1 = U.TIP_END + 8.0 if name == "P3_Shell" else \
            min(at(planes[name]["end"], zb), at(planes[name]["end"], zb + 1.5)) - PAD_END
        yy = np.linspace(y0, y1, max(8, int((y1 - y0) / 0.5)))
        w = np.interp(yy, ys, A) - PAD_EDGE
        outline = Polygon(np.r_[np.c_[w, yy], np.c_[-w[::-1], yy[::-1]]]).buffer(-1.5).buffer(1.5, quad_segs=16)
        tex = dict(pitch=GROOVE_PITCH)
        if name == "P3_Shell":   # side-profile region: palm face up to pad_h, rising over the rounded end to wrap
            yc, apex_y, za = sh.nose
            y_tip = apex_y + WALL_LOFT                       # outer end of the cap
            L_out = y_tip - yc
            Bst = np.array([o.bounds[1] for o in sh.outer])
            h = TIP["pad_h"]

            def zcut(y):
                # the pad's top edge in the side view. It must cross the cap surface steeply: along the fuller part it
                # runs h above the palm face (crossing the rounded palm-side corners); over the first half of the
                # rounded end it follows the scaled corners, then rises to za + wrap and runs level across the end
                # (crossing the end face square). It reaches the sides only where the end has narrowed by more than
                # the pad's thickness, so the pad never widens the finger.
                if y <= yc:
                    return float(np.interp(y, ys, Bst)) + 0.15 + (h - 0.15) * float(np.clip((y - y0) / 1.5, 0.0, 1.0))
                u = (min(y, y_tip) - yc) / L_out

                def low(u_):
                    s_ = np.sqrt(max(0.0, 1.0 - u_ * u_))
                    return za + (Bst[-1] - za) * s_ + h * s_
                if u <= 0.5:
                    return low(u)
                v = float(np.clip((u - 0.5) / 0.32, 0.0, 1.0))
                v = v * v * (3 - 2 * v)
                return low(0.5) + (za + TIP["wrap"] - low(0.5)) * v
            y_end = y_tip + PAD_T + 3.0
            yy = np.r_[np.linspace(y0, yc, 40), np.linspace(yc, y_tip, 41)[1:], y_end]
            yz = Polygon([(y, zcut(y)) for y in yy] + [(y_end, -40.0), (y0, -40.0)])
            w0 = float(A.max()) - PAD_EDGE + 0.7
            outline = box(-w0, y0, w0, y_end)   # nominal top view (reports); the pad region is yz
            z1 = za + TIP["wrap"]
            tex.update(yz=yz, wrap=dict(yc=yc, zc=za, ly=L_out + PAD_T, hz=za - (Bst[-1] - PAD_T)))
        pads.append(dict(name=pname, shell=name, outline=outline, z0=-25.0, z1=z1, kind="ribs", tex=tex))
    nail = dict(outline=rounded_rect(-3.3, 3.3, U.TIP_END - 10.0, U.TIP_END - 1.2, 2.2), z0=1.5, z1=25.0) if NAIL else None
    return pads, nail


def soften_ends(solid, planes, radii=(0.4, 0.25)):
    """Round the outer edge of each flat end face of a cover (the first radius that works); returns (solid, radius)."""
    edges = []
    for f in solid.Faces():
        if f.geomType() != "PLANE":
            continue
        n, c = f.normalAt(), f.Center()
        for a, b in planes.values():
            u = np.array([0.0, 1.0, -b]) / np.hypot(1.0, b)
            if abs(abs(n.y * u[1] + n.z * u[2]) - 1.0) < 1e-3 and abs(c.y - (a + b * c.z)) < 0.05:
                edges += f.outerWire().Edges()
    for r in radii if edges else ():
        try:
            out = solid.fillet(r, edges)
            if out.isValid() and len(out.Solids()) == 1:
                return out, r
        except Exception:  # noqa: BLE001
            pass
    return solid, None


def finish_shells(B, P3core):
    """Lofted shells + lands, pockets, pads, nail plate; cut by the end planes. Also trims the P3 core's tip to
    the cap cavity (shrunk by CLEAR), like the slim distal core."""
    t0 = time.time()
    specs, planes = B["specs"], B["planes"]
    pads, nail = pad_plan(planes, specs)
    B["pad_bounds"] = {p["shell"]: list(p["outline"].bounds) for p in pads}
    B["pad_yz"] = {p["shell"]: p["tex"]["yz"] for p in pads if "yz" in p["tex"]}
    B["nail"] = nail is not None
    parts, padparts, plain = {}, {}, {}
    for name, sp in specs.items():
        mine = [p for p in pads if p["shell"] == name]
        shell, pp, pl, cav = styled_shell(sp.body, mine, nail if (name == "P3_Shell" and nail) else None)
        pl_ = planes[name]
        keep = cq.Workplane().add(keep_plane(*pl_["start"], below=False))
        if "end" in pl_:
            keep = keep.intersect(cq.Workplane().add(keep_plane(*pl_["end"], below=True)))
        shell = shell.intersect(keep)
        soft, done = soften_ends(one_solid(shell.clean()), pl_)
        parts[name] = soft
        plain[name] = (pl, cav)
        for p in pp:
            smooth = one_solid(cq.Workplane().add(p["smooth"]).intersect(keep))
            padparts[p["name"]] = dict(smooth=smooth, textured=texture_mesh(smooth, sp.body, p["spec"]), fillet=p["fillet"])
        log(f"  {name}: shell + {len(pp)} pad(s){' + nail plate' if (name == 'P3_Shell' and nail) else ''} "
            f"(pad top-edge fillet: {[p['fillet'] for p in pp]}; end edges rounded: {done}) [{time.time() - t0:.0f} s]")
    # trim the fingertip core to the cap's cavity shrunk by CLEAR (its corners beyond the nose start)
    sp3 = specs["P3_Shell"]
    v0 = one_solid(P3core).Volume()
    done = None
    # the B-rep booleans of the core with the apex loft can fail silently (near-coincident faces where the cavity
    # hugs the core): try the trim from the blend station and from the first designed station, with the cavity
    # shrunk by CLEAR and slightly more, and keep the first result that is one valid solid and actually trimmed
    for y_trim in (sp3.trim_y, sp3.stations[int(np.argmax(np.asarray(sp3.stations) >= sp3.trim_y)) + 1]):
        tipbox = cq.Workplane().add(cq.Solid.makeBox(20, U.TIP_END + 10.0 - y_trim, 30, cq.Vector(-10, y_trim, -15)))
        before = cq.Workplane().add(cq.Solid.makeBox(20, y_trim + 60.0, 30, cq.Vector(-10, -60.0, -15)))
        for extra in (0.0, 0.02, 0.05):
            trim_body = cq.Workplane().add(sp3.body(-CLEAR - extra))
            for how, fn in (("cut", lambda: P3core.cut(tipbox.cut(trim_body))),
                            ("intersect", lambda: P3core.intersect(before.union(trim_body)))):
                try:
                    r = fn()
                    sol = one_solid(r)
                    if sol.isValid() and sol.Volume() < v0 - 1.0:
                        done = (r, how, y_trim, extra, sol.Volume())
                        break
                except Exception:  # noqa: BLE001
                    pass
            if done:
                break
        if done:
            break
    if done:
        P3core = done[0]
        log(f"  P3 core tip trimmed to the cap cavity - {CLEAR + done[3]:.2f} mm from y = {done[2]:.2f} ({done[1]}): "
            f"{v0:.0f} -> {done[4]:.0f} mm3")
    else:
        log("  WARNING: P3 core tip trim failed; the core overlaps its cap")
    return parts, padparts, pads, nail, P3core, plain


# ------------------------------------------------------------------ thumb: slim parts + a styled pad set
def thumb_parts():
    """Slim-finger shells with the styled pad set: chevron pad on the proximal shell, fingerprint pad on the distal
    shell (instead of the slim proud pad) and a nail plate. Same lands/pockets/glue gap as the fingers."""
    t0 = time.time()
    sd = json.loads((SF.HERE / "design.json").read_text())
    cores = SF.Cores()
    p_shell = cq.importers.importStep(str(SF.HERE / "step" / "Proximal_Shell_Slim.step")).val()
    d_shell = cq.importers.importStep(str(SF.HERE / "step" / "Distal_Shell_Slim.step")).val()
    pa, pb = sd["prox_end_plane"]
    da, db = sd["dist_base_plane"]
    cy, cz, cr = sd["mcp_cyl"]
    y0m, sm = sd["mcp_dorsal_plane"]
    keep_p = (cq.Workplane().add(keep_plane(pa, pb, below=True))
              .intersect(cq.Workplane().add(SF.x_cylinder(cy, cz, cr, -40, 40)))
              .intersect(cq.Workplane().add(keep_plane(y0m - sm * 9.5, sm, below=False))))
    keep_d = cq.Workplane().add(keep_plane(da, db, below=False))
    out = {}
    pro = taper_outline(3.6, 4.4, 17.0, 44.0)
    dist = u_outline(3.6, 64.0, 90.4)
    specs = [("Thumb_Proximal", p_shell, lambda off: SF.prox_body(cores, off), keep_p,
              dict(name="Thumb_Proximal_Pad", outline=pro, z0=-25.0, z1=-5.5, kind="ribs", tex=dict(pitch=GROOVE_PITCH)),
              None),
             ("Thumb_Distal", d_shell, lambda off: SF.dist_body(cores, off, SF.TIP_Y + (off - SF.WALL_LOFT)), keep_d,
              dict(name="Thumb_Distal_Pad", outline=dist, z0=-25.0, z1=-4.6, kind="ribs", tex=dict(pitch=GROOVE_PITCH)),
              dict(outline=rounded_rect(-3.6, 3.6, 81.0, 91.0, 2.4), z0=3.0, z1=25.0) if NAIL else None)]
    for name, stock_shell, bodies, keep, ps, nail in specs:
        extra, pp, _, _ = styled_shell(bodies, [ps], nail, off=SF.WALL_LOFT, with_shell=False)
        shell = cq.Workplane().add(stock_shell).union(extra)
        pocket = cq.Workplane().add(prism_xy(ps["outline"].buffer(PAD_GAP, quad_segs=16), ps["z0"], ps["z1"])).cut(
            cq.Workplane().add(bodies(SF.WALL_LOFT)))
        shell = shell.cut(pocket).intersect(keep)
        out[f"{name}_Shell"] = one_solid(shell.clean())
        smooth = one_solid(cq.Workplane().add(pp[0]["smooth"]).intersect(keep))
        out[ps["name"]] = dict(smooth=smooth, textured=texture_mesh(smooth, bodies, pp[0]["spec"]))
        log(f"  thumb {name}: slim shell + land/pocket + pad{' + nail plate' if nail else ''} (pad top-edge fillet "
            f"{pp[0]['fillet']}) [{time.time() - t0:.0f} s]")
    return out, dict(proximal_pad=list(pro.bounds), distal_pad=list(dist.bounds))


def predicted():
    """Planar joint angles at the ends of the servo travel (same drive map as the 3-phalanx study)."""
    f = np.array([-90.0, 90.0])
    s = U.D.solve(np.radians(np.interp(f, U.drive["f"], U.drive["psi"])))
    fs = float(np.interp(np.degrees(U.D.psi_s), U.drive["psi"], U.drive["f"]))
    return dict(f_straight_deg=round(fs, 2), mcp_at_f90=round(float(np.degrees(s["mcp"][1])), 2),
                pip_at_f90=round(float(np.degrees(s["pip"][1])), 2), dip_at_f90=round(float(np.degrees(s["dip"][1])), 2),
                mcp_at_fm90=round(float(np.degrees(s["mcp"][0])), 2), pip_at_fm90=round(float(np.degrees(s["pip"][0])), 2),
                dip_at_fm90=round(float(np.degrees(s["dip"][0])), 2))


# ------------------------------------------------------------------ export
NAMES = dict(link="UF_Link", P1="UF_P1_Proximal", P2="UF_P2_Middle", P3="UF_P3_Distal", L2="UF_L2_Coupler",
             P1_Shell="UF_P1_Shell", P2_Shell="UF_P2_Shell", P3_Shell="UF_P3_Shell",
             P1_Pad="UF_P1_Pad", P2_Pad="UF_P2_Pad", P3_Pad="UF_P3_Pad",
             pin_PIP="pin_2x12_PIP", pin_A="pin_2x12_A", pin_DIP="pin_2x12_DIP", pin_P4="pin_2x8_P4", pin_B="pin_2x8_B",
             pin_MCP="pin_2x16_MCP",
             Thumb_Proximal_Shell="Thumb_Proximal_Shell", Thumb_Distal_Shell="Thumb_Distal_Shell",
             Thumb_Proximal_Pad="Thumb_Proximal_Pad", Thumb_Distal_Pad="Thumb_Distal_Pad")
MATERIAL = dict(link="PA-CF (or PETG)", P1="PLA (or PETG)", P2="PA-CF (or PETG)", P3="PLA (or PETG)", L2="PA-CF (or PETG)",
                P1_Shell="TPU 82A", P2_Shell="TPU 82A", P3_Shell="TPU 82A", P1_Pad="TPU 82A grey", P2_Pad="TPU 82A grey",
                P3_Pad="TPU 82A grey", Thumb_Proximal_Shell="TPU 82A", Thumb_Distal_Shell="TPU 82A",
                Thumb_Proximal_Pad="TPU 82A grey", Thumb_Distal_Pad="TPU 82A grey")


def export_stl(shape, path, tol=0.01, ang=0.1):
    """STL with a watertight check; retries a few tessellation tolerances (OCC can leave a pinhole at a seam)."""
    for t in (tol, 0.012, 0.008, 0.015, 0.02):
        cq.exporters.export(cq.Workplane().add(shape), str(path), tolerance=t, angularTolerance=ang)
        m = trimesh.load(path)
        m.merge_vertices(digits_vertex=5)
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        if m.is_watertight:
            break
    m.export(path)
    return m


def fix_solid(s):
    """Heal a B-rep that came out of the booleans invalid (ShapeFix), keep it if it is then valid."""
    if s.isValid():
        return s
    from OCP.ShapeFix import ShapeFix_Shape
    sf = ShapeFix_Shape(s.wrapped)
    sf.Perform()
    fixed = cq.Shape.cast(sf.Shape())
    sols = fixed.Solids() if hasattr(fixed, "Solids") else [fixed]
    return sols[0] if len(sols) == 1 and sols[0].isValid() else s


def export_all(parts):
    """parts: name -> (solid, stl_only_solid or None). STEP of the solid, STL of the textured variant if given."""
    rows = {}
    for key, (solid, textured) in parts.items():
        name = NAMES[key] + ("" if key.startswith("Thumb") else U.SUFFIX)
        step = OUT_STEP / f"{name}{'_untextured' if textured is not None else ''}.step"
        cq.exporters.export(cq.Workplane().add(solid), str(step))
        back = cq.importers.importStep(str(step))   # what the user gets: the STEP file, re-imported
        step_valid = len(back.solids().vals()) == 1 and back.solids().vals()[0].isValid()
        if isinstance(textured, trimesh.Trimesh):
            path = OUT_STL / f"{name}.stl"
            textured.export(path)
            for _ in range(3):   # float32 STL vertices can collapse a few sliver faces: clean, re-export, re-check
                m = trimesh.load(path)
                if m.is_watertight:
                    break
                m.update_faces(m.nondegenerate_faces())
                m.update_faces(m.unique_faces())
                m.remove_unreferenced_vertices()
                trimesh.repair.fill_holes(m)
                trimesh.repair.fix_normals(m)
                m.export(path)
        else:
            m = export_stl(textured if textured is not None else solid, OUT_STL / f"{name}.stl")
        rows[key] = dict(file=f"{name}.stl", watertight=bool(m.is_watertight), volume_mm3=float(abs(m.volume)),
                         bounds=np.round(m.bounds, 3).tolist(), valid=bool(step_valid), solids=len(solid.Solids()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-thumb", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="phalanx length scale (shorter pinky variant, e.g. 0.88); "
                    "outputs get the suffix _S088 etc.")
    ap.add_argument("--figures-only", action="store_true", help="only redraw figures/ from the exported parts")
    args = ap.parse_args()
    t0 = time.time()
    if args.figures_only:
        import uf_checks
        import uf_figures
        P = U.layout()
        rows = json.loads(DESIGN.read_text())["parts"]
        uf_figures.all_figures(P, None, uf_checks.widths(rows), log)
        return
    for p in (OUT_STEP, OUT_STL, FIG):
        p.mkdir(parents=True, exist_ok=True)
    log(f"make_ultimate_finger.py report (all mm, finger design frame); phalanx length scale {U.SCALE}")
    P = U.layout()
    log(f"2D layout: {len(U.REG)} parts, motion sampled at {len(U.PSI)} Link angles "
        f"{np.degrees(U.PSI[0]):.1f}..{np.degrees(U.PSI[-1]):.1f} deg [{time.time() - t0:.0f} s]")
    S = make_cores(P)
    sw = Sweeps(body_meshes(S))
    B = build_shells(S, P, sw)
    shells, pads, padspec, nail, p3core, plain = finish_shells(B, S["P3"])
    S["P3"] = p3core
    for attempt in range(3):   # the B-rep booleans of the stock MCP end + new body are occasionally invalid: rebuild
        bad = [k for k in ("link", "P1", "P2", "L2") if not fix_solid(one_solid(S[k])).isValid()]
        if not bad:
            break
        log(f"  core(s) {bad} came out invalid, rebuilding them (attempt {attempt + 1})")
        S2 = make_cores(P)
        for k in bad:
            S[k] = S2[k]
    parts = {k: (fix_solid(one_solid(S[k])), None) for k in ("link", "P1", "P2", "P3", "L2", "pin_PIP", "pin_A", "pin_DIP",
                                                             "pin_P4", "pin_B", "pin_MCP")}
    parts.update({k: (v, None) for k, v in shells.items()})
    parts.update({k: (v["smooth"], v["textured"]) for k, v in pads.items()})
    thumb_info = None
    if not args.no_thumb and not U.SUFFIX:
        th, thumb_info = thumb_parts()
        for k, v in th.items():
            parts[k] = (v["smooth"], v["textured"]) if isinstance(v, dict) else (v, None)
    rows = export_all(parts)
    log(f"exported {len(rows)} parts [{time.time() - t0:.0f} s]")
    design = dict(
        units="mm, deg",
        frame="finger design frame of the stock CAD (x lateral = pin axes, y distal, z dorsal, MCP at the origin); "
              "parts drawn in place in the straight pose; UF_Link in the stock Link frame; thumb parts in the stock "
              "2-phalanx finger frame",
        linkage_source="cad/five_finger/three_segment/design.json (unchanged)",
        geometry=U.g3, drive=U.design3["drive"], predicted=predicted(),
        layers=dict(C=[-U.C_HALF, U.C_HALF], g1=list(U.BANDS["g1"]), M=list(U.M_BAND), g2=list(U.BANDS["g2"]), O=list(U.O_BAND),
                    tip_O=list(U.TIP_O), running_gap=U.LAT, in_plane_clearance=U.CLR),
        pins={k: dict(spec=U.PIN_SPEC[k], held_by=v[0], centre=U.POINTS[v[1]].tolist(), half_length=v[2]) for k, v in U.PINS.items()},
        holes=dict(press=U.HOLE_PRESS, running=U.HOLE_RUN),
        shells=dict(clear=CLEAR, clear_moving=CLEAR_MOV, wall=WALL, joint_clear=JOINT_CLEAR, planes=B["planes"],
                    land=LAND_H, rim=RIM, pad_gap=PAD_GAP, pad_t=PAD_T, groove=[GROOVE_W, GROOVE_D], rib_pitch=GROOVE_PITCH,
                    nail=NAIL_H if NAIL else None, sleeve_style=SLEEVE_STYLE, tip=TIP,
                    style={k: v.style_info for k, v in B["specs"].items()},
                    pads={p["name"]: list(np.round(p["outline"].bounds, 3)) for p in padspec}),
        thumb=dict(source="cad/five_finger/slim_fingers (Proximal_Slim, Distal_Slim cores; slim shells + styled pad set)",
                   pads=thumb_info),
        parts=rows, material={k: MATERIAL.get(k, "steel (hardened dowel ISO 8734)") for k in rows},
        sim=dict(bodies=dict(link=["link"], P1=["P1", "P1_Shell", "P1_Pad", "pin_MCP", "pin_PIP", "pin_A"],
                             P2=["P2", "P2_Shell", "P2_Pad", "pin_P4", "pin_DIP"], P3=["P3", "P3_Shell", "P3_Pad", "pin_B"],
                             L2=["L2"]),
                 thumb=dict(proximal_shell="Thumb_Proximal_Shell", proximal_pad="Thumb_Proximal_Pad",
                            distal_shell="Thumb_Distal_Shell", distal_pad="Thumb_Distal_Pad")),
    )
    DESIGN.write_text(json.dumps(design, indent=1) + "\n")
    (HERE / "_layout_cache.json").unlink(missing_ok=True)
    import uf_checks
    uf_checks.run(P, S, shells, pads, B, sw, rows, design, log, figures=not args.no_figures)
    log(f"\ndone [{time.time() - t0:.0f} s]")
    (HERE / f"report_cad{U.SUFFIX}.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
