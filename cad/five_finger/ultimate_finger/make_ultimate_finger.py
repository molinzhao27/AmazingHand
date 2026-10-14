#!/usr/bin/env python3
"""The "ultimate" AmazingHand finger: thin, styled, 3 phalanges, on the unchanged 2-servo drive.

Merges three earlier studies of the 5-digit hand:
  * cad/five_finger/three_segment   3 phalanges with the stock drive (linkage kept unchanged, see uf_layout.py);
  * cad/five_finger/slim_fingers    TPU covers hugging the cores (0.25 mm clearance, >= 1.2 mm wall), cover ends kept
                                    >= 0.5 mm from everything that sweeps past, exposed knuckles;
  * cad/five_finger/fingers         black TPU grip surfaces.

Covers (after the user's reference hands; uf_covers.py): implicit (voxel SDF) solids. The proximal and middle phalanges
are rounded capsules (superellipse sections, slightly convex back and palm lines, domed ends converging on the joint
pins); the distal one is 5 mm longer than in the 3-phalanx study (uf_layout.TIP_EXT) and tapers from the DIP into a
pointed, round-nosed ogive (nail side curving down, a gently convex pad side sweeping up to meet it). Every cover is
split into a white back shell and a black TPU glove over the palm half (transverse ribs, 1 mm pitch) that wraps the
fingertip. Cut rims are full-thickness (the skin is removed in whole columns). The covers are STL only. The thumb gets
the same covers on the slim-finger parts (it stays 2-phalanx).

New here: the cores are re-stacked for slimness (uf_layout.py: 3.0 mm centre links, 2 x 12 / 2 x 8 pins, 12.0 mm
knuckles instead of 16.2), the longer fingertip, the implicit covers, and the fingertip core trimmed to its cap (mesh
boolean).

Frame (mm): the finger design frame of the stock CAD: x lateral (pin axes), y distal, z dorsal (palm = -z), MCP at
the origin, every part drawn in place in the straight pose; the Link in the stock Link's own frame. The thumb parts
are in the stock 2-phalanx finger frame (as cad/five_finger/slim_fingers).

Writes step/*.step (rigid parts; UF_P3_Distal_untrimmed = the fingertip core before its tip trim), stl/*.stl (all
parts; the covers only as STL), figures/*.png, design.json and report_cad.txt.

Usage: .venv/bin/python cad/five_finger/ultimate_finger/make_ultimate_finger.py [--no-thumb] [--no-figures] [--scale 0.88]
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
from shapely.geometry import Polygon, box  # noqa: F401
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

# ------------------------------------------------------------------ cover parameters: see uf_covers.py
JOINT_CLEAR = 0.5     # covers: >= this from everything that sweeps past (whole servo travel)


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


# ------------------------------------------------------------------ implicit covers (uf_covers.py)
COVER_NAMES = {"P1": ("P1_Shell", "P1_Pad"), "P2": ("P2_Shell", "P2_Pad"), "P3": ("P3_Shell", "P3_Pad")}
THUMB_NAMES = {"prox": ("Thumb_Proximal_Shell", "Thumb_Proximal_Pad"), "dist": ("Thumb_Distal_Shell", "Thumb_Distal_Pad")}


def finger_covers(S):
    """White back shell + black ribbed glove per phalanx, as implicit solids (uf_covers.py)."""
    import uf_covers as C
    t0 = time.time()
    specs = C.finger_specs(U)
    bm = body_meshes(S)
    pts = {k: np.concatenate([SF.sample_surface(m, 0.2) for m in ms]) for k, ms in bm.items()}
    lp, l2p = pts["link"], pts["L2"]
    u = (U.P4L - U.O_L) / np.linalg.norm(U.P4L - U.O_L)
    arm = ((lp[:, 1:] - U.O_L) @ u >= 7.5) & (np.abs(lp[:, 0]) <= U.C_HALF + 0.05)   # the Link arm runs inside P1
    beye = np.linalg.norm(l2p[:, 1:] - U.B, axis=1) <= U.L2_R + 0.9                 # L2's B eye rises out of P2
    movers = {"P1": [("link", lp[arm])], "P2": [("L2", l2p[~beye])]}
    ko = {"P1": [("gimbal", pts["gimbal"]), ("link", lp[~arm]), ("P2", pts["P2"]), ("P3", pts["P3"]), ("L2", l2p)],
          "P2": [("gimbal", pts["gimbal"]), ("link", lp), ("P1", pts["P1"]), ("P3", pts["P3"]), ("L2", l2p[beye])],
          "P3": [("gimbal", pts["gimbal"]), ("link", lp), ("P1", pts["P1"]), ("P2", pts["P2"]), ("L2", l2p)]}

    def pose_rel(dst, src, k):
        return np.linalg.inv(U.pose(dst, k)) @ U.pose(src, k)
    out, info = C.build_covers(["P1", "P2", "P3"], pts, movers, ko, specs, pose_rel, range(len(U.PSI)), log, COVER_NAMES)
    log(f"implicit covers [{time.time() - t0:.0f} s]")
    return out, info, specs


def trim_p3_mesh(P3core, info, spec):
    """The P3 core's tip body trimmed to the cap's cavity shrunk by CLEAR (+0.05), as a mesh boolean against the
    cavity surface of the very field the cover was made from (exactly consistent with the cover). Returns the trimmed
    mesh (the STL / sim part); the STEP keeps the B-rep core before this trim."""
    import manifold3d as mf
    y_trim = spec["trim_from"]
    core = SF.to_trimesh(one_solid(P3core), 0.01)
    cav = info["trim_cavity"]
    M = lambda m: mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),  # noqa: E731
                                      tri_verts=np.asarray(m.faces, np.uint32)))
    beyond = mf.Manifold.cube([40.0, 60.0, 40.0]).translate([-20.0, y_trim, -20.0])
    out = M(core) - (beyond - M(cav))
    m = out.to_mesh()
    tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)
    trimesh.repair.fix_normals(tm)
    log(f"  P3 core tip trimmed to the cap cavity - {covers_mod().CLEAR + 0.05:.2f} mm from y = {y_trim:.2f} (mesh boolean): "
        f"{core.volume:.0f} -> {tm.volume:.0f} mm3, watertight {tm.is_watertight}")
    return tm


def thumb_specs():
    """The 2-phalanx thumb (stock finger frame of cad/five_finger/slim_fingers): capsule proximal, bullet distal."""
    def n_dist(Y):
        return 4.0 - 1.0 * np.clip((Y - 70.0) / 12.0, 0, 1)
    return dict(
        prox=dict(a=9.9, n=4.0, n_top=3.2, body=(13.0, 46.0), fit=(18.0, 42.0), domes=[(13.0, 2.0, 0.0), (46.0, 52.0, 0.0)],
                  bulge=0.5, bulge_palm=0.3, seam=0.15, bounds=((-11.6, 0.0, -13.5), (11.6, 57.0, 17.5))),
        dist=dict(a=7.6, n=n_dist, body=(58.0, 82.0), fit=(58.0, 82.0), domes=[(58.0, 52.0, 0.0), (82.0, 93.6, -0.6)],
                  bulge=0.2, seam=0.15, nail=dict(y=89.8, curve=0.06), rib_nose=(82.0, -0.6, 7.5),
                  bounds=((-9.6, 47.0, -12.5), (9.6, 96.0, 12.5))),
    )


def thumb_covers():
    """Thumb covers in the same style on the slim-finger parts (glued, no shell screws): proximal capsule, bullet
    distal with the nail and the glove wrapping the end."""
    import uf_covers as C
    t0 = time.time()
    cores = SF.Cores()
    prox_core, dist_core, _, _ = SF.make_cores(cores)
    hw = SF.hardware(SF.PROX_HALF - SF.SCREW["recess"], SF.DIST_HALF - SF.SCREW["recess"])
    bodies = {"prox": [SF.to_trimesh(prox_core, 0.02)] + hw["prox"][:2],      # pins (the shell screws are gone)
              "dist": [SF.to_trimesh(dist_core, 0.02)] + hw["dist"][:1],
              "link": [SF.load_stl("Link")] + hw["link"], "gimbal": [SF.load_stl("Gimbal")] + hw["gimbal"]}
    pts = {k: np.concatenate([SF.sample_surface(m, 0.2) for m in ms]) for k, ms in bodies.items()}
    poses = SF.fourbar(np.arange(SF.PHI_RANGE[0], SF.PHI_RANGE[1] + 1e-9, 0.25))

    def pose_rel(dst, src, k):
        R, t = SF.rel_pose(poses[k][dst], poses[k][src])
        T = np.eye(3)
        T[:2, :2], T[:2, 2] = R, t
        return T
    ko = {"prox": [("dist", pts["dist"]), ("link", pts["link"]), ("gimbal", pts["gimbal"])],
          "dist": [("prox", pts["prox"]), ("link", pts["link"]), ("gimbal", pts["gimbal"])]}
    out, info = C.build_covers(["prox", "dist"], pts, {}, ko, thumb_specs(), pose_rel, range(len(poses)), log, THUMB_NAMES)
    log(f"thumb covers [{time.time() - t0:.0f} s]")
    return out, info, bodies, pose_rel, len(poses)


def covers_mod():
    import uf_covers
    return uf_covers


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
        if solid is None:   # implicit cover: STL only
            textured.export(OUT_STL / f"{name}.stl")
            m = trimesh.load(OUT_STL / f"{name}.stl")
            (OUT_STEP / f"{name}.step").unlink(missing_ok=True)
            (OUT_STEP / f"{name}_untextured.step").unlink(missing_ok=True)
            rows[key] = dict(file=f"{name}.stl", watertight=bool(m.is_watertight), volume_mm3=float(abs(m.volume)),
                             bounds=np.round(m.bounds, 3).tolist(), valid="STL only", solids=len(m.split(only_watertight=False)))
            continue
        step = OUT_STEP / f"{name}{('_untrimmed' if key == 'P3' else '_untextured') if textured is not None else ''}.step"
        if key == "P3":
            (OUT_STEP / f"{name}.step").unlink(missing_ok=True)
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
    covers, cinfo, cspecs = finger_covers(S)
    p3_trimmed = trim_p3_mesh(S["P3"], cinfo["P3"], cspecs["P3"])
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
    parts["P3"] = (parts["P3"][0], p3_trimmed)   # STEP: the core before the tip trim; STL: trimmed to the cap
    parts.update({k: (None, v) for k, v in covers.items()})       # implicit covers: STL only
    thumb_info = None
    if not args.no_thumb and not U.SUFFIX:
        th, tinfo, tbodies, tpose, tn = thumb_covers()
        parts.update({k: (None, v) for k, v in th.items()})
        thumb_info = dict(skin={k: v["skin"] for k, v in tinfo.items()}, thin={k: v["thin"] for k, v in tinfo.items()},
                          bodies=tbodies, pose=tpose, n=tn, covers=th)
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
        covers=dict(method="implicit (voxel SDF) solids, cad/five_finger/ultimate_finger/uf_covers.py; STL only",
                    clear=covers_mod().CLEAR, clear_moving=covers_mod().CLEAR_MOV, wall=covers_mod().WALL,
                    keep_out_field=covers_mod().KO, keep_out_band=covers_mod().KO_W, rim_column_depth=covers_mod().COL_DEPTH,
                    joint_clear=JOINT_CLEAR, voxel=covers_mod().H,
                    ribs=dict(height=covers_mod().RIB_H, width=covers_mod().RIB_W, pitch=covers_mod().RIB_P),
                    style={k: {kk: vv for kk, vv in v.items() if kk not in ("design", "n") or not callable(vv)} for k, v in cspecs.items()},
                    skin={k: v["skin"] for k, v in cinfo.items()}, thin_rims={k: v["thin"][:2] for k, v in cinfo.items()}),
        thumb=dict(source="cad/five_finger/slim_fingers (Proximal_Slim, Distal_Slim cores); covers in the same implicit style, glued",
                   skin=thumb_info["skin"] if thumb_info else None),
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
    uf_checks.run(P, S, covers, cinfo, sw, rows, design, log, figures=not args.no_figures, thumb=thumb_info)
    log(f"\ndone [{time.time() - t0:.0f} s]")
    (HERE / f"report_cad{U.SUFFIX}.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
