"""Checks and figures for make_ultimate_finger.py (CAD-level; the MuJoCo verification is in
Demo/AHSimulation/ultimate_finger/)."""

import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import make_slim_fingers as SF
import uf_layout as U

HERE = U.HERE
STL = HERE / "stl"
FIG = HERE / "figures"
COL = dict(core=[0.80, 0.81, 0.83, 1], core2=[0.70, 0.72, 0.75, 1], link=[0.95, 0.45, 0.20, 1], l2=[0.20, 0.55, 0.95, 1],
           shell=[0.96, 0.96, 0.95, 1], pad=[0.07, 0.07, 0.08, 1], pin=[0.62, 0.64, 0.68, 1], gimbal=[0.75, 0.55, 0.8, 1])


def mesh(name):
    if not name.startswith("Thumb"):
        name += U.SUFFIX
    return trimesh.load(STL / f"{name}.stl")


# ------------------------------------------------------------------ pin-hole walls and thin features (2D slabs)
def hole_walls(P):
    """Minimum in-plane wall around each pin hole, per part and slab (from the carved 2D profiles)."""
    holes = dict(P1=[("PIP", U.HOLE_PRESS), ("A", U.HOLE_PRESS)], P2=[("PIP", U.HOLE_RUN), ("P4", U.HOLE_PRESS), ("DIP", U.HOLE_PRESS)],
                 P3=[("DIP", U.HOLE_RUN), ("B", U.HOLE_PRESS)], L2=[("A", U.HOLE_RUN), ("B", U.HOLE_RUN)],
                 link_arm=[("P4", U.HOLE_RUN)])
    out = []
    for part, hs in holes.items():
        for key, d in hs:
            c = np.asarray(U.P4L if part == "link_arm" else U.POINTS[key])
            walls = []
            for x0, x1, geom in P[part].slabs:
                for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                    if poly.contains(Point(*c)):
                        walls.append((x0, x1, poly.boundary.distance(Point(*c)) - d / 2))
            if walls:
                w = min(walls, key=lambda t: t[2])
                out.append((part, key, d, len(walls), w[2], (w[0], w[1])))
    return out


def thin_features(P, t=1.2):
    """Area of each part's slabs that is thinner than t in-plane (material removed by an opening of radius t/2)."""
    out = {}
    for part in ("P1", "P2", "P3", "L2", "link_arm"):
        tot, thin = 0.0, 0.0
        for x0, x1, geom in P[part].slabs:
            w = x1 - x0
            opened = geom.buffer(-t / 2, quad_segs=8).buffer(t / 2, quad_segs=8)
            tot += geom.area * w
            thin += geom.difference(opened).area * w
        out[part] = (thin, tot)
    return out


# ------------------------------------------------------------------ widths
def width_at(meshes, y, plane_normal=(0, 1, 0)):
    xs = []
    for m in meshes:
        s = m.section(plane_origin=[0, y, 0], plane_normal=list(plane_normal))
        if s is not None:
            xs.append(s.vertices[:, 0])
    if not xs:
        return None
    x = np.concatenate(xs)
    return float(x.max() - x.min()), float(x.min()), float(x.max())


def widths(rows):
    seg = dict(proximal=(["UF_P1_Proximal", "UF_P1_Shell", "UF_P1_Pad", "pin_2x12_PIP", "pin_2x12_A"], np.arange(1.0, 50.0, 0.5)),
               middle=(["UF_P2_Middle", "UF_P2_Shell", "UF_P2_Pad", "pin_2x12_DIP", "pin_2x8_P4"], np.arange(47.0, 76.0, 0.5)),
               distal=(["UF_P3_Distal", "UF_P3_Shell", "UF_P3_Pad", "pin_2x8_B"], np.arange(73.0, 93.5, 0.5)))
    out = {}
    for k, (names, ys) in seg.items():
        ms = [mesh(n) for n in names]
        shell = [m for n, m in zip(names, ms) if "Shell" in n][0]
        w_all = [(y, width_at(ms, y)) for y in ys]
        w_shell = [(y, width_at([shell], y)) for y in ys]
        w_all = [(y, w) for y, w in w_all if w]
        w_shell = [(y, w) for y, w in w_shell if w]
        out[k] = dict(max_all=max(w_all, key=lambda t: t[1][0]), max_shell=max(w_shell, key=lambda t: t[1][0]),
                      profile=[(float(y), round(w[0], 3)) for y, w in w_all])
    return out


# ------------------------------------------------------------------ shells
def mesh_section_wall(m, y, step=0.01):
    """In-plane wall of a tube-like mesh at y: the section's two loops resampled every `step` mm, min distance between
    them. (The exact B-rep version, make_slim_fingers.brep_section_wall, samples the section curves at 1 um deflection;
    on these lofted + padded shells that blew up to tens of GB, so the check uses the 0.01 mm tessellation.)"""
    sec = m.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    if sec is None:
        return None
    loops = [lp[:, [0, 2]] for lp in sec.discrete if len(lp) >= 4]
    if len(loops) != 2:
        return None
    loops.sort(key=lambda lp: -Polygon(lp).area)

    def dense(lp):
        lp = np.vstack([lp, lp[:1]])
        seg = np.linalg.norm(np.diff(lp, axis=0), axis=1)
        t = np.r_[0, np.cumsum(seg)]
        u = np.arange(0, t[-1], step)
        return np.c_[np.interp(u, t, lp[:, 0]), np.interp(u, t, lp[:, 1])]
    d, _ = cKDTree(dense(loops[0])).query(dense(loops[1]))
    return float(d.min())


def shell_walls(shells, B):
    """Section walls between the cut ends (both planes clear of the whole section), plus a 3-D sampled measure."""
    out = {}
    for name, solid in shells.items():
        pl = B["planes"][name]
        m = mesh(f"UF_{name}")
        z0, z1 = m.bounds[0, 2], m.bounds[1, 2]
        a, b = pl["start"]
        y0 = max(a + b * z0, a + b * z1) + 0.3
        if "end" in pl:
            a, b = pl["end"]
            y1 = min(a + b * z0, a + b * z1) - 0.3
        else:
            y1 = U.TIP_END - 1.0
        w = [(y, mesh_section_wall(m, y)) for y in np.arange(y0, y1, 0.5)]
        w = [(y, v) for y, v in w if v is not None]
        th, pts = SF.wall_thickness(m)
        keep = np.isfinite(th)
        # leave out the pad lands (their 0.8 mm rims are not walls), a nail plate if any and the cut end faces
        for side, (a_, b_) in pl.items():
            keep &= np.abs(pts[:, 1] - (a_ + b_ * pts[:, 2])) > 0.8
        zmid = 0.5 * (z0 + z1)
        pb = B.get("pad_bounds", {}).get(name)
        pyz = B.get("pad_yz", {}).get(name)
        if pyz is not None:   # side-profile pad (fingertip): leave out its land, rim and pocket in the side view
            import shapely
            keep &= ~shapely.contains_xy(pyz.buffer(0.1 + 0.8 + 0.4), pts[:, 1], pts[:, 2])
        elif pb is not None:
            xl = max(abs(pb[0]), abs(pb[2])) + 0.1 + 0.8 + 0.3   # pad + glue gap + land rim + margin
            keep &= ~((np.abs(pts[:, 0]) < xl) & (pts[:, 1] > pb[1] - 1.2) & (pts[:, 1] < pb[3] + 1.2) & (pts[:, 2] < zmid - 0.5))
        else:
            keep &= ~((np.abs(pts[:, 0]) < 5.6) & (pts[:, 2] < zmid - 2.0))
        if name == "P3_Shell" and B.get("nail", True):
            keep &= ~((np.abs(pts[:, 0]) < 4.4) & (pts[:, 2] > 1.0) & (pts[:, 1] > U.TIP_END - 11.5))
        th = th[keep]
        if w:
            ymin, vmin = min(w, key=lambda t: t[1])
            out[name] = (len(w), vmin, ymin, float(np.median([v for _, v in w])), float(np.percentile(th, 1)),
                         float(np.median(th)))
    return out


_TREES = {}


def min_dist(a_pts, b_mesh, cutoff=3.0):
    if id(b_mesh) not in _TREES:
        _TREES[id(b_mesh)] = (b_mesh, cKDTree(SF.sample_surface(b_mesh, 0.15)))
    tree = _TREES[id(b_mesh)][1]
    lo, hi = b_mesh.bounds[0] - cutoff, b_mesh.bounds[1] + cutoff
    q = a_pts[np.all((a_pts >= lo) & (a_pts <= hi), axis=1)]
    if not len(q):
        return np.inf
    d, _ = tree.query(q, distance_upper_bound=cutoff, workers=4)
    return float(d.min())


def run(P, S, shells, pads, B, sw, rows, design, log, figures=True):
    log("\n== parts (B-rep validity of the re-imported STEP; STL watertight; volume from the STL; bbox mm)")
    for k, r in rows.items():
        b = np.array(r["bounds"])
        log(f"  {k:<22} valid {r['valid']}  solids {r['solids']}  watertight {r['watertight']}  volume {r['volume_mm3']:8.1f} mm3  "
            f"x [{b[0, 0]:6.2f}, {b[1, 0]:6.2f}] y [{b[0, 1]:6.2f}, {b[1, 1]:6.2f}] z [{b[0, 2]:6.2f}, {b[1, 2]:6.2f}]")
    log("\n== pin-hole walls in the rigid parts (in-plane, from the carved profiles; rule >= 1.5 mm)")
    hw = hole_walls(P)
    for part, key, d, n, w, band in hw:
        log(f"  {part:<9} {key:<4} hole d{d:.2f}: {n} slab(s), thinnest wall {w:.2f} mm (|x| band {abs(band[0]):.2f}..{abs(band[1]):.2f})"
            + ("" if w >= 1.5 - 1e-6 else "  < 1.5"))
    log("\n== thin features: slab material narrower than 1.2 mm in-plane (volume, share of the part)")
    for part, (thin, tot) in thin_features(P).items():
        log(f"  {part:<9} {thin:7.1f} of {tot:7.1f} mm3 ({100 * thin / tot:4.1f} %)")
    log("\n== widths across the finger (x), including shells, pads and pins")
    W = widths(rows)
    for k, v in W.items():
        (ya, wa), (ys, wsh) = v["max_all"], v["max_shell"]
        log(f"  {k:<8} widest {wa[0]:6.2f} mm at y {ya:5.1f} (x {wa[1]:+.2f}..{wa[2]:+.2f}); shell alone {wsh[0]:6.2f} mm at y {ys:5.1f}")
    log("  knuckles (cores + pins): MCP end 16.2 (stock Proximal ears around the stock gimbal), PIP 12.0, DIP 12.0")
    log("\n== TPU shells: wall (sections of the 0.01 mm tessellation every 0.5 mm, two-loop sections)")
    for k, (n, vmin, ymin, med, p1, med3) in shell_walls(shells, B).items():
        log(f"  {k:<9} {n} sections between the cut ends: min {vmin:.3f} mm at y {ymin:.1f}, median {med:.3f}; 3-D sampled "
            f"(nearest opposite-facing surface, pad lands / nail / cut faces excluded): 1st percentile {p1:.2f}, median {med3:.2f} mm")
    log("\n== shells on their own core (straight finger): min gap core surface -> shell surface")
    for name, core in (("P1_Shell", "UF_P1_Proximal"), ("P2_Shell", "UF_P2_Middle"), ("P3_Shell", "UF_P3_Distal")):
        sm, cm = mesh(f"UF_{name}"), mesh(core)
        try:
            inside = trimesh.boolean.intersection([sm, cm], engine="manifold")
            vol = float(inside.volume) if not inside.is_empty else 0.0
        except ValueError:   # a mesh that is not a closed volume
            vol = float("nan")
        log(f"  {name:<9} overlap with its core {vol:.3f} mm3; min gap {min_dist(SF.sample_surface(cm, 0.2), sm):.3f} mm")
    log("\n== pads do not widen the finger: pad |x| vs shell half width at the pad's y range")
    for pname, shell in (("UF_P1_Pad", "UF_P1_Shell"), ("UF_P2_Pad", "UF_P2_Shell"), ("UF_P3_Pad", "UF_P3_Shell"),
                         ("Thumb_Proximal_Pad", "Thumb_Proximal_Shell"), ("Thumb_Distal_Pad", "Thumb_Distal_Shell")):
        if not (STL / f"{pname}{'' if pname.startswith('Thumb') else U.SUFFIX}.stl").exists():
            continue
        pm, sm = mesh(pname), mesh(shell)
        y0, y1 = pm.bounds[0, 1], pm.bounds[1, 1]
        sel = (sm.vertices[:, 1] >= y0) & (sm.vertices[:, 1] <= y1)
        log(f"  {pname:<20} |x| <= {np.abs(pm.vertices[:, 0]).max():.2f} vs shell |x| <= {np.abs(sm.vertices[sel, 0]).max():.2f} "
            f"over y {y0:.1f}..{y1:.1f}; thickness {PAD_INFO}")
    log("\n== joint sweeps (3-D, whole servo travel, 0.5 deg Link steps): shells vs everything moving past them")
    shell_frames = dict(P1_Shell="P1", P2_Shell="P2", P3_Shell="P3")
    shm = {k: mesh(f"UF_{k}") for k in shell_frames}
    padm = {k: mesh(f"UF_{k.split('_')[0]}_Pad") for k in shell_frames}
    for k in shell_frames:
        sw.add(shell_frames[k], shm[k], 0.3)
        sw.add(shell_frames[k], padm[k], 0.3)
    for k, fr in shell_frames.items():
        cells = []
        for other in ("gimbal", "link", "P1", "P2", "P3", "L2"):
            if other == fr:
                continue
            ds = [(min_dist(c, shm[k]), min_dist(c, padm[k])) for c in sw.swept_chunks(fr, other)]
            d_sh, d_pad = min(x[0] for x in ds), min(x[1] for x in ds)
            d = min(d_sh, d_pad)
            cells.append((f"{other} {d:.2f}" + (" (pad)" if d_pad < d_sh else "")) if np.isfinite(d) else f"{other} >3")
        log(f"  {k:<9} (+ its pad) min clearance to swept: " + ", ".join(cells))
    design["checks"] = dict(hole_walls=[dict(part=a, pin=b, hole=c, wall=round(w, 3)) for a, b, c, _, w, _ in hw],
                            widths={k: dict(widest=round(v["max_all"][1][0], 3), at_y=v["max_all"][0],
                                            shell=round(v["max_shell"][1][0], 3)) for k, v in W.items()})
    (HERE / f"design{U.SUFFIX}.json").write_text(json.dumps(design, indent=1) + "\n")
    if figures:
        import uf_figures
        uf_figures.all_figures(P, B, W, log)


PAD_INFO = "0.80 mm (0.45 in the land pocket + 0.35 proud), 0.50 under a 0.30 groove; ribs 0.55 / grooves 0.45 mm, 1.0 mm pitch"
