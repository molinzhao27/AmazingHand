#!/usr/bin/env python3
"""Print plates for the all-printed mock-up, sliced for an Anycubic Kobra X as Anycubic Slicer Next 3MF projects.

Every part of the mock-up (the real printed parts of AH_Right_5F_3S_Wrist plus the stand-ins from make_mockup.py) is
put in its print orientation, grouped by material, colour, layer height and supports, and sliced with the slicer's
command line, whose arrange spreads a group over as many 260 x 260 mm plates as it needs. Each group is one 3MF
project in plates/ with its plates in it (open it in Anycubic Slicer Next, check it, print plate by plate). The two
test plates come first:

  00_test_fit_PLA          the real hole sizes in a gauge block + every stand-in fit in three sizes
  00_test_switch_PLA_TPU   a PLA base with a TPU 95A ribbed pad on top: one filament change (PLA -> TPU)
  00_test_balls_PLA        just the fit test's two ball posts (block, stud pin, pressed-on ball) for the cups

Orientation: the build notes' orientation where they give one (pins vertical for the finger cores, the wrist parts as
make_wrist.PRINT_UP). The big curved parts (ORIENT) and the covers were chosen by slicing the candidates and keeping
the least support that still prints safely. Every other part takes the direction with the least overhang among 400
(below 45 deg, discounting faces on the bed) that keeps a usable bed contact and does not stand it tall and thin.
Supports use the presets' threshold (20 deg at 0.12 mm, 30 at 0.16 / 0.20).

Brims only where a part needs one (needs_brim): tall and slender (height > 3 x the narrow side of its first layer
and > 10 mm), a small base with height (first layer < 20 mm2 and > 6 mm tall), or a line contact (no first-layer
island wider than 1.2 mm). Those parts go in their own project (suffix _brim) with a 3 mm brim, 0.2 mm off the part
so it snaps off; every other plate has none. PLA and TPU 95A on the Kobra X's PEI plate do not warp enough to need
one otherwise.

Materials: every real part is white PLA except the finger gloves and the palm pad, which are black TPU 95A. The
stand-ins are black PLA (they look like the hardware they replace). The mock-up only tests fit and shape, so every
rigid part is PLA here (the real build wants PA-CF / PETG for some, see FIVE_FINGER.md).

Usage: .venv/bin/python cad/five_finger/mockup/make_plates.py [--only NAME_PREFIX] [--no-slice]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "wrist"))
OUT = HERE / "plates"
WORK = HERE / "plates" / ".work"

APP = "/Applications/AnycubicSlicerNext.app/Contents/MacOS/AnycubicSlicerNext"
PROFILES = Path("/Applications/AnycubicSlicerNext.app/Contents/Resources/profiles/Anycubic")
MACHINE = "Anycubic Kobra X 0.4 nozzle"
FIL = {"PLA": "Anycubic PLA @Anycubic Kobra X 0.4 nozzle", "TPU": "Anycubic TPU 95A @Anycubic Kobra X 0.4 nozzle"}
PROC = {0.12: "0.12mm High Quality @Anycubic Kobra X 0.4 nozzle", 0.16: "0.16mm Standard @Anycubic Kobra X 0.4 nozzle",
        0.20: "0.20mm Standard @Anycubic Kobra X 0.4 nozzle"}
COLOURS = {"white": "#F2F2F2", "black": "#1A1A1A"}
PLA_TEMP = (200, 210)   # nozzle, first layer (deg C)
BED = 260.0

C = "cad/five_finger"
UF = f"{C}/ultimate_finger/stl"
# (file, qty, material, colour, orientation rule): the real printed parts of AH_Right_5F_3S_Wrist (see
# cad/five_finger/assembly/report.txt). The stock Spacer is not printed: the one-piece rods replace rod + spacer.
REAL = [
    (f"{C}/stl/R-Hand_Plate_5F.stl", 1, "PLA", "white", "auto"),
    (f"{C}/stl/R-Top_Shell_5F.stl", 1, "PLA", "white", "auto"),
    (f"{C}/wrist/stl/R-Wrist_Hand_Adapter.stl", 1, "PLA", "white", "wrist"),
    ("cad/stl/Amazing Hand Parts - Finger_Frame-1.stl", 5, "PLA", "white", "auto"),
    ("cad/stl/Amazing Hand Parts - Finger_Frame-2.stl", 5, "PLA", "white", "auto"),
    ("cad/stl/Amazing Hand Parts - Gimbal.stl", 5, "PLA", "white", "auto"),
    ("cad/stl/Amazing Hand Parts - Link.stl", 1, "PLA", "white", "auto"),
    (f"{C}/slim_fingers/stl/Proximal_Slim.stl", 1, "PLA", "white", "pins"),
    (f"{C}/slim_fingers/stl/Distal_Slim.stl", 1, "PLA", "white", "pins"),
] + [(f"{UF}/UF_{n}{s}.stl", 3 if not s else 1, "PLA", "white", "pins")
     for n in ("Link", "P1_Proximal", "P2_Middle", "P3_Distal", "L2_Coupler") for s in ("", "_S088")] + [
    # covers: the back (Shell) white PLA, the glove (Pad) black TPU 95A
    (f"{UF}/UF_{p}_{k}{s}.stl", 3 if not s else 1, *(("PLA", "white") if k == "Shell" else ("TPU", "black")), "cover")
    for p in ("P1", "P2", "P3") for k in ("Shell", "Pad") for s in ("", "_S088")] + [
    (f"{UF}/Thumb_{p}_{k}.stl", 1, *(("PLA", "white") if k == "Shell" else ("TPU", "black")), "cover")
    for p in ("Proximal", "Distal") for k in ("Shell", "Pad")] + [
    (f"{C}/stl/R-Palm_Shell_5F.stl", 1, "PLA", "white", "auto"),
    (f"{C}/stl/R-Palm_Pad_Palm_5F.stl", 1, "TPU", "black", "auto"),
    (f"{C}/wrist/stl/Wrist_Gimbal_Ring.stl", 1, "PLA", "white", "wrist"),
    (f"{C}/wrist/stl/R-Wrist_Fork.stl", 1, "PLA", "white", "wrist"),
    (f"{C}/wrist/stl/R-Wrist_Servo_Frame.stl", 1, "PLA", "white", "wrist"),
    (f"{C}/wrist/stl/Wrist_Horn_Arm.stl", 2, "PLA", "white", "wrist"),
    (f"{C}/wrist/stl/Elbow_Flange.stl", 1, "PLA", "white", "wrist"),
    (f"{C}/wrist/stl/R-Forearm_Front.stl", 1, "PLA", "white", "wrist"),
    (f"{C}/wrist/stl/R-Forearm_Back.stl", 1, "PLA", "white", "wrist"),
]

# the big, curved parts, from slicing every axis and the best few of a support-volume estimate (0.20 mm, supports
# everywhere), keeping the least support among the orientations that are safe to print (low, or on a real base):
ORIENT = {
    "R-Hand_Plate_5F": (0, 0, -1),      # back face down: 11 g support (every other side 40+ g)
    "R-Top_Shell_5F": (0, 0, -1),       # roof down, 13 mm tall: 14 g support, 3 h 20 min (open side down: 28 g,
                                        # 4 h 25 min, a 130 mm roof on supports). Standing on its wrist end needs only
                                        # 7 g but would stand 135 mm tall on a point (0.1 mm2 first layer)
    "R-Palm_Shell_5F": (0, 0, 1),       # rim up, 58 mm tall: 31 g, 5 h 30 min (rim down: 46 g, 6 h 30 min). Standing
                                        # on its wrist end needs 22 g but is 137 mm tall on a 9 mm2 first layer
    "R-Palm_Pad_Palm_5F": (0, 0, -1),   # ribs up: 9 g. On edge would be 1 g, but an 85 mm-tall sheet of TPU wobbles
}
BIG = tuple(ORIENT)

# plate groups: name, material, colour, layer, supports (None / "plate": from the bed only / "all"), which parts
GROUPS = [
    ("01_hand_cores_PLA_white", "PLA", "white", 0.12, "plate", lambda p: p["src"] == "real" and p["rule"] in ("pins",)
     or p["file"].startswith("cad/stl/")),
    ("02_palm_PLA_white", "PLA", "white", 0.20, "all", lambda p: Path(p["file"]).stem in
     ("R-Hand_Plate_5F", "R-Top_Shell_5F", "R-Palm_Shell_5F", "R-Wrist_Hand_Adapter")),
    ("03_wrist_PLA_white", "PLA", "white", 0.20, "plate", lambda p: p["rule"] == "wrist" and "Forearm" not in p["file"]
     and "Hand_Adapter" not in p["file"]),
    ("04_forearm_PLA_white", "PLA", "white", 0.20, "plate", lambda p: "Forearm" in p["file"]),
    ("05_standins_PLA_black", "PLA", "black", 0.12, "plate", lambda p: p["src"] == "standin"),
    ("06_covers_PLA_white", "PLA", "white", 0.12, "plate", lambda p: p["rule"] == "cover" and p["material"] == "PLA"),
    ("07_TPU95A_black_pad", "TPU", "black", 0.16, "all", lambda p: Path(p["file"]).stem == "R-Palm_Pad_Palm_5F"),
    ("08_TPU95A_black_gloves", "TPU", "black", 0.16, "plate", lambda p: p["rule"] == "cover" and p["material"] == "TPU"),
]


def log(*a):
    print(*a, flush=True)


# ============================================================================ orientation
def fib(n):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def score(m, up, tol=0.3, tol_contact=0.05):
    """(overhang area below 45 deg off the bed, bed contact area) with `up` as print up. Contact counts only faces
    that lie flat on the bed (within 0.05 mm), so a part tilted a degree off its flat face scores no contact."""
    h = m.vertices @ up
    fn = m.face_normals @ up
    fz, fzmax = h[m.faces].min(1), h[m.faces].max(1)
    on_bed = fz <= h.min() + tol
    over = m.area_faces[(fn < -np.cos(np.radians(45))) & ~on_bed].sum()
    contact = m.area_faces[(fn < -0.999) & (fzmax <= h.min() + tol_contact)].sum()
    return over, contact


def flat_faces(m, n=40):
    """Print-up candidates that put one of the part's largest flat faces exactly on the bed."""
    if not len(m.facets):
        return []
    order = np.argsort(m.facets_area)[::-1][:n]
    return [-m.facets_normal[k] for k in order if m.facets_area[k] > 2.0]


def slender(m, u):
    """True if the part would stand tall and thin with `u` up: over 20 mm and over 3 x its footprint's short side."""
    h = m.vertices @ u
    e1 = np.cross(u, (1, 0, 0) if abs(u[0]) < 0.9 else (0, 1, 0))
    e1 /= np.linalg.norm(e1)
    xy = np.c_[m.vertices @ e1, m.vertices @ np.cross(u, e1)]
    _, ext = trimesh.bounds.oriented_bounds_2D(xy)
    H = h.max() - h.min()
    return H > 20 and H > 3 * min(ext)


def best_up(m, cands, min_contact=12.0):
    best = None
    for u in cands:
        u = np.asarray(u, float) / np.linalg.norm(u)
        over, contact = score(m, u)
        cost = over + (400.0 if contact < min_contact else 0.0) - 0.05 * contact
        cost += 300.0 if min_contact and slender(m, u) else 0.0   # it would wobble: lie it down if that is cheap
        if best is None or cost < best[0]:
            best = (cost, u, over, contact)
    return best[1], best[2], best[3]


def orient(m, rule, name, up=None):
    """Rotation (3x3) that takes the part to its print orientation (up -> +z)."""
    mm = m.simplify_quadric_decimation(face_count=30000) if len(m.faces) > 30000 else m
    if up is not None:
        u = np.asarray(up, float)
    elif rule == "pins":                       # pin axes vertical (cores print on a side face): +x or -x
        u = best_up(mm, [(1, 0, 0), (-1, 0, 0)], 0)[0]
    elif rule == "cover":   # low on the bed, not standing on an end (TPU wobbles): the back (Shell, PLA) on its seam,
        # dome up, supports hidden inside; the glove (Pad, TPU) on its outside, a few support tips that peel off
        u = np.array([0, 0, -1.0]) if "_Shell" in name else np.array([0, 0, 1.0])
    elif name in ORIENT:
        u = np.asarray(ORIENT[name], float)
    elif rule == "wrist":
        import make_wrist as MW
        u = MW.print_up_cad(name)
    else:
        u = best_up(mm, flat_faces(mm) + [tuple(v) for v in np.vstack([np.eye(3), -np.eye(3)])] + list(fib(200)))[0]
    u = u / np.linalg.norm(u)
    # rotation taking u to +z
    z = np.array([0, 0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), float(u @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


# ============================================================================ parts
def collect():
    parts = []
    for f, q, mat, col, rule in REAL:
        parts.append(dict(file=f, qty=q, material=mat, colour=col, rule=rule, src="real", up=None))
    manifest = json.loads((HERE / "stl" / "parts.json").read_text())
    for n, p in manifest.items():
        parts.append(dict(file=f"{C}/mockup/stl/{n}.stl", qty=p["qty"], material="PLA", colour="black", rule="auto",
                          src="standin", up=p["up"]))
    return parts


def needs_brim(m):
    """(True/False, reason) for a part in its print orientation, from its first layer and its height."""
    from shapely.ops import unary_union
    H = float(m.extents[2])
    sec = m.section(plane_origin=[0, 0, 0.1], plane_normal=[0, 0, 1])
    if sec is None:
        return True, "no first layer"
    planar, _ = sec.to_2D() if hasattr(sec, "to_2D") else sec.to_planar()
    islands = list(planar.polygons_full)
    if not islands:
        return True, "no first layer"

    def short_side(poly):
        r = poly.minimum_rotated_rectangle
        x, y = r.exterior.coords.xy
        e = [np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(2)]
        return min(e)

    area = sum(pl.area for pl in islands)
    w_fp = short_side(unary_union(islands).convex_hull)
    w_island = max(short_side(pl) for pl in islands)
    if H > 10 and H > 3 * w_fp:
        return True, f"tall and slender: {H:.0f} mm on a {w_fp:.1f} mm wide footprint"
    if area < 20 and H > 6:
        return True, f"small base: {area:.0f} mm2 under {H:.0f} mm"
    if w_island < 1.2:
        return True, f"line contact: {w_island:.2f} mm wide"
    return False, f"first layer {area:.0f} mm2, {w_fp:.1f} mm wide, {H:.0f} mm tall"


def prepare(p, folder):
    """Oriented copy on the bed, centred; returns its path, XY footprint (convex hull area, extents), overhang, bed
    contact and whether it needs a brim (with the reason)."""
    m = trimesh.load(REPO / p["file"], process=True)
    name = Path(p["file"]).stem
    up = None
    if p["src"] == "standin":
        up = p["up"] if p["up"] is not None else (0, 0, 1) if name.startswith(("Pin_", "Axle_", "Peg_", "Ball_D5",
                                                                                 "Stud_Pin_", "Ball_Stud_", "Insert_")) else None
    R = orient(m, p["rule"], name, up)
    m.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    out = folder / f"{name}.stl"
    m.export(out)
    xy = m.vertices[:, :2]
    from scipy.spatial import ConvexHull
    area = ConvexHull(xy).volume
    ext = m.extents
    over, contact = score(m, np.array([0, 0, 1.0]))
    brim, why = needs_brim(m)
    return out, area, ext, over, contact, brim, why


# ============================================================================ slicing
def preset(kind, name):
    return json.loads((PROFILES / kind / f"{name}.json").read_text())


def gcode_stats(path):
    """Time, grams, filament changes of one plate's G-code, and whether every printed move is on the bed."""
    txt = open(path, errors="ignore").read()
    grab = lambda pat: (re.findall(pat, txt) or ["?"])[-1]   # noqa: E731
    lo, hi, typ = [np.inf, np.inf], [-np.inf, -np.inf], ""
    for line in txt.splitlines():
        if line.startswith(";TYPE:"):
            typ = line[6:].strip()
        elif typ not in ("Custom", "") and line.startswith("G1") and " E" in line and "X" in line:
            mx, my = re.search(r"X(-?[\d.]+)", line), re.search(r"Y(-?[\d.]+)", line)
            if mx and my:
                x, y = float(mx.group(1)), float(my.group(1))
                lo, hi = [min(lo[0], x), min(lo[1], y)], [max(hi[0], x), max(hi[1], y)]
    fits = lo[0] >= 0 and lo[1] >= 0 and hi[0] <= BED and hi[1] <= BED
    return {"time": grab(r"estimated printing time \(normal mode\) = ([^\n=]+?)\s*(?:=|\n)"),
            "grams": grab(r"; filament used \[g\] = ([^\n]+)"), "fits_bed": bool(fits),
            "filament_changes": int(grab(r"; total filament change = (\d+)").replace("?", "0"))}


def process_json(layer, supports, two_materials=False, brim=False):
    """supports: None, "plate" (tree supports from the bed only) or "all" (also on the part's own surfaces)."""
    proc = preset("process", PROC[layer])
    proc.update({"name": f"{layer:.2f}mm mock-up", "wall_loops": "3", "sparse_infill_density": "20%",
                 "sparse_infill_pattern": "gyroid", "elefant_foot_compensation": "0.15",
                 "brim_type": "outer_only" if brim else "no_brim", "brim_width": "3", "brim_object_gap": "0.2",
                 "seam_position": "aligned", "reduce_crossing_wall": "1",   # avoid crossing walls
                 "enable_support": "1" if supports else "0", "support_type": "tree(auto)",
                 # support_threshold_angle: the preset's (20 deg at 0.12 mm, 30 at 0.16 / 0.20): steeper overhangs
                 # print fine in PLA at these layer heights
                 "support_on_build_plate_only": "0" if supports == "all" else "1",
                 # the Kobra X flushes at its purge station off the bed (machine change_filament_gcode,
                 # purge_in_prime_tower = 0): a one-change print needs no prime tower
                 "enable_prime_tower": "0"})
    return proc


def slice_plate(name, stls, material, colour, layer, supports, filament_ids=None, assemble=False, materials=None,
                brim=False):
    d = WORK / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    pj = d / "process.json"
    pj.write_text(json.dumps(process_json(layer, supports, two_materials=bool(materials), brim=brim), indent=1))
    fils = []
    for k, (mat, col) in enumerate(materials or [(material, colour)]):
        f = preset("filament", FIL[mat])
        f.update(filament_colour=[COLOURS[col]], default_filament_colour=[COLOURS[col]])
        if mat == "PLA":   # cooler than the preset's 205 / 215 (TPU 95A keeps its preset)
            f.update(nozzle_temperature=[str(PLA_TEMP[0])], nozzle_temperature_initial_layer=[str(PLA_TEMP[1])])
        fp = d / f"filament_{k + 1}.json"
        fp.write_text(json.dumps(f, indent=1))
        fils.append(str(fp))
    cmd = [APP, "--debug", "1", "--slice", "0", "--load-settings", f"{PROFILES / 'machine' / (MACHINE + '.json')};{pj}",
           "--load-filaments", ";".join(fils), "--load-filament-ids", ",".join(filament_ids or ["1"] * len(stls)),
           "--export-3mf", name + ".3mf", "--outputdir", str(d)]
    cmd += ["--assemble"] if assemble else ["--arrange", "1"]
    cmd += [str(s) for s in stls]
    r = subprocess.run(cmd, capture_output=True, text=True)
    gcodes = sorted(f for f in os.listdir(d) if re.match(r"plate_\d+\.gcode$", f))
    if r.returncode != 0 or not gcodes:
        return {"ok": False, "exit": r.returncode, "log": (r.stdout + r.stderr)[-1500:]}
    plates = [gcode_stats(d / g) for g in gcodes]   # the slicer's arrange adds plates when one is full
    st = {"plates": plates, "ok": all(p["fits_bed"] for p in plates),
          "filament_changes": sum(p["filament_changes"] for p in plates)}
    src = d / f"{name}.3mf"
    if src.exists() and st["ok"]:
        shutil.copy(src, OUT / f"{name}.3mf")
    return st


# ============================================================================ main
def run_job(job):
    st = slice_plate(job["name"], [it[0] for it in job["items"]], **job["kw"])
    st["parts"] = sorted({it[3] for it in job["items"]})
    log(f"{job['name']}: ok {st.get('ok')}, " + "; ".join(f"plate {k}: {p['time']}, {p['grams']} g"
                                                         for k, p in enumerate(st.get("plates", []), 1)))
    return {job["name"]: st}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="slice only the plates whose name starts with this")
    ap.add_argument("--no-slice", action="store_true")
    ap.add_argument("--jobs", type=int, default=4, help="plates sliced at once (each slicer takes ~1-2 GB)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    sfile = OUT / "summary.json"
    # a partial run (--only) keeps the others; a full run starts over (groups may have been renamed)
    summary = json.loads(sfile.read_text()) if sfile.exists() and args.only else {}
    jobs = []

    # ---------------------------------------------------------------- tests
    test = HERE / "stl" / "test"
    if "00_test_fit".startswith(args.only) or args.only.startswith("00_test_fit"):
        f = WORK / "_oriented_tests"
        f.mkdir(exist_ok=True)
        items = []
        for s in sorted(test.glob("Fit_*.stl")):
            p = dict(file=str(s.relative_to(REPO)), rule="auto", src="standin",
                     up=(0, 0, 1))   # every test piece is designed flat side down
            r = prepare(p, f)
            items.append((r[0], 0, 0, s.stem))
            if r[5]:
                log(f"  WARNING {s.stem} would want a brim ({r[6]}); the test plate has none")
        jobs.append(dict(name="00_test_fit_PLA", items=items,
                         kw=dict(material="PLA", colour="white", layer=0.12, supports=None)))
    if "00_test_balls".startswith(args.only) or args.only.startswith("00_test_balls"):
        f = WORK / "_oriented_tests"
        f.mkdir(exist_ok=True)
        items = [(prepare(dict(file=str(s.relative_to(REPO)), rule="auto", src="standin", up=(0, 0, 1)), f)[0], 0, 0,
                  s.stem) for s in sorted([*test.glob("Fit_Ball_Hand_*.stl"), *test.glob("Fit_Ball_Wrist_*.stl")])]
        jobs.append(dict(name="00_test_balls_PLA", items=items,
                         kw=dict(material="PLA", colour="white", layer=0.12, supports=None)))
    if "00_test_switch".startswith(args.only) or args.only.startswith("00_test_switch"):
        jobs.append(dict(name="00_test_switch_PLA_TPU",
                         items=[(test / "Switch_PLA_Base.stl", 0, 0, "Switch_PLA_Base"),
                                (test / "Switch_TPU_Pad.stl", 0, 0, "Switch_TPU_Pad")],
                         kw=dict(material=None, colour=None, layer=0.16, supports=None, filament_ids=["1", "2"],
                                 assemble=True, materials=[("PLA", "white"), ("TPU", "black")])))

    # ---------------------------------------------------------------- the mock-up
    parts = collect()
    taken = set()
    for gname, mat, col, layer, sup, sel in GROUPS:
        if args.only and not (gname.startswith(args.only) or args.only.startswith(gname[:2])):
            continue
        mine = [p for p in parts if id(p) not in taken and sel(p)]
        for p in mine:
            taken.add(id(p))
        if not mine:
            continue
        folder = WORK / f"_oriented_{gname}"
        folder.mkdir(exist_ok=True)
        items = []
        for p in mine:
            path, area, ext, over, contact, brim, why = prepare(p, folder)
            for _ in range(p["qty"]):
                items.append((path, area, ext, Path(p["file"]).stem, over, contact, brim))
            log(f"  {gname:24s} {Path(p['file']).stem:28s} x{p['qty']:<2d} {ext[0]:6.1f} x {ext[1]:6.1f} x {ext[2]:6.1f} mm"
                f"  overhang {over:6.0f} mm2  bed contact {contact:6.0f} mm2  {'BRIM: ' if brim else 'no brim: '}{why}")
        for brim in (False, True):   # a brim is a plate setting: the parts that need one get their own project
            mine_b = [it for it in items if bool(it[6]) == brim]
            if mine_b:
                jobs.append(dict(name=gname + ("_brim" if brim else ""), items=mine_b,
                                 kw=dict(material=mat, colour=col, layer=layer, supports=sup, brim=brim)))
    if args.no_slice:
        log("\n".join(f"{j['name']}: {len(j['items'])} parts" for j in jobs))
        return
    for f in OUT.glob("*.3mf"):   # stale plates of the groups being re-sliced (all of them on a full run)
        if not args.only or any(f.stem.startswith(j["name"].replace("_brim", "")) for j in jobs):
            f.unlink()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for out in ex.map(run_job, jobs):
            summary.update(out)
    sfile.write_text(json.dumps(summary, indent=1))
    log(json.dumps({k: {"ok": v.get("ok"), "plates": [(p["time"], p["grams"]) for p in v.get("plates", [])]}
                    for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
