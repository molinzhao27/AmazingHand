#!/usr/bin/env python3
"""Palm plate of the 5-digit hand (R-Hand_Plate_5F, L-Hand_Plate_5F) for a finger-module layout.
Needs CadQuery (pip install cadquery), mujoco, shapely, trimesh and manifold3d (the last two for the checks).

All CAD parts share one part-studio frame, in mm: x points from the index towards the pinky,
y up along the fingers (the index pivot is the origin) and z towards the back of the hand.

The stock R-Hand_Plate carries three finger modules (index, middle, ring) fanned about the palm normal,
the thumb and the foot that sits on the wrist interface. Where the finger modules go is set by
cad/five_finger/layout.py (LAYOUT, or --layout); the pinky module is the ring module moved by the
finger2 -> finger3 step D (-7.5 deg about z plus a translation, read from the stock MJCF, so it matches
the MuJoCo model).

--layout stock: the fanned stock layout plus a pinky; the original construction, unchanged. The plate
is the union of
  P_in  the stock plate, cut at D(L), where L is a seam between the middle and ring servo pairs;
  N     a strip copied from the middle unit just inside L, moved by D (fills the rounded top-right
        corner of the stock plate so the top edge runs on to the pinky unit);
  D(C)  C = the ring unit (outside L, above a line B just below its lower finger frame), moved by D;
  F     a web (back sheet plus an outer rim) from the pinky unit down to the plate's foot.

--layout parallel (default; any layout that moves the modules): every finger module is moved rigidly
by its layout transform (layout.apply: a turn about its stock knuckle pivot, then a shift to the new
pivot), exactly as the sim model moves the module's servos, finger frames, bushings and linkages. The
plate is rebuilt from
  units  A finger unit is the strip of the stock plate that carries one module, in the module's frame
         (u across the finger, v along it, origin on the knuckle pivot): |u| <= UNIT_HALF_W (15.05),
         UNIT_V (-52.3 .. 12). It holds everything the module is mounted on: the servo window; the servo
         pocket, U-shaped walls 2.75 mm thick that clamp both servos sideways (deep ribs between fingers,
         the shallow outer rim on the outside) and close 1.65 mm below the lower finger frame; the three
         finger-frame screw holes; the pivot hole and bushing recess (the notch in the top edge); and
         the top edge itself. In this frame the stock index, middle and ring units have the same window,
         holes, recess and pocket, but different outer walls (index: rim and rounded top corner on the
         index side, no deep wall there; ring: the same on the pinky side), so each is cut from its own
         place in the stock plate (unit_report() lists the differences) and the pinky unit is the ring
         unit moved by D. The unit width 2 x 15.05 mm is the module pitch: the stock ribs between pockets
         are >= 3.3 mm thick, so a unit takes only its own 2.75 mm of wall and no feature of a
         neighbour, and at the 30 mm pitch neighbouring units meet (0.10-0.15 mm overlap) and rebuild
         5.5 mm ribs. Where the legs run up into the index and ring U bottoms (LEG_TOPS), the material
         stays in place with the foot.
  Base   the stock plate below the moved units (below the line through their strip bottoms), in place:
         both legs, i.e. the foot on the wrist interface with its screw holes and the thumb side, the
         opening between the legs, and what is left of the stock ribs that tie the legs to the units.
         It reaches BASE_OVERLAP (0.6 mm) into each moved strip, never into a pocket.
  web    a back sheet (SHEET_Z) filling the outline between the moved units and Base, and the units'
         outline below v = WEB_V (-47), where their sheet has no hole or window; an outer rim rib
         (RIM_Z, RIM_W) along the new outline (index side from its rim down to the stock edge, and the
         new bottom edge from the right leg out to the pinky unit's rim), which also fills the outer
         units' bottom corners below their pockets at the rim level; back-sheet fillers along the joints
         between neighbouring units (the ring's rounded top corner and back edge beside the lower pinky
         unit, JOINT_W); and a skirt under every unit (its profile 1.5 mm above the strip bottom, run
         1.2 mm below it, SKIRT) so that it meets the stock below without slits.
  The two back-shell screw holes on the stock rib tops (index/middle, middle/ring) belong to no unit:
  they are filled before the units are cut and re-drilled, with the stock hole shape, on the ribs
  between neighbouring units (all three), at the highest point where the r1 pilot keeps SCREW_WALL of
  material and the r1.4 counterbore 1 mm. The positions are printed: make_5f_shells.py SCREWS must
  follow them.

Checks (--no-checks skips them):
  stock layout: every small hole/pin of the stock plate is unchanged, and the ring's are copied exactly
  onto the pinky unit;
  moved layouts: every stock hole, pin, window corner and pocket corner of each unit is found at its
  transformed place (same radius, axis and depth; probed at 24 points); each moved unit is the exact
  image of its stock strip (4000 random probes in the strip, all of its material present); both legs
  (foot, thumb side, the screw holes into the wrist interface) are unchanged; there are no other
  hole-like features; no slit where a unit meets the stock below it; and the minimum material between
  neighbouring units' windows, holes and pockets is printed;
  both: the plate does not intersect the palm-fixed module parts (servos, finger frames, bushings; poses
  from the stock MJCF moved like the sim moves them), the thumb's parts or the wrist interface more than
  the stock plate does; the part poses are cross-checked against Demo/.../AH_Right_5F if it is there.
  --sweep also settles every digit of that sim model on a 13 x 13 grid of its motor commands and checks
  that its moving parts hit the plate no more than in the stock hand (only the extreme hyperextended
  pose touches, the finger's own unit, as in the stock hand).

Writes cad/five_finger/step/R-Hand_Plate_5F.step, cad/five_finger/stl/R-Hand_Plate_5F.stl (mm, same
frame as the stock parts) and cad/five_finger/R-Hand_Plate_5F_preview.png (stock and new plate seen from
the back, the unit strips, the module footprints and the stock outline), plus the left plate
L-Hand_Plate_5F: the stock L-Hand_Plate is exactly the R-Hand_Plate mirrored about x = -50 (checked
here), so the left 5-finger plate is the right one mirrored the same way.

Usage: python make_5f_plate.py [--layout stock|parallel] [--no-checks] [--sweep]
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cadquery as cq
import mujoco
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import layout as LAYOUT  # noqa: E402

STOCK_STEP = REPO / "cad" / "step" / "Amazing Hand Parts - R-Hand_Plate.step"
STOCK_STEP_L = REPO / "cad" / "step" / "Amazing Hand Parts - L-Hand_Plate.step"
STOCK_MJCF = REPO / "Demo" / "AHSimulation" / "AHSimulation" / "AH_Right" / "mjcf"
OUT_STEP = HERE / "step" / "R-Hand_Plate_5F.step"
OUT_STL = HERE / "stl" / "R-Hand_Plate_5F.stl"
OUT_STEP_L = HERE / "step" / "L-Hand_Plate_5F.step"
OUT_STL_L = HERE / "stl" / "L-Hand_Plate_5F.stl"
OUT_PNG = HERE / "R-Hand_Plate_5F_preview.png"
MIRROR_X = LAYOUT.MIRROR_X  # L-Hand_Plate = R-Hand_Plate mirrored about the plane x = MIRROR_X

FINGERS = LAYOUT.FINGERS
SHEET_Z = (12.5, 15.5)  # back sheet of the stock plate
RIM_Z = (6.5, 15.5)  # outer rim rib of the stock plate
RIM_W = 2.0

# ---- stock (fanned) layout: the original construction
# Seam between the middle and ring units (plate frame, mm): passes between the two servo pairs,
# between the middle unit's right screw hole / pivot boss and the ring unit's, and between the
# lower finger frames (finger_frame_2). Every middle-unit feature is on one side, every ring-unit
# feature on the other.
L_BOT, L_TOP = np.array([42.8, -52.0]), np.array([47.5, -17.0])
# Bottom of the copied ring unit: parallel to the ring's lower finger frame, ~2 mm below it.
B_ANGLE = -13.5  # deg, ring finger fan angle in this frame
B_POINT = np.array([45.0, -62.0])
STRIP_W = 10.0  # mm, width of the middle-unit strip N copied next to the seam
FOOT_Y = -85.0  # plate bottom, sits on the wrist interface

# ---- moved layouts: finger units
UNIT_HALF_W = 15.05  # |u| of a unit strip (module frame): 2.75 mm of wall outside the 12.25 mm pocket face
UNIT_V = (-52.3, 12.0)  # v range: just below the U bottom (its outer faces at v = -52) .. above the top edge
FINGER_FRAME_1 = {"index": "finger_frame_1", "middle": "finger_frame_1_2", "ring": "finger_frame_1_3"}
# stock back-shell screw holes on the rib tops (r1.0 pilot from z 7.5, r1.4 x 0.8 counterbore)
SHELL_HOLES_STOCK = [(16.13, -11.65), (47.54, -20.08)]
SCREW_WALL = 1.5  # material kept around a re-drilled shell-screw pilot hole
# Tops of the legs (foot) where they run into a unit's U bottom, (unit, u range, v range, z range) in its
# module frame: they stay in place with the foot. Left leg: below the index pocket's open (rim-less) side,
# inside the leg's r8 fillet up to the pocket floor; right leg: its sloped top under the ring's U bottom.
LEG_TOPS = [("index", (-16.0, -1.0), (-60.0, -48.9), (-60.0, 6.5)), ("ring", (-2.0, 6.5), (-60.0, -47.0), (-60.0, -2.5))]
WEB_V = -47.0  # below this (module frame) a unit's sheet has no hole or window: the web may fill its outline
RIM_V = -42.0  # the outer rim rib runs down the outer units' sides from here (where their rim starts to curve in)
JOINT_W = 2.0  # half-width of the back-sheet filler along the joint between neighbouring units
OUTER_U = 14.98  # the web's outline along the outer units: 0.02 mm inside their rims' outer faces (|u| = 15)
BASE_OVERLAP = 0.6  # the stock kept in place (Base) reaches this far into the bottom of each moved unit strip
SKIRT = (1.5, 1.2)  # each unit's profile at 1.5 mm above its strip bottom is run down to 1.2 mm below it
FUZZY = 1e-5  # fuzzy tolerance of the final union (the stock faces sit on the web's planes to ~1e-7 mm)
LEFT_EDGE = (np.array([-15.0, -48.7992]), np.array([-9.0, -85.0]))  # stock outline below the index unit
WEB_LEFT_Y = -62.0  # the new index-side outline meets the stock edge here
RIGHT_LEG_CORNER = np.array([58.0, -85.0])  # bottom pinky-side corner of the right leg (the bottom rim starts
                                            # 0.8 mm into the leg, clear of its screw hole at x <= 56.4)
FOOT_TOP = -56.5  # the index corner fill stays above this (CAD y): the left leg below it is left exactly as stock


# ----------------------------------------------------------------------------- stock MJCF
def quat_mul(a, b):
    r = np.zeros(4)
    mujoco.mju_mulQuat(r, np.asarray(a, float), np.asarray(b, float))
    return r


def quat_conj(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q, v):
    r = np.zeros(3)
    mujoco.mju_rotVecQuat(r, np.asarray(v, float), np.asarray(q, float))
    return r


def quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float) / np.linalg.norm(q))
    return m.reshape(3, 3)


def get_pose(el):
    pos = np.array([float(x) for x in el.get("pos", "0 0 0").split()])
    quat = np.array([float(x) for x in el.get("quat", "1 0 0 0").split()])
    return pos, quat / np.linalg.norm(quat)


def part_geoms(body):
    """Map 'Part <name>' comments to the geom that follows them (direct children only)."""
    out, children = {}, list(body)
    for c, nxt in zip(children, children[1:]):
        if c.tag is ET.Comment and c.text.strip().startswith("Part ") and nxt.tag == "geom":
            out[c.text.strip()[5:]] = nxt
    return out


def stock_root(mjcf_dir=STOCK_MJCF):
    tree = ET.parse(mjcf_dir / "robot.xml", parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    return tree.getroot().find("worldbody/body")


def step_transform():
    """finger2 -> finger3 step in the plate frame: (rotation about z in deg, translation in mm)."""
    geoms = part_geoms(stock_root())
    plate_p, plate_q = get_pose(geoms["r_hand_plate"])
    p2, q2 = get_pose(geoms["finger_frame_1_2"])
    p3, q3 = get_pose(geoms["finger_frame_1_3"])
    qD = quat_mul(q3, quat_conj(q2))
    tD = p3 - rotate(qD, p2)
    # conjugate into the plate frame (and metres -> mm)
    qL = quat_mul(quat_conj(plate_q), quat_mul(qD, plate_q))
    tL = rotate(quat_conj(plate_q), rotate(qD, plate_p) + tD - plate_p) * 1000
    assert np.allclose(qL[1:3], 0, atol=1e-6) and abs(tL[2]) < 1e-6, "step is not a rotation about the plate normal"
    angle = 2 * np.degrees(np.arctan2(qL[3], qL[0]))
    return angle, tL[:2]


ANGLE, T = step_transform()


def module_frames():
    """{finger: (knuckle pivot (x, y), module angle about z in deg)} in the stock layout, from the finger
    frames (finger_frame_1*, whose origin is the knuckle pivot) of the stock MJCF; pinky = D(ring)."""
    geoms = part_geoms(stock_root())
    plate_p, plate_q = get_pose(geoms["r_hand_plate"])
    Rp = quat_mat(plate_q)
    out = {}
    for f, name in FINGER_FRAME_1.items():
        p, q = get_pose(geoms[name])
        Rc = Rp.T @ quat_mat(q)  # part orientation in the plate (CAD) frame
        assert abs(Rc[2, 2] - 1) < 1e-9, f"{name} is not turned about the plate normal only"
        origin = Rp.T @ (p - plate_p) * 1000
        assert np.allclose(origin[:2], LAYOUT.STOCK[f][0][:2], atol=1e-4), f"{f}: knuckle pivot differs from layout.py"
        ang = np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0]))
        # the MJCF quaternions carry 6 digits (-13.4999 deg); the plate itself is drawn at 0 / -6 / -13.5 deg
        assert abs(ang - round(ang * 2) / 2) < 1e-3, f"{f}: module angle {ang}"
        out[f] = (LAYOUT.STOCK[f][0][:2].copy(), round(ang * 2) / 2)
    assert np.allclose(d_pt(out["ring"][0]), LAYOUT.STOCK["pinky"][0][:2], atol=1e-4), "pinky pivot != D(ring pivot)"
    out["pinky"] = (LAYOUT.STOCK["pinky"][0][:2].copy(), out["ring"][1] + ANGLE)
    return out


# ----------------------------------------------------------------------------- geometry helpers
def rz2(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def D(shape):
    return shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), ANGLE).translate(cq.Vector(T[0], T[1], 0))


def d_pt(p, inverse=False):
    R = rz2(ANGLE)
    p = np.asarray(p, float)
    return (p - T) @ R if inverse else p @ R.T + T


def move(shape, tf_f):
    """Move a shape of one finger's module by its layout transform (layout.apply)."""
    phi, p0, p1 = tf_f
    if abs(phi) < 1e-12 and np.allclose(p0, p1):
        return shape
    c = cq.Vector(float(p0[0]), float(p0[1]), 0.0)
    return shape.rotate(c, c + cq.Vector(0, 0, 1), float(phi)).translate(cq.Vector(*map(float, p1 - p0)))


def move_pts(pts, tf_f):
    pts = np.asarray(pts, float)
    return LAYOUT.apply(tf_f, np.column_stack([pts, np.zeros(len(pts))]))[:, :2]


def prism(pts, z=(-60.0, 60.0)):
    wp = cq.Workplane("XY").workplane(offset=z[0])
    return wp.polyline([tuple(map(float, p)) for p in pts]).close().extrude(z[1] - z[0]).val()


def extrude(geom, z):
    """Shapely (Multi)Polygon -> list of CadQuery solids between z[0] and z[1]."""
    out = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.area < 1e-4:
            continue
        poly = shapely.geometry.polygon.orient(poly.simplify(1e-4), 1.0)
        s = prism(np.array(poly.exterior.coords)[:-1], z)
        holes = [prism(np.array(h.coords)[:-1], (z[0] - 1, z[1] + 1)) for h in poly.interiors]
        out.append(s.cut(*holes) if holes else s)
    return out


def line(a, b, ext=200.0):
    u = (b - a) / np.linalg.norm(b - a)
    return a - u * ext, b + u * ext


def side_region(a, b, left=True, far=400.0):
    """Half-plane prism on one side of the line a->b (left = counter-clockwise side)."""
    a, b = line(a, b)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]]) * (far if left else -far)
    return prism([a, b, b + n, a + n])


def intersect_lines(p, u, q, v):
    s = np.linalg.solve(np.column_stack([u, -v]), q - p)
    return p + s[0] * u


def section2d(shape, z, tol=0.005):
    """Cross-section of a solid at height z as a shapely (Multi)Polygon (wires discretised to tol)."""
    from OCP.BRepAdaptor import BRepAdaptor_CompCurve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection

    def ring(wire):
        d = GCPnts_QuasiUniformDeflection(BRepAdaptor_CompCurve(wire.wrapped), tol)
        return [(d.Value(i).X(), d.Value(i).Y()) for i in range(1, d.NbPoints() + 1)]

    polys = [Polygon(ring(f.outerWire()), [ring(w) for w in f.innerWires()])
             for f in cq.Workplane().add(shape).section(z).faces().vals()]
    return shapely.union_all([p.buffer(0) for p in polys]).buffer(0)


def classifier(shape):
    """Fast inside test for many points: returns f(points (n, 3)) -> bool array."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN

    c = BRepClass3d_SolidClassifier(shape.wrapped)

    def inside(pts):
        out = []
        for q in np.atleast_2d(pts):
            c.Perform(gp_Pnt(*map(float, q)), 1e-6)
            out.append(c.State() == TopAbs_IN)
        return np.array(out)
    return inside


def interior_points(shape, n=300, depth=0.1, seed=0):
    """Up to n random points at least depth mm inside a solid (from its tessellation)."""
    import trimesh

    v, t = shape.tessellate(0.01, 0.1)
    m = trimesh.Trimesh(np.array([(p.x, p.y, p.z) for p in v]), np.array(t))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(100):  # small batches keep trimesh's ray tests light
        pts = rng.uniform(m.bounds[0], m.bounds[1], size=(400, 3))
        pts = pts[m.contains(pts)]
        if len(pts):
            out += list(pts[trimesh.proximity.signed_distance(m, pts) > depth])
        if len(out) >= n:
            break
    return np.array(out[:n])


def step_roundtrip(shape):
    """The shape as written to and read back from STEP (the file is the deliverable; reading it back also
    settles the edge tolerances left by the booleans, so that its tessellation is a closed manifold)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "plate.step")
        cq.exporters.export(cq.Workplane().add(shape), path)
        return cq.importers.importStep(path).val()


def single_solid(shape, what):
    solids = shape.Solids()
    assert len(solids) == 1, f"{what}: {len(solids)} solids ({[round(s.Volume(), 3) for s in solids]})"
    assert solids[0].isValid(), f"{what}: invalid solid"
    return solids[0]


# ----------------------------------------------------------------------------- stock (fanned) layout
def build_stock(stock):
    """The original 5-finger plate: stock plate + pinky unit D(C) + strip N + web F (see the docstring)."""
    print(f"step D in plate frame: {ANGLE:.4f} deg about z, t = ({T[0]:.4f}, {T[1]:.4f}) mm")

    bu = np.array([np.cos(np.radians(B_ANGLE)), np.sin(np.radians(B_ANGLE))])
    B0, B1 = B_POINT, B_POINT + bu
    # left of L (pointing up) = middle side; right = ring side. Above B = left of B (pointing +x).
    ring_side_L = side_region(L_BOT, L_TOP, left=False)
    above_B = side_region(B0, B1, left=True)
    DL_bot, DL_top = d_pt(L_BOT), d_pt(L_TOP)

    P_in = stock.intersect(side_region(DL_bot, DL_top, left=True))
    C = stock.intersect(ring_side_L).intersect(above_B)
    DC = D(C)

    lu = (L_TOP - L_BOT) / np.linalg.norm(L_TOP - L_BOT)
    ln = np.array([-lu[1], lu[0]])  # towards the middle unit
    strip = prism([*line(L_BOT, L_TOP), *reversed(line(L_BOT + ln * STRIP_W, L_TOP + ln * STRIP_W))])
    band = side_region(d_pt(B0, inverse=True), d_pt(B1, inverse=True), left=True)  # D^-1(B): same band, middle unit
    N = D(stock.intersect(strip).intersect(band))

    # web F below the pinky unit: back sheet + outer rim, from the seam down to the foot
    DB0, DB1 = d_pt(B0), d_pt(B1)
    dbu = (DB1 - DB0) / np.linalg.norm(DB1 - DB0)
    dlu = (DL_top - DL_bot) / np.linalg.norm(DL_top - DL_bot)
    corner_in = intersect_lines(DL_bot, dlu, DB0, dbu)  # bottom of the seam
    # outer end of the copied unit's bottom edge: where B leaves the stock plate, moved by D
    xs = [x for x in np.arange(40, 90, 0.25)
          if stock.isInside(cq.Vector(*(B0 + bu * (x - B0[0]) / bu[0] + np.array([0, 0.3])), 14.0))]
    corner_out = d_pt(B0 + bu * (max(xs) - B0[0]) / bu[0])
    foot_out = np.array([corner_out[0] - 18.0, FOOT_Y])
    foot_in = np.array([58.0, FOOT_Y])
    # The web meets D(C) along the seam D(L) and its bottom D(B) (overlapping it by 0.05 mm so the
    # solids fuse) and overlaps the stock plate above the foot; it never reaches into the pinky unit.
    eps = 0.05
    seam_top = DL_bot + dlu * ((-61.0 - DL_bot[1]) / dlu[1])  # on D(L) where the stock edge meets it
    into_dc = np.array([dlu[1], -dlu[0]]) * eps  # normal to D(L), towards the pinky unit
    up = np.array([-dbu[1], dbu[0]]) * eps  # normal to D(B), into the pinky unit
    web = [seam_top + into_dc, corner_in + into_dc + up, corner_out + up, foot_out, foot_in,
           np.array([foot_in[0], -60.0])]
    F = prism(web, SHEET_Z)
    rim_dir = (foot_out - corner_out) / np.linalg.norm(foot_out - corner_out)
    rim_n = np.array([rim_dir[1], -rim_dir[0]])  # inwards (rim_dir runs right to left along the bottom edge)
    rim = prism([corner_out + up, foot_out, foot_out + rim_n * RIM_W, corner_out + up + rim_n * RIM_W], RIM_Z)
    print(f"seam D(L): {np.round(DL_bot, 2)} -> {np.round(DL_top, 2)}; pinky unit bottom {np.round(corner_in, 2)} "
          f"-> {np.round(corner_out, 2)}; web foot {np.round(foot_in, 1)} -> {np.round(foot_out, 1)}")

    plate = single_solid(P_in.fuse(N).fuse(DC).fuse(F).fuse(rim).clean(), "stock-layout plate")
    return plate, dict(shell_holes=[np.array(p) for p in SHELL_HOLES_STOCK] + [d_pt(SHELL_HOLES_STOCK[1])])


def check_features_stock(stock, plate):
    """Stock layout: every small hole/pin of the stock plate must be unchanged, and every ring-side one
    must have an exact copy on the pinky unit (probed at 8 x 3 points inside each feature)."""
    lu = (L_TOP - L_BOT) / np.linalg.norm(L_TOP - L_BOT)
    n_ring = np.array([lu[1], -lu[0]])
    lo_B = np.array([np.cos(np.radians(B_ANGLE)), np.sin(np.radians(B_ANGLE))])
    n_up = np.array([-lo_B[1], lo_B[0]])
    n_feat, n_copied = 0, 0
    for ft in features(stock):
        if ft["r"] > 1.6:
            continue
        ref = probe(stock, ft)
        assert ref == probe(plate, ft), f"feature at {ft['xy']} changed"
        n_feat += 1
        if np.dot(ft["xy"] - L_BOT, n_ring) > ft["r"] and np.dot(ft["xy"] - B_POINT, n_up) > ft["r"]:  # inside C
            assert ref == probe(plate, ft, lambda p: d_pt(p)), f"copy of {ft['xy']} differs"
            n_copied += 1
    print(f"feature check: {n_feat} stock holes/pins unchanged, {n_copied} copied exactly onto the pinky unit")


# ----------------------------------------------------------------------------- moved layouts
def unit_strip(frame, half_w=UNIT_HALF_W, v=UNIT_V, u=None):
    """Corners of a unit strip |u| <= half_w (or u[0] <= u <= u[1]), v[0] <= v <= v[1] in the module
    frame, in the CAD frame, counter-clockwise."""
    (c, ang) = frame
    u = (-half_w, half_w) if u is None else u
    loc = np.array([[u[0], v[0]], [u[1], v[0]], [u[1], v[1]], [u[0], v[1]]])
    return loc @ rz2(ang).T + c


def to_local(frame, pts):
    c, ang = frame
    return (np.asarray(pts, float) - c) @ rz2(ang)


def from_local(frame, pts):
    c, ang = frame
    return np.asarray(pts, float) @ rz2(ang).T + c


def skirt(unit, frame, top, depth):
    """The unit's cross-section at v = UNIT_V[0] + top (module frame), extruded down to UNIT_V[0] - depth."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    vdir, udir = rz2(frame[1]) @ np.array([0.0, 1.0]), rz2(frame[1]) @ np.array([1.0, 0.0])
    o = from_local(frame, [(0.0, UNIT_V[0] + top)])[0]
    plane = cq.Plane(origin=(o[0], o[1], 0.0), xDir=(udir[0], udir[1], 0.0), normal=(-vdir[0], -vdir[1], 0.0))
    out = []
    for f in cq.Workplane(plane).add(unit).section().faces().vals():
        d = -vdir * (top + depth)
        out.append(cq.Shape.cast(BRepPrimAPI_MakePrism(f.wrapped, gp_Vec(d[0], d[1], 0.0)).Shape()))
    return out


def shell_hole_tool(stock, xy):
    """The void of a stock back-shell screw hole at xy (exact stock shape), as a solid."""
    cyl = cq.Solid.makeCylinder(1.6, 9.0, cq.Vector(xy[0], xy[1], 6.5))
    void = cyl.cut(stock)
    vs = [s for s in void.Solids() if s.Volume() > 1.0]
    assert len(vs) == 1, f"shell hole at {xy}: {len(vs)} void pieces"
    return vs[0]


def build_moved(stock, tf):
    """Units moved by the layout transforms tf, the stock rest below them in place, and the web."""
    frames = module_frames()
    print("module frames (stock): " + ", ".join(f"{f} pivot {np.round(c, 4).tolist()} {a:+.4f} deg" for f, (c, a) in frames.items()))

    # back-shell screw holes on the stock rib tops: belong to no unit, filled before cutting the units
    tools = [shell_hole_tool(stock, xy) for xy in SHELL_HOLES_STOCK]
    hole_tool = tools[0].translate(cq.Vector(-SHELL_HOLES_STOCK[0][0], -SHELL_HOLES_STOCK[0][1], 0))  # at the origin
    vols = [t.Volume() for t in tools]
    assert np.allclose(vols, vols[0], rtol=1e-3) and 25 < vols[0] < 30, f"shell hole voids {vols}"
    stock_p = stock.fuse(*tools).clean()
    assert abs(stock_p.Volume() - stock.Volume() - sum(vols)) < 0.01

    strips = {f: unit_strip(frames[f]) for f in ("index", "middle", "ring")}
    # where the legs run up into the index and ring units' U bottoms: part of the foot, not of the units
    legs = [prism(unit_strip(frames[f], v=v, u=u), z) for f, u, v, z in LEG_TOPS]
    units = {f: stock_p.intersect(prism(strips[f])).cut(*legs) for f in strips}
    units["pinky"] = D(units["ring"])
    strips["pinky"] = d_pt(strips["ring"])
    moved = {f: move(units[f], tf[f]) for f in FINGERS}
    mframes = {f: (tf[f][2][:2], frames[f][1] + tf[f][0]) for f in FINGERS}  # module frames, new layout
    mstrip = {f: unit_strip(mframes[f]) for f in FINGERS}
    b = UNIT_HALF_W
    for f in FINGERS:
        assert np.allclose(mstrip[f], move_pts(strips[f], tf[f]), atol=1e-6)
        print(f"unit {f:6s}: {units[f].Volume():8.1f} mm3, turned {tf[f][0]:+.3f} deg, strip bottom "
              f"{np.round(mstrip[f][0], 2).tolist()} -> {np.round(mstrip[f][1], 2).tolist()}")
    # neighbouring moved strips may overlap a little (only wall material: walls end 2.8 mm from the edge)
    for f, g in zip(FINGERS, FINGERS[1:]):
        ov = Polygon(mstrip[f]).intersection(Polygon(mstrip[g]))
        width = ov.area / max(ov.length / 2, 1e-9)
        print(f"strips {f}/{g} overlap {width:.3f} mm")
        assert width < 0.5

    # Base: the stock plate below the moved units (below the line through their bottom corners) and
    # outside them, in place. It reaches up to each moved unit's bottom, so the legs stay tied to the
    # units at full depth; near a unit's bottom it may keep up to ~1 mm of the stock U bottom (a wedge
    # under the moved one), never a pocket, window or hole (the pockets end 3 mm above the strip bottom).
    # Base reaches BASE_OVERLAP into each unit's strip, which below its U bottom (v < -52) may be empty
    # where the unit came from: so no slit can open between a leg and the unit above it.
    above = {f: unit_strip(mframes[f], b, (UNIT_V[0] + BASE_OVERLAP, UNIT_V[1])) for f in FINGERS}
    env = [above[FINGERS[0]][0] - np.array([200.0, 0.0])]
    for f in FINGERS:
        env += [above[f][0], above[f][1]]
    env += [above[FINGERS[-1]][1] + np.array([200.0, 0.0])]
    env += [np.array([env[-1][0], -400.0]), np.array([env[0][0], -400.0])]
    assert Polygon(env).is_valid
    # the leg tops stay whole, except where they would reach into a moved unit's pocket (v > -49.5)
    guard = [prism(unit_strip(mframes[f], b, (-49.5, UNIT_V[1]))) for f in FINGERS]
    base = stock_p.intersect(prism(env).cut(*[prism(above[f]) for f in FINGERS]).fuse(*[g.cut(*guard) for g in legs]))

    # Web outline (2D): from the index unit's bottom corner down to the stock edge, through Base,
    # along the new bottom edge to the pinky unit's bottom corner, and back through the units at
    # v = WEB_V: below that a unit carries no hole or window in its sheet, so the web may fill the
    # unit's outline there (its bottom corners), never above.
    A = from_local(mframes[FINGERS[0]], [(-OUTER_U, UNIT_V[0])])[0]  # outer bottom corners (on the outer rim faces)
    BR = from_local(mframes[FINGERS[-1]], [(OUTER_U, UNIT_V[0])])[0]
    e0, e1 = LEFT_EDGE
    X = e0 + (e1 - e0) * (WEB_LEFT_Y - e0[1]) / (e1[1] - e0[1])
    inner = []
    for f in reversed(FINGERS):
        inner += list(from_local(mframes[f], [(OUTER_U if f == FINGERS[-1] else b, WEB_V),
                                              (-OUTER_U if f == FINGERS[0] else -b, WEB_V)]))
    web_poly = Polygon([A, X, (-6.0, -70.0), (55.0, -72.0), RIGHT_LEG_CORNER, BR, *inner])
    assert web_poly.is_valid, "web outline self-intersects"
    # keep out of the opening between the legs (as stock; Base carries its edge) and of Base's holes
    s14 = section2d(stock, 14.0)
    notch = [g for g in getattr(box(-3.5, -86.0, 52.5, -55.0).difference(s14), "geoms", [])
             if g.contains(Point(25.0, -75.0))]
    assert len(notch) == 1
    keep_out = [notch[0].buffer(0.3)]
    for ft in features(stock):  # holes in the sheet of the stock outside the units (the legs' back-shell holes)
        if ft["hole"] and ft["r"] <= 1.6 and ft["z"][1] > SHEET_Z[0] and not any(
                Polygon(strips[f]).buffer(-0.01).contains(Point(*ft["xy"])) for f in ("index", "middle", "ring")):
            if not any(np.hypot(*(ft["xy"] - np.array(h))) < 0.1 for h in SHELL_HOLES_STOCK):
                keep_out.append(Point(*ft["xy"]).buffer(ft["r"] + 0.4))
    keep_out = shapely.union_all(keep_out)

    def protected(v0):
        return shapely.union_all([Polygon(unit_strip(mframes[f], b - 0.05, (v0, UNIT_V[1] + 1))) for f in FINGERS])

    web2d = web_poly.difference(protected(WEB_V)).difference(keep_out)
    # outer rim rib along the new outline, from the index unit's rim (straight down to v = RIM_V) round
    # its bottom corner to the stock edge, and from the right leg along the new bottom edge round the
    # pinky unit's bottom corner up to its rim
    rims = [shapely.geometry.LineString([from_local(mframes[FINGERS[0]], [(-OUTER_U, RIM_V)])[0], A, X]),
            shapely.geometry.LineString([RIGHT_LEG_CORNER - [0.8, 0.0], RIGHT_LEG_CORNER, BR, from_local(mframes[FINGERS[-1]], [(OUTER_U, RIM_V)])[0]])]
    # ... and, at the rim level, the outer units' U bottoms outside their pockets (beyond the r7 pocket
    # corners + 0.3 mm, below the pocket floor - 0.3 mm), down to the new outline: the rim's rounded
    # corners would otherwise leave thin wedges against the leg top and the new rims. The index's
    # reaches across the left leg (to u = 3); the pinky's fills the strip down to the new bottom edge.
    v0 = UNIT_V[0] - 12.0
    pocket = shapely.union_all([Point(-5.25, -42.0).buffer(7.3), Point(5.25, -42.0).buffer(7.3), box(-5.25, -49.3, 5.25, -42.0)])
    # (u from/to the outer rim's face, OUTER_U; the pinky's starts 0.3 mm inside its deep left wall)
    corners = [box(-OUTER_U, v0, 3.0, -42.0).difference(pocket), box(-UNIT_HALF_W + 0.3, v0, OUTER_U, -42.0).difference(pocket)]
    outline = shapely.union_all([web_poly] + [Polygon(mstrip[f]) for f in FINGERS])
    corners = [Polygon(from_local(mframes[f], np.array(c.exterior.coords))).intersection(outline)
               for f, c in zip((FINGERS[0], FINGERS[-1]), corners)]
    corners[0] = corners[0].difference(box(-100, -200, 0, FOOT_TOP))  # not on the left leg
    rim2d = shapely.union_all([r.buffer(RIM_W, single_sided=True, join_style="mitre") for r in rims] + corners)
    rim2d = rim2d.difference(protected(RIM_V)).difference(keep_out)
    # Between neighbouring units the higher one's outline may round off next to the lower one's corner
    # (the ring's top corner beside the pinky) or have a rounded back edge (0.5 mm) along the joint:
    # fill the back sheet within JOINT_W of the joint, up to just below the lower of the two top edges.
    joints = []
    for a, c in zip(FINGERS, FINGERS[1:]):
        mid = (to_local(mframes[a], mframes[c][0])[0]) / 2
        tops = []
        for f, s in ((a, mid - JOINT_W), (c, mid + JOINT_W)):
            ln = shapely.geometry.LineString(from_local(mframes[a], [(s, -60.0), (s, 30.0)]))
            sec = section2d(moved[f], 13.0).intersection(ln)
            tops.append(max(to_local(mframes[a], np.array(g.coords))[:, 1].max() for g in getattr(sec, "geoms", [sec])))
        lo = max(UNIT_V[0], to_local(mframes[a], unit_strip(mframes[c], b, UNIT_V)[0])[1])  # the higher bottom
        joints.append(Polygon(from_local(mframes[a], [(mid - JOINT_W, lo), (mid + JOINT_W, lo),
                                                      (mid + JOINT_W, min(tops) - 0.6), (mid - JOINT_W, min(tops) - 0.6)])))
    # one back-sheet layer (web, rim and joints) and the rim below it, touching at z = SHEET_Z[0]
    web = extrude(shapely.union_all([web2d, rim2d, *joints]), SHEET_Z) + extrude(rim2d, (RIM_Z[0], SHEET_Z[0]))
    # Skirts: each unit's profile just above its strip bottom, run SKIRT[1] mm further down, so the unit
    # meets the stock kept in place below it (legs, ribs) without slits where their outlines differ. Kept
    # inside the outline, out of the opening between the legs, the legs' holes and every unit's pocket.
    clip = shapely.union_all([web_poly] + [Polygon(mstrip[f]) for f in FINGERS]).difference(keep_out)
    clip = prism(np.array(clip.exterior.coords)[:-1]).cut(*[prism(np.array(h.coords)[:-1]) for h in clip.interiors], *guard)
    for f in FINGERS:
        web += [p.intersect(clip) for p in skirt(moved[f], mframes[f], *SKIRT)]

    pieces = [moved[f] for f in FINGERS] + [base] + web
    plate = pieces[0].fuse(*pieces[1:], tol=FUZZY).clean()
    inside = classifier(plate)
    for i, p in enumerate(pieces):  # nothing may get lost in the boolean
        pts = interior_points(p)
        assert inside(pts).all(), f"piece {i} got lost in the union"
    plate = single_solid(plate, "plate before the shell-screw holes")

    # back-shell screw holes on the ribs between neighbouring units
    holes = shell_hole_positions(plate, mframes)
    plate = single_solid(plate.cut(*[hole_tool.translate(cq.Vector(h[0], h[1], 0)) for h in holes]).clean(), "plate")
    plate = single_solid(step_roundtrip(plate), "plate (read back from STEP)")
    info = dict(frames=frames, mframes=mframes, strips=strips, mstrip=mstrip, units=units, moved=moved, base=base,
                shell_holes=holes, hole_tool=hole_tool, web2d=web2d, rim2d=rim2d, web_poly=web_poly)
    return plate, info


def shell_hole_positions(plate, mframes):
    """On the line midway between neighbouring units, the highest point (from the lower knuckle down)
    where the r1.0 pilot keeps SCREW_WALL and the r1.4 counterbore 1 mm of material."""
    secs = {z: section2d(plate, z) for z in (7.6, 9.0, 10.5, 12.0, 12.45, 13.5, 14.65)}
    secs_cb = {z: section2d(plate, z) for z in (14.8, 15.3)}
    holes = []
    for a, c in zip(FINGERS, FINGERS[1:]):
        knuckle = to_local(mframes[a], mframes[c][0])
        mid, top = knuckle[0] / 2, min(0.0, knuckle[1])
        for dv in np.arange(0.0, 45.0, 0.25):
            p = from_local(mframes[a], [(mid, top - dv)])[0]
            if all(_clearance(g, p) >= 1.0 + SCREW_WALL for g in secs.values()) and \
                    all(_clearance(g, p) >= 1.4 + 1.0 for g in secs_cb.values()):
                break
        else:
            raise AssertionError(f"no room for a back-shell screw hole between {a} and {c}")
        wall = min(_clearance(g, p) for g in secs.values()) - 1.0
        print(f"back-shell screw hole {a}/{c}: ({p[0]:.2f}, {p[1]:.2f}), {dv:.1f} mm below the lower knuckle, "
              f"pilot wall {wall:.2f} mm")
        holes.append(p)
    return holes


def _clearance(geom, p):
    """Distance from p to the nearest boundary of geom if p is inside, else -1."""
    pt = Point(*p)
    return geom.boundary.distance(pt) if geom.contains(pt) else -1.0


# ----------------------------------------------------------------------------- features and checks
def features(shape, rmax=7.5):
    """Cylindrical faces with their axis along z: [{r, xy, z, hole, area}] (hole = concave)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_REVERSED

    out = []
    for f in shape.Faces():
        s = BRepAdaptor_Surface(f.wrapped)
        if s.GetType() != GeomAbs_Cylinder:
            continue
        c = s.Cylinder()
        d = c.Axis().Direction()
        if abs(d.Z()) < 0.999 or c.Radius() > rmax:
            continue
        xy = np.array([c.Axis().Location().X(), c.Axis().Location().Y()])
        u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
        props = BRepLProp_SLProps(s, (u0 + u1) / 2, (v0 + v1) / 2, 1, 1e-6)
        n, p = props.Normal(), props.Value()
        n = np.array([n.X(), n.Y()]) * (-1 if f.wrapped.Orientation() == TopAbs_REVERSED else 1)
        bb = f.BoundingBox()
        out.append(dict(r=c.Radius(), xy=xy, z=(bb.zmin, bb.zmax), hole=float(np.dot(n, np.array([p.X(), p.Y()]) - xy)) < 0,
                        area=f.Area(), bbox=(bb.xmin, bb.xmax, bb.ymin, bb.ymax)))
    return out


def probe(shape, ft, xf=None, region=None):
    """Inside/outside at 8 x 3 points just inside a cylindrical feature's circle (only those in region,
    a 2D polygon, if given), optionally moved by xf."""
    pts = [(ft["xy"] + 0.8 * ft["r"] * np.array([np.cos(t), np.sin(t)]), z)
           for t in np.linspace(0, 2 * np.pi, 8, endpoint=False) for z in np.linspace(ft["z"][0] + 0.1, ft["z"][1] - 0.1, 3)]
    if region is not None:
        pts = [(p, z) for p, z in pts if region.contains(Point(*p))]
    if xf is not None:
        pts = [(xf(p), z) for p, z in pts]
    return [shape.isInside(cq.Vector(float(p[0]), float(p[1]), float(z))) for p, z in pts]


def match(ft, fts, xy, tol=2e-3):
    return [g for g in fts if abs(g["r"] - ft["r"]) < 1e-4 and np.hypot(*(g["xy"] - xy)) < tol and g["hole"] == ft["hole"]
            and abs(g["z"][0] - ft["z"][0]) < tol and abs(g["z"][1] - ft["z"][1]) < tol]


def unit_report(stock, info):
    """Features of the three stock units in their module frames: which are common to all three."""
    frames, strips = info["frames"], info["strips"]
    loc = {}
    for f in ("index", "middle", "ring"):
        poly = Polygon(strips[f]).buffer(-0.01)
        loc[f] = [dict(ft, xy=to_local(frames[f], ft["xy"])) for ft in features(stock)
                  if poly.contains(Polygon([(ft["bbox"][0], ft["bbox"][2]), (ft["bbox"][1], ft["bbox"][2]),
                                            (ft["bbox"][1], ft["bbox"][3]), (ft["bbox"][0], ft["bbox"][3])]))]
    common = [ft for ft in loc["middle"] if all(match(ft, loc[g], ft["xy"], 5e-3) for g in ("index", "ring"))]
    print("stock units, module frame: " + ", ".join(f"{f} {len(v)} z-cylinder faces" for f, v in loc.items())
          + f"; {len(common)} common to all three (window corners, screw holes, pivot hole/recess, pocket corners)")
    for f in loc:
        own = [ft for ft in loc[f] if not all(match(ft, loc[g], ft["xy"], 5e-3) for g in loc if g != f)]
        if own:
            print(f"   only in {f}: " + "; ".join(f"r{ft['r']:.2f} at ({ft['xy'][0]:.2f}, {ft['xy'][1]:.2f}) z {ft['z'][0]:.1f}..{ft['z'][1]:.1f}"
                                            for ft in own))


def check_moved(stock, plate, info, tf):
    """Units exact and features at their transformed places; foot/thumb unchanged; no stray features."""
    frames, strips, moved = info["frames"], info["strips"], info["moved"]
    new_fts = features(plate)
    used = set()

    def xf_for(f):
        if f == "pinky":
            return lambda p: move_pts(d_pt(np.atleast_2d(p)), tf[f])[0]
        return lambda p: move_pts(np.atleast_2d(p), tf[f])[0]

    # 1. every hole/pin/window corner of each unit at its transformed place (same r, axis, depth)
    n_ok, n_all = 0, 0
    for f in FINGERS:
        src = "ring" if f == "pinky" else f
        poly = Polygon(strips[src]).buffer(-0.01)
        core = Polygon(unit_strip(frames[src], UNIT_HALF_W - JOINT_W - 0.1, (WEB_V + 0.5, UNIT_V[1])))
        if f == "pinky":
            core = Polygon(unit_strip(frames["ring"], UNIT_HALF_W - JOINT_W - 0.1, (WEB_V + 0.5, UNIT_V[1])))
        xf = xf_for(f)
        for ft in features(stock, rmax=6.3):
            bb = ft["bbox"]
            if not poly.contains(box(bb[0], bb[2], bb[1], bb[3])):
                continue
            n_all += 1
            m = match(ft, new_fts, xf(ft["xy"]))
            assert m, f"{f}: feature r{ft['r']:.2f} at {ft['xy']} not found at {xf(ft['xy'])}"
            assert probe(stock, ft, region=core) == probe(plate, ft, xf, core), f"{f}: feature at {ft['xy']} differs"
            used.update(id(g) for g in m)
            n_ok += 1
    print(f"unit features: {n_ok}/{n_all} stock holes, pins, window and pocket corners found at their moved places "
          f"(same radius, axis, depth; probed at 24 points each)")

    # each moved unit is the exact image of its stock strip: (a) inside the strip (less the joints with
    # its neighbours, JOINT_W + 0.1 mm, and below WEB_V only the pocket) the plate is solid exactly where
    # the moved unit is (4000 random probes); (b) the plate keeps all of the unit's material (probes inside
    # the unit), except where the back-shell screw holes cut into the joints
    rng = np.random.default_rng(1)
    in_plate = classifier(plate)
    holes2d = shapely.union_all([Point(*h).buffer(1.6) for h in info["shell_holes"]]) if info["shell_holes"] else Polygon()
    mf = info["mframes"]
    for f in FINGERS:
        w = UNIT_HALF_W - JOINT_W - 0.1
        top = Polygon(unit_strip(mf[f], w, (WEB_V + 0.5, UNIT_V[1])))
        low = Polygon(unit_strip(mf[f], 12.0, (-48.9, WEB_V + 0.6)))  # the pocket, down to its bottom (v = -49)
        x0, y0, x1, y1 = top.union(low).bounds
        pts = rng.uniform([x0, y0, -13.0], [x1, y1, 16.0], size=(12000, 3))
        keep = np.array([top.contains(Point(*q[:2])) or (low.contains(Point(*q[:2])) and q[2] < SHEET_Z[0] - 0.01) for q in pts])
        pts = pts[keep][:4000]
        a_, b_ = in_plate(pts), classifier(moved[f])(pts)
        inner = interior_points(moved[f], n=600, seed=2)
        inner = inner[[not holes2d.contains(Point(*q[:2])) for q in inner]]
        print(f"unit {f:6s}: plate = moved stock unit at {np.sum(a_ == b_)}/{len(pts)} probes in its strip "
              f"({b_.sum()} in material); {in_plate(inner).sum()}/{len(inner)} probes inside the unit are in the plate")
        bad = pts[a_ != b_]
        assert (a_ == b_).all() and in_plate(inner).all(), f"{f}: unit changed at {np.round(bad, 3).tolist()} " \
            f"(module frame {np.round(to_local(mf[f], bad[:, :2]), 3).tolist()}, plate {a_[a_ != b_]})"

    # 2. thumb side and foot unchanged (legs with their screw holes, thumb-side face, wrist-interface faces)
    foot = [box(-9.2, -85.0, -2.9, -57.0), box(-16.0, -85.0, -9.2, -63.0), box(51.9, -85.0, 58.0, -72.0)]
    in_stock = classifier(stock)
    pts = np.vstack([rng.uniform([b2.bounds[0], b2.bounds[1], -12.6], [b2.bounds[2], b2.bounds[3], 15.6], size=(1500, 3)) for b2 in foot])
    a_, b_ = in_plate(pts), in_stock(pts)
    n_foot = 0
    for ft in features(stock):
        if any(b2.buffer(-0.01).contains(Point(*ft["xy"])) for b2 in foot) and ft["r"] < 3.6:
            m = match(ft, new_fts, ft["xy"])
            assert m and probe(stock, ft) == probe(plate, ft), f"foot feature at {ft['xy']} changed"
            used.update(id(g) for g in m)
            n_foot += 1
    print(f"foot and thumb side (both legs): plate = stock at {np.sum(a_ == b_)}/{len(pts)} probes; "
          f"{n_foot} z-cylinder faces (holes, fillets) unchanged")
    assert (a_ == b_).all()
    _check_y_holes(stock, plate)

    # 3. no stray features: every concave z-cylinder of the plate up to the recess size is a moved unit
    #    feature (or a stock fillet cut by a unit strip), a foot feature, an edge of the stock kept in place
    #    (Base) or a re-drilled back-shell screw hole
    base_fts = features(info["base"])
    unit_fts = [ft for f in FINGERS for ft in features(moved[f])]  # incl. stock fillets cut by a strip's bottom
    tool_fts = [dict(ft, xy=ft["xy"] + h) for h in info["shell_holes"] for ft in features(info["hole_tool"])]
    stray = []
    for g in new_fts:
        if id(g) in used or not g["hole"] or g["r"] > 6.3:
            continue
        if any(abs(g["r"] - b["r"]) < 1e-4 and np.hypot(*(g["xy"] - b["xy"])) < 2e-3 for b in base_fts + unit_fts + tool_fts):
            continue
        stray.append(g)
    small = [f.Area() for f in plate.Faces() if f.Area() < 0.02]
    print(f"stray hole-like features: {len(stray)}; faces under 0.02 mm2: {len(small)}")
    for g in stray:
        print(f"   STRAY r{g['r']:.3f} at ({g['xy'][0]:.2f}, {g['xy'][1]:.2f}) z {g['z'][0]:.2f}..{g['z'][1]:.2f}")
    assert not stray


def check_seams(plate, info):
    """No slit where a moved unit meets what is below it (the stock kept in place, the rims): along the
    bottom of each unit strip, at every z, no gap of 0.6 mm or less between material below and material
    above may run on for 1 mm or more across the strip (a short one is the tip of an open notch)."""
    inside = classifier(plate)
    us, zs = np.arange(-14.9, 14.95, 0.5), np.arange(-2.4, 15.5, 1.5)
    vs = np.arange(UNIT_V[0] - 2.5, UNIT_V[0] + 3.5, 0.1)
    longest = 0.0
    for f in FINGERS:
        frame = info["mframes"][f]
        xy = from_local(frame, np.array([(u, v) for u in us for v in vs]))
        for z in zs:
            col = inside(np.column_stack([xy, np.full(len(xy), z)])).reshape(len(us), len(vs))
            thin = []
            for c in col:
                solid = np.flatnonzero(c)
                gaps = np.diff(solid) - 1 if len(solid) > 1 else np.zeros(1)
                thin.append(bool(((gaps > 0) & (gaps * 0.1 <= 0.6)).any()))
            run = max((len(r) for r in "".join("1" if t else "0" for t in thin).split("0")), default=0)
            longest = max(longest, run * 0.5)
            assert run < 2, f"{f}: slit under the unit at z = {z:.1f} ({run * 0.5:.1f} mm long)"
    print(f"seams under the units: no slit (no gap <= 0.6 mm runs 1 mm across a strip; longest {longest:.1f} mm)")


def _check_y_holes(stock, plate):
    """The foot's screw holes into the wrist interface (axis along y) are unchanged."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder

    def ys(shape):
        out = []
        for f in shape.Faces():
            s = BRepAdaptor_Surface(f.wrapped)
            if s.GetType() == GeomAbs_Cylinder and abs(s.Cylinder().Axis().Direction().Y()) > 0.999:
                bb = f.BoundingBox()
                out.append((round(s.Cylinder().Radius(), 4), round(bb.xmin, 3), round(bb.xmax, 3), round(bb.ymin, 3),
                            round(bb.ymax, 3), round(bb.zmin, 3), round(bb.zmax, 3)))
        return sorted(out)
    a, b = ys(stock), ys(plate)
    assert a == b, f"wrist-interface screw holes changed:\n{a}\n{b}"
    print(f"foot: {len(a)} screw-hole faces into the wrist interface unchanged")


def min_material(plate, info):
    """Minimum material between neighbouring units' features, from sections of the plate:
    - back sheet (windows, finger-frame screw holes, pivot holes, the back-shell screw holes and the
      foot's holes; every enclosed hole belongs to the unit whose axis is nearest, or to the foot);
    - servo pockets: thickness of the rib between neighbouring pockets, across the joint, over the
      servos (module frame v -42 .. -14) at several z."""
    mf = info["mframes"]
    screws = [Point(*h) for h in info["shell_holes"]]
    out = {}

    def keep(key, d, z):
        if d < out.get(key, (np.inf, 0))[0]:
            out[key] = (d, z)

    for z in (13.0, 14.0, 15.2):
        sec = section2d(plate, z)
        groups = {}
        for g in getattr(sec, "geoms", [sec]):
            for h in g.interiors:
                h = Polygon(h)
                if any(h.distance(p) < 1e-6 for p in screws):
                    key = "shell screw"
                else:
                    loc = {f: to_local(mf[f], np.array(h.centroid.coords))[0] for f in FINGERS}
                    inside = [f for f in FINGERS if abs(loc[f][0]) < UNIT_HALF_W and UNIT_V[0] < loc[f][1] < UNIT_V[1]]
                    key = min(inside, key=lambda f: abs(loc[f][0])) if inside else "foot"
                groups.setdefault(key, []).append(h)
        groups = {k: shapely.union_all(v) for k, v in groups.items()}
        for a, b in list(zip(FINGERS, FINGERS[1:])) + [(f, k) for f in FINGERS for k in ("foot", "shell screw")]:
            if a in groups and b in groups:
                keep(f"{a}/{b} (sheet)", groups[a].distance(groups[b]), z)
    for z in (0.0, 5.0, 8.0, 11.0, 12.3):
        sec = section2d(plate, z)
        for a, b in zip(FINGERS, FINGERS[1:]):
            mid = to_local(mf[a], mf[b][0])[0] / 2
            for v in np.arange(-42.0, -13.9, 0.5):  # along the servos (v -43 .. -10.5), below the rib tops
                p = Point(*from_local(mf[a], [(mid, v)])[0])
                if not sec.contains(p):
                    continue  # (above a pocket wall's top, or no wall at this z on either side)
                ln = shapely.geometry.LineString(from_local(mf[a], [(mid - 8, v), (mid + 8, v)]))
                seg = [g for g in getattr(sec.intersection(ln), "geoms", [sec.intersection(ln)]) if g.distance(p) < 1e-9]
                keep(f"{a}/{b} (pocket rib)", seg[0].length, z)
    for k, (d, z) in out.items():
        print(f"min material {k:28s}: {d:.2f} mm (z = {z})")
    return out


def part_meshes(tf):
    """Palm-fixed module parts (moved like the sim moves them), the thumb's parts and the wrist
    interface, as trimesh meshes in the CAD frame (mm): {name: (finger, mesh)}."""
    import trimesh

    root = stock_root()
    geoms = part_geoms(root)
    plate_p, plate_q = get_pose(geoms["r_hand_plate"])
    Rp = quat_mat(plate_q)
    frames = module_frames()
    cache = {}
    out = {}
    for name, g in geoms.items():
        mesh = g.get("mesh")
        if mesh not in ("scs0009", "finger_frame_1", "finger_frame_2", "bushing_0608_04", "r_wrist_interface"):
            continue
        if mesh not in cache:
            cache[mesh] = trimesh.load(STOCK_MJCF / "assets" / f"{mesh}.stl")
        p, q = get_pose(g)
        v = (cache[mesh].vertices @ quat_mat(q).T + p - plate_p) @ Rp * 1000
        m = trimesh.Trimesh(v, cache[mesh].faces, process=False)
        c = m.centroid[:2]
        owner = "thumb" if mesh != "r_wrist_interface" else "wrist"
        for f in ("index", "middle", "ring"):
            loc = to_local(frames[f], c)
            if mesh != "r_wrist_interface" and abs(loc[0]) < 16 and -55 < loc[1] < 12 and m.bounds[1, 2] > -10:
                owner = f
        out[name] = (owner, m)
    moved = {}
    for name, (owner, m) in out.items():
        if owner in ("index", "middle", "ring"):
            mv = m.copy()
            mv.vertices[:, :2] = move_pts(mv.vertices[:, :2], tf[owner])
            moved[name] = (owner, mv)
            if owner == "ring":
                pk = m.copy()
                pk.vertices[:, :2] = move_pts(d_pt(pk.vertices[:, :2]), tf["pinky"])
                moved[name + " (pinky copy)"] = ("pinky", pk)
        else:
            moved[name] = (owner, m)
    counts = {f: sum(1 for o, _ in moved.values() if o == f) for f in FINGERS + ["thumb", "wrist"]}
    assert all(counts[f] == 6 for f in FINGERS + ["thumb"]) and counts["wrist"] == 1, counts
    return moved


def check_parts(stock, plate, tf, model_dir=None):
    """Intersection volume of the plate with every palm-fixed part (moved), compared with the stock
    plate and the stock parts. Optionally cross-checks the part poses against a regenerated 5F model."""
    import manifold3d as mf

    def manifold(shape_or_mesh):
        from OCP.BRepTools import BRepTools

        for tol in (0.01, 0.02, 0.005, 0.04):
            if hasattr(shape_or_mesh, "tessellate"):
                BRepTools.Clean_s(shape_or_mesh.wrapped)  # drop face meshes cached at other tolerances
                v, t = shape_or_mesh.tessellate(tol, 0.1)
                v, t = np.array([(p.x, p.y, p.z) for p in v]), np.array(t)
            else:
                v, t = shape_or_mesh.vertices, shape_or_mesh.faces
            mesh = mf.Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=np.asarray(t, np.uint32))
            mesh.merge()  # the B-rep faces are meshed separately: weld their shared edges
            out = mf.Manifold(mesh)
            if out.status() == mf.Error.NoError and out.volume() > 0:
                return out
        raise AssertionError(f"no manifold mesh: {out.status()}")

    stock_tf = {f: (0.0, LAYOUT.STOCK[f][0], LAYOUT.STOCK[f][0]) for f in FINGERS}
    P_new, P_stock = manifold(plate), manifold(stock)
    new, old = part_meshes(tf), part_meshes(stock_tf)
    worst = {}
    for name, (owner, m) in new.items():
        v_new = (P_new ^ manifold(m)).volume()
        base_name = name.replace(" (pinky copy)", "")
        v_old = (P_stock ^ manifold(old[base_name][1])).volume()
        worst[owner] = max(worst.get(owner, (0, 0, "")), (v_new, v_old, name))
        assert v_new <= v_old + 0.05, f"{name} ({owner}): plate intersects it by {v_new:.3f} mm3 (stock plate {v_old:.3f})"
    print("plate vs parts (intersection volume, new / stock plate with stock parts): "
          + ", ".join(f"{o} {w[0]:.3f}/{w[1]:.3f} mm3" for o, w in worst.items()))
    if model_dir is not None:
        _crosscheck_model(model_dir, new)


def _crosscheck_model(model_dir, parts):
    """Worst offset (mm) between the palm-fixed parts of a regenerated 5F model and ours (None if absent)."""
    import trimesh

    path = Path(model_dir) / "robot.xml"
    if not path.exists():
        return None
    geoms = part_geoms(stock_root(Path(model_dir)))
    plate_p, plate_q = get_pose(geoms["r_hand_plate"])
    Rp = quat_mat(plate_q)
    mine = np.array([m.centroid for _, m in parts.values()])
    cache, dmax, n = {}, 0.0, 0
    for name, g in geoms.items():
        mesh = g.get("mesh")
        if mesh not in ("scs0009", "finger_frame_1", "finger_frame_2", "bushing_0608_04"):
            continue
        if mesh not in cache:
            cache[mesh] = trimesh.load(STOCK_MJCF / "assets" / f"{mesh}.stl").centroid
        p, q = get_pose(g)
        c = Rp.T @ (p + quat_mat(q) @ cache[mesh] - plate_p) * 1000
        dmax, n = max(dmax, np.linalg.norm(mine - c, axis=1).min()), n + 1
    print(f"{path.relative_to(REPO)}: {n} palm-fixed parts, worst offset from ours {dmax:.4f} mm"
          + ("" if dmax < 0.01 else "  (that model is not built with this layout)"))
    return dmax


def check_sweep(stock, plate, tf, model_dir, n_grid=13):
    """Moving parts of every digit (servo horns, push rods, fingers) settled on an n x n grid of its two
    motor commands (+-90 deg), against the plate: the intersection volume per digit may not exceed the
    stock hand's against the stock plate (pinky: the stock ring's). Needs the sim model built with this
    layout (Demo/AHSimulation/five_finger/build_5f_model.py)."""
    import manifold3d as mf
    import mujoco as mj

    sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "five_finger"))
    import verify_5f_model as V

    if (_crosscheck_model(model_dir, part_meshes(tf)) or 1.0) > 0.01:
        print("moving-parts sweep skipped: no sim model with this layout")
        return
    plate_p, plate_q = get_pose(part_geoms(stock_root())["r_hand_plate"])

    def worst(model, shape):
        v, t = shape.tessellate(0.01, 0.1)
        mesh = mf.Mesh(vert_properties=np.array([(p.x, p.y, p.z) for p in v], np.float32), tri_verts=np.array(t, np.uint32))
        mesh.merge()
        P = mf.Manifold(mesh)
        h = V.Hand(model)
        m, d = h.m, h.d
        poses, _ = V.settle_grid(h, np.linspace(-np.pi / 2, np.pi / 2, n_grid))
        out = {}
        for q in poses.values():
            d.qpos[:] = q
            mj.mj_kinematics(m, d)
            Rb, tb = d.xmat[V.ROOT_BODY].reshape(3, 3), d.xpos[V.ROOT_BODY]
            Rp, tp = Rb @ quat_mat(plate_q), tb + Rb @ plate_p  # world <- CAD frame (m)
            for dig in h.digits:
                for g in h.geoms[dig][0]:
                    if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                        continue
                    k = m.geom_dataid[g]
                    va, vn, fa, fn = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
                    X = (d.geom_xpos[g] + m.mesh_vert[va:va + vn] @ d.geom_xmat[g].reshape(3, 3).T - tp) @ Rp * 1000
                    M = mf.Manifold(mf.Mesh(vert_properties=X.astype(np.float32), tri_verts=m.mesh_face[fa:fa + fn].astype(np.uint32)))
                    out[dig] = max(out.get(dig, 0.0), (P ^ M).volume())
        return out

    new = worst(Path(model_dir), plate)
    old = worst(STOCK_MJCF, stock)
    old["pinky"] = old["ring"]
    print(f"moving parts vs plate over {n_grid}x{n_grid} motor poses, worst intersection per digit (new / stock): "
          + ", ".join(f"{dig} {new[dig]:.2f}/{old[dig]:.2f} mm3" for dig in new))
    for dig in new:
        assert new[dig] <= old[dig] + 0.5, f"{dig}: moving parts hit the plate more than in the stock hand"


# ----------------------------------------------------------------------------- preview and export
def preview(stock, plate, path, info, tf, layout):
    """Stock plate (left) and the new one (right), seen from the back, with the finger-unit strips, the
    module footprints (servos, finger frames and bushings cut at z = 10) and the stock outline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    colors = {"index": "tab:red", "middle": "tab:green", "ring": "tab:blue", "pinky": "tab:orange", "thumb": "tab:purple"}
    stock_tf = {f: (0.0, LAYOUT.STOCK[f][0], LAYOUT.STOCK[f][0]) for f in FINGERS}
    fig, axs = plt.subplots(1, 2, figsize=(18, 9.5))
    panels = ((axs[0], stock, "stock R-Hand_Plate", stock_tf, info.get("strips")),
              (axs[1], plate, f"R-Hand_Plate_5F, layout '{layout}'", tf, info.get("mstrip")))
    for ax, shape, title, tf_, strips in panels:
        verts, tris = shape.tessellate(0.1, 0.3)
        v = np.array([(p.x, p.y, p.z) for p in verts])
        t = v[np.array(tris)]
        t = t[np.argsort(t[:, :, 2].mean(1))]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        shade = 0.35 + 0.6 * np.abs(n[:, 2]) / (np.linalg.norm(n, axis=1) + 1e-12)
        ax.add_collection(PolyCollection(t[:, :, :2], facecolors=np.outer(shade, [0.75, 0.75, 0.8]), edgecolors="none"))
        for name, (owner, m) in part_meshes(tf_).items():
            if owner in colors and not (shape is stock and owner == "pinky"):
                sec = m.section(plane_origin=[0, 0, 10.0], plane_normal=[0, 0, 1])
                for e in ([] if sec is None else sec.discrete):
                    ax.plot(e[:, 0], e[:, 1], "-", c=colors[owner], lw=0.5, alpha=0.8)
        for f in FINGERS if strips else []:
            if shape is stock and f == "pinky":
                continue
            st = np.vstack([strips[f], strips[f][:1]])
            ax.plot(*st.T, "-" if shape is plate else ":", c=colors[f], lw=1.1, label=f"{f} unit")
        ax.set_xlim(-30, 115)
        ax.set_ylim(-95, 15)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{title} (seen from the back, plate frame, mm)")
    s_out = section2d(stock, 15.2)
    for g in getattr(s_out, "geoms", [s_out]):
        axs[1].plot(*g.exterior.xy, "k--", lw=0.8, label="stock outline")
    for h in info["shell_holes"]:
        axs[1].plot(*h, "kx", ms=6, label="back-shell screw")
    handles, labels = axs[1].get_legend_handles_labels()
    keep = {lab: hd for hd, lab in zip(handles, labels)}
    axs[1].legend(keep.values(), keep.keys(), loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=60)


def mirror(shape):
    return shape.mirror("YZ", cq.Vector(MIRROR_X, 0, 0))


def export(shape, step, stl):
    step.parent.mkdir(parents=True, exist_ok=True)
    stl.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(cq.Workplane().add(shape), str(step))
    cq.exporters.export(cq.Workplane().add(shape), str(stl), tolerance=0.02, angularTolerance=0.15)
    print(f"wrote {step.relative_to(REPO)}, {stl.relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=["stock", "parallel"], default=LAYOUT.LAYOUT, help="finger-module layout")
    ap.add_argument("--no-checks", action="store_true", help="skip the checks")
    ap.add_argument("--sweep", action="store_true", help="also check the moving finger parts over their motor grid "
                    "(needs the sim model AH_Right_5F built with the same layout)")
    args = ap.parse_args()
    tf = LAYOUT.transforms(args.layout)
    stock = cq.importers.importStep(str(STOCK_STEP)).val()
    model = REPO / "Demo" / "AHSimulation" / "AHSimulation" / "AH_Right_5F" / "mjcf"

    if args.layout == "stock":
        plate, info = build_stock(stock)
    else:
        plate, info = build_moved(stock, tf)
    bb = plate.BoundingBox()
    print(f"layout '{args.layout}': stock plate {stock.Volume() / 1000:.2f} cm3 -> 5-finger plate {plate.Volume() / 1000:.2f} cm3; "
          f"bbox x [{bb.xmin:.1f}, {bb.xmax:.1f}] y [{bb.ymin:.1f}, {bb.ymax:.1f}] z [{bb.zmin:.1f}, {bb.zmax:.1f}]")
    print("back-shell screw holes (make_5f_shells.py SCREWS): "
          + ", ".join(f"({h[0]:.1f}, {h[1]:.1f})" for h in info["shell_holes"]))

    if not args.no_checks:
        if args.layout == "stock":
            check_features_stock(stock, plate)
        else:
            unit_report(stock, info)
            check_moved(stock, plate, info, tf)
            check_seams(plate, info)
            min_material(plate, info)
        check_parts(stock, plate, tf, model)
        if args.sweep:
            check_sweep(stock, plate, tf, model)

    export(plate, OUT_STEP, OUT_STL)
    preview(stock, plate, OUT_PNG, info, tf, args.layout)

    stock_l = cq.importers.importStep(str(STOCK_STEP_L)).val()
    diff = mirror(stock).cut(stock_l).Volume() + stock_l.cut(mirror(stock)).Volume()
    assert diff < 1.0, f"L-Hand_Plate is not the mirrored R-Hand_Plate ({diff:.1f} mm3 differ)"
    export(mirror(plate), OUT_STEP_L, OUT_STL_L)


if __name__ == "__main__":
    main()
