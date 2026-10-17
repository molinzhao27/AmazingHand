#!/usr/bin/env python3
"""All-printed mock-up of the final 5-digit right hand (AH_Right_5F_3S_Wrist): printed stand-ins for every bought part.

The mock-up is for checking assembly and shape on a real print before buying servos and hardware. Every printed part
of the real build is used unchanged (cad/five_finger/**, cad/stl/*). Only the bought parts are replaced, by printed parts
that fill the same holes, so the assembly is the real one. The mock-up is posable by hand: the dummy servos carry a
plain output stub, the printed horns sit on it with a friction fit, and the push rods are one-piece printed rods with
snap-on ball cups at the exact ball-to-ball length of the sim.

  stand-in                    replaces                                       how it is made
  Dummy_SCS0009          x10  Feetech SCS0009                                 the sim's servo mesh, its output spline a plain
                                                                              d4 stub, ear holes for d1.75 filament pins
  Mock_Horn              x10  reworked cross horn + M2x9 ball stud + nut      the sim's horn mesh, its bore a friction fit on
                                                                              the stub, the d5 ball printed on it
  Rod_<digit>_M1/M2      x10  2 M2 ball joints + M2x18 rod + spacer          one piece: snap-on cups at the sim's ball centres
  Link_Ball              x10  the two ball studs on each finger's Link        d5 ball on a d2.1 axle through the Link
  Pin_D2_L*, Axle_D2.1_L*     the d2 dowels (stock and 3-phalanx fingers)     printed pins, lying on a 0.15 mm flat
  Bushing_0608           x10  igus bushing 0608-04                            the sim's mesh
  Washer_M2.5             x5  M2.5 large washer                               the sim's mesh
  Peg_2.5x6/8/16              2.5 mm thermoplastic screws                     d2.0 press peg with a pan head
  Dummy_STS3215           x2  Feetech STS3215                                 make_wrist's servo box + a d6 output stub
  Disc_Horn               x2  25T metal disc horn                             d24 x 3 disc, friction bore, 4 peg holes
  Peg_M2.5x6              x8  horn arm -> disc horn screws                    d2.4 press peg
  Bushing_F683            x4  F683ZZ flanged bearing                          printed flanged sleeve on the d3 pin
  Pin_D3_L10              x4  d3 x 10 dowel (gimbal ring)                     printed pin
  Ball_Stud_Top/Arm       x2  M3 ball studs (d5.8)                            printed stud, press-fit shank
  Rod_Wrist_M1/M2         x1  2 M3 ball links + M3 rod                        one piece: snap-on cups, 66.0 / 96.0 mm
  Insert_M3              x14  M3 heat-set insert                              printed press sleeve
  Peg_M3x10 / Peg_M3x8_CSK    M3 socket head / countersunk screws             d3.0 press peg into the sleeve

Not replaced: the four servo screws per finger (2 x 7 mm self-tappers into d1.5 pilot holes, too small for a printed
pin) take 1.75 mm filament pins through the dummy servo's ear holes (ream the frame's d1.5 holes to 1.8 mm), or are left
out (the servo pair is trapped between the two finger frames and the plate). The M5 stand screws, cables and
electronics are not part of the mock-up.

Fits are parameters at the top. Print the fit test first (test_fit.3mf, see README.md), pick the variants that fit on
your printer, set the parameters, and run this again.

Checks: every stand-in that has a place in the sim is put there and compared with the real printed parts around it
(surface samples, signed distance). The report lists the deepest overlap per stand-in, which should only be the
intended press fits.

Usage: .venv/bin/python cad/five_finger/mockup/make_mockup.py
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq
import manifold3d as mf
import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "assembly"))
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "wrist"))
import make_assembly as ASM  # noqa: E402  (sim placements of every part, in the hand's CAD frame, mm)
import make_wrist as MW  # noqa: E402  (wrist primitives, frames and the bought-part geometry)

wd = MW.wd
V = cq.Vector
STL_DIR = HERE / "stl"
TEST_DIR = HERE / "stl" / "test"

# ============================================================================ fits (mm, on the diameter)
# Tune these after printing test_fit.3mf: each one is printed there in three sizes (the value and +-0.1 mm).
PIN_D = 2.00          # printed d2 dowel. Holes: d1.95 press (P1, P2, P3, Gimbal), d2.15 running (P2, P3, Link, L2)
AXLE_D = 2.10         # axle through the Link's d2.2 ball-stud hole (and the horn's ball pin)
BALL_HOLE = 2.00      # Link_Ball's hole on that axle (press)
PEG_D = 2.00          # 2.5 mm self-tapper stand-in: passes the d2.8 clearance holes, presses into the d1.9 pilots
STUB_D = 4.00         # dummy SCS0009 output stub (the real spline is d4)
HORN_BORE = 3.90      # Mock_Horn bore on the stub: friction that holds a pose
BALL_D = 5.00         # hand ball studs (as the M2 ball joints)
CUP_CLEAR = 0.15      # radial gap ball / cup
CUP_LIP = 1.20        # the cup's mouth plane above the ball centre: higher = tighter snap (mouth d4.7 at 1.2)
CUP_SLITS = 2         # radial slits through the lip so it can open over the ball
SLIT_W = 0.6
WBALL_D = 5.80        # wrist ball studs
WCUP_LIP = 1.40
STS_STUB_D = 6.00     # dummy STS3215 output stub
DISC_BORE = 5.90      # Disc_Horn bore on that stub (friction)
M25_PEG_D = 2.40      # horn arm (d2.7 clearance) -> Disc_Horn (d2.3 holes)
DISC_HOLE = 2.30
F683_OD = 7.10        # Bushing_F683 in the d7.0 bearing seats (press)
F683_BORE = 3.25      # running on the d3 pin
PIN3_D = 3.05         # d3 dowel into the gimbal ring's d2.9 holes (press)
STUD_TOP_SHANK = 4.10   # top ball stud straight into the bar's d4.0 insert hole (press)
STUD_ARM_SHANK = 3.30   # arm ball stud into the horn arm's d3.2 hole (press)
INSERT_OD = 4.10      # Insert_M3 in the d4.0 insert holes (press)
INSERT_ID = 2.85
M3_PEG_D = 3.00       # Peg_M3 into Insert_M3 (press), through the d3.4 clearance holes
FLAT = 0.30           # pins and pegs print lying down, on a flat this deep (a d2 pin: 1.4 mm wide, no brim needed)
FILAMENT = 1.75       # filament pins (the servo ears' d2.0 holes)
CLEAR_FRAME = 0.15    # dummy servo trimmed this far off the finger frames

# ============================================================================ small helpers
def log(*a):
    print(*a, flush=True)


def tf(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def frame(z, x_hint=(1, 0, 0)):
    """Rotation whose z column is z (and x column as close to x_hint as possible)."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, (0, 1, 0)) if abs(z[1]) < 0.9 else np.cross(z, (1, 0, 0))
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def apply(T, P):
    return np.asarray(P, float) @ T[:3, :3].T + T[:3, 3]


def cyl(p, d, r, h):
    return MW.cyl(p, d, r, h)


def sphere(p, r):
    return MW.sphere(p, r)


def to_mesh(shape):
    """cq shape -> watertight trimesh (mm), with make_wrist's absolute tessellation. OCC leaves the odd zero-area sliver
    triangle where a sphere meets a cut (as in R-Hand_Plate_5F.stl); those are dropped."""
    tmp = STL_DIR / "_tmp.stl"
    MW.export_stl(shape, tmp)
    m = trimesh.load(tmp, process=True)
    tmp.unlink()
    if not m.is_watertight:
        m.update_faces(m.nondegenerate_faces(height=1e-6))
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        bodies = [b for b in m.split(only_watertight=False) if len(b.faces) > 3 and abs(b.volume) > 1e-3]
        m = trimesh.util.concatenate(bodies) if len(bodies) > 1 else bodies[0]
        m.merge_vertices()
    return m


def man(tm):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                               tri_verts=np.asarray(tm.faces, np.uint32)))


def unman(M):
    mesh = M.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)


def mcyl(p0, p1, r, n=64):
    """manifold3d cylinder from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    T = tf(frame(p1 - p0), p0)
    return mf.Manifold.cylinder(L, r, r, n).transform(T[:3, :4])


# ============================================================================ the sim: where every bought part is
@dataclass
class Sim:
    geoms: list
    by_name: dict
    motors: dict           # (digit, motor) -> (anchor, axis) in the CAD frame

    def mesh(self, name):
        """The geom's mesh file (mm) and its file -> CAD transform."""
        g = self.by_name[name]
        tm = trimesh.load(g.asset, process=True)
        return trimesh.Trimesh(tm.vertices * g.scale, tm.faces, process=True), g.T

    def placed(self, name):
        m, T = self.mesh(name)
        return trimesh.Trimesh(apply(T, m.vertices), m.faces, process=False)


def load_sim():
    m, geoms, _ = ASM.load_sim()
    ASM.map_cad_stls(geoms)
    plate = next(g for g in geoms if g.mesh == "r_hand_plate")
    W = ASM.inv(plate.T_world)
    for g in geoms:
        g.T = W @ g.T_world
    ASM.assign_names(geoms)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, ASM.KEYFRAME))
    mujoco.mj_forward(m, d)
    motors = {}
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jn.startswith("finger") and "_motor" in jn:
            motors[(ASM.DIGITS[int(jn[6])], int(jn[-1]))] = (apply(W, d.xanchor[j] * 1000), W[:3, :3] @ d.xaxis[j])
    return Sim(geoms, {g.name: g for g in geoms}, motors)


# ============================================================================ parts
@dataclass
class Part:
    name: str
    group: str                 # hand / wrist / test
    mesh: trimesh.Trimesh      # in the part's own frame (make_plates.py orients it for printing)
    places: list = field(default_factory=list)   # part frame -> CAD frame, one per instance in the sim
    qty: int = 0
    note: str = ""
    up: tuple = None           # print up in the part frame (None: make_plates.py picks)


PARTS = {}


def add(name, group, mesh, places=(), qty=None, note="", up=None):
    PARTS[name] = Part(name, group, mesh, list(places), qty if qty is not None else len(places), note, up)


# ---------------------------------------------------------------------------- pins and pegs (designed lying along x)
def pin(d, L, chamfer=0.3):
    """Pin of diameter d and length L along x, centred, with a flat FLAT deep at -z (it prints on the flat)."""
    r = d / 2
    s = cyl((-L / 2, 0, 0), (1, 0, 0), r, L)
    c = min(chamfer, r * 0.4)
    for sgn in (1, -1):   # chamfered ends (cones)
        x0 = sgn * L / 2
        s = s.cut(cyl((x0 - sgn * c, 0, 0), (sgn, 0, 0), r + 1, c + 1)).fuse(
            cq.Solid.makeCone(r, r - c, c, pnt=V(x0 - sgn * c, 0, 0), dir=V(sgn, 0, 0)))
    s = s.cut(MW.box(-L, L, -r - 1, r + 1, -r - 1, -r + FLAT))
    return s.clean()


def peg(d, L, head_d, head_h, csk=False):
    """Screw stand-in along x: head at x <= 0, shank of diameter d to x = L, flat at -z."""
    r = d / 2
    if csk:   # 90 deg countersunk head
        head = cq.Solid.makeCone(head_d / 2, r, head_d / 2 - r, pnt=V(-(head_d / 2 - r), 0, 0), dir=V(1, 0, 0))
        x_head = -(head_d / 2 - r)
    else:
        head = cyl((-head_h, 0, 0), (1, 0, 0), head_d / 2, head_h)
        x_head = -head_h
    s = head.fuse(cyl((0, 0, 0), (1, 0, 0), r, L - 0.3), cq.Solid.makeCone(r, r * 0.6, 0.3, pnt=V(L - 0.3, 0, 0), dir=V(1, 0, 0)))
    zf = -min(r, head_d / 2) + FLAT
    s = s.cut(MW.box(x_head - 1, L + 1, -head_d, head_d, -head_d, zf))
    return s.clean()


def pin_place(geom_mesh, T):
    """Part frame of a printed pin -> the sim pin (its long axis, centred on its mid point)."""
    v = apply(T, geom_mesh.vertices)
    c = 0.5 * (v.min(0) + v.max(0))
    c = v.mean(0)
    u, s, vt = np.linalg.svd(v - c)
    ax = vt[0]
    h = (v - c) @ ax
    mid = c + ax * 0.5 * (h.max() + h.min())
    L = h.max() - h.min()
    R = frame(ax)            # z along the pin
    R = R[:, [2, 0, 1]]      # part x = pin axis
    return tf(R, mid), L


# ---------------------------------------------------------------------------- hand
def build_hand(sim):
    # pins: every d2 dowel in the sim, grouped by length
    pins = {}
    for g in sim.geoms:
        lab = ASM.label_of(g)
        if lab.startswith(("Pin_2x", "pin_2x")):
            m, T = sim.mesh(g.name)
            P, L = pin_place(m, T)
            pins.setdefault(round(L), []).append(P)
    for L, places in sorted(pins.items()):
        add(f"Pin_D2_L{L}", "hand", to_mesh(pin(PIN_D, L)), places,
            note=f"d2 x {L} dowel: printed pin, or a {FILAMENT} mm filament piece (looser)")

    # bushings and washers: the sim meshes as they are
    for lab, name in (("Bushing_0608", "Bushing_0608"), ("Washer_M2.5", "Washer_M2.5")):
        gs = [g for g in sim.geoms if ASM.label_of(g) == lab]
        m, _ = sim.mesh(gs[0].name)
        add(name, "hand", m, [g.T for g in gs], note=f"replaces {lab} (same shape)")

    # pegs for the 2.5 mm thermoplastic screws: the gimbal screw is in the sim, the others are counted from the
    # assembly guide and FIVE_FINGER.md (frames -> plate 3 per digit, palm plate -> hand adapter 4, back shell 5)
    add("Peg_2.5x8", "hand", to_mesh(peg(PEG_D, 8.0, 4.2, 1.6)), qty=5 + 15 + 2,
        note="2.5x8 thermoplastic screw: gimbal (with the washer) 5, finger frames -> plate 3 per digit, back shell -> "
             "hand adapter 2")
    add("Peg_2.5x16", "hand", to_mesh(peg(PEG_D, 16.0, 4.2, 1.6)), qty=3,
        note="back shell -> palm plate ribs (2.5x16 thermoplastic screws)")
    add("Peg_2.5x6", "hand", to_mesh(peg(PEG_D, 6.0, 4.2, 1.6)), qty=4,
        note="hand adapter -> palm plate legs (2.5x6 thermoplastic screws)")

    # dummy servos: the sim's SCS0009 mesh. Its ears already have the d2.0 holes on the finger frames' screw axes
    # (they take 1.75 mm filament pins); its spline is a d4 stub with a d2.5 screw hole, which is filled
    servos = [g for g in sim.geoms if ASM.label_of(g) == "SCS0009_Servo"]
    sm, T0 = sim.mesh(servos[0].name)
    a, ax = sim.motors[(servos[0].group, servos[0].motor)]
    Ti = ASM.inv(T0)
    a_f, ax_f = apply(Ti, a[None])[0], Ti[:3, :3] @ ax
    h = (sm.vertices - a_f) @ ax_f
    tip = a_f + ax_f * h.min()                    # the stub's free end
    S = man(sm)
    if abs(STUB_D - 4.0) < 1e-6:
        S = S + mcyl(tip + ax_f * 0.3, tip + ax_f * 6.0, 1.5)
    else:   # replace the stub (leaves a 0.5 mm groove round its root, so no faces coincide)
        root = tip + ax_f * 3.65                  # the servo's round boss face
        S = S - mcyl(tip - ax_f, root + ax_f * 0.5, 2.3) + mcyl(tip, root + ax_f * 1.0, STUB_D / 2)
    # the CAD servo's ears sit 0.6 mm inside Finger_Frame-2 (as in the stock model; the real ears are thinner): a printed
    # dummy would not go in, so it is trimmed to the frames with CLEAR_FRAME of clearance. Every digit's drive module
    # is the same rigid copy, so the index's two servos give the cut for all ten.
    cut = None
    for g in (sim.by_name["index_SCS0009_Servo_M1"], sim.by_name["index_SCS0009_Servo_M2"]):
        Ti = ASM.inv(g.T)
        for fr in ("index_Finger_Frame-1", "index_Finger_Frame-2"):
            F = sim.placed(fr)
            if len(F.faces) > 6000:
                F = F.simplify_quadric_decimation(face_count=6000)
            for dv in ((0, 0, 0), (0, 0, CLEAR_FRAME), (0, 0, -CLEAR_FRAME), (0, CLEAR_FRAME, 0), (0, -CLEAR_FRAME, 0),
                       (CLEAR_FRAME, 0, 0), (-CLEAR_FRAME, 0, 0)):
                Fm = man(trimesh.Trimesh(apply(Ti, F.vertices + np.asarray(dv)), F.faces))
                cut = Fm if cut is None else cut + Fm
    S = S - cut
    add("Dummy_SCS0009", "hand", unman(S), [g.T for g in servos],
        note=f"servo stand-in; its ear holes (d2.0) take {FILAMENT} mm filament pins into the finger frames")

    # horns with the ball on them
    horns = [g for g in sim.geoms if ASM.label_of(g) == "Servo_Horn"]
    hm, Th = sim.mesh(horns[0].name)
    ball_f, stud_f, axis_f = [], [], []
    for g in horns:
        Ti = ASM.inv(g.T)
        b = sim.by_name[f"{g.group}_Rotule_Ball_M{g.motor}"]
        bm, Tb = sim.mesh(b.name)
        c = apply(Tb, bm.vertices).mean(0)
        a, ax = sim.motors[(g.group, g.motor)]
        ball_f.append(apply(Ti, c[None])[0])
        axis_f.append(Ti[:3, :3] @ ax)
        stud_f.append(apply(Ti, a[None])[0])
    ball_f, axis_f, anchor_f = np.mean(ball_f, 0), np.mean(axis_f, 0), np.mean(stud_f, 0)
    H = man(hm)
    # the stud: from the ball centre towards the horn (along the servo axis), into the horn arm
    hv = hm.vertices
    toward = axis_f if np.dot(hv.mean(0) - ball_f, axis_f) > 0 else -axis_f
    H = H + mf.Manifold.sphere(BALL_D / 2, 96).translate(tuple(ball_f)) + mcyl(ball_f, ball_f + toward * 4.0, 1.1)
    # friction bore on the stub: a sleeve inside the horn's d4 bore
    hb = (hv - anchor_f) @ axis_f
    face = anchor_f + axis_f * (hb.max() if np.dot(axis_f, toward) > 0 else hb.min())   # the face on the servo
    inward = -toward                                                                    # into the horn from it
    sleeve = mcyl(face - inward * 0.01, face + inward * 2.8, 2.3) - mcyl(face - inward, face + inward * 4, HORN_BORE / 2)
    H = H + sleeve
    add("Mock_Horn", "hand", unman(H), [g.T for g in horns], up=tuple(-toward),   # prints on its hub face
        note="cross horn + M2 ball stud + nut in one piece; slides onto the Dummy_SCS0009 stub (friction)")

    # Link balls and their axles
    lb, axles = [], {}
    ball_shape = None
    for dig in ("index", "middle", "ring", "pinky", "thumb"):
        names = [f"{dig}_Rotule_Ball_1", f"{dig}_Rotule_Ball_2"]
        C, A = [], []
        for n in names:
            bm, Tb = sim.mesh(n)
            C.append(apply(Tb, bm.vertices).mean(0))
            A.append(Tb[:3, :3] @ np.array([0, 0, 1.0]))
        u = (C[1] - C[0]) / np.linalg.norm(C[1] - C[0])
        for c, sgn in ((C[0], -1), (C[1], 1)):
            lb.append(tf(frame(sgn * u), c))        # ball frame: z points away from the Link
        L = np.linalg.norm(C[1] - C[0]) + 2 * 2.0
        P = tf(frame(u)[:, [2, 0, 1]], 0.5 * (C[0] + C[1]))
        axles.setdefault(round(L * 2) / 2, []).append(P)
    s = sphere((0, 0, 0), BALL_D / 2).intersect(MW.box(-3, 3, -3, 3, -2.0, 2.0))
    s = s.cut(cyl((0, 0, -3), (0, 0, 1), BALL_HOLE / 2, 6)).clean()
    add("Link_Ball", "hand", to_mesh(s), lb, note="d5 ball pressed on the Link axle, one each side of the Link")
    for L, places in sorted(axles.items()):
        add(f"Axle_D2.1_L{L:g}", "hand", to_mesh(pin(AXLE_D, L)), places,
            note="through the Link's d2.2 ball-stud hole, a Link_Ball pressed on each end")

    # one-piece push rods
    for dig in ("index", "middle", "ring", "pinky", "thumb"):
        for mot in (1, 2):
            build_hand_rod(sim, dig, mot)


def mbox(c, R, half):
    """manifold3d box centred at c with axes R (columns) and half sizes."""
    return mf.Manifold.cube(tuple(2 * np.asarray(half, float)), True).transform(tf(R, c)[:3, :4])


def cup(c, mouth, ball_d, r_out, lip, floor, slits=0, slit_w=0.6, slit_dir=None):
    """Snap-on ball cup at c, mouth towards `mouth` (manifold3d): returns (outer solid, [tools to subtract])."""
    c = np.asarray(c, float)
    m = np.asarray(mouth, float) / np.linalg.norm(mouth)
    rb = ball_d / 2 + CUP_CLEAR
    outer = mcyl(c - m * (rb + floor), c + m * lip, r_out, 96)
    r_mouth = np.sqrt(max(rb ** 2 - lip ** 2, 0.0))
    tools = [mf.Manifold.sphere(rb, 96).translate(tuple(c)), mcyl(c + m * (lip - 0.01), c + m * (lip + 5), r_mouth, 64)]
    R = frame(m, slit_dir if slit_dir is not None else (1, 0, 0))
    for k in range(slits):   # through the lip only (ball centre plane -> mouth), 45 deg off the rod
        a = np.pi * k / slits + np.pi / 4
        u = R[:, 0] * np.cos(a) + R[:, 1] * np.sin(a)
        Rs = np.column_stack([u, np.cross(m, u), m])
        tools.append(mbox(c + m * (lip + 0.5) / 2, Rs, (r_out + 1, slit_w / 2, (lip + 0.5) / 2 + 0.01)))
    return outer, tools


def carve(solids, tools):
    body = solids[0]
    for x in solids[1:]:
        body = body + x
    for t in tools:
        body = body - t
    return unman(body)


def build_hand_rod(sim, dig, mot):
    hb = sim.by_name[f"{dig}_Rotule_Ball_M{mot}"]
    rod = sim.by_name[f"{dig}_Rod_M2_L18_M{mot}"]
    bm, Tb = sim.mesh(hb.name)
    cA = apply(Tb, bm.vertices).mean(0)
    rv = sim.placed(rod.name).vertices
    rc = rv.mean(0)
    # the Link-side ball: the one of the two nearest to the rod's line, beyond the rod's far end
    cands = []
    for n in (f"{dig}_Rotule_Ball_1", f"{dig}_Rotule_Ball_2"):
        m2, T2 = sim.mesh(n)
        cands.append((n, apply(T2, m2.vertices).mean(0), T2))
    u = rc - cA
    u /= np.linalg.norm(u)
    nB, cB, TB = min(cands, key=lambda t: np.linalg.norm(np.cross(t[1] - cA, u)))
    a, ax = sim.motors[(dig, mot)]
    mA = ax if np.dot(ax, a - cA) > 0 else -ax                       # cup A mouth: towards the horn
    link = sim.placed(f"{dig}_UF_Link" if f"{dig}_UF_Link" in sim.by_name else
                      f"{dig}_UF_Link_S088" if f"{dig}_UF_Link_S088" in sim.by_name else f"{dig}_Link")
    axB = TB[:3, :3] @ np.array([0, 0, 1.0])
    mB = axB if np.dot(axB, link.vertices.mean(0) - cB) > 0 else -axB   # cup B mouth: towards the Link
    oA, tA = cup(cA, mA, BALL_D, 3.75, CUP_LIP, 0.8, CUP_SLITS, SLIT_W, cB - cA)
    oB, tB = cup(cB, mB, BALL_D, 3.75, CUP_LIP, 0.8, CUP_SLITS, SLIT_W, cA - cB)
    d = cB - cA
    L = np.linalg.norm(d)
    mesh = carve([oA, oB, mcyl(cA, cB, 1.5)], tA + tB)
    # part frame: origin at cup A's centre, x along the rod, z = cup A's mouth (prints mouth up, rod lying flat)
    R = frame(mA, d)
    P = tf(R, cA)
    mesh.apply_transform(ASM.inv(P))
    add(f"Rod_{dig}_M{mot}", "hand", mesh, [P], up=(0, 0, 1),
        note=f"push rod, {L:.2f} mm between ball centres: cup with the mouth up on the horn ball, the other on the "
             f"Link ball")


# ---------------------------------------------------------------------------- wrist
def w2c(shape):
    return MW.w_to_cad(shape)


def build_wrist(sim):
    Rw = wd.R_W2CAD
    # dummy servos: body box + output stub through the disc horn
    b = wd.servo_box(1)
    body = MW.box(b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1])
    sh = np.array([wd.X_TOP_FACE, wd.Y_SHAFT[1], wd.Z_SHAFT])
    dummy = MW.fuse(body, cyl(sh - (0.5, 0, 0), (1, 0, 0), STS_STUB_D / 2, 0.5 + wd.HORN_GAP + wd.HORN_T))
    P1 = tf(Rw, wd.C_CAD)
    Rz = tf(MW.wd.rot_z(np.pi), (0, 0, 0))
    T180 = tf(np.eye(3), (0, MW.Y_PAIR, 0)) @ Rz @ tf(np.eye(3), (0, -MW.Y_PAIR, 0))
    add("Dummy_STS3215", "wrist", to_mesh(dummy), [P1, P1 @ T180],
        note="STS3215 stand-in (make_wrist's servo box) with a plain d6 output stub")

    # disc horn: d24 x 3, friction bore, 4 peg holes on a 14 mm circle at 45 deg
    x0 = wd.X_TOP_FACE + wd.HORN_GAP
    sh = np.array([x0, wd.Y_SHAFT[1], wd.Z_SHAFT])
    disc = cyl(sh, (1, 0, 0), wd.HORN_D / 2, wd.HORN_T)
    tools = [cyl(sh - (1, 0, 0), (1, 0, 0), DISC_BORE / 2, 5)]
    for k in range(4):
        t = np.radians(45 + 90 * k)
        tools.append(cyl(sh + (-1, 7 * np.cos(t), 7 * np.sin(t)), (1, 0, 0), DISC_HOLE / 2, 5))
    add("Disc_Horn", "wrist", to_mesh(MW.cut(disc, *tools)), [P1, P1 @ T180],
        note="25T disc horn stand-in: slides onto the Dummy_STS3215 stub; the Wrist_Horn_Arm pegs onto it")
    add("Peg_M2.5x6", "wrist", to_mesh(peg(M25_PEG_D, 6.5, 4.5, 1.8)), qty=8,
        note="Wrist_Horn_Arm -> Disc_Horn (4 per arm)")

    # gimbal: printed bearings and pins
    a = np.array([1.0, 0, 0])   # built on the +x lug, placed on all four
    brg = MW.cut(MW.fuse(cyl(a * wd.LUG_IN, a, wd.BRG_FD / 2, wd.BRG_FT),
                         cyl(a * (wd.LUG_IN + wd.BRG_FT), a, F683_OD / 2, wd.BRG_B)),
                 cyl(a * (wd.LUG_IN - 1), a, F683_BORE / 2, 10))
    places_b, places_p = [], []
    for name, ax in MW.AXES.items():
        a = np.asarray(ax, float)
        R = frame(a, (0, 1, 0) if abs(a[1]) < 0.9 else (1, 0, 0)) @ frame((1, 0, 0), (0, 1, 0)).T
        places_b.append(P1 @ tf(R))
        places_p.append(P1 @ tf(R))
    add("Bushing_F683", "wrist", to_mesh(brg), places_b, note="F683ZZ flanged bearing stand-in (4 lugs)")
    pins3 = []
    for Tp in places_p:
        c = apply(Tp, np.array([[wd.LUG_OUT - wd.PIN_L / 2, 0, 0]]))[0]
        pins3.append(tf(Tp[:3, :3], c))
    add("Pin_D3_L10", "wrist", to_mesh(pin(PIN3_D, wd.PIN_L)), pins3, note="d3 x 10 gimbal pin (press into the ring)")

    # ball studs
    arm_in = wd.X_TOP_FACE + wd.HORN_GAP + wd.HORN_T
    for kind, H, shank, depth in (("Top", wd.STUD_TOP_H, STUD_TOP_SHANK, 5.5), ("Arm", wd.STUD_ARM_H, STUD_ARM_SHANK, wd.ARM_T)):
        s = MW.fuse(cyl((0, 0, 0), (0, 0, 1), 3.0, 1.5), cyl((0, 0, 0), (0, 0, 1), 1.75, H),
                    sphere((0, 0, H), WBALL_D / 2), cyl((0, 0, -depth), (0, 0, 1), shank / 2, depth + 0.01))
        places = []
        for i in (1, 2):
            sg = wd.SIDE[i]
            if kind == "Top":
                seat, ball = np.array([sg * wd.BAR_X, 0, wd.E_LEVER]), wd.P_TOP_HAND[i]
            else:
                bot = wd.ball_bottom(i, 0.0)
                seat, ball = np.array([sg * arm_in, bot[1], bot[2]]), bot
            places.append(P1 @ tf(frame(ball - seat), seat))
        add(f"Ball_Stud_{kind}", "wrist", to_mesh(s), places,
            note="M3 ball stud stand-in: " + ("shank presses into the lever bar's insert hole" if kind == "Top"
                                               else "shank presses through the horn arm (no nut)"))

    # one-piece rods
    for i in (1, 2):
        top, bot = wd.P_TOP_HAND[i], wd.ball_bottom(i, 0.0)
        sg = wd.SIDE[i]
        m_top = np.array([-sg, 0, 0.0])          # towards the lever bar (inwards)
        m_bot = np.array([sg, 0, 0.0])           # towards the horn arm (outwards)
        oA, tA = cup(bot, m_bot, WBALL_D, 4.75, WCUP_LIP, 1.0, CUP_SLITS, SLIT_W, top - bot)
        oB, tB = cup(top, m_top, WBALL_D, 4.75, WCUP_LIP, 1.0, CUP_SLITS, SLIT_W, bot - top)
        d = top - bot
        mesh = carve([oA, oB, mcyl(bot, top, 2.0)], tA + tB)
        P = P1 @ tf(frame(m_bot, d), bot)
        mesh.apply_transform(tf(frame(m_bot, d)).T @ tf(np.eye(3), -bot))
        add(f"Rod_Wrist_M{i}", "wrist", mesh, [P], up=(0, 0, 1),
            note=f"wrist push rod, {np.linalg.norm(d):.1f} mm between ball centres (cups snap onto the studs)")

    # heat-set inserts and M3 screws
    sl = MW.cut(cyl((0, 0, 0), (0, 0, 1), INSERT_OD / 2, 5.0), cyl((0, 0, -1), (0, 0, 1), INSERT_ID / 2, 7))
    add("Insert_M3", "wrist", to_mesh(sl), qty=14 - 2,
        note="M3 heat-set insert stand-in, pressed into the d4.0 insert holes (the bar's 2 take the ball studs)")
    add("Peg_M3x10", "wrist", to_mesh(peg(M3_PEG_D, 10.0, 5.5, 3.0)), qty=8, note="fork cap 4, elbow flange 4")
    add("Peg_M3x8_CSK", "wrist", to_mesh(peg(M3_PEG_D, 8.0, 6.0, 0, csk=True)), qty=4, note="servo frame")


# ============================================================================ test pieces
def build_tests():
    """Fit test (PLA): the real holes in a gauge block + the stand-ins in three sizes each (value, -0.1, +0.1)."""
    out = {}
    # hole gauge: one hole per size used in the real parts, vertical (as the cores print), small to large from the notch
    sizes = [1.50, 1.90, 1.95, 2.15, 2.20, 2.30, 2.70, 2.80, 2.90, 3.20, 4.00, 7.00, 8.15]
    pitch = [max(5.0, d + 3.0) for d in sizes]
    blk = MW.box(0, sum(pitch) + 4, 0, 10, 0, 6)
    tools, x = [MW.box(-1, 3, -1, 3, 4, 7)], 2.0          # corner notch next to the d1.5 hole
    for d, p in zip(sizes, pitch):
        tools.append(cyl((x + p / 2, 5, -1), (0, 0, 1), d / 2, 8))
        x += p
    out["Fit_Hole_Gauge"] = MW.cut(blk, *tools)
    for dd in (-0.1, 0.0, 0.1):
        out[f"Fit_Pin_D{PIN_D + dd:.2f}"] = pin(PIN_D + dd, 12)
        out[f"Fit_Axle_D{AXLE_D + dd:.2f}"] = pin(AXLE_D + dd, 12)
        out[f"Fit_Peg_D{PEG_D + dd:.2f}"] = peg(PEG_D + dd, 8, 4.2, 1.6)
        out[f"Fit_Pin3_D{PIN3_D + dd:.2f}"] = pin(PIN3_D + dd, 10)
        out[f"Fit_PegM3_D{M3_PEG_D + dd:.2f}"] = peg(M3_PEG_D + dd, 8, 5.5, 3.0)
        hub = MW.fuse(cyl((0, 0, 0), (0, 0, 1), 2.9, 4.5), MW.box(0, 9, -1.5, 1.5, 0, 1.0))
        out[f"Fit_Horn_Bore_D{HORN_BORE + dd:.2f}"] = MW.cut(hub, cyl((0, 0, -1), (0, 0, 1), (HORN_BORE + dd) / 2, 3.8))
        ring = MW.cut(cyl((0, 0, 0), (0, 0, 1), 5.0, 4.0), cyl((0, 0, -1), (0, 0, 1), (BALL_HOLE + dd) / 2, 6))
        out[f"Fit_Ball_Hole_D{BALL_HOLE + dd:.2f}"] = ring
        out[f"Fit_PegM25_D{M25_PEG_D + dd:.2f}"] = peg(M25_PEG_D + dd, 6.5, 4.5, 1.8)
        out[f"Fit_Arm_Shank_D{STUD_ARM_SHANK + dd:.2f}"] = pin(STUD_ARM_SHANK + dd, 8)
        out[f"Fit_Insert_OD{INSERT_OD + dd:.2f}"] = MW.cut(cyl((0, 0, 0), (0, 0, 1), (INSERT_OD + dd) / 2, 5.0),
                                                         cyl((0, 0, -1), (0, 0, 1), INSERT_ID / 2, 7))
        a = np.array([0, 0, 1.0])
        out[f"Fit_F683_OD{F683_OD + dd:.2f}"] = MW.cut(MW.fuse(cyl((0, 0, 0), a, wd.BRG_FD / 2, wd.BRG_FT),
                                                              cyl((0, 0, wd.BRG_FT), a, (F683_OD + dd) / 2, wd.BRG_B)),
                                                      cyl((0, 0, -1), a, F683_BORE / 2, 10))
        out[f"Fit_Disc_Bore_D{DISC_BORE + dd:.2f}"] = MW.cut(cyl((0, 0, 0), (0, 0, 1), 6.0, 3.0),
                                                            cyl((0, 0, -1), (0, 0, 1), (DISC_BORE + dd) / 2, 5))
    out["Fit_Servo_Stub"] = MW.fuse(MW.box(-6, 6, -6, 6, 0, 4), cyl((0, 0, 4), (0, 0, 1), STUB_D / 2, 3.2))
    out["Fit_STS_Stub"] = MW.fuse(MW.box(-8, 8, -8, 8, 0, 4), cyl((0, 0, 4), (0, 0, 1), STS_STUB_D / 2, 3.5))
    # ball posts and cups (three mouth heights each): the cup prints mouth up and snaps down onto the post's ball
    for bd, lip, rout, floor, tag in ((BALL_D, CUP_LIP, 3.75, 0.8, "Hand"), (WBALL_D, WCUP_LIP, 4.75, 1.0, "Wrist")):
        out[f"Fit_Ball_{tag}"] = MW.fuse(MW.box(-5, 5, -5, 5, 0, 3), cyl((0, 0, 3), (0, 0, 1), 1.1, 1.0 + bd / 2),
                                          sphere((0, 0, 4.0 + bd / 2), bd / 2))
        for dl in (-0.2, 0.0, 0.2):
            c = np.array([0, 0, bd / 2 + CUP_CLEAR + floor])
            o, t = cup(c, (0, 0, 1), bd, rout, lip + dl, floor, CUP_SLITS, SLIT_W, (1, 0, 0))
            arm = mf.Manifold.cube((rout + 10, 3.0, 3.0)).translate((0, -1.5, 0))
            out[f"Fit_Cup_{tag}_Lip{lip + dl:.1f}"] = carve([o, arm], t)
    return out


def build_switch_test():
    """PLA base with a TPU 95A ribbed pad on top: the two materials meet on one flat layer (z = 4 mm), so the print has
    exactly one filament change. (A pad sunk into a recess, like the palm pad, shares layers with the PLA round it
    and costs a change on every one of them.)"""
    base = MW.box(0, 40, 0, 20, 0, 4)
    pad = MW.fuse(MW.box(4, 36, 4, 16, 4, 5.2), *[MW.box(4.5, 35.5, y, y + 0.9, 5.2, 5.7) for y in np.arange(5.0, 15.0, 1.8)])
    return base, pad


# ============================================================================ checks
def check_fits(sim, parts):
    """Each placed stand-in vs the real printed parts near it: the deepest point of its surface inside one (mm)."""
    bought_wrist = ("wrist_f683", "wrist_pin_d3", "wrist_stud", "wrist_link", "wrist_rod", "wrist_sts3215")
    printed = [g for g in sim.geoms
               if not (g.mesh in ASM.STOCK and ASM.STOCK[g.mesh][1] is None)        # stock bought parts
               and not ASM.label_of(g).startswith("pin_2x")                          # the 3-phalanx fingers' pins
               and not g.mesh.startswith(bought_wrist)
               and g.mesh != "spacer"]                                               # replaced by the one-piece rods
    cache, rows = {}, {}
    for p in parts.values():
        if not p.places:
            continue
        pts0, _ = trimesh.sample.sample_surface(p.mesh, 1200, seed=1)
        worst = (0.0, "-")
        for P in p.places:
            pts = apply(P, pts0)
            lo, hi = pts.min(0) - 0.5, pts.max(0) + 0.5
            for g in printed:
                if g.name not in cache:
                    m = sim.placed(g.name)
                    if len(m.faces) > 40000:
                        m = m.simplify_quadric_decimation(face_count=40000)
                    cache[g.name] = m
                m = cache[g.name]
                if np.any(m.bounds[0] > hi) or np.any(m.bounds[1] < lo):
                    continue
                # > 0: inside the printed part. In chunks: on the big thin shells every point has thousands of
                # candidate triangles, and one call on all points took ~10 GB
                sd = np.concatenate([trimesh.proximity.signed_distance(m, pts[k:k + 100])
                                     for k in range(0, len(pts), 100)])
                m._cache.clear()
                if sd.max() > worst[0]:
                    worst = (float(sd.max()), g.name)
        rows[p.name] = worst
    return rows


# ============================================================================ main
def main():
    t0 = time.time()
    STL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    sim = load_sim()
    log(f"sim: {len(sim.geoms)} mesh geoms, {len(sim.motors)} finger motors")
    build_hand(sim)
    build_wrist(sim)
    log(f"{len(PARTS)} stand-ins built ({time.time() - t0:.0f} s)")
    for f in STL_DIR.glob("*.stl"):
        f.unlink()
    for p in PARTS.values():
        assert p.mesh.is_watertight, f"{p.name} is not watertight"
        p.mesh.export(STL_DIR / f"{p.name}.stl")
    tests = build_tests()
    for f in TEST_DIR.glob("*.stl"):
        f.unlink()
    for n, sh in tests.items():
        (sh if isinstance(sh, trimesh.Trimesh) else to_mesh(sh)).export(TEST_DIR / f"{n}.stl")
    pla, tpu = build_switch_test()
    MW.export_stl(pla, TEST_DIR / "Switch_PLA_Base.stl")
    MW.export_stl(tpu, TEST_DIR / "Switch_TPU_Pad.stl")
    log(f"{len(tests) + 2} test pieces")

    fits = check_fits(sim, PARTS)
    L = ["All-printed mock-up of AH_Right_5F_3S_Wrist: printed stand-ins for the bought parts "
         "(generated by make_mockup.py; do not edit)", "",
         "Fits (mm, diameter): " + ", ".join(f"{k} {v}" for k, v in FITS.items()), "",
         f"{'stand-in':24s} {'qty':>4s}  {'deepest overlap with a printed part':>36s}  note"]
    for p in sorted(PARTS.values(), key=lambda p: (p.group, p.name)):
        d, n = fits.get(p.name, (None, ""))
        ov = f"{d:6.2f} mm  {n[:26]:26s}" if d is not None else f"{'(no place in the sim)':>36s}"
        L.append(f"{p.name:24s} {p.qty:4d}  {ov}  {p.note}")
    (HERE / "report.txt").write_text("\n".join(L) + "\n")
    manifest = {p.name: {"group": p.group, "qty": p.qty, "up": p.up, "note": p.note} for p in PARTS.values()}
    (HERE / "stl" / "parts.json").write_text(json.dumps(manifest, indent=1))
    log("\n".join(L))
    log(f"done in {time.time() - t0:.0f} s")


FITS = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, float) and k not in ("FLAT",)}

if __name__ == "__main__":
    main()
