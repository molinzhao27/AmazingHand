#!/usr/bin/env python3
"""Assembled CAD of the final 5-digit right hand (sim model AH_Right_5F_3S_Wrist), every part in its built position.

The parts were generated one by one (CadQuery STEP + STL, mm, in the stock hand's CAD frame) and were only ever put
together inside the MuJoCo model. This script takes that assembly from the sim and writes it as CAD:

  * AH_Right_5F_3S_Wrist_assembly.step   every part that has a B-rep source, as a STEP assembly (one sub-assembly per
                                          digit + palm + wrist, parts named and coloured). Every part is its own
                                          product, so CAD tools that label parts by product (FreeCAD, Fusion) show
                                          index_P1_Proximal, middle_P1_Proximal, ... rather than one shared name
                                          (~90 MB; pcurves are kept: an importer that rebuilds them by projection
                                          changes the forearm's volume by 0.1 %);
  * AH_Right_5F_3S_Wrist_assembly.3mf    the whole build as meshes (exactly the sim meshes), same names and frame:
                                          the only place the mesh-only parts are (palm shells and pad, finger and
                                          thumb covers and pads, the fingertip cores UF_P3_Distal, and the bought
                                          servos, horns, screws, ball joints, bushings, rods and pins of the hand);
  * report.txt                            the part list with each part's source and its verified placement.
load_in_freecad.py (a FreeCAD macro) opens both with every part as its own object; FreeCAD's own 3MF import merges
the file into one mesh.

How: the model is loaded with mujoco, the `zero` keyframe applied (mj_forward), and every visible mesh geom's world
pose read (geom_xpos / geom_xmat). MuJoCo compiles each mesh into its inertial frame (v = R(mesh_quat)^T (s x -
mesh_pos)), so the pose of the mesh *file* is geom_pose . (R(mesh_quat)^T, -R^T mesh_pos), with the file scaled to mm
by s * 1000 (the custom parts are in mm with scale 0.001, the onshape-to-robot stock parts in m). The sim assets are
byte-identical copies of the cad/ STLs (matched by md5), and the STEP of a part is the one next to its STL (or, for
the stock parts in m, the cad/step part registered onto the sim STL by ICP; the wrist's bought-part envelopes are
rebuilt as solids by cad/five_finger/wrist/make_wrist.py). The assembly frame is the hand's CAD frame (the palm
plate's), in mm: x index -> pinky, y along the fingers, z towards the back of the hand.

Checks: every placed B-rep part is tessellated and compared with the sim geom's world vertices (sim -> CAD and CAD ->
sim distances); both written files are re-imported and every part is compared again with the sim (and, for the STEP,
with its volume before export).

Usage: .venv/bin/python cad/five_finger/assembly/make_assembly.py [--no-3mf]
"""

import argparse
import hashlib
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq
import mujoco
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Trsf
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MJCF = REPO / "Demo" / "AHSimulation" / "AHSimulation" / "AH_Right_5F_3S_Wrist" / "mjcf"
SCENE = MJCF / "scene_shells.xml"
KEYFRAME = "zero"
NAME = "AH_Right_5F_3S_Wrist_assembly"
OUT_STEP, OUT_3MF, REPORT = HERE / f"{NAME}.step", HERE / f"{NAME}.3mf", HERE / "report.txt"
CAD = REPO / "cad"
STOCK_STEP = CAD / "step"
sys.path.insert(0, str(CAD / "five_finger" / "wrist"))

TOL_SIM_TO_CAD = 0.05     # mm: the sim mesh's vertices must lie this close to the placed B-rep surface
TOL_CAD_TO_SIM = 1.0      # mm: the B-rep vs the sim mesh, which is the STL's chordal error (the cad/ STLs are
#                           exported with cadquery's relative tolerance: up to ~0.8 mm on the forearm); catches a
#                           STEP that has a feature its STL lacks
TOL_REIMPORT = 1e-3       # mm: a part read back from the written STEP / 3MF vs where it was put
TOL_VOLUME = 1e-4         # relative: a part's volume read back from the STEP (an inside-out face is >> 1 %)

DIGITS = {1: "index", 2: "middle", 3: "ring", 4: "thumb", 5: "pinky"}
GROUPS = ["palm", "index", "middle", "ring", "pinky", "thumb", "wrist"]

# stock parts (onshape-to-robot exports, in m, in their link frames) -> (name, STEP in cad/step or None)
STOCK = {
    "finger_frame_1": ("Finger_Frame-1", "Amazing Hand Parts - Finger_Frame-1.step"),
    "finger_frame_2": ("Finger_Frame-2", "Amazing Hand Parts - Finger_Frame-2.step"),
    "gimbal": ("Gimbal", "Amazing Hand Parts - Gimbal.step"),
    "link": ("Link", "Amazing Hand Parts - Link.step"),
    "spacer": ("Spacer", "BallJoint_Rod - Spacer.step"),
    "scs0009": ("SCS0009_Servo", None),
    "custom_servo_horn": ("Servo_Horn", None),
    "ph_pan_head_screw_m2x0_40_x_10__2803432263e518bbd16bccbbef8784ed": ("Screw_M2x10_Pan", None),
    "plain_washer_large_grade_a_m2_5__9a369f0dc77bf9c598cdf3fb468977e5": ("Washer_M2.5", None),
    "std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default": ("Screw_2.5x8_Torx", None),
    "thermo_screw_m2_5x5": ("Screw_2.5x5_Thermo", None),
    "bushing_0608_04": ("Bushing_0608", None),
    "m2_rod_l18": ("Rod_M2_L18", None),
    "rotule_ball": ("Rotule_Ball", None),
    "rotule_lever": ("Rotule_Lever", None),
    "parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999": ("Pin_2x10", None),
    "parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b": ("Pin_2x16", None),
}
# the palm-fixed parts of a digit (on the hand's root body): digit from the nearest servo shaft or finger base pivot
ROOT_DIGIT_PARTS = {"scs0009", "finger_frame_1", "finger_frame_2", "bushing_0608_04"}
# the sim splits the hand adapter: its interface plate is the hand's root mesh (r_wrist_interface), the neck rides on
# the wrist; both are one printed part, R-Wrist_Hand_Adapter
ADAPTER_PIECES = {"r_wrist_interface": "palm", "wrist_r_wrist_hand_adapter_neck": "neck"}
# cad STL name -> part name where they differ (the second horn arm is the same part, turned)
RENAME = {"Wrist_Horn_Arm_motor2_placed": "Wrist_Horn_Arm"}


def log(*a):
    print(*a, flush=True)


# ============================================================================ rigid transforms (4x4, mm)
def tf(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def inv(T):
    return tf(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def orthonormal(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])
    return tf(U @ Vt, T[:3, 3])


def quat_R(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def cq_location(T):
    T = orthonormal(T)
    t = gp_Trsf()
    t.SetValues(*T[0], *T[1], *T[2])
    return cq.Location(t)


# ============================================================================ point to mesh distance
def _closest_on_tris(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (n, 3) (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    dot = lambda u, v: np.einsum("ij,ij->i", u, v)   # noqa: E731
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        q = a + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]                            # face
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        q[m] = (b + (c - b) * w[:, None])[m]                                                     # edge BC
        w = d2 / (d2 - d6)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        q[m] = (a + ac * w[:, None])[m]                                                          # edge AC
        m = (d6 >= 0) & (d5 <= d6)
        q[m] = c[m]                                                                              # vertex C
        v = d1 / (d1 - d3)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        q[m] = (a + ab * v[:, None])[m]                                                          # edge AB
        m = (d3 >= 0) & (d4 <= d3)
        q[m] = b[m]                                                                              # vertex B
        m = (d1 <= 0) & (d2 <= 0)
        q[m] = a[m]                                                                              # vertex A
    bad = ~np.isfinite(q).all(1)                                                                 # degenerate triangle
    q[bad] = a[bad]
    return q


def _normals(tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


class Surface:
    """Closest points on a triangle mesh: the k nearest (by centroid) of the mesh split into small triangles, then,
    for points that come out farther than `exact_above`, every triangle whose bounding box is closer (exact)."""

    def __init__(self, V, F, max_edge, exact_above=0.004):
        V, F = np.asarray(V, float), np.asarray(F)
        self.tri0, self.normal0 = V[F], _normals(V[F])
        self.lo, self.hi = self.tri0.min(1), self.tri0.max(1)
        V, F = trimesh.remesh.subdivide_to_size(V, F, max_edge=max_edge, max_iter=30)
        self.tri, self.exact_above = V[F], exact_above
        self.normal = _normals(self.tri)
        self.tree = cKDTree(self.tri.mean(1))

    def closest(self, P, k=16):
        P = np.asarray(P, float)
        _, idx = self.tree.query(P, k=k)
        best = np.full(len(P), np.inf)
        q_best, n_best = np.zeros_like(P), np.zeros_like(P)
        for j in range(k):
            t = self.tri[idx[:, j]]
            q = _closest_on_tris(P, t[:, 0], t[:, 1], t[:, 2])
            d = np.linalg.norm(q - P, axis=1)
            m = d < best
            best[m], q_best[m], n_best[m] = d[m], q[m], self.normal[idx[m, j]]
        for i in np.nonzero(best > self.exact_above)[0]:
            p = P[i]
            c = np.nonzero(np.linalg.norm(np.maximum(0, np.maximum(self.lo - p, p - self.hi)), axis=1) < best[i])[0]
            if len(c):
                t = self.tri0[c]
                q = _closest_on_tris(np.repeat(p[None], len(c), 0), t[:, 0], t[:, 1], t[:, 2])
                d = np.linalg.norm(q - p, axis=1)
                j = d.argmin()
                if d[j] < best[i]:
                    best[i], q_best[i], n_best[i] = d[j], q[j], self.normal0[c[j]]
        return best, q_best, n_best


def edge_for(V):
    """Triangle size for the closest-point search: 0.6 mm on the small parts, up to 1.5 mm on the forearm."""
    return float(np.clip(np.linalg.norm(np.ptp(V, axis=0)) / 100, 0.6, 1.5))


def subsample(P, n, seed=0):
    if len(P) <= n:
        return P
    return P[np.random.default_rng(seed).choice(len(P), n, replace=False)]


# ============================================================================ registration (stock parts in m)
def _octahedral():
    out = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
            G = np.zeros((3, 3))
            G[range(3), perm] = s
            if np.linalg.det(G) > 0:
                out.append(G)
    return out


def _kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return tf(R, cb - R @ ca)


def register(cad_mesh, cad_surface, file_mesh):
    """Rigid T with file_mesh ~ T(cad_mesh): PCA starts over the 24 axis matchings, point-to-point ICP on surface
    samples, then point-to-plane ICP of the file's vertices (exact surface points) on the B-rep's tessellation."""
    A = trimesh.sample.sample_surface(cad_mesh, 6000, seed=1)[0]
    B = trimesh.sample.sample_surface(file_mesh, 6000, seed=2)[0]
    tree_b = cKDTree(B)

    def axes(P):
        w, U = np.linalg.eigh(np.cov((P - P.mean(0)).T))
        return U if np.linalg.det(U) > 0 else U * [1, 1, -1]

    Ua, Ub = axes(A), axes(B)
    best = (np.inf, None)
    for G in _octahedral():
        R = Ub @ G @ Ua.T
        T = tf(R, B.mean(0) - R @ A.mean(0))
        for _ in range(25):
            d, i = tree_b.query(apply(T, A))
            T = _kabsch(A, B[i])
        d, _ = tree_b.query(apply(T, A))
        if d.mean() < best[0]:
            best = (d.mean(), T)
    T = best[1]
    # point-to-plane refinement in the CAD frame: file vertices -> B-rep surface
    X = subsample(np.asarray(file_mesh.vertices), 20000)
    for _ in range(15):
        P = apply(inv(T), X)
        _, Q, N = cad_surface.closest(P)
        J = np.hstack([np.cross(P, N), N])
        r = np.einsum("ij,ij->i", P - Q, N)
        x = np.linalg.lstsq(J, -r, rcond=None)[0]
        dT = tf(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:])
        T = T @ inv(dT)
        if np.linalg.norm(x) < 1e-9:
            break
    return orthonormal(T)


# ============================================================================ the sim
@dataclass
class Geom:
    gid: int
    mesh: str
    asset: Path
    scale: float           # file units -> mm
    T_world: np.ndarray    # file (mm) -> world (mm)
    world_verts: np.ndarray
    faces: np.ndarray
    rgba: tuple
    body: str
    group: str = "palm"
    motor: int = 0
    cad_stl: Path = None
    T: np.ndarray = None   # file (mm) -> assembly frame (mm)
    name: str = ""


def load_sim():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME))
    mujoco.mj_forward(m, d)
    name = lambda t, i: mujoco.mj_id2name(m, t, i)   # noqa: E731
    # motor joints (digit, motor, world anchor) by body, and each finger's base pivot (its gimbal, joint "passive2 (N)")
    motors, pivots = {}, []
    for j in range(m.njnt):
        jn = name(mujoco.mjtObj.mjOBJ_JOINT, j)
        if jn.startswith("finger") and "_motor" in jn:
            motors[m.jnt_bodyid[j]] = (int(jn[6]), int(jn[-1]), d.xanchor[j] * 1000)
        elif jn.startswith("passive2 ("):
            pivots.append((int(jn[-2]), 0, d.xanchor[j] * 1000))
    geoms = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or (m.geom_rgba[g][3] == 0 and m.geom_matid[g] < 0):
            continue
        mid = m.geom_dataid[g]
        mesh = name(mujoco.mjtObj.mjOBJ_MESH, mid)
        s = float(m.mesh_scale[mid][0])
        assert np.allclose(m.mesh_scale[mid], s), mesh
        Rq, p = quat_R(m.mesh_quat[mid]), m.mesh_pos[mid]
        Rg, tg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        R = Rg @ Rq.T
        T_world = tf(R, 1000 * (tg - R @ p))
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        wv = (m.mesh_vert[a:a + n].astype(float) @ Rg.T + tg) * 1000
        rgba = tuple(m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g])
        asset = MJCF / "assets" / m.paths[m.mesh_pathadr[mid]:].split(b"\0", 1)[0].decode()
        geo = Geom(g, mesh, asset, s * 1000, T_world, wv, m.mesh_face[fa:fa + fn].copy(), rgba,
                   name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
        # digit: the nearest motor joint up the tree; the horn and rod bodies (motor body and its child) get its motor
        b, depth = m.geom_bodyid[g], 0
        while b and b not in motors:
            b, depth = m.body_parentid[b], depth + 1
        if b:
            geo.group, geo.motor = DIGITS[motors[b][0]], motors[b][1] if depth <= 1 else 0
        elif mesh in ROOT_DIGIT_PARTS:   # the nearest servo shaft (motor anchor) or finger base pivot
            tree = cKDTree(wv)
            k = min(list(motors.values()) + pivots, key=lambda v: tree.query(v[2])[0])
            geo.group, geo.motor = DIGITS[k[0]], k[1] if mesh == "scs0009" else 0
        elif mesh.startswith("wrist_") or mesh in ADAPTER_PIECES:
            geo.group = "wrist"
        geoms.append(geo)
    qpos = {name(mujoco.mjtObj.mjOBJ_JOINT, j): d.qpos[m.jnt_qposadr[j]] for j in range(m.njnt)
            if name(mujoco.mjtObj.mjOBJ_JOINT, j).startswith("wrist_") and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    return m, geoms, qpos


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def map_cad_stls(geoms):
    index = defaultdict(list)
    for f in sorted((CAD / "five_finger").rglob("*.stl")):
        index[md5(f)].append(f)
    for g in geoms:
        hits = index.get(md5(g.asset), [])
        if len(hits) > 1:   # identical files (e.g. the MCP pin is the same at x0.88): keep the scale's own name
            pk = g.mesh.startswith("ufpk_")
            hits = [h for h in hits if h.stem.endswith("_S088") == pk] or hits
        g.cad_stl = hits[0] if hits else None


# ============================================================================ sources
@dataclass
class Source:
    key: str
    label: str
    kind: str                      # "step" | "brep" (built here) | "mesh"
    path: Path = None
    shape: object = None           # cq.Shape (B-rep, in its own frame, mm)
    T_file_cad: np.ndarray = field(default_factory=lambda: np.eye(4))   # B-rep frame -> file frame (mm)
    registered: bool = False
    surface: Surface = None        # fine tessellation of the B-rep, for the checks
    tess: tuple = None


def import_step(path):
    vals = cq.importers.importStep(str(path)).vals()
    return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)


def tessellate(shape, tol=0.002, ang=0.05):
    """Triangulation with an absolute chordal deflection (cadquery's tessellate() uses a relative one), then the
    triangulation is dropped again so it is not written to the STEP."""
    BRepTools.Clean_s(shape.wrapped)
    BRepMesh_IncrementalMesh(shape.wrapped, tol, False, ang, True)
    V, F, off = [], [], 0
    for f in shape.Faces():
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
        if tri is None:
            continue
        T = loc.Transformation()
        V += [tri.Node(i).Transformed(T).Coord() for i in range(1, tri.NbNodes() + 1)]
        t = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1 + off
        F.append(t[:, [0, 2, 1]] if f.wrapped.Orientation() == TopAbs_REVERSED else t)
        off += tri.NbNodes()
    BRepTools.Clean_s(shape.wrapped)
    return np.array(V), np.vstack(F)


def label_of(g):
    assert g.mesh in STOCK or g.mesh in ADAPTER_PIECES or g.cad_stl is not None, f"no source for {g.mesh}"
    if g.mesh in STOCK:
        return STOCK[g.mesh][0]
    if g.mesh in ADAPTER_PIECES:
        return "R-Wrist_Hand_Adapter"
    stem = g.cad_stl.stem
    if g.mesh.startswith("thumb_"):
        return stem[len("Thumb_"):]
    if g.mesh.startswith(("uf_", "ufpk_")):
        return stem.replace("UF_", "") if stem.startswith(("UF_P", "UF_L2")) else stem
    if g.mesh.startswith("wrist_") and g.cad_stl.parent.name == "ref":
        return RENAME.get(stem, "wrist_" + stem)
    return RENAME.get(stem, stem)


def make_sources(geoms, bought):
    """One source per distinct part: its B-rep (if any) and the transform from the B-rep's frame to the file's."""
    srcs = {}
    for g in geoms:
        key = "R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.mesh
        if key in srcs:
            continue
        label = label_of(g)
        if key == "R-Wrist_Hand_Adapter":
            s = Source(key, label, "step", CAD / "five_finger" / "wrist" / "step" / "R-Wrist_Hand_Adapter.step")
        elif g.mesh in STOCK:
            step = STOCK[g.mesh][1]
            s = Source(key, label, "step", STOCK_STEP / step) if step else Source(key, label, "mesh")
            s.registered = bool(step)
        elif g.mesh == "wrist_wrist_horn_arm_motor2_placed":
            s = Source(key, label, "step", CAD / "five_finger" / "wrist" / "step" / "Wrist_Horn_Arm.step", registered=True)
        elif g.cad_stl is not None and g.cad_stl.parent.name == "ref" and g.cad_stl.stem in bought:
            s = Source(key, label, "brep", g.cad_stl.parent.parent.parent / "make_wrist.py",
                       shape=bought[g.cad_stl.stem])
        else:
            step = g.cad_stl.parent.parent / "step" / (g.cad_stl.stem + ".step")
            s = Source(key, label, "step", step) if step.exists() else Source(key, label, "mesh")
        srcs[key] = s
    return srcs


def load_breps(srcs, geoms):
    by_key = {}
    for g in geoms:
        by_key.setdefault("R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.mesh, g)
    for key, s in srcs.items():
        if s.kind == "mesh":
            continue
        t0 = time.time()
        if s.kind == "step":
            s.shape = import_step(s.path)
        V, F = tessellate(s.shape)
        s.tess = (V, F)
        s.surface = Surface(V, F, max_edge=edge_for(V))
        if s.registered:
            g = by_key[key]
            tm = trimesh.load(g.asset, process=False)
            file_mesh = trimesh.Trimesh(tm.vertices * g.scale, tm.faces)
            s.T_file_cad = register(trimesh.Trimesh(V, F), s.surface, file_mesh)
        log(f"  {s.label:28s} {s.kind:5s} {'registered' if s.registered else '':10s} {len(F):7d} tri  "
            f"{time.time() - t0:5.1f} s")


def wrist_envelopes(MW):
    """The wrist's bought-part envelopes as solids in the CAD frame (make_wrist.bought_parts_neutral). The ball studs
    are rebuilt with the ball's sphere axis along the stud (the same solid): with make_wrist's z-axis sphere, two of
    the four come back from a STEP inside out (the sphere face reverses on import, in OCCT and FreeCAD alike)."""
    wd, out = MW.wd, MW.bought_parts_neutral()

    def stud(seat, ball, flange_d=6.0):   # make_wrist.stud_env
        seat, ball = np.asarray(seat, float), np.asarray(ball, float)
        d = ball - seat
        sphere = cq.Solid.makeSphere(5.8 / 2, cq.Vector(*ball), cq.Vector(*d), -90, 90, 360)
        return MW.fuse(MW.cyl(seat, d, flange_d / 2, 1.5), MW.cyl(seat, d, 1.75, np.linalg.norm(d)), sphere)

    arm_in = wd.X_TOP_FACE + wd.HORN_GAP + wd.HORN_T
    for i in (1, 2):
        s, bot = wd.SIDE[i], wd.ball_bottom(i, 0.0)
        out[f"stud_top_{i}"] = stud((s * wd.BAR_X, 0, wd.E_LEVER), wd.P_TOP_HAND[i])
        out[f"stud_arm_{i}"] = stud((s * arm_in, bot[1], bot[2]), bot)
    return {k: MW.w_to_cad(v) for k, v in out.items()}


# ============================================================================ names
def assign_names(geoms):
    for g in geoms:
        base = label_of(g)
        if g.group in DIGITS.values():
            base = f"{g.group}_{base}" + (f"_M{g.motor}" if g.motor else "")
        g.name = base
    count = Counter(g.name for g in geoms if g.mesh not in ADAPTER_PIECES)
    seen = Counter()
    for g in sorted(geoms, key=lambda g: g.gid):
        if g.mesh in ADAPTER_PIECES or count[g.name] == 1:
            continue
        seen[g.name] += 1
        g.name = f"{g.name}_{seen[g.name]}"


# ============================================================================ STEP
def build_step(geoms, srcs):
    top = cq.Assembly(name="AH_Right_5F_3S_Wrist")
    subs = {grp: cq.Assembly(name=grp) for grp in GROUPS}
    placed, used = {}, set()   # part name -> (source, T: B-rep frame -> assembly, geom)
    for g in geoms:
        key = "R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.mesh
        s = srcs[key]
        if s.kind == "mesh" or g.name in placed:
            continue
        T = g.T @ s.T_file_cad
        placed[g.name] = (s, T, g)
        shape = s.shape.copy() if key in used else s.shape   # a copy is a product of its own (named per part)
        used.add(key)
        subs[g.group].add(shape, name=g.name, loc=cq_location(T), color=cq.Color(*map(float, g.rgba)))
    for grp in GROUPS:
        if subs[grp].children:
            top.add(subs[grp])
    top.export(str(OUT_STEP))
    return placed


def check_step_reimport(placed):
    """Read the STEP back (assembly structure, names, locations) and compare every part's vertices and volume."""
    asm = cq.Assembly.importStep(str(OUT_STEP))
    leaves = {}

    def walk(a, loc):
        loc = loc * a.loc
        if a.obj is not None and not a.children:
            leaves[a.name] = (a.obj, loc)
        for c in a.children:
            walk(c, loc)

    walk(asm, cq.Location())
    worst, worst_vol, missing = 0.0, 0.0, sorted(set(placed) - set(leaves))
    for n, (s, T, g) in placed.items():
        if n not in leaves:
            continue
        obj, loc = leaves[n]
        shape = obj.val() if isinstance(obj, cq.Workplane) else obj
        P0 = np.array([v.toTuple() for v in s.shape.Vertices()])
        P1 = np.array([v.toTuple() for v in shape.moved(loc).Vertices()])
        P0 = apply(T, P0)
        worst_vol = max(worst_vol, abs(shape.Volume() / s.shape.Volume() - 1))
        if len(P0) != len(P1):
            worst = np.inf
            continue
        worst = max(worst, float(cKDTree(P1).query(P0)[0].max()))
    return len(leaves), missing, sorted(set(leaves) - set(placed)), worst, worst_vol


# ============================================================================ 3MF
def hexcol(rgba):
    return "#" + "".join(f"{int(round(255 * c)):02X}" for c in rgba[:3])


def fmt_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return " ".join(f"{v:.9g}" for v in list(R.T.ravel()) + list(t))


def write_3mf(geoms, srcs):
    """Mesh objects (one per distinct sim mesh and colour, in the file's frame, mm) and one named object per part
    whose components place them; build items reference the named objects."""
    mats, objs, parts = {}, {}, defaultdict(list)
    for g in geoms:
        key = "R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.name
        c = hexcol(g.rgba)
        mats.setdefault(c, len(mats))
        objs.setdefault((g.mesh, c), None)
        parts[key].append(g)
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
           f'<metadata name="Title">{NAME}</metadata>',
           '<metadata name="Application">cad/five_finger/assembly/make_assembly.py</metadata>',
           "<resources>", '<basematerials id="1">']
    xml += [f'<base name="{c}" displaycolor="{c}"/>' for c in mats]
    xml.append("</basematerials>")
    oid = 2
    by_mesh = {g.mesh: g for g in geoms}
    for (mesh, c) in objs:
        g = by_mesh[mesh]
        tm = trimesh.load(g.asset, process=True)
        V, F = tm.vertices * g.scale, tm.faces
        objs[(mesh, c)] = oid
        src = srcs["R-Wrist_Hand_Adapter" if mesh in ADAPTER_PIECES else mesh]
        oname = (src.label + "_" + ADAPTER_PIECES[mesh]) if mesh in ADAPTER_PIECES else src.label
        xml.append(f'<object id="{oid}" type="model" name="{oname}" pid="1" pindex="{mats[c]}"><mesh><vertices>')
        xml.append("\n".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in V))
        xml.append("</vertices><triangles>")
        xml.append("\n".join(f'<triangle v1="{a}" v2="{b}" v3="{c_}"/>' for a, b, c_ in F))
        xml.append("</triangles></mesh></object>")
        oid += 1
    items = []
    for grp in GROUPS:
        for key, gs in parts.items():
            if gs[0].group != grp:
                continue
            comps = "".join(f'<component objectid="{objs[(g.mesh, hexcol(g.rgba))]}" transform="{fmt_T(g.T)}"/>'
                            for g in gs)
            xml.append(f'<object id="{oid}" type="model" name="{key}"><components>{comps}</components></object>')
            items.append(f'<item objectid="{oid}"/>')
            oid += 1
    xml += ["</resources>", "<build>"] + items + ["</build>", "</model>"]
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
          'relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-'
          '3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/'
            '2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/'
            '3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(OUT_3MF, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", "\n".join(xml))
    return len(parts), len(objs)


def read_3mf(path):
    """Minimal 3MF reader (core spec, one level of components): {part name: placed vertices}."""
    ns = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"

    def T3(attr):   # "m00 m01 m02 m10 ... m32": row-vector convention, p' = [p 1] M
        T = np.eye(4)
        if attr:
            T[:3, :] = np.array(attr.split(), float).reshape(4, 3).T
        return T

    names, meshes, comps, items, verts, cur = {}, {}, defaultdict(list), [], [], None
    with zipfile.ZipFile(path) as z, z.open("3D/3dmodel.model") as f:
        for ev, el in ET.iterparse(f, events=("start", "end")):
            tag = el.tag.replace(ns, "")
            if ev == "start":
                if tag == "object":
                    cur, verts = el.get("id"), []
                    names[cur] = el.get("name")
                continue
            if tag == "vertex":
                verts.append((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
            elif tag == "component":
                comps[cur].append((el.get("objectid"), T3(el.get("transform"))))
            elif tag == "object" and verts:
                meshes[cur] = np.array(verts)
            elif tag == "item":
                items.append((el.get("objectid"), T3(el.get("transform"))))
            if tag in ("vertex", "triangle", "mesh", "vertices", "triangles"):
                el.clear()
    out = {}
    for oid, T in items:
        out[names[oid]] = np.vstack([apply(T @ Tc, meshes[c]) for c, Tc in comps[oid]] if comps[oid]
                                    else [apply(T, meshes[oid])])
    return out, len(meshes)


def check_3mf_reimport(geoms):
    """Read the 3MF back and compare every part (by name) with the sim geoms' world vertices."""
    parts, n_mesh = read_3mf(OUT_3MF)
    worst, missing = 0.0, []
    for g in geoms:
        key = "R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.name
        if key not in parts:
            missing.append(key)
            continue
        P = apply(ASM_FROM_WORLD, g.world_verts)
        worst = max(worst, float(cKDTree(parts[key]).query(P)[0].max()))
    return len(parts), n_mesh, missing, worst


# ============================================================================ checks
def check_placement(placed, srcs, geoms):
    """For every placed B-rep: the sim geom's world vertices -> the B-rep surface, and the B-rep's tessellation ->
    the sim mesh, both in the B-rep's frame."""
    rows = {}
    pieces = defaultdict(list)
    for g in geoms:
        pieces["R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.name].append(g)
    for n, (s, T, g0) in placed.items():
        to_cad = inv(T) @ ASM_FROM_WORLD
        sim_pts, sim_V, sim_F, off = [], [], [], 0
        for g in pieces[n]:
            W = apply(to_cad, g.world_verts)
            # the sim's adapter pieces are capped at their cut (CAD y = MOUNT_Y); the cap is not a surface of the part
            keep = np.abs(W[:, 1] - ADAPTER_CUT_Y) > 1e-3 if g.mesh in ADAPTER_PIECES else slice(None)
            sim_pts.append(subsample(W[keep], 30000))
            sim_V.append(W)
            sim_F.append(g.faces + off)
            off += len(W)
        d1, _, _ = s.surface.closest(np.vstack(sim_pts))
        sim_surf = Surface(np.vstack(sim_V), np.vstack(sim_F), max_edge=edge_for(s.tess[0]))
        d2, _, _ = sim_surf.closest(subsample(s.tess[0], 30000))
        rows[n] = (float(d1.max()), float(np.percentile(d1, 99)), float(d2.max()))
    return rows


def check_meshes(geoms):
    """Mesh parts: the file placed by its 3MF transform vs the sim geom's world vertices."""
    worst = 0.0
    for g in geoms:
        tm = trimesh.load(g.asset, process=False)
        P = apply(g.T, tm.vertices * g.scale)
        W = apply(ASM_FROM_WORLD, g.world_verts)
        worst = max(worst, float(cKDTree(W).query(P)[0].max()))
    return worst


# ============================================================================ main
ASM_FROM_WORLD = np.eye(4)
ADAPTER_CUT_Y = None


def main():
    global ASM_FROM_WORLD, ADAPTER_CUT_Y
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-3mf", action="store_true", help="skip the 3MF (mesh) companion")
    args = ap.parse_args()
    t0 = time.time()
    import make_wrist as MW   # cad/five_finger/wrist: the bought-part envelopes as solids, in the CAD frame
    bought = wrist_envelopes(MW)
    ADAPTER_CUT_Y = MW.wd.MOUNT_Y

    m, geoms, wrist_q = load_sim()
    log(f"{SCENE.relative_to(REPO)}: keyframe '{KEYFRAME}', {len(geoms)} mesh geoms")
    map_cad_stls(geoms)
    plate = next(g for g in geoms if g.mesh == "r_hand_plate")
    ASM_FROM_WORLD = inv(plate.T_world)
    for g in geoms:
        g.T = ASM_FROM_WORLD @ g.T_world
    assign_names(geoms)
    srcs = make_sources(geoms, bought)
    log(f"{len(srcs)} distinct parts; B-rep sources:")
    load_breps(srcs, geoms)

    placed = build_step(geoms, srcs)
    log(f"wrote {OUT_STEP.relative_to(REPO)} ({OUT_STEP.stat().st_size / 1e6:.1f} MB, {len(placed)} parts)")
    n_leaves, missing, extra, worst_step, worst_vol = check_step_reimport(placed)
    log(f"  re-imported: {n_leaves} parts, missing {missing}, extra {extra}, vertices moved <= {worst_step:.2g} mm, "
        f"volumes within {worst_vol:.1e}")

    rows = check_placement(placed, srcs, geoms)
    worst_mesh = check_meshes(geoms)
    n_parts_3mf = n_obj = n_back_3mf = worst_3mf = None
    missing_3mf = []
    if not args.no_3mf:
        n_parts_3mf, n_obj = write_3mf(geoms, srcs)
        log(f"wrote {OUT_3MF.relative_to(REPO)} ({OUT_3MF.stat().st_size / 1e6:.1f} MB, {n_parts_3mf} parts, "
            f"{n_obj} meshes)")
        n_back_3mf, n_mesh_back, missing_3mf, worst_3mf = check_3mf_reimport(geoms)
        log(f"  re-imported: {n_back_3mf} parts ({n_mesh_back} meshes), missing {missing_3mf}, "
            f"sim vertices within {worst_3mf:.2g} mm")

    # ---------------------------------------------------------------- report
    bad = [n for n, r in rows.items() if r[0] > TOL_SIM_TO_CAD or r[2] > TOL_CAD_TO_SIM]
    parts = {}
    for g in geoms:
        parts.setdefault("R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.name, g)
    mesh_only = sorted(n for n in parts if n not in placed)
    L = [f"Assembled CAD of {MJCF.parent.name} (generated by make_assembly.py; do not edit)", "",
         f"Sim: {SCENE.relative_to(REPO)}, keyframe '{KEYFRAME}' (wrist at "
         + ", ".join(f"{k} {np.degrees(v):+.3f} deg" for k, v in wrist_q.items()) + ").",
         "Frame: the hand's CAD frame (the palm plate's, as every part in cad/), mm: x index -> pinky, y along the "
         "fingers, z towards the back.",
         f"{len(geoms)} sim mesh geoms = {len(parts)} parts (the hand adapter is split in the sim: interface plate + neck).",
         "",
         f"{OUT_STEP.name}: {len(placed)} parts as B-rep (STEP of the part, or the solid it is built from), in "
         "sub-assemblies " + ", ".join(GROUPS) + ".",
         f"  re-imported: {n_leaves} parts, names " + ("all match" if not missing and not extra else
                                                      f"missing {missing}, extra {extra}")
         + f", vertices within {worst_step:.1e} mm of where they were put, volumes within {worst_vol:.1e} (relative).",
         ]
    if not args.no_3mf:
        L += [f"{OUT_3MF.name}: all {n_parts_3mf} parts as meshes (the sim meshes, {n_obj} distinct), same names and "
              "frame.",
              f"  re-imported: {n_back_3mf} parts" + (f", missing {missing_3mf}" if missing_3mf else "")
              + f"; every sim vertex within {worst_3mf:.1e} mm of its part's vertices."]
    L += ["",
          f"Placement check (B-rep parts): sim -> CAD = distance of the sim mesh's world vertices to the placed B-rep "
          f"surface, tessellated to 0.002 mm (max / 99th percentile, limit {TOL_SIM_TO_CAD} mm); CAD -> sim = the "
          f"B-rep's surface to the sim mesh (max, limit {TOL_CAD_TO_SIM} mm): mostly the sim STL's own chordal "
          "error (the cad/ STLs are exported with cadquery's relative tolerance), it would catch a STEP feature that "
          "the STL lacks.",
          f"Mesh parts: the file placed by its 3MF transform vs the sim's world vertices: <= {worst_mesh:.1e} mm.",
          "Result: " + ("every B-rep part is in place." if not bad else f"OUT OF TOLERANCE: {bad}"), "",
          f"{'part':42s} {'in':5s} {'source':66s} {'sim->CAD max/p99':>17s} {'CAD->sim':>9s}"]
    for grp in GROUPS:
        for n, g in sorted(parts.items()):
            if g.group != grp:
                continue
            key = "R-Wrist_Hand_Adapter" if g.mesh in ADAPTER_PIECES else g.mesh
            s = srcs[key]
            if n in placed:
                where = "STEP"
                src = str(s.path.relative_to(REPO)) + (" (ICP onto the sim STL)" if s.registered else "")
                r = rows[n]
                dev = f"{r[0]:8.4f} /{r[1]:7.4f} {r[2]:9.4f}"
            else:
                where = "3MF"
                src = str((g.cad_stl or g.asset).relative_to(REPO)) + (" (m)" if g.scale == 1000 else "")
                dev = ""
            L.append(f"{n:42s} {where:5s} {src:66s} {dev}")
    kinds = Counter(label_of(parts[n]) for n in mesh_only)
    L += ["", f"Mesh-only parts ({len(mesh_only)}, in the 3MF only; no B-rep of them exists in the repo: the shells, "
          "covers and pads are implicit/voxel solids, UF_P3_Distal is trimmed by a mesh boolean, the hand's bought "
          "parts are the onshape-to-robot meshes):",
          "  " + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())), ""]
    REPORT.write_text("\n".join(L) + "\n")
    log(f"wrote {REPORT.relative_to(REPO)}; {len(placed)} B-rep parts, {len(mesh_only)} mesh-only; "
        f"worst sim->CAD {max(r[0] for r in rows.values()):.4f} mm, CAD->sim {max(r[2] for r in rows.values()):.4f} mm"
        f"{'; OUT OF TOLERANCE: ' + str(bad) if bad else ''}; {time.time() - t0:.0f} s")
    fail = (bad or missing or extra or worst_step > TOL_REIMPORT or worst_vol > TOL_VOLUME or missing_3mf
            or (worst_3mf or 0) > TOL_REIMPORT)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
