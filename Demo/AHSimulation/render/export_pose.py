#!/usr/bin/env python3
"""Pose the MuJoCo hand and export the visible geometry as a GLB for the Blender beauty renders.

Loads a scene, drives the hand to a pose (per-digit servo commands (m1, m2) in degrees) and the wrist to
(phi, psi) degrees, settles it in the simulation and writes
  <out>.glb   one node per visible geom, in MuJoCo world coordinates (metres, Z up; the GLB itself is Y up,
              Blender's glTF importer turns it back to Z up). Meshes are shared between geoms of the same mesh;
              the node transform places them. Every node carries a material named after its class.
  <out>.json  sidecar: per node its geom/mesh/body name, material class, effective rgba, group
              (hand = palm and everything on it, fingers = the moving digit bodies, arm = wrist + forearm) and
              digit (index / middle / ring / pinky / thumb, from the actuator that drives the body's chain);
              bounding boxes per group and a suggested camera target (the hand's bounding box).

Material classes: white_plastic (shells, finger covers, forearm), black_tpu (grip pads), metal (links, pins,
screws, bearings, cores), dark_plastic (servos, frames, plates), other.

Usage (project venv):
  python export_pose.py --pose OpenHand --wrist -15 0 --out /tmp/open.glb
  python export_pose.py --pose Peace --digit thumb=80,-80 --wrist -20 5 --out /tmp/peace.glb
  python export_pose.py --list-poses
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
AHSIM = HERE.parent                                  # Demo/AHSimulation
REPO = AHSIM.parent.parent
SIM_DIR = AHSIM / "AHSimulation"
DEFAULT_MODEL = "AH_Right_5F_3S_Wrist"
DEFAULT_POSES = AHSIM / "ultimate_finger" / "data" / "hand_poses.json"
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "wrist"))
import wrist_design as wd  # noqa: E402

DIGITS = {"index": "finger1", "middle": "finger2", "ring": "finger3", "thumb": "finger4", "pinky": "finger5"}

# Material class rules, first match wins, matched against "<mesh name> <geom name> <material name>" (lowercase).
CLASS_RULES = [
    ("black_tpu", r"pad"),
    ("white_plastic", r"shell|cover|forearm"),
    ("dark_plastic", r"scs0009|sts3215|servo|motor|frame|plate|flange|interface|lever|horn_arm"),
    ("metal", r"screw|torx|bolt|nut|pin|rod|bearing|f683|bushing|washer|rotule_ball|stud|spacer|link|_l2|gimbal"
              r"|fork|adapter|(^|_)p\d(\s|$)|^distal|^proximal|core|coupler"),
]

# Groups for framing: the hand root is the body carrying the palm shell (or, on a stock model, the hand plate).
HAND_ROOT_MESHES = ("palm_shell", "hand_plate")


def log(*a):
    print(*a, flush=True)


def load_poses(path):
    poses = {}
    try:  # the stock 5-digit demo poses, as render_showcase.py does
        sys.path.insert(0, str(AHSIM / "five_finger"))
        from verify_5f_model import DEMO_POSES_5F
        poses.update({k: {d: list(v) for d, v in p.items()} for k, p in DEMO_POSES_5F.items()})
    except Exception as e:  # noqa: BLE001 (optional source)
        log("note: DEMO_POSES_5F not available:", e)
    if path:
        poses.update(json.loads(Path(path).read_text()))
    return poses


def name(m, obj, i):
    return mujoco.mj_id2name(m, obj, i) if i >= 0 else None


def effective_rgba(m, g):
    """MuJoCo's own rule: the geom rgba unless it is the default (0.5 0.5 0.5 1) and the geom has a material."""
    rgba = m.geom_rgba[g].copy()
    mat = m.geom_matid[g]
    if mat >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        rgba = m.mat_rgba[mat].copy()
    return rgba


def classify(mesh, geom, mat, rgba):
    key = " ".join(s.lower() for s in (mesh or "", geom or "", mat or "") if s).strip()
    for cls, pat in CLASS_RULES:
        if re.search(pat, key):
            return cls
    lum = float(np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]))
    if lum > 0.88:
        return "white_plastic"
    return "other"


def settle(m, d, target, steps=1500, ramp=600):
    """Ramp the actuators from the keyframe to the target and let the closed chains settle."""
    mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m, d)
    for k in range(steps):
        d.ctrl[:] = target * min(1.0, (k + 1) / ramp)
        mujoco.mj_step(m, d)
    return float(np.abs(d.qvel).max())


def hidden_geoms(m):
    """Floor planes and the mocap target spheres (render_showcase.py hides them too)."""
    hide = set()
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or name(m, mujoco.mjtObj.mjOBJ_GEOM, g) == "floor":
            hide.add(g)
    for b in range(m.nbody):
        if m.body_mocapid[b] >= 0:
            hide.update(range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))
    return hide


def subtree(m, root):
    out = {root}
    for b in range(root + 1, m.nbody):
        if m.body_parentid[b] in out:
            out.add(b)
    return out


def primitive_mesh(m, g):
    t, s = m.geom_type[g], m.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=s[0])
    if t == G.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(height=2 * s[1], radius=s[0], count=[32, 16])
    if t == G.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=s[0], height=2 * s[1], sections=48)
    if t == G.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * s[:3])
    if t == G.mjGEOM_ELLIPSOID:
        me = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        me.vertices *= s[:3]
        return me
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"model dir under {SIM_DIR.relative_to(REPO)}")
    ap.add_argument("--scene", type=Path, help="scene XML (overrides --model)")
    ap.add_argument("--poses", type=Path, default=DEFAULT_POSES, help="JSON {pose: {digit: [m1, m2] deg}}")
    ap.add_argument("--pose", default="OpenHand")
    ap.add_argument("--digit", action="append", default=[], metavar="DIGIT=M1,M2",
                    help="override one digit's command (deg), e.g. thumb=60,-20; repeatable")
    ap.add_argument("--wrist", type=float, nargs=2, default=(0.0, 0.0), metavar=("PHI", "PSI"),
                    help="wrist flexion (> 0 palm-ward) and deviation (> 0 radial), deg")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", type=Path, help="output .glb (sidecar .json next to it)")
    ap.add_argument("--list-poses", action="store_true")
    args = ap.parse_args()

    poses = load_poses(args.poses)
    if args.list_poses:
        for k, v in poses.items():
            log(f"{k:<16}", "  ".join(f"{d}={tuple(c)}" for d, c in v.items()))
        return
    if args.out is None:
        ap.error("--out is required")
    if args.pose not in poses:
        ap.error(f"unknown pose {args.pose!r}; have {sorted(poses)}")
    cmd = {d: list(map(float, c)) for d, c in poses[args.pose].items()}
    for s in args.digit:
        dig, val = s.split("=")
        cmd[dig] = [float(v) for v in val.split(",")]

    scene = args.scene or SIM_DIR / args.model / "mjcf" / "scene_shells.xml"
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    act = {name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    target = np.zeros(m.nu)
    for dig, (a1, a2) in cmd.items():
        ids = [act.get(f"{DIGITS[dig]}_motor{k}") for k in (1, 2)]
        if None in ids:
            log(f"note: no actuators for {dig} ({DIGITS[dig]}_motor1/2) in this model; skipped")
            continue
        target[ids] = np.radians([a1, a2])
    phi, psi = wd.clamp_wrist(*np.radians(args.wrist))
    if "wrist_motor1" in act:
        target[[act["wrist_motor1"], act["wrist_motor2"]]] = wd.servo_ik(phi, psi)
    target = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]) \
        if m.actuator_ctrllimited.any() else target
    vmax = settle(m, d, target, args.steps)
    log(f"settled {args.pose} wrist ({np.degrees(phi):.1f}, {np.degrees(psi):.1f}) deg; max |qvel| {vmax:.2e}")

    # groups
    hand_root = None
    for g in range(m.ngeom):
        mn = name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else ""
        if mn and any(k in mn for k in HAND_ROOT_MESHES):
            hand_root = int(m.geom_bodyid[g])
            break
    hand_bodies = subtree(m, hand_root) if hand_root is not None else set(range(m.nbody))

    def group_of(b):
        if b not in hand_bodies:
            return "arm"
        return "hand" if b == hand_root or hand_root is None else "fingers"

    # digit of each finger body: the hand root's child chain that a finger<N>_motor actuator drives
    def top_child(b):
        while b > 0 and m.body_parentid[b] != hand_root:
            b = m.body_parentid[b]
        return b if b > 0 else None

    digit_of_top = {}
    for i in range(m.nu):
        an = name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        dig = next((k for k, v in DIGITS.items() if an.startswith(v + "_")), None)
        if dig and m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and hand_root is not None:
            tc = top_child(int(m.jnt_bodyid[m.actuator_trnid[i, 0]]))
            if tc is not None:
                digit_of_top[tc] = dig

    def digit_of(b):
        return digit_of_top.get(top_child(b)) if b in hand_bodies and b != hand_root else None

    # Z-up (MuJoCo, Blender) -> Y-up (glTF): (x, y, z) -> (x, z, -y)
    C = np.eye(4)
    C[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    sc = trimesh.Scene()
    hide = hidden_geoms(m)
    nodes, used_names, mesh_cache = [], set(), {}
    pts = {"hand": [], "fingers": [], "arm": []}
    ntri = 0
    for g in range(m.ngeom):
        if g in hide or m.geom_group[g] > 2:
            continue
        rgba = effective_rgba(m, g)
        if rgba[3] <= 0:
            continue
        t = m.geom_type[g]
        mesh_id = int(m.geom_dataid[g]) if t == mujoco.mjtGeom.mjGEOM_MESH else -1
        mname = name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else None
        gname = name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        matname = name(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[g])
        body = name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        cls = classify(mname, gname, matname, rgba)
        key = (mesh_id, cls, tuple(np.round(rgba, 3))) if mesh_id >= 0 else ("prim", g)
        if key not in mesh_cache:
            if mesh_id >= 0:
                va, vn = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
                fa, fn = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
                tm = trimesh.Trimesh(m.mesh_vert[va:va + vn].astype(np.float64), m.mesh_face[fa:fa + fn].copy(),
                                     process=False)
                tm.merge_vertices()  # shared vertices -> smooth shading in Blender
            else:
                tm = primitive_mesh(m, g)
                if tm is None:
                    log(f"skip geom {g} ({gname}): unsupported type {t}")
                    continue
            tm.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                name=cls, baseColorFactor=np.clip(np.round(rgba * 255), 0, 255).astype(np.uint8),
                metallicFactor=1.0 if cls == "metal" else 0.0, roughnessFactor=0.4))
            geom_key = f"m{len(mesh_cache):03d}_{re.sub(r'__[0-9a-f]{20,}', '', mname or 'prim')[:40]}"
            mesh_cache[key] = (geom_key, tm)
        geom_key, tm = mesh_cache[key]
        # short unique node names: Blender truncates object names at 63 bytes (the CAD screw names are longer)
        nn = f"g{g:03d}_{re.sub(r'__[0-9a-f]{20,}', '', gname or mname or 'prim')[:40]}"
        assert nn not in used_names
        used_names.add(nn)
        T = np.eye(4)
        T[:3, :3] = d.geom_xmat[g].reshape(3, 3)
        T[:3, 3] = d.geom_xpos[g]
        if geom_key not in sc.geometry:
            sc.add_geometry(tm, geom_name=geom_key, node_name=nn, transform=C @ T)
        else:
            sc.graph.update(frame_to=nn, frame_from=sc.graph.base_frame, matrix=C @ T, geometry=geom_key)
        grp = group_of(int(m.geom_bodyid[g]))
        pts[grp].append(tm.vertices @ T[:3, :3].T + T[:3, 3])
        ntri += len(tm.faces)
        nodes.append(dict(name=nn, geom=gname, mesh=mname, material=matname, body=body, group=grp,
                          digit=digit_of(int(m.geom_bodyid[g])), cls=cls, rgba=[round(float(x), 4) for x in rgba]))

    def bbox(arrs):
        if not arrs:
            return None
        P = np.concatenate(arrs)
        lo, hi = P.min(0), P.max(0)
        return dict(min=lo.round(5).tolist(), max=hi.round(5).tolist(), center=((lo + hi) / 2).round(5).tolist(),
                    size=(hi - lo).round(5).tolist())

    boxes = {"hand": bbox(pts["hand"] + pts["fingers"]), "fingers": bbox(pts["fingers"]), "arm": bbox(pts["arm"]),
             "all": bbox(pts["hand"] + pts["fingers"] + pts["arm"])}
    hand_frame = None
    if hand_root is not None:
        hand_frame = dict(pos=d.xpos[hand_root].round(6).tolist(),
                          xmat=d.xmat[hand_root].reshape(3, 3).round(6).tolist())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    glb = trimesh.exchange.gltf.export_glb(sc, include_normals=False)
    args.out.write_bytes(glb)
    side = dict(
        scene=str(Path(scene).resolve().relative_to(REPO)) if REPO in Path(scene).resolve().parents else str(scene),
        pose=args.pose, commands_deg=cmd,
        wrist_deg=[round(float(np.degrees(phi)), 3), round(float(np.degrees(psi)), 3)],
        units="m", up="+Z (MuJoCo world; the GLB is Y-up and Blender's importer restores Z-up)",
        camera_target=boxes["hand"]["center"] if boxes["hand"] else None, bbox=boxes, hand_frame=hand_frame,
        classes=sorted({n["cls"] for n in nodes}), nodes=nodes)
    args.out.with_suffix(".json").write_text(json.dumps(side, indent=1))
    from collections import Counter
    cnt = Counter(n["cls"] for n in nodes)
    log(f"wrote {args.out} ({len(glb) / 1e6:.1f} MB, {len(nodes)} nodes, {len(sc.geometry)} meshes, "
        f"{ntri / 1e6:.2f} M triangles) and {args.out.with_suffix('.json').name}; classes {dict(cnt)}")


if __name__ == "__main__":
    main()
