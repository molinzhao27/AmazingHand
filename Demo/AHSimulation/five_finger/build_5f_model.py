#!/usr/bin/env python3
"""Generate the 5-digit (4 fingers + thumb) right-hand MJCF from the stock AH_Right model.

The pinky (finger5) is a rigid copy of the ring finger (finger3): every finger3 part that is
fixed to the palm (servos, finger frames, bushings, closing sites) and both of its motor chains
are copied and moved by the same rigid transform that takes the middle finger (finger2) onto the
ring finger (finger3), i.e. a 7.5 deg rotation about the palm-normal (x) axis plus the matching
translation. The pinky therefore relates to the ring finger exactly the way the ring finger
relates to the middle finger in the stock hand.

finger5 is appended after the thumb, so qpos[0:68], ctrl[0:8] and mocap[0:4] keep the stock
layout; finger5 adds qpos[68:85], ctrl[8:10] and mocap[4].

Usage:  python build_5f_model.py            (writes AHSimulation/AH_Right_5F/mjcf)
        python build_5f_model.py --keyframe (also regenerates keyframes.xml by settling at ctrl=0)
"""

import argparse
import copy
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "AHSimulation"
SRC = SIM_DIR / "AH_Right" / "mjcf"
DST = SIM_DIR / "AH_Right_5F" / "mjcf"

# Palm plate mesh used by the 5-finger model. Swapped for the widened plate once it exists.
PLATE_MESH = "r_hand_plate_5f.stl"
PLATE_MESH_SRC = HERE.parent.parent.parent / "cad" / "five_finger" / "stl" / "R-Hand_Plate_5F.stl"

# Root-level parts (identified by their "Part ..." comment) of finger2 and finger3, in matching order.
F2_PARTS = ["scs0009_5", "scs0009_3", "finger_frame_1_2", "finger_frame_2", "bushing_0608_04", "bushing_0608_04_4"]
F3_PARTS = ["scs0009_7", "scs0009_6", "finger_frame_1_3", "finger_frame_2_2", "bushing_0608_04_3", "bushing_0608_04_7"]
# Instance names for the new finger5 parts (continuing the onshape-to-robot numbering).
F5_PARTS = ["scs0009_9", "scs0009_10", "finger_frame_1_5", "finger_frame_2_5", "bushing_0608_04_9", "bushing_0608_04_10"]

F2_SITES = ["closing_1 (2)_2", "closing_1 (2)_2_z"]
F3_SITES = ["closing_1 (3)_2", "closing_1 (3)_2_z"]

# Top-level bodies of each chain: (motor1 chain, motor2 chain).
F2_BODIES = ["custom_servo_horn_2", "rotule_ball_4"]
F3_BODIES = ["custom_servo_horn_3", "rotule_ball_6"]

# Every named element in finger3's chains and root sites -> its finger5 name.
RENAME = {
    # bodies
    "custom_servo_horn_3": "custom_servo_horn_5",
    "ball_link_4": "ball_link_8",
    "rotule_ball_5": "rotule_ball_9",
    "std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default_3": "std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default_5",
    "parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999_3": "parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999_5",
    "parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b_3": "parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b_5",
    "rotule_ball_6": "rotule_ball_10",
    "ball_link_5": "ball_link_9",
    # joints
    "finger3_motor1": "finger5_motor1",
    "finger3_motor2": "finger5_motor2",
    "passive_ball7": "passive_ball13",
    "passive_ball8": "passive_ball14",
    "passive_ball9": "passive_ball15",
    "passive2 (3)": "passive2 (5)",
    "passive4 (3)": "passive4 (5)",
    "passive_5 (3)": "passive_5 (5)",
    # sites
    "closing_1 (3)_1": "closing_1 (5)_1",
    "closing_1 (3)_1_z": "closing_1 (5)_1_z",
    "closing_1 (3)_2": "closing_1 (5)_2",
    "closing_1 (3)_2_z": "closing_1 (5)_2_z",
    "closing_3 (3)_1": "closing_3 (5)_1",
    "closing_3 (3)_1_z": "closing_3 (5)_1_z",
    "closing_3 (3)_2": "closing_3 (5)_2",
    "closing_3 (3)_2_z": "closing_3 (5)_2_z",
    "closing_ball3_1": "closing_ball5_1",
    "closing_ball3_2": "closing_ball5_2",
    "tip3": "tip5",
}
RENAME_RE = re.compile(
    "|".join(rf"(?<![\w(]){re.escape(k)}(?![\w)])" for k in sorted(RENAME, key=len, reverse=True))
)


# ---------------------------------------------------------------- pose helpers
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


def get_pose(el):
    pos = np.array([float(x) for x in el.get("pos", "0 0 0").split()])
    quat = np.array([float(x) for x in el.get("quat", "1 0 0 0").split()])
    return pos, quat / np.linalg.norm(quat)


def apply(T, pose):
    (tD, qD), (p, q) = T, pose
    return rotate(qD, p) + tD, quat_mul(qD, q)


def same_pose(a, b, tol=2e-5):
    (pa, qa), (pb, qb) = a, b
    return np.linalg.norm(pa - pb) < tol and min(np.linalg.norm(qa - qb), np.linalg.norm(qa + qb)) < tol


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def set_pose(el, pose):
    pos, quat = pose
    el.set("pos", fmt(pos))
    el.set("quat", fmt(quat))


# ---------------------------------------------------------------- xml helpers
def parse(path):
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def is_comment(el):
    return el.tag is ET.Comment


def part_geoms(body):
    """Map 'Part <name>' comments to the geom that follows them (direct children only)."""
    out, children = {}, list(body)
    for c, nxt in zip(children, children[1:]):
        if is_comment(c) and c.text.strip().startswith("Part ") and nxt.tag == "geom":
            out[c.text.strip()[5:]] = (c, nxt)
    return out


def child_by_name(parent, tag, name):
    for el in parent:
        if el.tag == tag and el.get("name") == name:
            return el
    raise KeyError(f"<{tag} name='{name}'> not found")


def preceding_comment(parent, el):
    children = list(parent)
    i = children.index(el)
    return children[i - 1] if i > 0 and is_comment(children[i - 1]) else None


def rename_subtree(el):
    for sub in el.iter():
        if is_comment(sub):
            text = sub.text
            if text.strip().startswith("Part "):
                sub.text = f" Part {text.strip()[5:]} (finger5 copy) "
            else:
                sub.text = RENAME_RE.sub(lambda m: RENAME[m.group(0)], text)
        elif sub.get("name") is not None:
            name = sub.get("name")
            if name not in RENAME:
                raise KeyError(f"no finger5 name for '{name}'")
            sub.set("name", RENAME[name])
        for attr in ("site1", "site2", "joint"):
            if sub.get(attr) in RENAME:
                sub.set(attr, RENAME[sub.get(attr)])


def indent(tree):
    ET.indent(tree, space="  ")


# ---------------------------------------------------------------- build
def finger_step_transform(root_body):
    """Rigid transform taking finger2 onto finger3, checked on every fixed part and chain root."""
    geoms = part_geoms(root_body)
    p2 = get_pose(geoms["finger_frame_1_2"][1])
    p3 = get_pose(geoms["finger_frame_1_3"][1])
    qD = quat_mul(p3[1], quat_conj(p2[1]))
    tD = p3[0] - rotate(qD, p2[0])
    T = (tD, qD)
    pairs = [(geoms[a][1], geoms[b][1]) for a, b in zip(F2_PARTS, F3_PARTS)]
    pairs += [(child_by_name(root_body, "site", a), child_by_name(root_body, "site", b)) for a, b in zip(F2_SITES, F3_SITES)]
    pairs += [(child_by_name(root_body, "body", a), child_by_name(root_body, "body", b)) for a, b in zip(F2_BODIES, F3_BODIES)]
    for e2, e3 in pairs:
        assert same_pose(apply(T, get_pose(e2)), get_pose(e3)), f"finger2->finger3 step is not rigid for {e2.attrib}"
    angle = 2 * np.degrees(np.arctan2(np.linalg.norm(qD[1:]), qD[0]))
    print(f"finger2->finger3 step: {angle:.3f} deg about {np.round(qD[1:] / np.linalg.norm(qD[1:]), 4)}, "
          f"t = {np.round(tD * 1000, 3)} mm (verified on {len(pairs)} elements)")
    return T


def build_robot(plate_mesh):
    tree = parse(SRC / "robot.xml")
    mj = tree.getroot()
    root_body = mj.find("worldbody/body")
    assert root_body.get("name") == "r_wrist_interface"
    T = finger_step_transform(root_body)

    geoms = part_geoms(root_body)
    children = list(root_body)
    first_body = next(i for i, c in enumerate(children) if c.tag == "body")
    insert_at = first_body - 1  # before the comment that precedes the first body

    new_fixed = [ET.Comment(" finger5 (pinky): fixed parts, copied from finger3 and moved by the finger2->finger3 step ")]
    for f3, f5 in zip(F3_PARTS, F5_PARTS):
        _, geom = geoms[f3]
        g = copy.deepcopy(geom)
        set_pose(g, apply(T, get_pose(geom)))
        new_fixed += [ET.Comment(f" Part {f5} "), g]
    for name in F3_SITES:
        site = child_by_name(root_body, "site", name)
        s = copy.deepcopy(site)
        set_pose(s, apply(T, get_pose(site)))
        rename_subtree(s)
        new_fixed += [ET.Comment(f" Frame {s.get('name')} "), s]
    for i, el in enumerate(new_fixed):
        root_body.insert(insert_at + i, el)

    root_body.append(ET.Comment(" finger5 (pinky): motor chains, copied from finger3 and moved by the finger2->finger3 step "))
    for name in F3_BODIES:
        body = child_by_name(root_body, "body", name)
        b = copy.deepcopy(body)
        set_pose(b, apply(T, get_pose(body)))
        rename_subtree(b)
        root_body.append(ET.Comment(f" Link {b.get('name')} "))
        root_body.append(b)

    actuator = mj.find("actuator")
    for name in ("finger3_motor1", "finger3_motor2"):
        a = copy.deepcopy(child_by_name(actuator, "position", name))
        rename_subtree(a)
        actuator.append(a)

    equality = mj.find("equality")
    eqs = list(equality)
    last_closing = max(i for i, e in enumerate(eqs) if e.get("site1").startswith("closing_") and "ball" not in e.get("site1"))
    f3_closing = [e for e in eqs if "(3)" in e.get("site1")]
    f3_ball = [e for e in eqs if e.get("site1") == "closing_ball3_1"]
    for k, e in enumerate(f3_closing):
        c = copy.deepcopy(e)
        rename_subtree(c)
        equality.insert(last_closing + 1 + k, c)
    for e in f3_ball:
        c = copy.deepcopy(e)
        rename_subtree(c)
        equality.append(c)

    if plate_mesh:
        for mesh in mj.find("asset").iter("mesh"):
            if mesh.get("file") == "r_hand_plate.stl":
                mesh.set("name", "r_hand_plate")
                mesh.set("file", plate_mesh)

    indent(tree)
    text = ET.tostring(mj, encoding="unicode").replace(" />", "/>")
    header = (
        '<?xml version="1.0" ?>\n'
        "<!-- Generated using onshape-to-robot -->\n"
        "<!-- Onshape https://cad.onshape.com/documents/430ff184cf3dd9557aaff2be/w/e3658b7152c139971d22c688/e/a5789448a382c9e7e5c5750a -->\n"
        "<!-- 5-digit variant: finger5 (pinky) added by Demo/AHSimulation/five_finger/build_5f_model.py. Do not edit by hand. -->\n"
    )
    (DST / "robot.xml").write_text(header + text + "\n")
    return T


def build_scene(T):
    text = (SRC / "scene.xml").read_text()
    tree = parse(SRC / "scene.xml")
    f3 = child_by_name(tree.getroot().find("worldbody"), "body", "finger3_target")
    pos, quat = apply(T, get_pose(f3))
    target5 = (
        f'        <body name="finger5_target" pos="{fmt(pos)}" quat="{fmt(quat)}" mocap="true">\n'
        '          <geom type="sphere" size=".01" contype="0" conaffinity="0" rgba="1 0. 0. .2"/>\n'
        "        </body>\n\n"
    )
    anchor = "    </worldbody>"
    assert anchor in text
    text = text.replace(anchor, target5 + anchor)
    (DST / "scene.xml").write_text(text)


def write_keyframe(qpos):
    (DST / "keyframes.xml").write_text(
        "<mujoco>\n  <keyframe>\n    <key name=\"zero\"\n         qpos=\""
        + " ".join(f"{x:.6g}" for x in qpos)
        + "\"\n         />\n  </keyframe>\n</mujoco>\n"
    )


def settle_keyframe():
    """Settle the new model at ctrl = 0 from the stock keyframe (+ finger3's values for finger5)."""
    stock = mujoco.MjModel.from_xml_path(str(SRC / "scene.xml"))
    q_stock = stock.key_qpos[0].copy()

    (DST / "keyframes.xml").write_text("<mujoco><keyframe/></mujoco>\n")  # the old 68-value key would not compile
    m = mujoco.MjModel.from_xml_path(str(DST / "scene.xml"))
    d = mujoco.MjData(m)

    def qadr(model, joint):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]

    d.qpos[: stock.nq] = q_stock
    f3 = slice(qadr(stock, "finger3_motor1"), qadr(stock, "finger4_motor1"))  # finger3 = both of its chains
    d.qpos[stock.nq:] = q_stock[f3]
    d.ctrl[:] = 0
    for _ in range(5000):  # 10 s
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
    print(f"settled keyframe: max |qvel| = {np.abs(d.qvel).max():.2e}, "
          f"stock part max |dq| = {np.abs(d.qpos[:stock.nq] - q_stock).max():.2e}, "
          f"finger5 vs finger3 max |dq| = {np.abs(d.qpos[stock.nq:] - d.qpos[f3]).max():.2e}")
    write_keyframe(d.qpos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyframe", action="store_true", help="regenerate keyframes.xml by settling at ctrl=0")
    ap.add_argument("--stock-plate", action="store_true", help="use the stock 3-finger palm plate mesh")
    args = ap.parse_args()

    (DST / "assets").mkdir(parents=True, exist_ok=True)
    for stl in SRC.glob("assets/*.stl"):
        shutil.copy2(stl, DST / "assets" / stl.name)
    for f in ("joints_properties.xml", "additional.xml"):
        shutil.copy2(SRC / f, DST / f)

    plate = None
    if not args.stock_plate and PLATE_MESH_SRC.exists():
        shutil.copy2(PLATE_MESH_SRC, DST / "assets" / PLATE_MESH)
        plate = PLATE_MESH
    print(f"palm plate mesh: {plate or 'r_hand_plate.stl (stock, 3 fingers + thumb)'}")

    T = build_robot(plate)
    build_scene(T)
    if args.keyframe or not (DST / "keyframes.xml").exists():
        settle_keyframe()

    m = mujoco.MjModel.from_xml_path(str(DST / "scene.xml"))
    print(f"wrote {DST.relative_to(HERE.parent.parent.parent)}: nbody={m.nbody} njnt={m.njnt} nq={m.nq} "
          f"nu={m.nu} neq={m.neq} nkey={m.nkey} nmocap={m.nmocap}")


if __name__ == "__main__":
    main()
