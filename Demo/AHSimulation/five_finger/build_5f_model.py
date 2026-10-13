#!/usr/bin/env python3
"""Generate the 5-digit (4 fingers + thumb) MJCF from the stock AH_Right (or AH_Left) model.

The pinky (finger5) is a rigid copy of the ring finger (finger3): every finger3 part that is
fixed to the palm (servos, finger frames, bushings, closing sites) and both of its motor chains
are copied and moved by the same rigid transform that takes the middle finger (finger2) onto the
ring finger (finger3), i.e. a 7.5 deg rotation about the palm-normal (x) axis plus the matching
translation. The pinky therefore relates to the ring finger exactly the way the ring finger
relates to the middle finger in the stock hand.

finger5 is appended after the thumb, so qpos[0:68], ctrl[0:8] and mocap[0:4] keep the stock
layout; finger5 adds qpos[68:85], ctrl[8:10] and mocap[4].

Usage:  python build_5f_model.py              (writes AHSimulation/AH_Right_5F/mjcf)
        python build_5f_model.py --side left  (writes AHSimulation/AH_Left_5F/mjcf)
        add --keyframe to regenerate keyframes.xml by settling at ctrl=0
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
CAD_STL = HERE.parent.parent.parent / "cad" / "five_finger" / "stl"
# Widened palm plates come from cad/five_finger/make_5f_plate.py (the stock plates have 3 finger mounts).
SIDES = {
    "right": dict(src=SIM_DIR / "AH_Right" / "mjcf", dst=SIM_DIR / "AH_Right_5F" / "mjcf", root="r_wrist_interface",
                  plate="r_hand_plate", plate_src=CAD_STL / "R-Hand_Plate_5F.stl"),
    "left": dict(src=SIM_DIR / "AH_Left" / "mjcf", dst=SIM_DIR / "AH_Left_5F" / "mjcf", root="l_wrist_interface",
                 plate="l_hand_plate", plate_src=CAD_STL / "L-Hand_Plate_5F.stl"),
}
SRC, DST = SIDES["right"]["src"], SIDES["right"]["dst"]

# Root-level parts (by their "Part ..." comment) of finger2/finger3 in the right model, and the
# finger5 names they get; build_robot() finds them geometrically and checks against these.
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
    """Rigid transform taking finger2 onto finger3 (from the finger pivot sites), the finger3 parts
    fixed to the palm (the parts that are an exact image of a finger2 part) and their finger5 names.
    The step is checked on every matched part, closing site and chain root."""
    p2 = get_pose(child_by_name(root_body, "site", F2_SITES[0]))
    p3 = get_pose(child_by_name(root_body, "site", F3_SITES[0]))
    qD = quat_mul(p3[1], quat_conj(p2[1]))
    tD = p3[0] - rotate(qD, p2[0])
    T = (tD, qD)

    geoms = part_geoms(root_body)
    matched = [(n2, n3) for n3, (_, g3) in geoms.items() for n2, (_, g2) in geoms.items()
               if n2 != n3 and g2.get("mesh") == g3.get("mesh") and same_pose(apply(T, get_pose(g2)), get_pose(g3))]
    assert len(matched) == 6 and len({n3 for _, n3 in matched}) == 6, f"finger2/finger3 parts not found: {matched}"
    pairs = [(geoms[a][1], geoms[b][1]) for a, b in matched]
    pairs += [(child_by_name(root_body, "site", a), child_by_name(root_body, "site", b)) for a, b in zip(F2_SITES, F3_SITES)]
    pairs += [(child_by_name(root_body, "body", a), child_by_name(root_body, "body", b)) for a, b in zip(F2_BODIES, F3_BODIES)]
    for e2, e3 in pairs:
        assert same_pose(apply(T, get_pose(e2)), get_pose(e3)), f"finger2->finger3 step is not rigid for {e2.attrib}"

    # finger5 names continue the per-mesh instance numbering; the part nearest the motor1 horn first
    horn = get_pose(child_by_name(root_body, "body", F3_BODIES[0]))[0]
    used = {}
    for name, (_, g) in geoms.items():
        mesh = g.get("mesh")
        suffix = name[len(mesh) + 1:] if name.startswith(mesh + "_") else ""
        used[mesh] = max(used.get(mesh, 1), int(suffix) if suffix.isdigit() else 1)
    order = ["scs0009", "finger_frame_1", "finger_frame_2", "bushing_0608_04"]  # servos, frames, bushings
    f3 = sorted((n3 for _, n3 in matched),
                key=lambda n: (order.index(geoms[n][1].get("mesh")), np.linalg.norm(get_pose(geoms[n][1])[0] - horn)))
    new = {}
    for n3 in f3:
        mesh = geoms[n3][1].get("mesh")
        used[mesh] += 1
        new[n3] = f"{mesh}_{used[mesh]}"

    angle = 2 * np.degrees(np.arctan2(np.linalg.norm(qD[1:]), qD[0]))
    print(f"finger2->finger3 step: {angle:.3f} deg about {np.round(qD[1:] / np.linalg.norm(qD[1:]), 4)}, "
          f"t = {np.round(tD * 1000, 3)} mm (verified on {len(pairs)} elements)")
    return T, new


def build_robot(side, plate_mesh):
    src, dst = side["src"], side["dst"]
    tree = parse(src / "robot.xml")
    mj = tree.getroot()
    root_body = mj.find("worldbody/body")
    assert root_body.get("name") == side["root"]
    T, f5_names = finger_step_transform(root_body)
    if side["root"] == "r_wrist_interface":
        assert f5_names == dict(zip(F3_PARTS, F5_PARTS)), f5_names

    geoms = part_geoms(root_body)
    children = list(root_body)
    first_body = next(i for i, c in enumerate(children) if c.tag == "body")
    insert_at = first_body - 1  # before the comment that precedes the first body

    new_fixed = [ET.Comment(" finger5 (pinky): fixed parts, copied from finger3 and moved by the finger2->finger3 step ")]
    for f3, f5 in f5_names.items():
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

    if plate_mesh:  # same part frame as the stock plate, but in mm like the other cad/ exports
        for mesh in mj.find("asset").iter("mesh"):
            if mesh.get("file") == side["plate"] + ".stl":
                mesh.set("name", side["plate"])
                mesh.set("file", plate_mesh)
                mesh.set("scale", "0.001 0.001 0.001")

    indent(tree)
    text = ET.tostring(mj, encoding="unicode").replace(" />", "/>")
    header = "".join((src / "robot.xml").read_text().splitlines(keepends=True)[:3]) + (
        "<!-- 5-digit variant: finger5 (pinky) added by Demo/AHSimulation/five_finger/build_5f_model.py. Do not edit by hand. -->\n"
    )
    (dst / "robot.xml").write_text(header + text + "\n")
    return T


def build_scene(side, T):
    text = (side["src"] / "scene.xml").read_text()
    tree = parse(side["src"] / "scene.xml")
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
    (side["dst"] / "scene.xml").write_text(text)


# 5-finger palm/back shells and grip pads from cad/five_finger/make_5f_shells.py: (part, rgba)
SHELL_PARTS = [("Top_Shell", "0.94 0.94 0.95 1"), ("Palm_Shell", "0.86 0.87 0.89 1"),
               ("Palm_Pad_Band", "0.33 0.34 0.36 1"), ("Palm_Pad_Palm", "0.33 0.34 0.36 1")]


def build_shell_scene(side):
    """scene_shells.xml: scene.xml plus the shells and pads as static visual meshes on the palm."""
    letter = "R" if side["root"].startswith("r_") else "L"
    srcs = [CAD_STL / f"{letter}-{part}_5F.stl" for part, _ in SHELL_PARTS]
    if not all(p.exists() for p in srcs):
        print("shells not found, skipping scene_shells.xml (run cad/five_finger/make_5f_shells.py)")
        return
    root = parse(side["dst"] / "robot.xml").getroot().find("worldbody/body")
    pos, quat = get_pose(part_geoms(root)[side["plate"]][1])  # all CAD parts share the plate's frame
    assets, geoms = [], []
    for (part, rgba), src in zip(SHELL_PARTS, srcs):
        name = f"{letter.lower()}_{part.lower()}_5f"
        shutil.copy2(src, side["dst"] / "assets" / f"{name}.stl")
        assets.append(f'        <mesh name="{name}" file="{name}.stl" scale="0.001 0.001 0.001"/>')
        geoms.append(f'          <geom type="mesh" mesh="{name}" rgba="{rgba}" contype="0" conaffinity="0" group="2"/>')
    text = (side["dst"] / "scene.xml").read_text()
    shells = ("    <asset>\n" + "\n".join(assets) + "\n    </asset>\n\n"
              "    <worldbody>\n"
              f'        <!-- palm/back shells and grip pads (static: the palm is fixed to the world) -->\n'
              f'        <body name="shells" pos="{fmt(pos)}" quat="{fmt(quat)}">\n' + "\n".join(geoms) + "\n        </body>\n"
              "    </worldbody>\n")
    anchor = '    <include file="keyframes.xml"/>'
    assert anchor in text
    (side["dst"] / "scene_shells.xml").write_text(text.replace(anchor, shells + anchor))


def write_keyframe(dst, qpos):
    (dst / "keyframes.xml").write_text(
        "<mujoco>\n  <keyframe>\n    <key name=\"zero\"\n         qpos=\""
        + " ".join(f"{x:.6g}" for x in qpos)
        + "\"\n         />\n  </keyframe>\n</mujoco>\n"
    )


def settle_keyframe(side):
    """Settle the new model at ctrl = 0 from the stock keyframe (+ finger3's values for finger5)."""
    src, dst = side["src"], side["dst"]
    stock = mujoco.MjModel.from_xml_path(str(src / "scene.xml"))
    q_stock = stock.key_qpos[0].copy()

    (dst / "keyframes.xml").write_text("<mujoco><keyframe/></mujoco>\n")  # the old 68-value key would not compile
    m = mujoco.MjModel.from_xml_path(str(dst / "scene.xml"))
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
    write_keyframe(dst, d.qpos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyframe", action="store_true", help="regenerate keyframes.xml by settling at ctrl=0")
    ap.add_argument("--stock-plate", action="store_true", help="use the stock 3-finger palm plate mesh")
    ap.add_argument("--side", choices=SIDES, default="right")
    args = ap.parse_args()
    side = SIDES[args.side]
    src, dst = side["src"], side["dst"]

    (dst / "assets").mkdir(parents=True, exist_ok=True)
    for stl in src.glob("assets/*.stl"):
        shutil.copy2(stl, dst / "assets" / stl.name)
    for f in ("joints_properties.xml", "additional.xml"):
        shutil.copy2(src / f, dst / f)

    plate = None
    if not args.stock_plate and side["plate_src"].exists():
        plate = side["plate"] + "_5f.stl"
        shutil.copy2(side["plate_src"], dst / "assets" / plate)
    print(f"palm plate mesh: {plate or side['plate'] + '.stl (stock, 3 fingers + thumb)'}")

    T = build_robot(side, plate)
    build_scene(side, T)
    build_shell_scene(side)
    if args.keyframe or not (dst / "keyframes.xml").exists():
        settle_keyframe(side)

    m = mujoco.MjModel.from_xml_path(str(dst / "scene.xml"))
    print(f"wrote {dst.relative_to(HERE.parent.parent.parent)}: nbody={m.nbody} njnt={m.njnt} nq={m.nq} "
          f"nu={m.nu} neq={m.neq} nkey={m.nkey} nmocap={m.nmocap}")


if __name__ == "__main__":
    main()
