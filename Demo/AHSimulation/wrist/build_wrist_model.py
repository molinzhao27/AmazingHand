#!/usr/bin/env python3
"""Generate AH_Right_5F_Wrist: the 5-digit AmazingHand on the powered wrist and forearm base.

Derived from AH_Right_5F (copied and transformed, the source is never edited):

  world
   `- wrist_forearm            fixed (turned by wrist_design.WORLD_QUAT so the forearm stands upright): forearm
       |                       shells, fork, servo frame, elbow flange, 2 x STS3215
       |- wrist_gimbal_ring    hinge wrist_flexion   (W -x = root +y; > 0 palm-ward)
       |   `- wrist_hand_adapter   hinge wrist_deviation (W +z = root -x; > 0 radial / towards the thumb)
       |       `- r_wrist_interface    the whole AH_Right_5F hand subtree, unchanged
       |                               (+ a lumped inertial from hand_mass_model.py)
       |- wrist_horn1          hinge wrist_motor1 (ulnar servo, upper)  -> wrist_rod1 (ball joint)
       `- wrist_horn2          hinge wrist_motor2 (radial servo, lower) -> wrist_rod2 (ball joint)
  equality: connect wrist_rodN_top <-> wrist_ballN_hand  (the rods' upper ball joints)

All geometry and kinematic parameters come from cad/five_finger/wrist/wrist_design.py and the meshes
from cad/five_finger/wrist/stl (make_wrist.py). The hand root sits at the world origin at the neutral
wrist pose; the forearm axis runs along the hand's mean finger direction (wrist_design.HAND_YAW) through the
centroid of the four knuckles, and the world is turned so that it is vertical (the hand leans by HAND_YAW).
Body positions inside wrist_forearm are in the hand-root frame.

Layout: qpos[0] wrist_flexion, qpos[1] wrist_deviation, qpos[2:87] the AH_Right_5F hand (same order),
qpos[87:97] the linkage (horn1, rod1 ball, horn2, rod2 ball). ctrl[0:10] are the finger actuators
(unchanged), ctrl[10] wrist_motor1, ctrl[11] wrist_motor2.

Usage: python build_wrist_model.py [--keyframe]
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent / "five_finger"))
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "wrist"))
sys.path.insert(0, str(HERE))
import wrist_design as wd  # noqa: E402
from build_5f_model import SHELL_PARTS, child_by_name, fmt, get_pose, home_ctrl, indent, parse, write_keyframe  # noqa: E402

SIM_DIR = HERE.parent / "AHSimulation"
SRC = SIM_DIR / "AH_Right_5F" / "mjcf"
DST = SIM_DIR / "AH_Right_5F_Wrist" / "mjcf"
CAD = REPO / "cad" / "five_finger" / "wrist" / "stl"

# printed parts on the fixed forearm / moving bodies: (stl stem, body, rgba)
FIXED_MESHES = [("R-Forearm_Front", "0.92 0.92 0.93 1"), ("R-Forearm_Back", "0.95 0.95 0.96 1"),
                ("R-Wrist_Fork", "0.80 0.81 0.83 1"), ("R-Wrist_Servo_Frame", "0.55 0.56 0.58 1"),
                ("Elbow_Flange", "0.25 0.26 0.28 1"), ("ref/STS3215_motor1", "0.13 0.13 0.15 1"),
                ("ref/STS3215_motor2", "0.13 0.13 0.15 1"), ("ref/F683ZZ_+x", "0.75 0.76 0.8 1"),
                ("ref/F683ZZ_-x", "0.75 0.76 0.8 1")]
RING_MESHES = [("Wrist_Gimbal_Ring", "0.80 0.81 0.83 1")] + [(f"ref/pin_d3x10_{a}", "0.6 0.6 0.65 1") for a in ("+x", "-x", "+z", "-z")]
# the hand adapter's neck (its interface plate is the hand root's r_wrist_interface mesh, see build_5f_model.py)
ADAPTER_MESHES = [("ref/R-Wrist_Hand_Adapter_neck", "0.80 0.81 0.83 1"), ("ref/F683ZZ_+z", "0.75 0.76 0.8 1"),
                  ("ref/F683ZZ_-z", "0.75 0.76 0.8 1"), ("ref/stud_top_1", "0.75 0.76 0.8 1"), ("ref/stud_top_2", "0.75 0.76 0.8 1")]
HORN_MESHES = {1: [("Wrist_Horn_Arm", "0.25 0.26 0.28 1"), ("ref/stud_arm_1", "0.75 0.76 0.8 1")],
               2: [("ref/Wrist_Horn_Arm_motor2_placed", "0.25 0.26 0.28 1"), ("ref/stud_arm_2", "0.75 0.76 0.8 1")]}
ROD_MESHES = {i: [(f"ref/link_bot_{i}", "0.2 0.2 0.22 1"), (f"ref/link_top_{i}", "0.2 0.2 0.22 1"),
                  (f"ref/rod_{i}", "0.7 0.7 0.74 1")] for i in (1, 2)}

# masses (g) of bought parts (real masses; their envelopes only give the CoM / inertia shape)
PLA_G_CM3 = 1.00
BOUGHT_G = {"F683ZZ": 0.6, "pin_d3x10": 0.55, "stud": 1.0, "link": 1.2, "rod_per_mm": 0.055}
EXTRA_G = {"adapter": 2 * 0.5 + 4 * 0.8, "ring": 4 * 0.1, "horn": 3.0 + 1.0 + 4 * 0.3}   # inserts + screws / disc horn + nut


def mesh_name(stem):
    return "wrist_" + Path(stem).name.lower().replace("-", "_").replace("+", "p")


def mass_props(stem, body_pos_root):
    """(mass kg, com in body frame, inertia 3x3 about com in body frame) of a part's STL (CAD mm, neutral pose)."""
    t = trimesh.load(CAD / f"{stem}.stl")
    if t.volume < 0:
        t.invert()
    name = Path(stem).name
    vol = t.volume                                                     # mm3
    I1 = t.moment_inertia                                              # density 1: mm5
    if name.startswith("F683ZZ"):
        mass = BOUGHT_G["F683ZZ"]
    elif name.startswith("pin_"):
        mass = BOUGHT_G["pin_d3x10"]
    elif name.startswith("stud"):
        mass = BOUGHT_G["stud"]
    elif name.startswith("link"):
        mass = BOUGHT_G["link"]
    elif name.startswith("rod_"):
        mass = BOUGHT_G["rod_per_mm"] * t.extents.max()
    else:
        mass = vol / 1000 * PLA_G_CM3
    m_kg = mass / 1000
    rho = m_kg / vol                                                   # kg / mm3
    com = wd.cad_to_root(t.center_mass) - body_pos_root
    I = wd.R_CAD2ROOT @ (I1 * rho) @ wd.R_CAD2ROOT.T * 1e-6             # kg m2
    return m_kg, com, I


def combine(items):
    """Combine (m, com, I_about_com) items into one."""
    M = sum(m for m, _, _ in items)
    c = sum(m * cc for m, cc, _ in items) / M
    I = np.zeros((3, 3))
    for m, cc, Ic in items:
        d = cc - c
        I += Ic + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, c, I


def inertial_el(parent, m, com, I):
    I = 0.5 * (I + I.T)
    w = np.linalg.eigvalsh(I)
    assert w.min() > 0, f"non positive-definite inertia {w}"
    el = ET.Element("inertial", pos=fmt(com), mass=f"{m:.6g}",
                    fullinertia=" ".join(f"{v:.6g}" for v in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])))
    parent.insert(0, el)
    return el


def add_mesh_geoms(body, meshes, body_pos_root, assets, extra_mass=0.0):
    items = []
    for stem, rgba in meshes:
        name = mesh_name(stem)
        if name not in assets:
            shutil.copy2(CAD / f"{stem}.stl", DST / "assets" / f"{name}.stl")
            assets[name] = True
        ET.SubElement(body, "geom", type="mesh", mesh=name, rgba=rgba, contype="0", conaffinity="0", group="2",
                      pos=fmt(wd.PLATE_POS - body_pos_root), quat=fmt(wd.PLATE_QUAT))
        items.append(mass_props(stem, body_pos_root))
    if extra_mass:
        m, c, I = combine(items)
        items.append((extra_mass / 1000, c, np.zeros((3, 3))))
    return combine(items)


def hand_inertial():
    """Lumped inertial for r_wrist_interface: the hand mass model (with 5F shells and pads, fingers at the
    keyframe) minus the inertials the AH_Right_5F moving finger bodies already carry."""
    from check_shells import shells
    from hand_mass_model import mass_model
    from verify_5f_model import Hand
    h = Hand(SRC, shells("5f", "R"))
    h.reset()
    mujoco.mj_kinematics(h.m, h.d)
    mujoco.mj_comPos(h.m, h.d)
    total = mass_model(h.m, h.d)
    m, d = h.m, h.d
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_interface")
    items = []
    for b in range(m.nbody):
        if b != root and m.body_rootid[b] == root and m.body_mass[b] > 0:
            R = d.ximat[b].reshape(3, 3)
            items.append((m.body_mass[b], d.xipos[b].copy(), R @ np.diag(m.body_inertia[b]) @ R.T))
    mf, cf, If = combine(items)
    M, C, IC = total["mass"], total["com"], total["inertia"]
    mr = M - mf
    cr = (M * C - mf * cf) / mr
    # inertia about the origin, subtract, shift to cr
    def about_origin(m_, c_, I_):
        return I_ + m_ * (c_ @ c_ * np.eye(3) - np.outer(c_, c_))
    Ir = about_origin(M, C, IC) - about_origin(mf, cf, If)
    Ir = Ir - mr * (cr @ cr * np.eye(3) - np.outer(cr, cr))
    return dict(total=total, fingers=(mf, cf), root=(mr, cr, Ir))


DEFAULTS = """
<default>
  <default class="wrist_passive">
    <joint damping="0.01" frictionloss="0.005" armature="0.0005"/>
  </default>
  <default class="wrist_rod">
    <joint damping="0.0005" frictionloss="0" armature="2e-06"/>
  </default>
  <default class="sts3215_12v">
    <!-- Feetech STS3215 12 V version (C018, 1:345, 30 kg.cm = 2.94 N.m stall @ 12 V). damping / frictionloss /
         armature / kp: the sts3215_345 identification in joints_properties.xml (Open Duck Mini); forcerange:
         the datasheet stall torque (the identified class uses 3.23). -->
    <joint damping="0.56" frictionloss="0.068" armature="0.027"/>
    <position kp="17.11" kv="0.0" forcerange="-2.94 2.94"/>
  </default>
</default>
"""


def build_robot(with_shells):
    tree = parse(SRC / "robot.xml")
    mj = tree.getroot()
    wb = mj.find("worldbody")
    hand = wb.find("body")
    assert hand.get("name") == "r_wrist_interface"
    wb.remove(hand)
    # defaults go after the existing ones
    idx = max(i for i, el in enumerate(list(mj)) if el.tag == "default") + 1
    mj.insert(idx, ET.fromstring(DEFAULTS))

    assets = {}
    asset_el = mj.find("asset")
    c_root = wd.C_ROOT
    forearm = ET.SubElement(wb, "body", name="wrist_forearm", pos="0 0 0", quat=fmt(wd.WORLD_QUAT))
    m, c, I = add_mesh_geoms(forearm, FIXED_MESHES, np.zeros(3), assets)
    inertial_el(forearm, m + 2 * wd.SERVO_MASS, c, I)     # (fixed: only for completeness)

    ring = ET.SubElement(forearm, "body", name="wrist_gimbal_ring", pos=fmt(c_root))
    ET.SubElement(ring, "joint", name="wrist_flexion", type="hinge", axis=fmt(wd.dir_w_to_root([-1, 0, 0])),
                  range=f"{-wd.FLEX_LIMIT:.6g} {wd.FLEX_LIMIT:.6g}", **{"class": "wrist_passive"})
    inertial_el(ring, *add_mesh_geoms(ring, RING_MESHES, c_root, assets, EXTRA_G["ring"]))

    adapter = ET.SubElement(ring, "body", name="wrist_hand_adapter", pos="0 0 0")
    ET.SubElement(adapter, "joint", name="wrist_deviation", type="hinge", axis=fmt(wd.dir_w_to_root([0, 0, 1])),
                  range=f"{-wd.DEV_LIMIT:.6g} {wd.DEV_LIMIT:.6g}", **{"class": "wrist_passive"})
    inertial_el(adapter, *add_mesh_geoms(adapter, ADAPTER_MESHES, c_root, assets, EXTRA_G["adapter"]))
    for i in (1, 2):
        ET.SubElement(adapter, "site", name=f"wrist_ball{i}_hand", pos=fmt(wd.w_to_root(wd.P_TOP_HAND[i]) - c_root),
                      size="0.002", group="3")
    ET.SubElement(adapter, "site", name="wrist_centre", pos="0 0 0", size="0.003", group="3")

    # the hand: unchanged subtree, re-rooted under the adapter, with a realistic lumped inertial
    hand.set("pos", fmt(-c_root))
    hi = hand_inertial()
    old = hand.find("inertial")
    hand.remove(old)
    mr, cr, Ir = hi["root"]
    inertial_el(hand, mr, cr, Ir)
    hand.insert(0, ET.Comment(f" lumped inertial: hand mass model {hi['total']['mass'] * 1000:.1f} g "
                              f"(Demo/AHSimulation/wrist/hand_mass_model.py) minus the finger bodies' own "
                              f"{hi['fingers'][0] * 1000:.1f} g "))
    if with_shells:
        for part, rgba in SHELL_PARTS:
            name = f"r_{part.lower()}_5f"
            ET.SubElement(hand, "geom", type="mesh", mesh=name, rgba=rgba, contype="0", conaffinity="0", group="2",
                          pos=fmt(wd.PLATE_POS), quat=fmt(wd.PLATE_QUAT))
            if asset_el.find(f"mesh[@name='{name}']") is None:
                ET.SubElement(asset_el, "mesh", name=name, file=f"{name}.stl", scale="0.001 0.001 0.001")
    adapter.append(hand)

    for i in (1, 2):
        sh = wd.w_to_root(wd.servo_shaft(i))
        horn = ET.SubElement(forearm, "body", name=f"wrist_horn{i}", pos=fmt(sh))
        # positive = clockwise seen from the horn (Feetech position increasing): about -x_W for the
        # ulnar servo (horn towards +x_W), about +x_W for the radial servo
        ET.SubElement(horn, "joint", name=f"wrist_motor{i}", type="hinge",
                      axis=fmt(wd.dir_w_to_root([-wd.SIDE[i], 0, 0])), **{"class": "sts3215_12v"})
        inertial_el(horn, *add_mesh_geoms(horn, HORN_MESHES[i], sh, assets, EXTRA_G["horn"]))
        bb = wd.w_to_root(wd.ball_bottom(i, 0.0))
        rod = ET.SubElement(horn, "body", name=f"wrist_rod{i}", pos=fmt(bb - sh))
        ET.SubElement(rod, "joint", name=f"wrist_rod{i}_ball", type="ball", **{"class": "wrist_rod"})
        inertial_el(rod, *add_mesh_geoms(rod, ROD_MESHES[i], bb, assets))
        ET.SubElement(rod, "site", name=f"wrist_rod{i}_top", pos=fmt(wd.w_to_root(wd.P_TOP_HAND[i]) - bb), size="0.002", group="3")

    for name in assets:
        ET.SubElement(asset_el, "mesh", name=name, file=f"{name}.stl", scale="0.001 0.001 0.001")

    act = mj.find("actuator")
    for i in (1, 2):
        ET.SubElement(act, "position", name=f"wrist_motor{i}", joint=f"wrist_motor{i}",
                      ctrlrange=f"{-wd.SERVO_LIMIT:.6g} {wd.SERVO_LIMIT:.6g}", **{"class": "sts3215_12v"})
    eq = mj.find("equality")
    for i in (1, 2):
        # stiffer impedance than the hand's default (0.99 0.999): the rods carry tens of N, and with the default
        # the closure stretches ~0.35 mm (= ~1-4 deg of wrist error); measured effect in reports/
        ET.SubElement(eq, "connect", name=f"wrist_rod{i}_closure", site1=f"wrist_rod{i}_top", site2=f"wrist_ball{i}_hand",
                      solimp="0.9999 0.9999 0.001 0.5 2")

    indent(tree)
    text = ET.tostring(mj, encoding="unicode").replace(" />", "/>")
    header = "".join((SRC / "robot.xml").read_text().splitlines(keepends=True)[:4]) + (
        "<!-- Powered wrist + forearm base added by Demo/AHSimulation/wrist/build_wrist_model.py. Do not edit by hand. -->\n")
    out = DST / ("robot_shells.xml" if with_shells else "robot.xml")
    out.write_text(header + text + "\n")
    return hi


def root_to_world(p):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, wd.WORLD_QUAT)
    return R.reshape(3, 3) @ np.asarray(p, float)


def check_knuckle_centre():
    """The forearm axis is laid out through wrist_design.KNUCKLE_CENTRE: check it against the hand model."""
    root = parse(SRC / "robot.xml").getroot().find("worldbody/body")
    piv = np.array([(get_pose(child_by_name(root, "site", f"closing_1 ({n})_2"))[0] - wd.PLATE_POS) @ wd.R_CAD2ROOT * 1000
                    for n in (1, 2, 3, 5)])
    c = piv.mean(0)[:2]
    assert np.allclose(c, wd.KNUCKLE_CENTRE, atol=0.05), f"knuckle centroid {c} != wrist_design.KNUCKLE_CENTRE {wd.KNUCKLE_CENTRE}"
    return c


def build_scenes():
    elbow_z = root_to_world(wd.w_to_root([0, wd.ELBOW_Y, 0]))[2]
    text = (SRC / "scene.xml").read_text()
    a = '<geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />'
    assert a in text
    text = text.replace(a, f'<geom name="floor" size="0 0 0.05" pos="0 0 {elbow_z:.4f}" type="plane" material="groundplane" />')
    text = text.replace('<global azimuth="160" elevation="-20" />', '<global azimuth="150" elevation="-15" />')
    text = text.replace('        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />',
                        '        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />\n'
                        '        <light pos="0.6 0.6 0.6" dir="-1 -1 -1" diffuse="0.35 0.35 0.35" />')
    (DST / "scene.xml").write_text(text)
    (DST / "scene_shells.xml").write_text(text.replace('<include file="robot.xml" />', '<include file="robot_shells.xml" />'))


def settle_keyframe():
    src = mujoco.MjModel.from_xml_path(str(SRC / "scene.xml"))
    (DST / "keyframes.xml").write_text("<mujoco><keyframe/></mujoco>\n")
    m = mujoco.MjModel.from_xml_path(str(DST / "scene.xml"))
    d = mujoco.MjData(m)
    a0 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger1_motor1")]
    d.qpos[a0:a0 + src.nq] = src.key_qpos[0]
    for i in (1, 2):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"wrist_rod{i}_ball")
        d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 4] = [1, 0, 0, 0]
    d.ctrl[:] = home_ctrl(m)   # 0, except the thumb: layout.clamp_thumb(0, 0), inside its command envelope
    for _ in range(5000):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
    jf = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wrist_flexion")]
    jd = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wrist_deviation")]
    print(f"settled keyframe (home ctrl: 0, thumb at layout.clamp_thumb(0, 0); 10 s): max |qvel| {np.abs(d.qvel).max():.2e}, wrist sag under gravity: "
          f"flexion {np.degrees(d.qpos[jf]):+.3f} deg, deviation {np.degrees(d.qpos[jd]):+.3f} deg; "
          f"hand qpos vs AH_Right_5F key max |dq| {np.abs(d.qpos[a0:a0 + src.nq] - src.key_qpos[0]).max():.2e}")
    write_keyframe(DST, d.qpos, d.ctrl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyframe", action="store_true", help="regenerate keyframes.xml by settling at the home ctrl (0; the thumb at layout.clamp_thumb(0, 0))")
    ap.add_argument("--hand", default="AH_Right_5F",
                    help="hand model to put on the wrist, e.g. AH_Right_5F_Slim (writes <hand>_Wrist)")
    args = ap.parse_args()
    global SRC, DST
    if args.hand != "AH_Right_5F":
        SRC, DST = SIM_DIR / args.hand / "mjcf", SIM_DIR / f"{args.hand}_Wrist" / "mjcf"
    (DST / "assets").mkdir(parents=True, exist_ok=True)
    for stale in (DST / "assets").glob("wrist_*.stl"):          # wrist meshes are re-copied from cad/ below
        stale.unlink()
    for stl in SRC.glob("assets/*.stl"):
        shutil.copy2(stl, DST / "assets" / stl.name)
    for stl in (SIM_DIR / "AH_Right_5F" / "mjcf" / "assets").glob("r_*_5f.stl"):  # palm shells + pads
        shutil.copy2(stl, DST / "assets" / stl.name)
    for f in ("joints_properties.xml", "additional.xml"):
        shutil.copy2(SRC / f, DST / f)
    c = check_knuckle_centre()
    print(f"forearm axis: {np.degrees(wd.HAND_YAW):.1f} deg from CAD y, through the knuckle centroid ({c[0]:.2f}, {c[1]:.2f}) mm; "
          f"wrist centre CAD {np.round(wd.C_CAD, 2)} mm")
    hi = build_robot(False)
    build_robot(True)
    build_scenes()
    mr, cr, _ = hi["root"]
    print(f"hand mass model {hi['total']['mass'] * 1000:.1f} g (CoM root {np.round(hi['total']['com'] * 1000, 1)} mm); "
          f"finger bodies carry {hi['fingers'][0] * 1000:.1f} g -> lumped root inertial {mr * 1000:.1f} g at {np.round(cr * 1000, 1)} mm")
    if args.keyframe or not (DST / "keyframes.xml").exists():
        settle_keyframe()
    m = mujoco.MjModel.from_xml_path(str(DST / "scene.xml"))
    print(f"wrote {DST.relative_to(REPO)}: nbody={m.nbody} njnt={m.njnt} nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq} "
          f"nkey={m.nkey} nmocap={m.nmocap}")
    print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
    mujoco.MjModel.from_xml_path(str(DST / "scene_shells.xml"))


if __name__ == "__main__":
    main()
