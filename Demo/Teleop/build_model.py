"""Build MuJoCo scenes for the AmazingHand, optionally with an extra pinky finger.

The stock AmazingHand (Pollen Robotics) has 3 fingers + thumb. This script clones
finger 3 (ring) into a new finger 5 (pinky), placed one more "fan step" further
along the palm, so a 4-finger + thumb fork can be simulated before its CAD exists.

The stock model is copied from Demo/AHSimulation/AHSimulation/AH_Right/mjcf into
--model-dir the first time (models/ is git-ignored).

Outputs, next to the stock robot.xml:
  scene_4f.xml   stock hand (index, middle, ring, thumb)
  robot_5f.xml   hand with the cloned pinky
  scene_5f.xml   scene for robot_5f.xml

Usage:  python build_model.py [--model-dir models/AH_Right] [--pinky-extra-deg 0]
"""

import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

STOCK = Path(__file__).resolve().parent.parent / "AHSimulation" / "AHSimulation" / "AH_Right" / "mjcf"

SCENE = """<mujoco model="amazinghand_{tag}">
  <include file="{robot}"/>
  <option timestep="0.002"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1280" offheight="1280"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.3 0 0.6" dir="-0.5 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 -0.02" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def pose_matrix(el):
    """4x4 transform from an element's pos/quat attributes (MuJoCo wxyz quats)."""
    T = np.eye(4)
    pos = el.get("pos")
    quat = el.get("quat")
    if pos:
        T[:3, 3] = [float(v) for v in pos.split()]
    if quat:
        q = np.array([float(v) for v in quat.split()])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
        T[:3, :3] = R.reshape(3, 3)
    return T


def set_pose(el, T):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
    el.set("pos", " ".join(f"{v:.9g}" for v in T[:3, 3]))
    el.set("quat", " ".join(f"{v:.9g}" for v in q))


def rot_x(deg):
    a = np.radians(deg)
    T = np.eye(4)
    T[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


def rename(name):
    for old, new in (("finger3", "finger5"), ("tip3", "tip5"), ("(3)", "(5)"), ("ball3", "ball5")):
        if old in name:
            return name.replace(old, new)
    return name + "_f5"


def build_5f(model_dir: Path, extra_deg: float):
    tree = ET.parse(model_dir / "robot.xml")
    root = tree.getroot()
    palm = root.find("./worldbody/body[@name='r_wrist_interface']")

    def child(tag, name):
        return palm.find(f"./{tag}[@name='{name}']")

    # Finger mounts on the palm: the site where each finger frame bolts on.
    mounts = {k: pose_matrix(child("site", f"closing_1 ({k})_2")) for k in (1, 2, 3, 4)}
    # Motor bodies hanging off the palm (motor1 horn, motor2 lever) per finger.
    motor_bodies = {}
    for body in palm.findall("./body"):
        for j in body.findall("./joint"):
            if j.get("name", "").startswith("finger"):
                motor_bodies.setdefault(int(j.get("name")[6]), []).append(body)

    # Palm-level geoms (servos, finger frames, bushings) belong to the nearest finger anchor.
    anchors = []
    for k in (1, 2, 3, 4):
        anchors.append((k, mounts[k][:3, 3]))
        for b in motor_bodies[k]:
            anchors.append((k, pose_matrix(b)[:3, 3]))
    finger3_geoms = []
    for g in palm.findall("./geom"):
        if g.get("mesh") in ("r_wrist_interface", "r_hand_plate"):
            continue
        p = pose_matrix(g)[:3, 3]
        k = min(anchors, key=lambda a: np.linalg.norm(a[1] - p))[0]
        if k == 3:
            finger3_geoms.append(g)

    # Fan step from finger 2 to finger 3 (a rotation about the palm x axis), applied once more.
    step = mounts[3] @ np.linalg.inv(mounts[2])
    D = rot_x(extra_deg) @ step

    site_map = {}
    new_elems = []
    for g in finger3_geoms:
        g2 = copy.deepcopy(g)
        set_pose(g2, D @ pose_matrix(g))
        new_elems.append(g2)
    for s in palm.findall("./site"):
        if "(3)" in s.get("name", ""):
            s2 = copy.deepcopy(s)
            set_pose(s2, D @ pose_matrix(s))
            site_map[s.get("name")] = rename(s.get("name"))
            s2.set("name", site_map[s.get("name")])
            new_elems.append(s2)
    joint_map = {}
    for b in motor_bodies[3]:
        b2 = copy.deepcopy(b)
        set_pose(b2, D @ pose_matrix(b))
        for el in b2.iter():
            name = el.get("name")
            if name is None:
                continue
            el.set("name", rename(name))
            if el.tag == "site":
                site_map[name] = rename(name)
            elif el.tag == "joint":
                joint_map[name] = rename(name)
        new_elems.append(b2)
    for el in new_elems:
        palm.append(el)

    eq = root.find("./equality")
    for c in list(eq):
        s1, s2 = c.get("site1"), c.get("site2")
        if s1 in site_map and s2 in site_map:
            c2 = copy.deepcopy(c)
            c2.set("site1", site_map[s1])
            c2.set("site2", site_map[s2])
            eq.append(c2)
    act = root.find("./actuator")
    for a in list(act):
        if a.get("joint") in joint_map:
            a2 = copy.deepcopy(a)
            a2.set("name", joint_map[a.get("name")])
            a2.set("joint", joint_map[a.get("joint")])
            act.append(a2)

    out = model_dir / "robot_5f.xml"
    ET.indent(tree)
    tree.write(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=Path(__file__).parent / "models" / "AH_Right", type=Path)
    ap.add_argument("--pinky-extra-deg", default=0.0, type=float,
                    help="extra fan angle for the pinky beyond the ring->pinky step (deg)")
    args = ap.parse_args()

    if not (args.model_dir / "robot.xml").exists():
        shutil.copytree(STOCK, args.model_dir, ignore=shutil.ignore_patterns("*.part"), dirs_exist_ok=True)
        print(f"copied the stock model from {STOCK}")
    build_5f(args.model_dir, args.pinky_extra_deg)
    for tag, robot in (("4f", "robot.xml"), ("5f", "robot_5f.xml")):
        path = args.model_dir / f"scene_{tag}.xml"
        path.write_text(SCENE.format(tag=tag, robot=robot))
        m = mujoco.MjModel.from_xml_path(str(path))
        motors = [m.actuator(i).name for i in range(m.nu)]
        print(f"{path.name}: nq={m.nq} nu={m.nu} motors={motors}")


if __name__ == "__main__":
    main()
