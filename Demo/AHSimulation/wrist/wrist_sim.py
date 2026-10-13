#!/usr/bin/env python3
"""Shared helpers for the AH_Right_5F_Wrist checks and renders: model bookkeeping, IK-driven wrist poses."""

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent / "five_finger"))
sys.path.insert(0, str(REPO / "cad" / "five_finger" / "wrist"))
import wrist_design as wd  # noqa: E402
from check_shells import shells  # noqa: E402
from verify_5f_model import CHAINS, DEMO_POSES_5F  # noqa: E402

MJCF = HERE.parent / "AHSimulation" / "AH_Right_5F_Wrist" / "mjcf"


def jid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def aid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


class WristModel:
    def __init__(self, scene="scene.xml", payload_g=0.0, payload_site=None):
        """payload_g > 0 adds a point mass rigidly attached to the hand root at the OpenHand fingertip
        centroid (index..pinky, CAD), i.e. '200 g at the fingertips'."""
        if payload_g:
            spec = mujoco.MjSpec.from_file(str(MJCF / scene))
            hand = spec.body("r_wrist_interface")
            p = wd.cad_to_root(PAYLOAD_CAD) if payload_site is None else payload_site
            b = hand.add_body(name="payload", pos=p)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], rgba=[0.9, 0.3, 0.2, 0.6],
                       contype=0, conaffinity=0, group=2, mass=payload_g / 1000)
            self.m = spec.compile()
        else:
            self.m = mujoco.MjModel.from_xml_path(str(MJCF / scene))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.q_flex = m.jnt_qposadr[jid(m, "wrist_flexion")]
        self.q_dev = m.jnt_qposadr[jid(m, "wrist_deviation")]
        self.v_flex = m.jnt_dofadr[jid(m, "wrist_flexion")]
        self.v_dev = m.jnt_dofadr[jid(m, "wrist_deviation")]
        self.a_wrist = [aid(m, "wrist_motor1"), aid(m, "wrist_motor2")]
        self.q_motor = [m.jnt_qposadr[jid(m, f"wrist_motor{i}")] for i in (1, 2)]
        self.finger_ctrl = {dig: [aid(m, f"{f}_motor{k}") for k in (1, 2)] for dig, (f, _) in CHAINS.items()}

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)

    def wrist_angles(self):
        return np.array([self.d.qpos[self.q_flex], self.d.qpos[self.q_dev]])

    def servo_angles(self):
        return np.array([self.d.qpos[a] for a in self.q_motor])

    def ctrl_for(self, hand_pose=None, wrist=(0.0, 0.0), base=None):
        """ctrl vector: finger pose (dict of digit -> (m1, m2) deg, right hand) + wrist (phi, psi) rad via IK."""
        c = np.zeros(self.m.nu) if base is None else base.copy()
        if hand_pose is not None:
            for dig, (a1, a2) in hand_pose.items():
                c[self.finger_ctrl[dig]] = np.radians([a1, a2])
        th = wd.servo_ik(*wrist)
        assert np.all(np.isfinite(th)), f"wrist pose {np.degrees(wrist)} unreachable"
        c[self.a_wrist] = th
        return c

    def move(self, target, ramp=0.6, hold=0.6, record=None):
        m, d = self.m, self.d
        start = d.ctrl.copy()
        nr, nh = int(ramp / m.opt.timestep), int(hold / m.opt.timestep)
        for k in range(nr + nh):
            d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / max(nr, 1))
            mujoco.mj_step(m, d)
            if record is not None:
                record(m, d)


# OpenHand fingertip centroid of index..pinky (CAD mm, from the 5F model): where "200 g at the fingertips" sits
PAYLOAD_CAD = np.array([63.8, 70.3, -9.5])

HAND_POSES = {"OpenHand": DEMO_POSES_5F["OpenHand"], "CloseHand": DEMO_POSES_5F["CloseHand"], "keyframe": None}
