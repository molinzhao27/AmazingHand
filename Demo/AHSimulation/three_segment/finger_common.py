"""Shared helpers for the 3-segment finger study (AmazingHand, 5-digit variant).

Conventions
-----------
Finger design frame ("CAD frame", mm): the frame every stock finger STEP/STL is drawn in.
Origin = MCP / abduction-axis intersection, x = pin axes (lateral), y = along the straight
finger (distal), z = dorsal (so flexion moves the finger towards -z). In this frame the stock
finger is straight (MCP = PIP = 0) and every part sits at its assembled position.

Joint angles are measured between the CAD frames of neighbouring parts:
    abduction = rotation of the gimbal about the finger frame's z axis,
    link angle psi = flexion of the Link relative to the gimbal (0 = the stock straight pose),
    MCP = flexion of P1 (proximal) relative to the gimbal,
    PIP = flexion of P2 relative to P1, DIP = flexion of P3 relative to P2.
Flexion is positive towards the palm (a rotation of -angle about +x).
Motor commands: f = (m1 - m2) / 2 drives flexion, a = (m1 + m2) / 2 drives abduction.
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM_5F = REPO / "Demo" / "AHSimulation" / "AHSimulation" / "AH_Right_5F" / "mjcf"
MODEL_DIR = HERE / "model"
REPORT_DIR = HERE / "reports"
RENDER_DIR = HERE / "renders"
DATA_DIR = HERE / "data"
CAD_DIR = REPO / "cad" / "five_finger" / "three_segment"
DESIGN_JSON = CAD_DIR / "design.json"
FIG_DIR = CAD_DIR / "figures"

# Feetech SCS0009 at 6 V (reseller data): 2.3 kg.cm stall torque, ~0.1 s/60 deg no-load.
SERVO_STALL_NM = 2.3 * 9.80665 / 100.0  # 0.2256 N m

# Stock index-finger (finger1) chain bodies in AH_Right_5F/robot.xml
STOCK_NAMES = dict(
    horn1="custom_servo_horn", rod1="ball_link", link="rotule_ball",
    gimbal="std00333_plast_tcb_torx_2_5x8__configuration_copy_of_default",
    distal="parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999",
    proximal="parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b",
    horn2="rotule_ball_2", rod2="m2_rod_l18",
)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- small geometry helpers
def rotx(beta):
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def flex(theta):
    """Rotation of a flexion by theta (rad): a rotation of -theta about +x."""
    return rotx(-theta)


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def mat_of(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def H(R=np.eye(3), p=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def Hinv(T):
    R, p = T[:3, :3], T[:3, 3]
    return H(R.T, -R.T @ p)


def xangle(R):
    """Rotation angle about x of a (pure x-) rotation matrix, as a flexion angle (rad)."""
    return -np.arctan2(R[2, 1], R[1, 1])


def zangle(R):
    return np.arctan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------- finger model wrapper
class Finger:
    """A one-digit model (see build_finger_models.py) with helpers to settle and measure it.

    roles: {role: (body name, mesh name or None)}; the role's CAD frame is the frame of that mesh's
    geom as written in the XML (the STL frame), or the body frame itself when mesh is None.
    """

    def __init__(self, xml_path, roles, tip_site="tip1"):
        self.spec = mujoco.MjSpec.from_file(str(xml_path))
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.cad = {}
        for role, (body, mesh) in roles.items():
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            assert b >= 0, body
            if mesh is None:
                self.cad[role] = (b, H())
            else:
                g = [g for g in self.spec.body(body).geoms if g.meshname == mesh][0]
                self.cad[role] = (b, H(mat_of(g.quat), np.array(g.pos)))
        self.tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tip_site)
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger1_motor{k}") for k in (1, 2)]
        self.mjnt = [m.actuator_trnid[a, 0] for a in self.act]
        self.mdof = [m.jnt_dofadr[j] for j in self.mjnt]
        self.reset()

    # -- state
    def reset(self):
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        else:
            mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)

    def eq_err(self):
        x = self.d.site_xpos
        return np.array([np.linalg.norm(x[self.m.eq_obj1id[e]] - x[self.m.eq_obj2id[e]]) for e in range(self.m.neq)])

    def move(self, m1, m2, ramp=0.3, hold=0.3, check=True):
        """Ramp both motor commands (rad) to (m1, m2) and hold; returns max eq error seen (m)."""
        m, d = self.m, self.d
        start = d.ctrl[self.act].copy()
        target = np.array([m1, m2])
        nr, nh = max(1, int(round(ramp / m.opt.timestep))), int(round(hold / m.opt.timestep))
        worst = 0.0
        for k in range(nr + nh):
            d.ctrl[self.act] = start + (target - start) * min(1.0, (k + 1) / nr)
            mujoco.mj_step(m, d)
            if check:
                if not np.all(np.isfinite(d.qpos)):
                    raise RuntimeError("non-finite state")
                worst = max(worst, self.eq_err().max())
        return worst

    # -- measurements
    def frame(self, role):
        b, T = self.cad[role]
        return H(self.d.xmat[b].reshape(3, 3), self.d.xpos[b]) @ T

    def rel(self, a, b):
        return Hinv(self.frame(a)) @ self.frame(b)

    def to_frame(self, role, p_world):
        return (Hinv(self.frame(role)) @ np.r_[p_world, 1])[:3]

    def measure(self):
        """Joint angles (deg), tip position in the finger frame (mm), motor angles (deg)."""
        out = dict(
            abd=np.degrees(zangle(self.rel("frame", "gimbal")[:3, :3])),
            psi=np.degrees(xangle(self.rel("gimbal", "link")[:3, :3])),
            mcp=np.degrees(xangle(self.rel("gimbal", "P1")[:3, :3])),
            pip=np.degrees(xangle(self.rel("P1", "P2")[:3, :3])),
            tip=self.to_frame("frame", self.d.site_xpos[self.tip]) * 1e3,
            m1=np.degrees(self.d.qpos[self.m.jnt_qposadr[self.mjnt[0]]]),
            m2=np.degrees(self.d.qpos[self.m.jnt_qposadr[self.mjnt[1]]]),
            eq=self.eq_err().max() * 1e3,
        )
        if "P3" in self.cad:
            out["dip"] = np.degrees(xangle(self.rel("P2", "P3")[:3, :3]))
        return out

    # -- statics through the closed chains
    def motor_jacobian(self, site=None):
        """d(site position, world, m) / d(motor angles, rad) through every loop closure: 3 x 2.

        Velocities that keep every equality constraint satisfied: J_c v = 0. With the two motor
        speeds given, the passive speeds are the minimum-norm solution (the spins of the ball joints
        about their rods are free and do not move anything else)."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        J = np.zeros((m.neq * 3, m.nv))
        for e in range(m.neq):
            s1, s2 = m.eq_obj1id[e], m.eq_obj2id[e]
            j1, j2 = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, j1, None, s1)
            mujoco.mj_jacSite(m, d, j2, None, s2)
            J[3 * e:3 * e + 3] = j1 - j2
        mdof = np.array(self.mdof)
        odof = np.setdiff1d(np.arange(m.nv), mdof)
        Jo, Jm = J[:, odof], J[:, mdof]
        vo = -np.linalg.lstsq(Jo, Jm, rcond=1e-10)[0]
        resid = np.abs(Jo @ vo + Jm).max()
        V = np.zeros((m.nv, 2))
        V[mdof, [0, 1]] = 1.0
        V[odof] = vo
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, None, self.tip if site is None else site)
        return jp @ V, resid, V


def stock_roles():
    n = STOCK_NAMES
    return dict(frame=("r_wrist_interface", "finger_frame_1"), gimbal=(n["gimbal"], "gimbal"),
                link=(n["link"], "link"), P1=(n["proximal"], "proximal"), P2=(n["distal"], "distal"))


def three_seg_roles():
    n = STOCK_NAMES
    return dict(frame=("r_wrist_interface", "finger_frame_1"), gimbal=(n["gimbal"], "gimbal"),
                link=(n["link"], "link3s"), P1=("p1", None), P2=("p2", None), P3=("p3", None), L2=("l2", None))


def flexion_ctrl(f_deg, a_deg=0.0):
    """Motor commands (rad) for flexion f and abduction a (deg)."""
    return np.radians(a_deg + f_deg), np.radians(a_deg - f_deg)
