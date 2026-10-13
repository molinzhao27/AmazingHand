#!/usr/bin/env python3
"""Design parameters and linkage kinematics of the AmazingHand 5F powered wrist.

Single source of truth for the CAD script (make_wrist.py), the MuJoCo generator
(Demo/AHSimulation/wrist/build_wrist_model.py), the checks and the control code (IK).
numpy only, no CAD dependency.

Frames
------
CAD : the stock AmazingHand part-studio frame (mm): x from index towards pinky, y up along the
      fingers, z towards the back of the hand. All STEP/STL files are written in this frame,
      at the neutral wrist pose.
W   : CAD axes, origin at the wrist centre C (intersection of the two gimbal axes).
root: the MJCF frame of the hand root body `r_wrist_interface` (m): X_root = POS + R @ x_cad / 1000.

Mechanism (see README.md)
-------------------------
Ring gimbal (Cardan joint) under the hand: the forearm fork carries the flexion axis (W x, proximal),
the ring carries the deviation axis (W z, distal) into the hand adapter. Two STS3215 servos in the
forearm (axes along W x, stacked in y) drive two push rods with ball joints that end on a lever bar
behind the gimbal (W +z side). Both rods up = palm-ward flexion, one up / one down = deviation,
so both servos share the load on either axis (same idea as the AmazingHand finger).

Angles
------
phi  (wrist_flexion)  > 0: palm-ward flexion (fingers towards the palm, CAD -z)
psi  (wrist_deviation)> 0: radial deviation (fingers towards the thumb / index side, CAD -x)
theta_i (servo)        > 0: clockwise seen from the horn (Feetech position increasing).
motor1 = ulnar (pinky-side) servo, upper; motor2 = radial (index-side) servo, lower.
With this naming the wrist follows the finger convention:
    flexion ~ (m1 - m2) / 2,  radial deviation ~ (m1 + m2) / 2.
"""

import numpy as np

# ----------------------------------------------------------------------------- frames
PLATE_POS = np.array([-0.0115, 0.0245, 0.1022])            # m, pose of every CAD part in the hand root
PLATE_QUAT = np.array([0.5, 0.5, -0.5, -0.5])
R_CAD2ROOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # = quat2mat(PLATE_QUAT)

INTERFACE_AXIS = np.array([24.5, -11.5])        # (x, z) CAD of the R-Wrist_Interface ring spigot axis
INTERFACE_BOTTOM = -102.2                       # CAD y of the wrist interface's lowest face
H_FLANGE = 25.0                                 # wrist centre is this far below the interface bottom
C_CAD = np.array([INTERFACE_AXIS[0], INTERFACE_BOTTOM - H_FLANGE, INTERFACE_AXIS[1]])  # (24.5, -127.2, -11.5)


def cad_to_root(p_cad):
    """CAD mm -> root m (points, (..., 3))."""
    return PLATE_POS + np.asarray(p_cad, float) @ R_CAD2ROOT.T / 1000.0


def w_to_cad(p_w):
    return np.asarray(p_w, float) + C_CAD


def w_to_root(p_w):
    return cad_to_root(w_to_cad(p_w))


def dir_w_to_root(v):
    return np.asarray(v, float) @ R_CAD2ROOT.T


C_ROOT = cad_to_root(C_CAD)                     # (0, 0, -0.025)

# ----------------------------------------------------------------------------- gimbal (W frame, mm)
# Compact gimbal: every part on the flexion axis stays inside |x| < 16 mm, because at the palm-ward
# flexion + deviation corners the push rods (at |x| ~ 19-21) pass within ~2.5 mm of that axis.
RING_R_OUT = 10.0          # gimbal ring outer radius (ring axis = W y)
RING_R_IN = 5.5            # d11 bore for the hand's bus cable (crimp its connector after threading it)
RING_H = 8.0               # ring height (y)
PIN_D = 3.0                # 4 x dowel pin d3 x 10 (ISO 8734), press fit in the ring
PIN_L = 10.0
BRG_D, BRG_B = 7.0, 3.0    # F683ZZ flanged ball bearing 3x7x3 in each outer lug
BRG_FD, BRG_FT = 8.1, 0.7
LUG_IN, LUG_OUT = 10.5, 15.5   # radial position of the inner / outer faces of all four lugs
LUG_R = 5.5                # lug boss radius around a pin

# ----------------------------------------------------------------------------- hand adapter (W)
FLANGE_T = 6.5             # flange thickness; its underside is at y = H_FLANGE - FLANGE_T = 18.5
FLANGE_R = 27.0            # 2.5 mm larger than the interface spigot (d49), room for the screw counterbores
BAR_X = 14.5               # lever bar half-length (|x| <= BAR_X), clear of the fork lugs at |x| >= 17.5
BAR_Y = 5.0                # bar half-height (y)
BAR_Z = (17.5, 25.0)       # bar z range (behind the gimbal; the back lug block fills z = LUG_IN..25 at |x| <= LUG_R)
STUD_TOP_H = 6.5           # ball centre above the stud seat (M3 ball stud, ball d5.8)
E_LEVER = 0.5 * (BAR_Z[0] + BAR_Z[1])           # 21.25: flexion lever (z of the top ball centres)
D_LEVER = BAR_X + STUD_TOP_H                    # 21.0 : deviation lever (|x| of the top ball centres)
# top ball centres in the hand frame (= W at the neutral pose); motor1 = ulnar (+x), motor2 = radial (-x)
P_TOP_HAND = {1: np.array([+D_LEVER, 0.0, E_LEVER]), 2: np.array([-D_LEVER, 0.0, E_LEVER])}

# ----------------------------------------------------------------------------- servos (W)
# Feetech STS3215 (C001, 7.4 V version, 1:345): 45.2 x 24.7 x 35 mm, 55 g, output 10 mm from the
# gear end, centred in the 24.7 width. Mounted with the output axis along W x, horn facing outwards,
# body length along W z (gear end towards the back), so the arm swings in the y-z plane.
SERVO_L, SERVO_W, SERVO_H = 45.2, 24.7, 35.0
SERVO_SHAFT_FROM_END = 10.0
SERVO_MASS = 0.055
HORN_D, HORN_T, HORN_GAP = 24.0, 3.0, 0.5   # metal 25T disc horn envelope (d24 kept as conservative)
ARM_T = 4.0                # printed horn arm thickness
STUD_ARM_H = 6.0           # ball stud on the arm, pointing inwards (towards the servo body)
R_HORN = 26.0              # arm radius (servo axis -> ball centre)
X_TOP_FACE = D_LEVER - (HORN_GAP + HORN_T) + STUD_ARM_H   # 23.5: |x| of the servo top (horn) face
Z_SHAFT = E_LEVER - R_HORN                                # -4.75: rods vertical at the neutral pose
Y_SHAFT = {1: -66.0, 2: -96.0}                           # motor1 (ulnar) upper, motor2 (radial) lower
SIDE = {1: +1.0, 2: -1.0}                                # x side of each servo / rod
KAPPA = {1: +1.0, 2: -1.0}  # +theta (CW from the horn) moves the ball up (+1) or down (-1)


def servo_shaft(i):
    """W position of servo i's output axis at the horn face plane."""
    return np.array([SIDE[i] * X_TOP_FACE, Y_SHAFT[i], Z_SHAFT])


def ball_bottom(i, theta):
    """W centre of the arm ball of servo i at servo angle theta (rad)."""
    return np.array([SIDE[i] * D_LEVER, Y_SHAFT[i] + KAPPA[i] * R_HORN * np.sin(theta), Z_SHAFT + R_HORN * np.cos(theta)])


ROD_L = {i: float(np.linalg.norm(P_TOP_HAND[i] - ball_bottom(i, 0.0))) for i in (1, 2)}   # 66 / 96 mm

# servo body box in W (x, y, z ranges)
def servo_box(i):
    s = SIDE[i]
    x = sorted([s * X_TOP_FACE, s * (X_TOP_FACE - SERVO_H)])
    y = [Y_SHAFT[i] - SERVO_W / 2, Y_SHAFT[i] + SERVO_W / 2]
    z = [Z_SHAFT + SERVO_SHAFT_FROM_END - SERVO_L, Z_SHAFT + SERVO_SHAFT_FROM_END]
    return np.array([x, y, z])


# ----------------------------------------------------------------------------- forearm (W)
CAP_Y = (-32.0, -27.0)     # fork cap plate (top of the forearm)
SPLIT_Z = -10.0            # Forearm_Front (palm side, z < SPLIT_Z) / Forearm_Back split plane
WALL = 2.5
FOREARM_BOTTOM = -215.0    # bottom of the forearm shells (top face of the elbow flange)
FLANGE_PLATE_T = 6.0
ELBOW_Y = FOREARM_BOTTOM - FLANGE_PLATE_T     # -221: flange underside (stand / table)
# outer forearm key sections, interpolated smoothly in y (make_wrist.forearm_profile):
# (y, x half-width, z front, z back, front corner radius, back corner radius)
FOREARM_KEYS = [(-32.0, 35.5, -35.0, 31.5, 14.0, 9.0),
                (-52.0, 35.5, -45.0, 31.5, 14.0, 9.0),
                (-128.0, 35.5, -45.0, 31.5, 14.0, 9.0),
                (FOREARM_BOTTOM, 41.0, -52.0, 37.0, 22.0, 16.0)]   # flare towards the elbow; bay for the bus adapter
ARM_TIP_R = 4.5            # horn arm tip radius around the ball stud (M3 nut pocket inside)

# ----------------------------------------------------------------------------- operating range
FLEX_MAX = np.radians(60.0)
DEV_MAX = np.radians(25.0)
FLEX_LIMIT = np.radians(65.0)    # joint stops in the sim; hard stops on the hardware at ~66 / ~31 deg (see README)
DEV_LIMIT = np.radians(30.0)
SERVO_LIMIT = np.radians(75.0)   # actuator ctrlrange
# Coupled operating ROM (like a human wrist): full +-25 deg deviation up to +-40 deg flexion, tapering to
# +-15 deg at +-60 deg. Outside it the push rods come within ~1 mm of their servo's gear end (and at the
# 60 / 25 corners they would touch it).
ROM_FULL_DEV_UP_TO = np.radians(40.0)
ROM_DEV_AT_MAX_FLEX = np.radians(15.0)


def rom_dev_limit(phi):
    """Max |deviation| (rad) of the operating ROM at flexion phi (rad)."""
    a = abs(phi)
    if a <= ROM_FULL_DEV_UP_TO:
        return DEV_MAX
    t = min(1.0, (a - ROM_FULL_DEV_UP_TO) / (FLEX_MAX - ROM_FULL_DEV_UP_TO))
    return DEV_MAX + t * (ROM_DEV_AT_MAX_FLEX - DEV_MAX)


def thumb_flexion_min(phi):
    """Lowest thumb flexion (m1 - m2)/2, rad, allowed at wrist flexion phi (rad): an over-extended thumb
    (< -70 deg; OpenHand uses -35 deg) reaches the forearm's radial side at palm-ward wrist flexion >= 50 deg."""
    return np.radians(-70.0) if phi > np.radians(40.0) else -np.pi / 2


def clamp_wrist(phi, psi):
    """Clamp a wrist command (rad) into the operating ROM (what the controller should do before the IK)."""
    phi = float(np.clip(phi, -FLEX_MAX, FLEX_MAX))
    lim = rom_dev_limit(phi)
    return phi, float(np.clip(psi, -lim, lim))


# ----------------------------------------------------------------------------- kinematics
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def hand_rot(phi, psi):
    """Hand orientation in W: flexion about -x (proximal, fork->ring), then deviation about +z (ring->hand)."""
    return rot_x(-phi) @ rot_z(psi)


def ball_top(i, phi, psi):
    return hand_rot(phi, psi) @ P_TOP_HAND[i]


def servo_ik(phi, psi):
    """Servo angles (theta1, theta2) [rad] for wrist angles (phi, psi) [rad]; nan if unreachable."""
    out = []
    for i in (1, 2):
        P = ball_top(i, phi, psi)
        a, b = P[1] - Y_SHAFT[i], P[2] - Z_SHAFT
        c2 = ROD_L[i] ** 2 - (SIDE[i] * D_LEVER - P[0]) ** 2
        K = (R_HORN ** 2 + a * a + b * b - c2) / (2 * R_HORN)
        A, B = KAPPA[i] * a, b                       # A sin(t) + B cos(t) = K
        rho = np.hypot(A, B)
        if abs(K) > rho:
            out.append(np.nan)
            continue
        base, off = np.arcsin(K / rho), np.arctan2(B, A)
        cand = [(t + np.pi) % (2 * np.pi) - np.pi for t in (base - off, np.pi - base - off)]
        out.append(min(cand, key=abs))               # the branch through the neutral pose
    return np.array(out)


def _residual(q, th):
    phi, psi = q
    return np.array([np.linalg.norm(ball_bottom(i, th[k]) - ball_top(i, phi, psi)) - ROD_L[i] for k, i in enumerate((1, 2))])


def servo_fk(theta, q0=(0.0, 0.0), iters=30):
    """Wrist angles (phi, psi) for servo angles theta (Newton on the two rod-length constraints)."""
    q = np.array(q0, float)
    for _ in range(iters):
        r = _residual(q, theta)
        if np.abs(r).max() < 1e-10:
            break
        J = np.zeros((2, 2))
        for k in range(2):
            dq = np.zeros(2)
            dq[k] = 1e-7
            J[:, k] = (_residual(q + dq, theta) - r) / 1e-7
        q = q - np.linalg.solve(J, r)
    return q


def jacobian(phi, psi, h=1e-6):
    """d(theta)/d(phi, psi) (2x2). Statics: tau_servo = J^-T tau_joint (virtual work)."""
    J = np.zeros((2, 2))
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h
        J[:, k] = (servo_ik(phi + dq[0], psi + dq[1]) - servo_ik(phi - dq[0], psi - dq[1])) / (2 * h)
    return J


def misalignment(phi, psi):
    """Ball-link misalignment (deg) at the top and bottom joint of each rod: angle between the rod
    and the plane normal to the ball stud (studs along +-x: hand x axis on top, W x at the bottom)."""
    th = servo_ik(phi, psi)
    R = hand_rot(phi, psi)
    out = {}
    for k, i in enumerate((1, 2)):
        u = ball_top(i, phi, psi) - ball_bottom(i, th[k])
        u /= np.linalg.norm(u)
        out[(i, "top")] = np.degrees(abs(np.arcsin(np.clip(u @ (R @ np.array([1.0, 0, 0])), -1, 1))))
        out[(i, "bottom")] = np.degrees(abs(np.arcsin(np.clip(u[0], -1, 1))))
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("C (CAD mm)", C_CAD, " C (root m)", C_ROOT)
    print("e (flexion lever) %.2f mm, d (deviation lever) %.2f mm, horn radius %.1f mm" % (E_LEVER, D_LEVER, R_HORN))
    print("rod lengths", ROD_L, " servo top face |x| %.2f, shaft z %.2f" % (X_TOP_FACE, Z_SHAFT))
    for i in (1, 2):
        print(f"servo {i} box (W):", servo_box(i).tolist())
    for phi, psi in [(0, 0), (60, 0), (-60, 0), (0, 25), (0, -25), (60, 25), (60, -25), (-60, 25), (-60, -25)]:
        th = servo_ik(np.radians(phi), np.radians(psi))
        back = np.degrees(servo_fk(th)) if np.all(np.isfinite(th)) else [np.nan] * 2
        print(f"phi {phi:+4d} psi {psi:+4d} -> theta {np.degrees(th)} -> fk {back}")
