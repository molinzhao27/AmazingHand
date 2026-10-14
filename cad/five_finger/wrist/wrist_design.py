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
W   : origin at the wrist centre C (intersection of the two gimbal axes), axes = the CAD axes turned by
      HAND_YAW about CAD z (the palm normal): W y is the forearm axis, pointing to the fingers.
      x_cad = C_CAD + R_W2CAD @ x_w.
root: the MJCF frame of the hand root body `r_wrist_interface` (m): X_root = POS + R @ x_cad / 1000.

Hand on the forearm
-------------------
The stock AmazingHand fans its fingers towards the pinky side (index along CAD y, each next finger
7.5 deg further out), so with a fourth finger the hand leans ~11 deg to the ulnar side. The forearm
axis therefore runs along the mean finger direction (HAND_YAW from CAD y) and through the centroid
of the four knuckles, like the middle-finger axis of a human hand: the fingers fan symmetrically
about the forearm and the knuckle line arcs down from the middle finger to the pinky. The hand
adapter absorbs the angle: its flange top sits on the stock wrist interface (CAD axes), its
underside and the gimbal below it are in W.

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
INTERFACE_PADS_Y = -101.0                       # CAD y of the recessed faces the adapter flange mates with
H_FLANGE = 25.0                                 # W y of the flange's nominal top (the gimbal is laid out from it)
FLANGE_T = 6.5             # flange thickness at the rim's thinnest point; its underside is at W y = H_FLANGE - FLANGE_T = 18.5
FLANGE_R = 28.0            # 3.5 mm larger than the interface spigot (d49): room for the screw counterbores, and the
                           # flange (centred on the spigot) still covers the gimbal lugs, which sit 5 mm off its axis
FLANGE_MIN = FLANGE_T + 1.2                     # thinnest flange section (rim, CAD y), = the stock (yaw 0) flange

PALM_EXT = 20.0            # mm the wrist sits further from the hand than the adapter needs: a longer palm (the palm
                           # shells reach down to the wrist, so the thumb sits higher above it, as on a human hand)
HAND_YAW = np.radians(10.5)                     # forearm axis from CAD y towards the pinky (the fingers point at
                                                # 0.3 / 6.3 / 13.8 / 21.3 deg, mean 10.4)
KNUCKLE_CENTRE = np.array([48.32, -13.82])      # CAD (x, y): centroid of the index..pinky knuckle pivots (closing_1 (n)_2)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R_W2CAD = rot_z(-HAND_YAW)                      # W y = (sin yaw, cos yaw, 0) in CAD
R_W2ROOT = R_CAD2ROOT @ R_W2CAD


def _wrist_centre():
    """C on the forearm axis (through KNUCKLE_CENTRE along W y), as high as the adapter allows. The flange is a
    cylinder of radius FLANGE_R about the interface's spigot axis (CAD y), from the pad plane CAD y =
    INTERFACE_PADS_Y down to the W plane y = H_FLANGE - FLANGE_T (as at zero yaw, so the gimbal below keeps its
    clearances); it must be at least FLANGE_MIN thick where that plane is highest, on the radial rim. PALM_EXT
    lowers C further (a longer adapter, so a longer palm)."""
    s_, c_ = np.sin(HAND_YAW), np.cos(HAND_YAW)
    x_rim = INTERFACE_AXIS[0] - FLANGE_R
    t = c_ * (KNUCKLE_CENTRE[1] - INTERFACE_PADS_Y + FLANGE_MIN) + (H_FLANGE - FLANGE_T) - (x_rim - KNUCKLE_CENTRE[0]) * s_
    t += PALM_EXT
    return np.array([KNUCKLE_CENTRE[0] - t * s_, KNUCKLE_CENTRE[1] - t * c_, INTERFACE_AXIS[1]])


C_CAD = _wrist_centre()                         # (22.58, -152.69, -11.5); 141 mm below the knuckle centroid


def cad_to_root(p_cad):
    """CAD mm -> root m (points, (..., 3))."""
    return PLATE_POS + np.asarray(p_cad, float) @ R_CAD2ROOT.T / 1000.0


def w_to_cad(p_w):
    return np.asarray(p_w, float) @ R_W2CAD.T + C_CAD


def cad_to_w(p_cad):
    return (np.asarray(p_cad, float) - C_CAD) @ R_W2CAD


def w_to_root(p_w):
    return cad_to_root(w_to_cad(p_w))


def dir_w_to_root(v):
    return np.asarray(v, float) @ R_W2ROOT.T


C_ROOT = cad_to_root(C_CAD)
# The sim stands the forearm upright: the world is the root frame turned so that W y is world z (the hand root
# stays at the world origin). In the root frame W y is (0, -sin yaw, cos yaw), so this is a turn of -yaw about x.
WORLD_QUAT = np.array([np.cos(-HAND_YAW / 2), np.sin(-HAND_YAW / 2), 0.0, 0.0])
INTERFACE_TOP_W = cad_to_w([INTERFACE_AXIS[0], INTERFACE_PADS_Y, INTERFACE_AXIS[1]])   # flange top centre (on the spigot axis)

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
# Outer forearm key sections: superellipses |x / a|^n + |(z - zm) / b|^n = 1 with b = zm - zf, n = nf on the palm side
# and b = zb - zm, n = nb on the back, interpolated in y by a monotone cubic (make_wrist.forearm_profile).
# (y, x half-width a, z front zf, z back zb, z of the widest line zm, nf, nb)
FOREARM_KEYS = [(-32.0, 41.5, -44.5, 34.5, -3.0, 2.8, 2.3),    # wrist end under the cap: narrowest; rounder back corners
                                                             # clear the hand at full extension, the rod slots still fit
                (-50.0, 44.3, -48.0, 36.5, -3.0, 2.8, 2.8),    # servo region: holds the frame and the horn-arm tips
                (-90.0, 46.0, -49.0, 37.4, -3.2, 2.8, 2.8),
                (-130.0, 48.0, -50.4, 38.7, -3.5, 2.75, 2.75),
                (-168.0, 50.0, -52.0, 40.0, -4.0, 2.7, 2.7),   # muscle belly (bay for the bus adapter)
                (FOREARM_BOTTOM, 48.8, -51.2, 39.2, -4.0, 2.7, 2.7)]   # eases into the elbow flange
CUFF_Y = (-45.0, -48.0)    # wrist cuff: the outer surface stands CUFF_PROUD proud from the cap down to CUFF_Y[0],
CUFF_PROUD = 1.5           # blended out by CUFF_Y[1] (<= 37 deg, prints standing)
CAP_FILLET = 3.0           # rounded top edge of the fork cap: the top of the cuff ring
GROOVE_Y = -43.5           # panel line near the lower edge of the cuff: V groove GROOVE_W wide and GROOVE_D deep from
                           # this height down (in the cuff band, so the wall stays >= WALL + CUFF_PROUD - GROOVE_D)
GROOVE_W, GROOVE_D = 1.0, 0.5
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
    """Lowest thumb flexion (m1 - m2)/2, rad, allowed at wrist flexion phi (rad). With the longer palm
    (PALM_EXT = 20 mm) even the fully over-extended thumb (-90 deg) clears the forearm by >= 7.6 mm over the whole
    ROM (Demo/AHSimulation/wrist/check_thumb_rule.py), so there is no limit. Without it (PALM_EXT = 0) an
    over-extended thumb (< -70 deg; OpenHand uses -35 deg) reached the forearm's radial side at palm-ward wrist
    flexion >= 50 deg."""
    if PALM_EXT >= 20.0:
        return -np.pi / 2
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
    print("C (CAD mm)", C_CAD, " C (root m)", C_ROOT, " hand yaw %.1f deg" % np.degrees(HAND_YAW))
    print("forearm axis at the knuckle centroid's height: CAD x %.2f (centroid %.2f)"
          % (C_CAD[0] + (KNUCKLE_CENTRE[1] - C_CAD[1]) * np.tan(HAND_YAW), KNUCKLE_CENTRE[0]))
    print("flange top centre (W)", INTERFACE_TOP_W)
    print("e (flexion lever) %.2f mm, d (deviation lever) %.2f mm, horn radius %.1f mm" % (E_LEVER, D_LEVER, R_HORN))
    print("rod lengths", ROD_L, " servo top face |x| %.2f, shaft z %.2f" % (X_TOP_FACE, Z_SHAFT))
    for i in (1, 2):
        print(f"servo {i} box (W):", servo_box(i).tolist())
    for phi, psi in [(0, 0), (60, 0), (-60, 0), (0, 25), (0, -25), (60, 25), (60, -25), (-60, 25), (-60, -25)]:
        th = servo_ik(np.radians(phi), np.radians(psi))
        back = np.degrees(servo_fk(th)) if np.all(np.isfinite(th)) else [np.nan] * 2
        print(f"phi {phi:+4d} psi {psi:+4d} -> theta {np.degrees(th)} -> fk {back}")
