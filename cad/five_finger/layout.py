#!/usr/bin/env python3
"""Finger-module layout of the 5-digit hand: where each finger's drive module sits on the palm.

A finger module is everything that belongs to one finger: its two SCS0009 servos, finger frames,
gimbal bushings, both servo-horn -> push-rod -> finger linkages, and the finger itself. The stock
AmazingHand fans the modules (index 0, middle 6, ring 13.5, pinky 21 deg about the palm normal) at a
~33.3 mm knuckle pitch. With a fourth finger and the forearm centred on the four fingers
(cad/five_finger/wrist), the "parallel" layout instead stands every module parallel to the forearm
axis (W y) at PITCH mm:

  * each module is moved rigidly, so the finger, its drive and its kinematics are unchanged;
  * the knuckles keep their stock height along the forearm (the knuckle arc: index and middle level,
    ring 5 mm and pinky 14 mm lower) and their centroid stays on the forearm axis, so the wrist
    design (wrist_design.KNUCKLE_CENTRE, HAND_YAW) does not change;
  * PITCH = 30 mm is the tightest the stock drive allows: the widest parts of a module are the
    levers at the lower ends of the push rods (28.4 mm across), and at 30 mm parallel fingers keep
    1.2-1.8 mm between neighbouring drives over a +-30 deg abduction x full flexion envelope
    (the fanned stock layout: 0.8-1.2 mm). The gaps between the slim fingers shrink from ~14 to
    ~10.6 mm and the palm gets ~10 mm narrower.

Used by Demo/AHSimulation/five_finger/build_5f_model.py (sim) and cad/five_finger/make_5f_plate.py
(palm plate). Frames: the CAD part-studio frame (mm; x index -> pinky, y up the fingers, z to the
back) and the wrist frame W of cad/five_finger/wrist/wrist_design.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "wrist"))
import wrist_design as wd  # noqa: E402

FINGERS = ["index", "middle", "ring", "pinky"]
LAYOUT = "parallel"  # "parallel" | "stock"
PITCH = 30.0  # mm, knuckle pitch across the forearm axis (parallel layout)

# Stock 5-digit layout (AH_Right_5F; the pinky is the ring moved by the middle -> ring step), measured
# from the model: knuckle pivot (site closing_1 (n)_2, CAD mm) and the straight finger's direction
# (tip - knuckle at the straight command, deg from CAD y towards the pinky).
STOCK = {
    "index": (np.array([0.0, 0.0, 3.5]), 0.305),
    "middle": (np.array([32.8, -6.0457, 3.5]), 6.305),
    "ring": (np.array([65.0, -17.0583, 3.5]), 13.805),
    "pinky": (np.array([95.4871, -32.1796, 3.5]), 21.305),
}
MIRROR_X = -50.0  # the left-hand parts are the right ones mirrored about x = MIRROR_X


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def transforms(layout=None, pitch=None):
    """Right hand: {finger: (phi_deg, pivot_old, pivot_new)}. A stock-layout CAD point p of that finger's
    module moves to rot_z(phi) @ (p - pivot_old) + pivot_new (phi counter-clockwise about CAD z)."""
    layout = layout or LAYOUT
    pitch = pitch or PITCH
    out = {}
    if layout == "stock":
        return {f: (0.0, STOCK[f][0], STOCK[f][0]) for f in FINGERS}
    w_old = {f: wd.cad_to_w(STOCK[f][0]) for f in FINGERS}
    cx = np.mean([w_old[f][0] for f in FINGERS])
    for k, f in enumerate(FINGERS):
        w_new = w_old[f].copy()
        w_new[0] = cx + (k - 1.5) * pitch
        new = wd.w_to_cad(w_new)
        new[2] = STOCK[f][0][2]
        phi = STOCK[f][1] - np.degrees(wd.HAND_YAW)  # turns the finger onto the forearm axis
        out[f] = (phi, STOCK[f][0], new)
    return out


def mirrored(tf):
    """The same layout for the left hand (parts mirrored about x = MIRROR_X)."""
    m = lambda p: np.array([2 * MIRROR_X - p[0], p[1], p[2]])  # noqa: E731
    return {f: (-phi, m(p0), m(p1)) for f, (phi, p0, p1) in tf.items()}


def apply(tf_f, p):
    """Move CAD point(s) p (..., 3) of one finger's module."""
    phi, p0, p1 = tf_f
    return (np.asarray(p, float) - p0) @ rot_z(phi).T + p1


# ----------------------------------------------------------------------------- thumb module
# The thumb (finger4) is a stock finger module too: two SCS0009 side by side, finger frames, bushings, gimbal, horns,
# push rods and the thumb. Its module frame is the frame of its knuckle site closing_1 (4)_2: origin on the knuckle
# pivot, u across the thumb, v along it (from the servos to the knuckle), w towards the servo backs (the pad faces -w).
# The index module's frame is the CAD frame, so a module frame is given by its axes in CAD: v's azimuth in the plate
# plane (deg from CAD +y towards CAD -x, the radial side), v's elevation (deg, + towards the back of the hand) and the
# roll of (u, w) about v (deg; 0 = lying flat like a finger, w = back of the hand).
# The stock hand screws the thumb into the wrist interface, its servos standing almost square to the palm (v 70 deg
# out of the plate plane) and hanging below it: its servo pack reaches CAD z = -51 mm (the finger drives: -15.8 mm).
# THUMB_MOUNTS["slim"] moves the whole module rigidly (thumb, drive, kinematics and calibration unchanged) so that the
# pack lies flatter in the lower radial corner of the palm; see THUMB_NOTE.
THUMB_MOUNT = "slim"  # "slim" | "stock"
THUMB_MOUNTS = {
    "stock": dict(azimuth=90.0, elevation=-70.0, roll=90.0, pivot=np.array([0.0, -76.0, -39.5778])),
    # the module on its side (servos stacked across the palm's thickness, u ~ along the plate normal), its long axis
    # along the radial edge of the palm, 7.5 deg off the fingers' direction and tilted 32 deg palm-ward; the knuckle
    # 14 mm in front of the plate, radial of the index module's foot (a 101.4 deg turn of the stock module)
    "slim": dict(azimuth=-7.5, elevation=-32.0, roll=92.0, pivot=np.array([-28.0, -52.5, -13.8])),
}
THUMB_NOTE = """slim mount: the thinnest mount that keeps a human-like thumb (pad-to-pad pinch with the index and middle
fingertips, OK sign, across the palm for the peace sign, on the curled fingers in a fist, spread out for the open hand)
with its servo pack above the wrist neck (W y >= 45), from a search over the module's turn and pivot (FIVE_FINGER.md,
"Thumb mount"); the closed palm shell then rules out folding it into the palm (THUMB_ENVELOPE).
The pack (servos, finger frames, bushings, and the horn/rod sweep over the full +-90 deg command grid) spans CAD z -25.9
... +18.5 mm (stock: -51.1 ... +7.2), W x -85.8 ... -47.0, W y 44.8 ... 102.9; it clears the finger modules, the plate's
finger units and legs, the hand adapter and the fingers' working sweeps."""


# Thumb command envelope of a mount (right hand, motor commands (m1, m2) in deg): m2_min(m1) <= m2 <= m2_max(m1), linear
# between the tabulated m1 (none outside them). Lying along the palm's radial edge, the slim-mount thumb can be commanded
# into the palm: backwards into the plate and the index module, flexed down into its own unit, or folded up against the
# underside of the finger modules. m2_max: over a 7.5 deg grid of +-90 deg, measured on AH_Right_5F, AH_Right_5F_3S and
# AH_Left_5F (mirrored), the moving parts clear every palm part (plate, hand adapter, servos and finger frames of the
# finger modules) with one 7.5 deg grid step to spare. m2_min and the lower end of m1: the thumb clears the closed palm
# shell and grip pad (cad/five_finger/make_5f_shells.py) by >= 1.2 mm on AH_Right_5F and AH_Right_5F_3S; folded deeper
# into the palm it rises to within 1 mm of the finger modules' undersides, where no palm shell fits. The envelope covers
# 22 % of the +-90 deg command square (58 % without the shells). The controller should clamp thumb commands to it
# (thumb_command_ok, clamp_thumb). The stock mount has no such limit.
THUMB_ENVELOPE = {
    "slim": dict(m1=np.arange(-67.5, 90.1, 7.5),
                 m2_min=np.array([30, 30, 30, 30, 30, 30, 30, 30, 22.5, 22.5, 22.5, 22.5, 22.5, 30, 30, 37.5, 45, 52.5, 60, 67.5,
                                  75, 82.5], float),
                 m2_max=np.array([30, 37.5, 45, 52.5, 67.5, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90],
                                 float)),
}


def thumb_command_ok(m1, m2, mount=None, side="right"):
    """Whether the thumb command (m1, m2) (deg) is inside the mount's envelope (left hand: mirrored, (-m2, -m1))."""
    env = THUMB_ENVELOPE.get(mount or THUMB_MOUNT)
    if env is None:
        return True
    if side == "left":
        m1, m2 = -m2, -m1
    lo = np.interp(m1, env["m1"], env["m2_min"])
    hi = np.interp(m1, env["m1"], env["m2_max"])
    return bool(lo - 1e-9 <= m2 <= hi + 1e-9 and env["m1"][0] - 1e-9 <= m1 <= env["m1"][-1] + 1e-9)


def clamp_thumb(m1, m2, mount=None, side="right"):
    """The nearest command inside the envelope, keeping m1 (deg)."""
    env = THUMB_ENVELOPE.get(mount or THUMB_MOUNT)
    if env is None:
        return m1, m2
    if side == "left":
        m1, m2 = -m2, -m1
    m1 = float(np.clip(m1, env["m1"][0], env["m1"][-1]))
    m2 = float(np.clip(m2, np.interp(m1, env["m1"], env["m2_min"]), np.interp(m1, env["m1"], env["m2_max"])))
    return (-m2, -m1) if side == "left" else (m1, m2)


def thumb_axes(azimuth, elevation, roll):
    """Module axes (columns u, v, w) in CAD for v at (azimuth, elevation) and the roll of (u, w) about v."""
    p, e, g = np.radians([azimuth, elevation, roll])
    v = np.array([-np.sin(p) * np.cos(e), np.cos(p) * np.cos(e), np.sin(e)])
    w0 = np.array([0.0, 0.0, 1.0]) - v * v[2]
    w0 /= np.linalg.norm(w0)
    u0 = np.cross(v, w0)
    return np.column_stack([np.cos(g) * u0 + np.sin(g) * w0, v, -np.sin(g) * u0 + np.cos(g) * w0])


def thumb_frame(mount=None):
    """(axes, pivot) of the thumb module in CAD for a mount (default THUMB_MOUNT)."""
    t = THUMB_MOUNTS[mount or THUMB_MOUNT]
    return thumb_axes(t["azimuth"], t["elevation"], t["roll"]), np.asarray(t["pivot"], float)


def thumb_transform(mount=None):
    """Right hand: (R, pivot_old, pivot_new). A stock-mount CAD point p of the thumb module moves to
    R @ (p - pivot_old) + pivot_new (R a rotation in CAD)."""
    a0, p0 = thumb_frame("stock")
    a1, p1 = thumb_frame(mount)
    return a1 @ a0.T, p0, p1


def thumb_mirrored(t):
    """The same thumb transform for the left hand (parts mirrored about x = MIRROR_X)."""
    R, p0, p1 = t
    M = np.diag([-1.0, 1.0, 1.0])
    m = lambda p: np.array([2 * MIRROR_X - p[0], p[1], p[2]])  # noqa: E731
    return M @ R @ M, m(p0), m(p1)


def apply_thumb(t, p):
    """Move CAD point(s) p (..., 3) of the thumb module."""
    R, p0, p1 = t
    return (np.asarray(p, float) - p0) @ R.T + p1


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    tf = transforms()
    for f, (phi, p0, p1) in tf.items():
        print(f"{f:6s} turn {phi:+7.3f} deg about the knuckle {p0} -> knuckle {p1}  (W {wd.cad_to_w(p1)})")
    k = np.array([p1 for _, _, p1 in tf.values()])
    print("knuckle centroid CAD", k.mean(0)[:2], "wrist_design.KNUCKLE_CENTRE", wd.KNUCKLE_CENTRE)
    R, p0, p1 = thumb_transform()
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"thumb mount '{THUMB_MOUNT}': turn {ang:.2f} deg about the knuckle {p0} -> knuckle {p1} (W {wd.cad_to_w(p1)}); "
          f"module axes u, v, w (CAD):\n{thumb_frame()[0].T}")
