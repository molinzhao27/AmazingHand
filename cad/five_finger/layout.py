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


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    tf = transforms()
    for f, (phi, p0, p1) in tf.items():
        print(f"{f:6s} turn {phi:+7.3f} deg about the knuckle {p0} -> knuckle {p1}  (W {wd.cad_to_w(p1)})")
    k = np.array([p1 for _, _, p1 in tf.values()])
    print("knuckle centroid CAD", k.mean(0)[:2], "wrist_design.KNUCKLE_CENTRE", wd.KNUCKLE_CENTRE)
