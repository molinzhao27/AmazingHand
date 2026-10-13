#!/usr/bin/env python3
"""Regenerate the 5-digit palm/back shells and grip pads around the 3-phalanx fingers.

The shells of cad/five_finger (make_5f_shells.py) are generated from the swept volumes of the stock
fingers; the 3-phalanx fingers close further (MCP 89 instead of 75 deg), so their proximal shells
reach the stock palm shell's top edge in a full fist (reports/verify_hand5_3seg_shells.txt). This
wrapper runs make_5f_shells.py unchanged, but on model/hand5_3seg, and writes the result to
cad/five_finger/three_segment/stl/palm_shells/ (the existing cad/five_finger/stl is not touched).

Usage: python regen_palm_shells.py [--grid 19] [--vox 0.5]
"""

import sys

import mujoco

from finger_common import CAD_DIR, HERE, MODEL_DIR, REPO

sys.path.insert(0, str(HERE.parent / "five_finger"))
sys.path.insert(0, str(REPO / "cad" / "five_finger"))
import make_5f_shells as S  # noqa: E402
import verify_5f_model as V  # noqa: E402

OUT = CAD_DIR / "stl" / "palm_shells"
_Hand = V.Hand


def hand_3seg(_path, *a, **k):
    """V.Hand on the 3-phalanx hand, mesh geoms only (the new pin cylinders are not meshes)."""
    h = _Hand(MODEL_DIR / "hand5_3seg", *a, **k)
    mesh = lambda gs: [g for g in gs if h.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]  # noqa: E731
    h.geoms = {key: (mesh(x), mesh(y)) for key, (x, y) in h.geoms.items()}
    return h


if __name__ == "__main__":
    S.V.Hand = hand_3seg
    S.OUT_STL = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    S.main()
