#!/usr/bin/env python3
"""Widen the stock R-Hand_Plate with a fourth finger (pinky) mount. Needs CadQuery (pip install cadquery).

All CAD parts share one part-studio frame, in mm: x points from the index towards the pinky,
y up along the fingers (the index pivot is the origin) and z towards the back of the hand.

The pinky is the ring finger moved by the finger2 -> finger3 step D (-7.5 deg about z plus a
translation), the same transform used for the MuJoCo model (it is read from the stock MJCF, so
the two cannot drift apart). The new plate is the union of:

  P_in  the stock plate, cut at D(L), where L is a seam between the middle and ring servo pairs
        (D(L) runs along the stock plate's pinky-side edge, so almost nothing is removed);
  N     a strip copied from the middle unit just inside L, moved by D: fills the rounded
        top-right corner of the stock plate so the top edge runs on to the pinky unit;
  D(C)  C = the ring finger's unit (servo window, pivot boss, screw holes, ribs, outer rim), i.e.
        the stock plate outside L and above B (a line just below the ring's lowest bracket), moved
        by D: an exact copy of the ring mount, so the stock finger module and servo mounting
        pattern fit unchanged;
  F     a web (back sheet plus an outer rim, same thickness as the stock ones) from the bottom of
        the pinky unit down to the plate's foot on the wrist interface.

Writes cad/five_finger/step/R-Hand_Plate_5F.step, cad/five_finger/stl/R-Hand_Plate_5F.stl (mm,
same frame as the stock parts) and cad/five_finger/R-Hand_Plate_5F_preview.png, plus the left
plate L-Hand_Plate_5F: the stock L-Hand_Plate is exactly the R-Hand_Plate mirrored about x = -50
(checked here), so the left 5-finger plate is the right one mirrored the same way.
"""

import sys
from pathlib import Path

import cadquery as cq
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "Demo" / "AHSimulation" / "five_finger"))
from build_5f_model import SRC, get_pose, parse, part_geoms, quat_conj, quat_mul, rotate  # noqa: E402

STOCK_STEP = REPO / "cad" / "step" / "Amazing Hand Parts - R-Hand_Plate.step"
STOCK_STEP_L = REPO / "cad" / "step" / "Amazing Hand Parts - L-Hand_Plate.step"
OUT_STEP = HERE / "step" / "R-Hand_Plate_5F.step"
OUT_STL = HERE / "stl" / "R-Hand_Plate_5F.stl"
OUT_STEP_L = HERE / "step" / "L-Hand_Plate_5F.step"
OUT_STL_L = HERE / "stl" / "L-Hand_Plate_5F.stl"
MIRROR_X = -50.0  # L-Hand_Plate = R-Hand_Plate mirrored about the plane x = MIRROR_X

# Seam between the middle and ring units (plate frame, mm): passes between the two servo pairs,
# between the middle unit's right screw hole / pivot boss and the ring unit's, and between the
# lower finger frames (finger_frame_2). Every middle-unit feature is on one side, every ring-unit
# feature on the other.
L_BOT, L_TOP = np.array([42.8, -52.0]), np.array([47.5, -17.0])
# Bottom of the copied ring unit: parallel to the ring's lower finger frame, ~2 mm below it.
B_ANGLE = -13.5  # deg, ring finger fan angle in this frame
B_POINT = np.array([45.0, -62.0])
STRIP_W = 10.0  # mm, width of the middle-unit strip N copied next to the seam
SHEET_Z = (12.5, 15.5)  # back sheet of the stock plate
RIM_Z = (6.5, 15.5)  # outer rim rib of the stock plate
RIM_W = 2.0
FOOT_Y = -85.0  # plate bottom, sits on the wrist interface


def step_transform():
    """finger2 -> finger3 step in the plate frame: (rotation about z in deg, translation in mm)."""
    root = parse(SRC / "robot.xml").getroot().find("worldbody/body")
    geoms = part_geoms(root)
    plate_p, plate_q = get_pose(geoms["r_hand_plate"][1])
    p2, q2 = get_pose(geoms["finger_frame_1_2"][1])
    p3, q3 = get_pose(geoms["finger_frame_1_3"][1])
    qD = quat_mul(q3, quat_conj(q2))
    tD = p3 - rotate(qD, p2)
    # conjugate into the plate frame (and metres -> mm)
    qL = quat_mul(quat_conj(plate_q), quat_mul(qD, plate_q))
    tL = rotate(quat_conj(plate_q), rotate(qD, plate_p) + tD - plate_p) * 1000
    assert np.allclose(qL[1:3], 0, atol=1e-6) and abs(tL[2]) < 1e-6, "step is not a rotation about the plate normal"
    angle = 2 * np.degrees(np.arctan2(qL[3], qL[0]))
    return angle, tL[:2]


ANGLE, T = step_transform()


def D(shape):
    return shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), ANGLE).translate(cq.Vector(T[0], T[1], 0))


def d_pt(p, inverse=False):
    a = np.radians(ANGLE)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return R.T @ (np.asarray(p) - T) if inverse else R @ np.asarray(p) + T


def prism(pts, z=(-60.0, 60.0)):
    wp = cq.Workplane("XY").workplane(offset=z[0])
    return wp.polyline([tuple(map(float, p)) for p in pts]).close().extrude(z[1] - z[0]).val()


def line(a, b, ext=200.0):
    u = (b - a) / np.linalg.norm(b - a)
    return a - u * ext, b + u * ext


def side_region(a, b, left=True, far=400.0):
    """Half-plane prism on one side of the line a->b (left = counter-clockwise side)."""
    a, b = line(a, b)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]]) * (far if left else -far)
    return prism([a, b, b + n, a + n])


def intersect_lines(p, u, q, v):
    s = np.linalg.solve(np.column_stack([u, -v]), q - p)
    return p + s[0] * u


def build():
    stock = cq.importers.importStep(str(STOCK_STEP)).val()
    print(f"step D in plate frame: {ANGLE:.4f} deg about z, t = ({T[0]:.4f}, {T[1]:.4f}) mm")

    bu = np.array([np.cos(np.radians(B_ANGLE)), np.sin(np.radians(B_ANGLE))])
    B0, B1 = B_POINT, B_POINT + bu
    # left of L (pointing up) = middle side; right = ring side. Above B = left of B (pointing +x).
    ring_side_L = side_region(L_BOT, L_TOP, left=False)
    above_B = side_region(B0, B1, left=True)
    DL_bot, DL_top = d_pt(L_BOT), d_pt(L_TOP)

    P_in = stock.intersect(side_region(DL_bot, DL_top, left=True))
    C = stock.intersect(ring_side_L).intersect(above_B)
    DC = D(C)

    lu = (L_TOP - L_BOT) / np.linalg.norm(L_TOP - L_BOT)
    ln = np.array([-lu[1], lu[0]])  # towards the middle unit
    strip = prism([*line(L_BOT, L_TOP), *reversed(line(L_BOT + ln * STRIP_W, L_TOP + ln * STRIP_W))])
    band = side_region(d_pt(B0, inverse=True), d_pt(B1, inverse=True), left=True)  # D^-1(B): same band, middle unit
    N = D(stock.intersect(strip).intersect(band))

    # web F below the pinky unit: back sheet + outer rim, from the seam down to the foot
    DB0, DB1 = d_pt(B0), d_pt(B1)
    dbu = (DB1 - DB0) / np.linalg.norm(DB1 - DB0)
    dlu = (DL_top - DL_bot) / np.linalg.norm(DL_top - DL_bot)
    corner_in = intersect_lines(DL_bot, dlu, DB0, dbu)  # bottom of the seam
    # outer end of the copied unit's bottom edge: where B leaves the stock plate, moved by D
    xs = [x for x in np.arange(40, 90, 0.25)
          if stock.isInside(cq.Vector(*(B0 + bu * (x - B0[0]) / bu[0] + np.array([0, 0.3])), 14.0))]
    corner_out = d_pt(B0 + bu * (max(xs) - B0[0]) / bu[0])
    foot_out = np.array([corner_out[0] - 18.0, FOOT_Y])
    foot_in = np.array([58.0, FOOT_Y])
    # The web meets D(C) along the seam D(L) and its bottom D(B) (overlapping it by 0.05 mm so the
    # solids fuse) and overlaps the stock plate above the foot; it never reaches into the pinky unit.
    eps = 0.05
    seam_top = DL_bot + dlu * ((-61.0 - DL_bot[1]) / dlu[1])  # on D(L) where the stock edge meets it
    into_dc = np.array([dlu[1], -dlu[0]]) * eps  # normal to D(L), towards the pinky unit
    up = np.array([-dbu[1], dbu[0]]) * eps  # normal to D(B), into the pinky unit
    web = [seam_top + into_dc, corner_in + into_dc + up, corner_out + up, foot_out, foot_in,
           np.array([foot_in[0], -60.0])]
    F = prism(web, SHEET_Z)
    rim_dir = (foot_out - corner_out) / np.linalg.norm(foot_out - corner_out)
    rim_n = np.array([rim_dir[1], -rim_dir[0]])  # inwards (rim_dir runs right to left along the bottom edge)
    rim = prism([corner_out + up, foot_out, foot_out + rim_n * RIM_W, corner_out + up + rim_n * RIM_W], RIM_Z)
    print(f"seam D(L): {np.round(DL_bot, 2)} -> {np.round(DL_top, 2)}; pinky unit bottom {np.round(corner_in, 2)} "
          f"-> {np.round(corner_out, 2)}; web foot {np.round(foot_in, 1)} -> {np.round(foot_out, 1)}")

    plate = P_in.fuse(N).fuse(DC).fuse(F).fuse(rim).clean()
    solids = plate.Solids()
    assert len(solids) == 1, f"result has {len(solids)} solids"
    plate = solids[0]
    assert plate.isValid()
    print(f"stock plate {stock.Volume() / 1000:.2f} cm3 -> 5-finger plate {plate.Volume() / 1000:.2f} cm3; "
          f"removed from stock {stock.cut(plate).Volume():.1f} mm3; "
          f"bbox x [{plate.BoundingBox().xmin:.1f}, {plate.BoundingBox().xmax:.1f}] "
          f"y [{plate.BoundingBox().ymin:.1f}, {plate.BoundingBox().ymax:.1f}]")
    return stock, plate


def preview(stock, plate, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    for ax, shape, title in ((axs[0], stock, "stock R-Hand_Plate"), (axs[1], plate, "R-Hand_Plate_5F")):
        verts, tris = shape.tessellate(0.1, 0.3)
        v = np.array([(p.x, p.y, p.z) for p in verts])
        t = v[np.array(tris)]
        t = t[np.argsort(t[:, :, 2].mean(1))]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        shade = 0.35 + 0.6 * np.abs(n[:, 2]) / (np.linalg.norm(n, axis=1) + 1e-12)
        ax.add_collection(PolyCollection(t[:, :, :2], facecolors=np.outer(shade, [0.75, 0.75, 0.8]), edgecolors="none"))
        ax.set_xlim(-20, 120)
        ax.set_ylim(-95, 15)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{title} (seen from the back, plate frame, mm)")
    for p, c in ((L_BOT, "g"), (d_pt(L_BOT), "r")):
        q = L_TOP if c == "g" else d_pt(L_TOP)
        axs[1].plot(*np.array(line(p, q, 25)).T, c + "--", lw=1)
    plt.tight_layout()
    plt.savefig(path, dpi=60)


def mirror(shape):
    return shape.mirror("YZ", cq.Vector(MIRROR_X, 0, 0))


def export(shape, step, stl):
    step.parent.mkdir(parents=True, exist_ok=True)
    stl.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(cq.Workplane().add(shape), str(step))
    cq.exporters.export(cq.Workplane().add(shape), str(stl), tolerance=0.02, angularTolerance=0.15)
    print(f"wrote {step.relative_to(REPO)}, {stl.relative_to(REPO)}")


def main():
    stock, plate = build()
    export(plate, OUT_STEP, OUT_STL)
    preview(stock, plate, HERE / "R-Hand_Plate_5F_preview.png")

    stock_l = cq.importers.importStep(str(STOCK_STEP_L)).val()
    diff = mirror(stock).cut(stock_l).Volume() + stock_l.cut(mirror(stock)).Volume()
    assert diff < 1.0, f"L-Hand_Plate is not the mirrored R-Hand_Plate ({diff:.1f} mm3 differ)"
    export(mirror(plate), OUT_STEP_L, OUT_STL_L)


if __name__ == "__main__":
    main()
