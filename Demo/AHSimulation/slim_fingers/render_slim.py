#!/usr/bin/env python3
"""Stock vs slim fingers, side by side (offscreen MuJoCo renders).

Writes to slim_fingers/renders/:
  compare_<pose>.jpg          stock | slim, palm and back views (bare mechanism + finger shells)
  compare_shells_<pose>.jpg   the same with the 5-finger palm/back shells and grip pads
  finger_closeup.jpg          one finger (middle) from the side and from the back, open and fully flexed
  extremes.jpg                the slim hand at the poses where the stock shells touch (hyperextension,
                              full flexion, ring/pinky abducted together)

Usage: python render_slim.py [--size 420]
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "five_finger"))
from render_5f_model import make_hand, settle  # noqa: E402
from verify_5f_model import DEMO_POSES_5F  # noqa: E402

VIEWS = {"palm": (180, -12, 0.34), "back": (0, -12, 0.34), "3/4 palm": (140, -28, 0.36), "pinky side": (90, -10, 0.34)}
LOOKAT = np.array([0.012, -0.03, 0.085])
POSES = {"OpenHand": DEMO_POSES_5F["OpenHand"], "CloseHand": DEMO_POSES_5F["CloseHand"],
         "SpreadHand": DEMO_POSES_5F["SpreadHand"], "Perfect": DEMO_POSES_5F["Perfect"]}


def shoot(h, r, view, lookat=LOOKAT, dist=None):
    az, el, d = VIEWS[view] if isinstance(view, str) else view
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist or d
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    r.update_scene(h.d, cam, opt)
    return r.render()


def label(img, text):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([4, 3, 8 + 6 * len(text), 16], fill=(20, 20, 20))
    dr.text((7, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def grid(rows):
    return np.concatenate([np.concatenate(r, 1) for r in rows], 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--out", type=Path, default=HERE / "renders")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hands = {}
    for tag, model in (("stock", "AH_Right_5F"), ("slim", "AH_Right_5F_Slim")):
        for sh in (False, True):
            hands[(tag, sh)] = make_hand(model, args.size, sh)

    for sh in (False, True):
        for pname, pose in POSES.items():
            rows = []
            for views in (("palm", "back"), ("3/4 palm", "pinky side")):
                row = []
                for v in views:
                    for tag in ("stock", "slim"):
                        h, r = hands[(tag, sh)]
                        settle(h, pose)
                        row.append(label(shoot(h, r, v), f"{tag} - {pname} - {v}"))
                rows.append(row)
            f = args.out / f"compare{'_shells' if sh else ''}_{pname}.jpg"
            Image.fromarray(grid(rows)).save(f, quality=88)
            print("wrote", f.relative_to(HERE))

    # close-up of the middle finger: side and back, open and closed
    rows = []
    for pname, pose in (("open", dict(middle=(-35, 35))), ("hyperextended", dict(middle=(-90, 90))),
                        ("flexed", dict(middle=(90, -90)))):
        row = []
        for tag in ("stock", "slim"):
            h, r = hands[(tag, False)]
            settle(h, pose)
            tip = h.d.site("tip2").xpos
            mid = (tip + np.array([0.0, -0.01, 0.075])) / 2
            for v, (az, el) in (("side", (90, -5)), ("back", (0, -25))):
                row.append(label(shoot(h, r, (az, el, 0.16), lookat=mid), f"{tag} - middle {pname} - {v}"))
        rows.append(row)
    f = args.out / "finger_closeup.jpg"
    Image.fromarray(grid(rows)).save(f, quality=90)
    print("wrote", f.relative_to(HERE))

    # extremes
    ext = {"hyperextended (-90,+90)": {d: (-90, 90) for d in ("index", "middle", "ring", "pinky", "thumb")},
           "fully flexed (+90,-90)": {d: (90, -90) for d in ("index", "middle", "ring", "pinky", "thumb")},
           "ring+pinky abducted together": dict(ring=(-90, -90), pinky=(90, 90))}
    rows = []
    for pname, pose in ext.items():
        row = []
        for tag in ("stock", "slim"):
            h, r = hands[(tag, False)]
            settle(h, pose)
            for v in ("3/4 palm", "back"):
                row.append(label(shoot(h, r, v), f"{tag} - {pname} - {v}"))
        rows.append(row)
    f = args.out / "extremes.jpg"
    Image.fromarray(grid(rows)).save(f, quality=88)
    print("wrote", f.relative_to(HERE))


if __name__ == "__main__":
    main()
