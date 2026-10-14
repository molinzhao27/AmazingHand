#!/usr/bin/env python3
"""Neighbour-finger clearance baseline for another 5-digit model on the same finger layout, with the exact method
of verify_hand.py / five_finger/verify_5f_model.py (9 x 9 grid per digit, 6561 pose pairs per finger pair, surface
samples every 1 mm refined to 0.3 mm, "touching" < 0.225 mm). Used to compare AH_Right_5F_3S with the stock-finger
AH_Right_5F after the hand moved to the parallel 30 mm finger layout (cad/five_finger/layout.py).

Writes reports/baseline_<model>.txt.   Usage: python baseline_neighbours.py [--model AH_Right_5F] [--grid 9]
"""

import argparse
import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "five_finger"))
sys.path.insert(0, str(HERE))
import verify_5f_model as V  # noqa: E402
from verify_hand import Tee  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="AH_Right_5F")
    ap.add_argument("--grid", type=int, default=9)
    args = ap.parse_args()
    with open(HERE / "reports" / f"baseline_{args.model}.txt", "w") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        h = V.Hand(V.SIM_DIR / args.model / "mjcf")
        V.log(f"== neighbour-finger baseline: {args.model} (current layout), {args.grid}x{args.grid} grid per digit")
        tables, poses = V.check_interference(h, args.grid, 1.0, 0.3)
        keys = list(poses)
        table, who = V.clearance_table(h, poses, V.digit_geoms(h, "index"), V.digit_geoms(h, "middle"), 1e-3, 0.3e-3,
                                       0.225e-3)
        V.report_pair(h, "index (A) vs middle (B)", keys, table, who, 0.225e-3)


if __name__ == "__main__":
    main()
