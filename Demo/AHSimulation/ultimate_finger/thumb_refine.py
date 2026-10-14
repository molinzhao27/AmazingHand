#!/usr/bin/env python3
"""Thumb commands for the poses where the thumb meets the ultimate index finger (AH_Right_5F_3S).

The re-mapped stock commands make the thumb pass through the 3-phalanx index in CloseHand (thumb folded into the
palm where the index fingertip now curls) and slightly in Perfect and the coarse Pinched pose (verify_hand.py, section
6). For each such pose the thumb's two motor commands are searched, the other digits held: a coarse 7.5 deg grid,
then a 1.5 deg grid around the best, with every thumb part outside the target >= 0.2 mm from every digit (sampled
surfaces). Sampled distances cannot see interpenetration, so the fine candidates are then exact-checked in order of
preference and the first one whose thumb has no exact-boolean overlap with any finger is kept; the result is reported
with exact mesh booleans for every digit pair.

  CloseHand  fingers fully curled; the thumb's distal phalanx rests on the curled index's middle phalanx (a fist
             with the thumb outside). Preference: closest approach.
  Perfect    "OK" sign: thumb tip on the index fingertip, index at f = 30 / 45 and a = -10 / 0 / +10.
  Pinched    thumb tip on the index fingertip, index at f = 0 / 10 / 20 and a = -10 / 0 / +10.
             Target for both: the thumb's distal pad + shell touching the index's fingertip pad + cap (0.1-0.4 mm
             sampled); preference among touching candidates: the smallest pad-to-pad gap. (Pad-on-pad alone is not
             reachable: the 0.35 mm proud pads meet at an angle, so the TPU caps touch first; the search with the
             pads as the only target stops 3.8 mm short for any index abduction from -20 to +20 deg.)

Writes reports/thumb_poses.txt and updates data/hand_poses.json (thumb and index commands of these poses).
Usage: python thumb_refine.py [--poses CloseHand,Perfect,Pinched]
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_hand as VH  # noqa: E402

V = VH.V
SP = 0.5e-3   # search sampling; the final check uses 0.3 mm samples and exact mesh booleans
T1, T2 = (0.0, 75.0), (-82.5, 22.5)   # coarse thumb window (deg): every opposed thumb pose found on the full grid lies inside
TOUCH = 0.4e-3   # sampled distance that counts as touching (the exact check rejects interpenetration)


def search(h, base, thumb_target, digit_target, pads=None, centre=None, span=None, step=7.5):
    """Grid over the thumb (centre +- span, or the window T1 x T2) with base held. Returns every pose whose other thumb
    parts are >= 0.2 mm from every digit as (distance thumb_target <-> digit_target, min other clearance, (t1, t2),
    pad-to-pad distance or None)."""
    tgt_a = VH.Group_cache(h, "thumb", set(thumb_target), SP)
    tgt_b = VH.Group_cache(h, digit_target[0], set(digit_target[1]), SP)
    pad_a = VH.Group_cache(h, "thumb", set(pads[0]), SP) if pads else None
    pad_b = VH.Group_cache(h, digit_target[0], set(pads[1]), SP) if pads else None
    others = []
    for d in h.digits:
        if d == "thumb":
            continue
        excl = set(digit_target[1]) if d == digit_target[0] else set()
        others.append(V.Group(h, [g for g in h.geoms[d][0] if g not in VH.mesh_geoms(h, d, excl)], SP))
    rest_a = V.Group(h, [g for g in h.geoms["thumb"][0] if g not in VH.mesh_geoms(h, "thumb", set(thumb_target))], SP)
    tb = V.Group(h, h.geoms[digit_target[0]][0], SP)
    thumb_all = V.Group(h, h.geoms["thumb"][0], SP)
    VH.settle(h, dict(base, thumb=centre if centre else (0.0, 0.0)))
    if centre is None:
        v1, v2 = np.arange(T1[0], T1[1] + 0.01, step), np.arange(T2[0], T2[1] + 0.01, step)
    else:
        v1 = np.arange(centre[0] - span, centre[0] + span + 1e-6, step)
        v2 = np.arange(centre[1] - span, centre[1] + span + 1e-6, step)
    found = []
    for i, t1 in enumerate(v1):
        for t2 in (v2 if i % 2 == 0 else v2[::-1]):
            if max(abs(t1), abs(t2)) > 90:
                continue
            h.d.ctrl[h.ctrl["thumb"]] = np.radians([t1, t2])
            for _ in range(140):
                mujoco.mj_step(h.m, h.d)
            dt, _ = V.group_distance(tgt_a, tgt_a.poses(h.d), tgt_b, tgt_b.poses(h.d), 0.03)
            clear = min([V.group_distance(thumb_all, thumb_all.poses(h.d), o, o.poses(h.d), 1e-3)[0] for o in others]
                        + [V.group_distance(rest_a, rest_a.poses(h.d), tb, tb.poses(h.d), 1e-3)[0]])
            if clear < 0.2e-3:
                continue
            dp = V.group_distance(pad_a, pad_a.poses(h.d), pad_b, pad_b.poses(h.d), 0.03)[0] if pads else None
            found.append((float(dt), float(clear), (round(float(t1), 1), round(float(t2), 1)),
                          None if dp is None else float(dp)))
    return found


def coarse_score(e):
    return e[0] + (e[3] or 0.0)


def preference(found, pads):
    """Order in which fine candidates are exact-checked."""
    if not pads:
        return sorted(found, key=lambda e: e[0])
    touch = sorted([e for e in found if e[0] <= TOUCH], key=lambda e: e[3])
    return touch + sorted([e for e in found if e[0] > TOUCH], key=coarse_score)


def first_clean(h, exact, base, ordered, n_max=25, min_dt=0.1e-3):
    """First candidate whose thumb has no exact overlap with any other digit."""
    thumb = h.geoms["thumb"][0]
    fingers = [g for d in h.digits if d != "thumb" for g in h.geoms[d][0]]
    tried = 0
    for e in ordered:
        if e[0] < min_dt:
            continue
        VH.settle(h, dict(base, thumb=e[2]))
        vol, depth, _ = exact.overlap(thumb, fingers)
        tried += 1
        if vol <= 1e-3:
            return e, tried
        if tried >= n_max:
            break
    return None, tried


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", default="CloseHand,Perfect,Pinched")
    args = ap.parse_args()
    poses = json.loads((HERE / "data" / "hand_poses.json").read_text())
    with open(HERE / "reports" / "thumb_poses.txt", "w") as fh, contextlib.redirect_stdout(VH.Tee(sys.stdout, fh)):
        h = V.Hand(VH.HAND)
        exact = VH.Exact(h)
        tip = (["thumb_distal_pad", "thumb_distal_shell"], ("index", ["uf_p3_pad", "uf_p3_shell"]))
        pads = (["thumb_distal_pad"], ["uf_p3_pad"])
        V.log("== thumb commands where the thumb meets the ultimate index (coarse 7.5 deg grid, then 1.5 deg)")
        for name in args.poses.split(","):
            pose = {k: tuple(v) for k, v in poses[name].items()}
            candidates, pd = [], None
            if name == "CloseHand":
                candidates.append((pose, (["thumb_distal_pad", "thumb_distal_shell"], ("index", ["uf_p2_shell", "uf_p2_pad"]))))
            elif name in ("Perfect", "Pinched"):
                pd = pads
                for fi in ((30.0, 45.0) if name == "Perfect" else (0.0, 10.0, 20.0)):
                    for ai in (-10.0, 0.0, 10.0):
                        candidates.append((dict(pose, index=VH.cmd(fi, ai)), tip))
            coarse = []
            for cand, (ta, tb) in candidates:
                base = {k: v for k, v in cand.items() if k != "thumb"}
                found = search(h, base, ta, tb, pd)
                if not found:
                    V.log(f"  {name:<10} index {cand['index']}: no coarse thumb command clear of the other digits")
                    continue
                c0 = min(found, key=coarse_score)
                coarse.append((c0, cand, ta, tb))
                V.log(f"  {name:<10} index {cand['index']}: coarse thumb {c0[2]} {'/'.join(ta)} to {tb[0]} "
                      f"{'/'.join(tb[1])} {c0[0] * 1e3:.2f} mm" + (f", pad to pad {c0[3] * 1e3:.2f} mm" if pd else ""))
            best = None
            for c0, cand, ta, tb in sorted(coarse, key=lambda c: coarse_score(c[0]))[:2]:
                base = {k: v for k, v in cand.items() if k != "thumb"}
                found = search(h, base, ta, tb, pd, centre=c0[2], span=9.0, step=1.5) or [c0]
                res, tried = first_clean(h, exact, base, preference(found, pd))
                if res is None:
                    V.log(f"  {name:<10} index {cand['index']}: no overlap-free fine candidate in {tried} tried")
                    continue
                V.log(f"  {name:<10} index {cand['index']}: fine thumb {res[2]} ({tried} exact-checked): {'/'.join(ta)} "
                      f"to {tb[0]} {'/'.join(tb[1])} {res[0] * 1e3:.2f} mm (sampled)"
                      + (f", pad to pad {res[3] * 1e3:.2f} mm" if pd else "")
                      + f", every other thumb part >= {min(res[1], 1e-3) * 1e3:.2f} mm, no exact overlap")
                key = (res[0] > TOUCH, res[3]) if pd else (res[0],)
                if best is None or key < best[2]:
                    best = (res, cand, key)
            if best is None:
                V.log(f"  {name}: kept the previous commands")
                continue
            res, cand, _ = best
            new = dict(cand, thumb=res[2])
            VH.settle(h, new)
            groups = {d: V.Group(h, h.geoms[d][0] + h.geoms[d][1], 0.3e-3) for d in h.digits}
            VH.all_pair_report(h, f"{name} {new} - every digit pair:", groups, exact)
            poses[name] = {k: list(v) for k, v in new.items()}
            (HERE / "data" / "hand_poses.json").write_text(json.dumps(poses, indent=1) + "\n")
            V.log(f"  updated data/hand_poses.json ({name})")


if __name__ == "__main__":
    main()
