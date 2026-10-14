#!/usr/bin/env python3
"""Find, check and render named hand poses (peace sign, middle finger, ...) on any 5-digit variant.

Poses are written as intent per digit (straight or curled, plus abduction), so the same definition
works for the stock-width, slim and 3-phalanx fingers:
  * a finger's "straight" command is found per model: the flexion command f = (m1 - m2)/2 that puts
    its fingertip farthest from its knuckle pivot (at zero abduction a = (m1 + m2)/2);
  * straight fingers are parallel to the forearm at zero spread: the stock hand fans its fingers
    7.5 deg apart, so each gets a base abduction (found per model, against the powered wrist's
    forearm axis) and the pose's spread is added to it - the fingertips are ~34 mm apart instead of
    42-45 mm, and a V is symmetric about the forearm;
  * "curled" is full flexion, f = +90 deg (zero abduction);
  * the thumb is either tucked (fully flexed into the palm, like the stock Victory demo) or folded
    onto target fingertips: the grid command bringing its tip closest to them without touching
    any digit. (The AmazingHand thumb only opposes towards the index: it cannot reach the curled
    ring/pinky, so a peace sign uses the tucked thumb.)
Each pose is then settled and every pair of digits is checked for clearance (surface samples; for
touching pairs the penetration depth is estimated with signed distances). Fingers are checked on the
hand-only model; the wrist (if a wrist model is given for rendering) does not change finger-finger
geometry.

Usage: python hand_poses.py --model AH_Right_5F_Slim [--render-model AH_Right_5F_Slim_Wrist]
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent / "cad" / "five_finger" / "wrist"))
from verify_5f_model import SIM_DIR, Group, Hand, group_distance  # noqa: E402

DIGIT_ID = {"index": 1, "middle": 2, "ring": 3, "thumb": 4, "pinky": 5}
FINGERS = ["index", "middle", "ring", "pinky"]
SPREAD = 30.0  # deg of motor abduction a = (m1 + m2)/2 for a spread finger (stock Victory uses ~25)
TOUCH = 0.225e-3

TUCK = (90.0, -90.0)  # thumb fully flexed into the palm
# name: (digits -> ("straight" | "curled", spread in deg of motor abduction a, added to the parallel base),
#        thumb: "tuck" | target fingertip(s) | None, wrist (flexion, deviation) deg for renders)
POSES = {
    "PeaceSign": ({"index": ("straight", +SPREAD), "middle": ("straight", -SPREAD),
                   "ring": ("curled", 0.0), "pinky": ("curled", 0.0)}, "tuck", (-10.0, 5.0)),
    "MiddleFinger": ({"index": ("curled", 0.0), "middle": ("straight", 0.0),
                      "ring": ("curled", 0.0), "pinky": ("curled", 0.0)}, ["index"], (0.0, 0.0)),
    "Fist": ({d: ("curled", 0.0) for d in FINGERS}, ["index", "middle"], (0.0, 0.0)),
    "Open": ({d: ("straight", 0.0) for d in FINGERS}, None, (0.0, 0.0)),  # relaxed: fingers parallel
}


def log(*a):
    print(*a, flush=True)


class PoseModel:
    def __init__(self, model):
        self.h = Hand(SIM_DIR / model / "mjcf")
        m = self.h.m
        mesh = lambda gs: [g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]  # noqa: E731
        self.h.geoms = {k: (mesh(a), mesh(b)) for k, (a, b) in self.h.geoms.items()}
        self.groups = {dig: Group(self.h, sum(self.h.geoms[dig], []), 1e-3) for dig in self.h.digits}

    def settle(self, cmd_deg, steps=900):
        h = self.h
        h.reset()
        target = np.zeros(h.m.nu)
        for dig, (m1, m2) in cmd_deg.items():
            target[h.ctrl[dig]] = np.radians([m1, m2])
        start = h.d.ctrl.copy()
        for k in range(steps):
            h.d.ctrl[:] = start + (target - start) * min(1.0, (k + 1) / (steps * 0.5))
            mujoco.mj_step(h.m, h.d)

    def site(self, name):
        return self.h.d.site(name).xpos.copy()

    def straight_command(self):
        """Flexion command putting each fingertip farthest from its knuckle (a = 0)."""
        best = {d: (-1, None) for d in FINGERS}
        for f in np.arange(-90, 1, 2.0):
            self.settle({d: (f, -f) for d in FINGERS}, steps=500)
            for d in FINGERS:
                n = DIGIT_ID[d]
                r = np.linalg.norm(self.site(f"tip{n}") - self.site(f"closing_1 ({n})_2"))
                if r > best[d][0]:
                    best[d] = (r, f)
        return {d: float(v[1]) for d, v in best.items()}

    def direction(self, d):
        """Finger direction (tip - knuckle) in degrees from the forearm axis (W y), + towards the pinky."""
        import wrist_design as wd
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, wd.PLATE_QUAT)
        n = DIGIT_ID[d]
        v = (self.site(f"tip{n}") - self.site(f"closing_1 ({n})_2")) @ R.reshape(3, 3) @ wd.R_W2CAD
        return np.degrees(np.arctan2(v[0], v[1]))

    def parallel_abduction(self, straight):
        """Abduction a per finger that makes the straight finger parallel to the forearm axis."""
        dirs = {}
        for a in (0.0, 10.0):
            self.settle({d: (a + straight[d], a - straight[d]) for d in FINGERS}, steps=600)
            dirs[a] = {d: self.direction(d) for d in FINGERS}
        return {d: round(float(-dirs[0.0][d] * 10.0 / (dirs[10.0][d] - dirs[0.0][d])), 1) for d in FINGERS}

    def clearance(self, a, b, groups=None):
        g = groups or self.groups
        return group_distance(g[a], g[a].poses(self.h.d), g[b], g[b].poses(self.h.d), 0.010, skip_root_pair=True)


def finger_commands(pose, straight, base):
    cmd = {}
    for d, (state, a) in pose.items():
        f = straight[d] if state == "straight" else 90.0
        a = a + base[d] if state == "straight" else a
        m1, m2 = a + f, a - f
        lim = max(abs(m1), abs(m2))
        if lim > 90:  # keep both motors in range: give up abduction, not flexion
            shift = (lim - 90) * np.sign(a)
            m1, m2 = m1 - shift, m2 - shift
        cmd[d] = (round(float(m1), 1), round(float(m2), 1))
    return cmd


def search_thumb(pm, cmd, targets):
    """Thumb command folding the thumb tip closest to the target fingertips without touching."""
    best = (np.inf, None, None)
    grid = np.arange(-90, 91, 15.0)
    for t1 in grid:
        for t2 in grid:
            c = dict(cmd, thumb=(t1, t2))
            pm.settle(c, steps=600)
            tgt = np.mean([pm.site(f"tip{DIGIT_ID[t]}") for t in targets], axis=0)
            dist_t = np.linalg.norm(pm.site("tip4") - tgt)
            if dist_t >= best[0]:
                continue
            clear = min(pm.clearance("thumb", d)[0] for d in pm.h.digits if d != "thumb")
            if clear > 0.4e-3:
                best = (dist_t, (float(t1), float(t2)), clear)
    return best


def penetration(pm, ga, gb):
    """Max depth (m) of geom ga's surface samples inside geom gb's mesh (0 if just touching)."""
    m, d = pm.h.m, pm.h.d

    def world_mesh(g):
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        return trimesh.Trimesh(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T, f, process=False)

    a, b = world_mesh(ga), world_mesh(gb)
    if not b.is_watertight:
        return float("nan")
    pts = a.sample(4000, return_index=False)
    sd = trimesh.proximity.signed_distance(b, pts)  # > 0 inside
    return max(0.0, float(sd.max()))


def check_pose(pm, cmd):
    pm.settle(cmd)
    out = []
    groups = {dig: Group(pm.h, sum(pm.h.geoms[dig], []), 0.3e-3) for dig in pm.h.digits}
    digits = pm.h.digits
    for i, a in enumerate(digits):
        for b in digits[i + 1:]:
            dist, who = pm.clearance(a, b, groups)
            if not np.isfinite(dist):
                continue
            depth = penetration(pm, *who) if dist < TOUCH else 0.0
            out.append((a, b, dist, depth, who))
    return out


def render(model, cmds, wrists, out, size=420):
    import wrist_design as wd

    m = mujoco.MjModel.from_xml_path(str(SIM_DIR / model / "mjcf" / "scene_shells.xml"))
    m.vis.global_.offwidth, m.vis.global_.offheight = 2600, 2600
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fl >= 0:
        m.geom_rgba[fl, 3] = 0
    for bdy in range(m.nbody):
        if m.body_mocapid[bdy] >= 0:
            m.geom_rgba[m.body_geomadr[bdy]:m.body_geomadr[bdy] + m.body_geomnum[bdy], 3] = 0
    d = mujoco.MjData(m)
    act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    has_wrist = "wrist_motor1" in act
    r = mujoco.Renderer(m, size, size)
    rows = []
    for name, cmd in cmds.items():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        target = np.zeros(m.nu)
        for dig, (m1, m2) in cmd.items():
            target[[act[f"finger{DIGIT_ID[dig]}_motor1"], act[f"finger{DIGIT_ID[dig]}_motor2"]]] = np.radians([m1, m2])
        if has_wrist:
            target[[act["wrist_motor1"], act["wrist_motor2"]]] = wd.servo_ik(*wd.clamp_wrist(*np.radians(wrists[name])))
        for k in range(1500):
            d.ctrl[:] = target * min(1.0, (k + 1) / 600)
            mujoco.mj_step(m, d)
        row = []
        look = [0.012, -0.03, 0.09] if not has_wrist else [0.01, -0.01, 0.06]
        dist = 0.34 if not has_wrist else 0.45
        for view, az, el in (("palm", 180, -8), ("back", 0, -8), ("3/4", 140, -18)):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
            r.update_scene(d, cam, opt)
            im = Image.fromarray(r.render())
            ImageDraw.Draw(im).text((8, 6), f"{name} - {view}", fill=(255, 255, 255))
            row.append(np.asarray(im))
        rows.append(np.concatenate(row, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(out, quality=88)
    log("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="AH_Right_5F_Slim", help="hand-only model to find and check the poses on")
    ap.add_argument("--render-model", help="model to render (e.g. a *_Wrist model); default = --model")
    ap.add_argument("--poses", nargs="*", default=["Open", "PeaceSign", "MiddleFinger"])
    args = ap.parse_args()

    pm = PoseModel(args.model)
    straight = pm.straight_command()
    log(f"model {args.model}: straight flexion command f = (m1 - m2)/2 per finger (deg): {straight}")
    base = pm.parallel_abduction(straight)
    log(f"  abduction a = (m1 + m2)/2 that makes each straight finger parallel to the forearm (deg): {base}")
    cmds = {}
    for name in args.poses:
        pose, targets, _ = POSES[name]
        cmd = finger_commands(pose, straight, base)
        if targets == "tuck":
            cmd["thumb"] = TUCK
            log(f"{name}: thumb tucked {TUCK}")
        elif targets:
            dist_t, thumb, clear = search_thumb(pm, cmd, targets)
            cmd["thumb"] = thumb
            log(f"{name}: thumb folded to {thumb} (tip {dist_t * 1e3:.1f} mm from the {'/'.join(targets)} fingertip"
                f"{'s' if len(targets) > 1 else ''}, {clear * 1e3:.2f} mm from every other digit)")
        else:
            cmd["thumb"] = (-35.0, 35.0)
        cmds[name] = cmd
        log(f"  commands (m1, m2) deg: " + ", ".join(f"{d} {cmd[d]}" for d in ["index", "middle", "ring", "pinky", "thumb"]))
        pairs = check_pose(pm, cmd)
        close = [p for p in pairs if p[2] < 3e-3]
        log(f"  digit pairs closer than 3 mm: {len(close)} of {len(pairs)} within 10 mm")
        for a, b, dist, depth, who in close:
            state = "TOUCH" if dist < TOUCH else "clear"
            log(f"    {a:<6}/{b:<6} {dist * 1e3:5.2f} mm {state}"
                + (f", penetration {depth * 1e3:.2f} mm" if dist < TOUCH else ""))
    tag = args.render_model or args.model
    render(tag, cmds, {n: POSES[n][2] for n in cmds}, HERE / "renders" / f"poses_{tag}.jpg")


if __name__ == "__main__":
    main()
