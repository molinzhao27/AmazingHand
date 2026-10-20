"""Live webcam hand tracking -> AmazingHand servo commands -> MuJoCo simulation.

Pipeline (one window, runs with plain `python` on macOS, no mjpython or dora needed):
  camera frame(s) -> hand tracker (MediaPipe, or WiLoR with --tracker wilor; 21 3D keypoints)
  -> One Euro filter -> per-digit curl + abduction angles
  -> with two cameras, per-finger fusion weighted by how well each camera sees the hand
  -> AmazingHand motor targets (2 servos per digit) -> MuJoCo position actuators
  -> offscreen render shown next to the camera image.

The motor targets are exactly what the real hand takes (degrees per servo), so the
same numbers can later go to the Feetech SCS0009 servos (see --log).

The robot is the final 5-digit hand, AH_Right_5F_3S_Wrist (FIVE_FINGER.md), with its wrist held
straight; --model stock5 / stock4 drives the stock hand that build_model.py sets up instead.

Keys:  q/Esc quit   o calibrate OPEN (hand flat, fingers together)
       f calibrate FIST   r reset calibration   [ ] rotate robot view   v palm/back view
"""

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np

from multicam import CameraStream, fuse, list_cameras, palm_cosine
from trackers import CONNECTIONS, make_tracker

HERE = Path(__file__).parent

# MediaPipe landmark chains per digit: (base, j1, j2, j3, tip). Fingers start at the wrist.
CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
# Which robot finger (MJCF "fingerN_motor1/2") each human digit drives.
ROBOT_FINGER = {"index": 1, "middle": 2, "ring": 3, "thumb": 4, "pinky": 5}
FINGERS = ["index", "middle", "ring", "pinky"]          # thumb side to little-finger side

# Retargeting ranges, in degrees. Motor convention (matches the AmazingHand examples):
#   motor1 = flex + abd, motor2 = -flex + abd, i.e. flex = (m1 - m2)/2 and abd = (m1 + m2)/2.
# Stock fingers: flex -35 ~ straight, +90 ~ fully closed.
FLEX_OPEN, FLEX_CLOSED = -35.0, 90.0
THUMB_FLEX_OPEN, THUMB_FLEX_CLOSED = -20.0, 80.0
ABD_GAIN, ABD_MAX = 1.5, 35.0
THUMB_ABD_GAIN, THUMB_ABD_MAX = 1.0, 40.0
MOTOR_LIMIT = 90.0
SERVO_SPEED = 600.0             # deg/s, the SCS0009's no-load speed (0.1 s per 60 deg)

# Final hand (AH_Right_5F_3S_Wrist, FIVE_FINGER.md "Styled 3-phalanx fingers (final)"): the 3-phalanx fingers are
# straight at flex -56 and must not go below -60. The sim does not collide the digits with each other, so the
# mapping keeps them apart itself: spread is limited to the peace sign's +-30 and fades out by the time a finger is
# curled as far as in the pinch, and neighbouring fingers may turn towards each other by at most FINAL_ABD_CONVERGE
# in total (>= 1.3 mm apart).
FINAL_FLEX_OPEN, FINAL_FLEX_CLOSED, FINAL_FLEX_MIN = -56.0, 90.0, -60.0
FINAL_ABD_MAX, FINAL_ABD_CONVERGE = 30.0, 20.0
# Its thumb sits on the slim mount and moves nothing like a finger, so it is blended between checked poses (m1, m2)
# from Demo/AHSimulation/ultimate_finger/data/hand_poses.json and five_finger/hand_poses.py (TUCK, FIST_THUMB):
THUMB_OPEN = (-52.5, 45.0)      # extended along the radial side of the palm (OpenHand)
THUMB_TUCK = (48.0, 41.0)       # across the palm towards the curled ring and pinky (Victory)
THUMB_POINT = (-1.0, 23.5)      # on the curled middle while the index points (Index_Pointing)
THUMB_PINCH = (-7.5, 42.0)      # pad to pad with the index fingertip at flex 0 (Pinched)
THUMB_FIST = (-19.5, 45.0)      # on the side of the curled index (CloseHand)
S_PINCH = -FINAL_FLEX_OPEN / (FINAL_FLEX_CLOSED - FINAL_FLEX_OPEN)   # index curl of the pinch
# How far the thumb may close (0..1 of the way from open to its rest pose) before it comes within 0.5 mm of a
# finger, over the index curl (rows, 0..1 in eighths) and the middle curl (columns, in quarters), with the ring and
# pinky curled and the index and middle spread towards the thumb as far as they can be. Recompute with
# thumb_limits.py when the poses above change.
THUMB_CLOSE_MAX = np.array([
    [1.00, 1.00, 0.70, 0.85, 1.00],
    [1.00, 1.00, 0.85, 0.95, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00],
    [0.70, 0.70, 0.70, 0.70, 0.70],
    [0.65, 0.65, 0.65, 0.65, 0.65],
    [0.70, 0.70, 0.70, 0.70, 0.70],
    [0.80, 0.80, 0.80, 0.80, 0.80],
    [0.85, 0.85, 0.85, 0.85, 0.85],
    [0.95, 0.95, 0.95, 0.95, 0.95],
])
# and kept inside the mount's envelope, m2_min(m1) <= m2 <= m2_max(m1), linear between the tabulated m1
# (copied from THUMB_ENVELOPE["slim"] in cad/five_finger/layout.py).
THUMB_ENV_M1 = np.arange(-67.5, 90.1, 7.5)
THUMB_ENV_M2_MIN = np.array([30, 30, 30, 30, 30, 30, 30, 30, 22.5, 22.5, 22.5, 22.5, 22.5, 30, 30, 37.5, 45, 52.5, 60,
                             67.5, 75, 82.5])
THUMB_ENV_M2_MAX = np.array([30, 37.5, 45, 52.5, 67.5, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                             90])

SIM_MODELS = HERE.parent / "AHSimulation" / "AHSimulation"


@dataclass(frozen=True)
class Robot:
    """A simulated hand, the human digits it follows, and its finger ranges (deg). Defaults: the stock hand."""
    scene: Path
    digits: tuple = ("index", "middle", "ring", "thumb", "pinky")
    flex_open: float = FLEX_OPEN        # finger flex that holds a finger straight
    flex_closed: float = FLEX_CLOSED    # fully curled
    flex_min: float = FLEX_OPEN         # never command less flex than this
    abd_max: float = ABD_MAX
    abd_fade_end: float = 1.0           # curl (0..1) by which a finger's abduction has faded out
    abd_converge: float = np.inf        # most two neighbouring fingers may turn towards each other
    thumb_poses: bool = False           # thumb blended between checked poses (final hand) instead of flex + abd
    lookat: tuple = (0.02, -0.025, 0.085)   # render camera
    distance: float = 0.30


ROBOTS = {
    "final": Robot(SIM_MODELS / "AH_Right_5F_3S_Wrist" / "mjcf" / "scene_shells.xml",
                   flex_open=FINAL_FLEX_OPEN, flex_closed=FINAL_FLEX_CLOSED, flex_min=FINAL_FLEX_MIN,
                   abd_max=FINAL_ABD_MAX, abd_fade_end=S_PINCH, abd_converge=FINAL_ABD_CONVERGE, thumb_poses=True,
                   lookat=(0.015, -0.015, 0.105), distance=0.32),
    "stock5": Robot(HERE / "models" / "AH_Right" / "scene_5f.xml"),
    "stock4": Robot(HERE / "models" / "AH_Right" / "scene_4f.xml", digits=("index", "middle", "ring", "thumb"),
                    lookat=(0.02, -0.01, 0.085)),
}


@dataclass
class Calibration:
    """Human angle ranges. Defaults are rough; press o / f to measure your own."""
    curl_open: dict = field(default_factory=lambda: {
        "thumb": 20.0, "index": 30.0, "middle": 30.0, "ring": 30.0, "pinky": 30.0})
    curl_fist: dict = field(default_factory=lambda: {
        "thumb": 80.0, "index": 200.0, "middle": 210.0, "ring": 210.0, "pinky": 200.0})
    abd_neutral: dict = field(default_factory=lambda: {
        "thumb": 35.0, "index": 8.0, "middle": 0.0, "ring": -6.0, "pinky": -14.0})


class OneEuro:
    """One Euro filter (Casiez et al. 2012), vectorised. Smooth when still, responsive when moving."""

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


def angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def hand_angles(p):
    """p: (21, 3) world landmarks. Returns {digit: (curl_deg, abd_deg)}.

    curl = sum of flexion at the three joints of the digit (0 = straight).
    abd  = signed angle of the proximal bone in the palm plane, + towards the thumb side.
           For the thumb, abd is the spread angle between thumb and index metacarpals.
    """
    up = p[9] - p[0]                      # wrist -> middle MCP
    up /= np.linalg.norm(up)
    lat = p[5] - p[17]                    # pinky MCP -> index MCP (towards thumb)
    lat -= np.dot(lat, up) * up
    lat /= np.linalg.norm(lat)
    normal = np.cross(up, lat)            # palm normal (sign depends on handedness; unused for signs)

    out = {}
    for name, (a, b, c, d, e) in CHAINS.items():
        bones = [p[b] - p[a], p[c] - p[b], p[d] - p[c], p[e] - p[d]]
        if name == "thumb":
            curl = angle(bones[1], bones[2]) + angle(bones[2], bones[3])     # MCP + IP
            abd = angle(p[2] - p[1], p[5] - p[0])                            # thumb/index spread
        else:
            curl = angle(bones[0], bones[1]) + angle(bones[1], bones[2]) + angle(bones[2], bones[3])
            prox = bones[1] - np.dot(bones[1], normal) * normal              # project to palm plane
            abd = np.degrees(np.arctan2(np.dot(prox, lat), np.dot(prox, up)))
        out[name] = (curl, abd)
    return out


def lerp(a, b, s):
    return tuple(x + s * (y - x) for x, y in zip(a, b))


def clamp_thumb(m1, m2):
    """Nearest final-hand thumb command inside the slim mount's envelope, keeping m1 (deg)."""
    m1 = float(np.clip(m1, THUMB_ENV_M1[0], THUMB_ENV_M1[-1]))
    lo, hi = np.interp(m1, THUMB_ENV_M1, THUMB_ENV_M2_MIN), np.interp(m1, THUMB_ENV_M1, THUMB_ENV_M2_MAX)
    return m1, float(np.clip(m2, lo, hi))


def interp2(table, x, y):
    """Bilinear lookup in a table spanning 0..1 on both axes."""
    i, j = x * (table.shape[0] - 1), y * (table.shape[1] - 1)
    i0, j0 = min(int(i), table.shape[0] - 2), min(int(j), table.shape[1] - 2)
    u, v = i - i0, j - j0
    return float((1 - u) * ((1 - v) * table[i0, j0] + v * table[i0, j0 + 1])
                 + u * ((1 - v) * table[i0 + 1, j0] + v * table[i0 + 1, j0 + 1]))


def thumb_rest(curls):
    """Where the final-hand thumb rests when fully closed, by gesture: across the palm (index and middle straight,
    peace sign), on the curled middle (index pointing), on the index fingertip (pinch), or on the side of the
    curled index (fist)."""
    si = curls.get("index", 1.0)
    if si < S_PINCH:
        return lerp(lerp(THUMB_TUCK, THUMB_POINT, curls.get("middle", 1.0)), THUMB_PINCH, si / S_PINCH)
    return lerp(THUMB_PINCH, THUMB_FIST, (si - S_PINCH) / (1 - S_PINCH))


def final_thumb(s, curls):
    """Final-hand thumb command for thumb curl s: from open towards its rest pose, stopping short of the fingers."""
    s = min(s, interp2(THUMB_CLOSE_MAX, curls.get("index", 1.0), curls.get("middle", 1.0)))
    return clamp_thumb(*lerp(THUMB_OPEN, thumb_rest(curls), s))


def limit_convergence(abd, limit, reach):
    """Pull neighbouring fingers apart where they turn towards each other by more than `limit` in total.
    abd is + towards the thumb, so the finger on the little-finger side may not exceed its neighbour by more.
    Each finger's share of the move goes towards the thumb only as far as its reach allows."""
    names = [n for n in FINGERS if n in abd]
    for _ in range(len(names)):
        for a, b in zip(names, names[1:]):
            excess = abd[b] - abd[a] - limit
            if excess > 0:
                up = max(0.0, min(excess / 2, reach[a] - abd[a]))
                abd[a] += up
                abd[b] -= excess - up
    return abd


def to_motors(flex, a):
    return (float(np.clip(flex + a, -MOTOR_LIMIT, MOTOR_LIMIT)), float(np.clip(-flex + a, -MOTOR_LIMIT, MOTOR_LIMIT)))


def retarget(angles, calib, robot):
    """Human angles -> {robot_finger: (motor1_deg, motor2_deg)} plus normalised curl per digit."""
    curls, flex, abd, reach = {}, {}, {}, {}
    for name in robot.digits:
        curl, a = angles[name]
        lo, hi = calib.curl_open[name], calib.curl_fist[name]
        s = curls[name] = float(np.clip((curl - lo) / max(hi - lo, 1e-3), 0.0, 1.0))
        if name != "thumb":
            flex[name] = max(robot.flex_open + s * (robot.flex_closed - robot.flex_open), robot.flex_min)
            # Abduction is unreliable on a curled finger, so fade it out as the finger closes.
            reach[name] = robot.abd_max * max(0.0, 1 - s / robot.abd_fade_end)
            abd[name] = np.clip(ABD_GAIN * (a - calib.abd_neutral[name]), -robot.abd_max, robot.abd_max) \
                * reach[name] / robot.abd_max
        elif not robot.thumb_poses:
            flex[name] = THUMB_FLEX_OPEN + s * (THUMB_FLEX_CLOSED - THUMB_FLEX_OPEN)
            # Thumb moving in towards the index (smaller spread) = + abduction on the robot.
            abd[name] = np.clip(THUMB_ABD_GAIN * (calib.abd_neutral[name] - a), -THUMB_ABD_MAX, THUMB_ABD_MAX)
    if np.isfinite(robot.abd_converge):
        limit_convergence(abd, robot.abd_converge, reach)
    motors = {ROBOT_FINGER[n]: to_motors(flex[n], abd[n]) for n in flex}
    if robot.thumb_poses and "thumb" in robot.digits:
        motors[ROBOT_FINGER["thumb"]] = final_thumb(curls["thumb"], curls)
    return motors, curls


class Sim:
    def __init__(self, robot: Robot, size: int):
        self.m = mujoco.MjModel.from_xml_path(str(robot.scene))
        self.d = mujoco.MjData(self.m)
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)     # the settled rest pose (final hand)
        # Only the finger servos are driven; anything else (the final hand's wrist) holds its rest command.
        self.act = {ROBOT_FINGER[n]: (self.m.actuator(f"finger{ROBOT_FINGER[n]}_motor1").id,
                                      self.m.actuator(f"finger{ROBOT_FINGER[n]}_motor2").id) for n in robot.digits}
        self.target = self.d.ctrl.copy()
        for b in range(self.m.nbody):                          # hide the IK target markers
            if self.m.body_mocapid[b] >= 0:
                self.m.geom_rgba[self.m.body_geomadr[b]:self.m.body_geomadr[b] + self.m.body_geomnum[b], 3] = 0
        mujoco.mj_forward(self.m, self.d)
        vis = self.m.vis.global_
        vis.offwidth, vis.offheight = max(vis.offwidth, size), max(vis.offheight, size)
        self.renderer = mujoco.Renderer(self.m, size, size)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = robot.lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = robot.distance, 180.0, -12.0

    def command(self, motors_deg):
        for k, (m1, m2) in motors_deg.items():
            if k in self.act:
                self.target[list(self.act[k])] = np.radians([m1, m2])

    def advance(self, seconds):
        # The servos move at most SERVO_SPEED, like the real ones, so a jump in the command doesn't kick the sim.
        step = np.radians(SERVO_SPEED) * self.m.opt.timestep
        for _ in range(max(1, int(seconds / self.m.opt.timestep))):
            self.d.ctrl[:] += np.clip(self.target - self.d.ctrl, -step, step)
            mujoco.mj_step(self.m, self.d)

    def render(self):
        self.renderer.update_scene(self.d, self.cam)
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)


class VideoSource:
    """A recorded clip, read one frame per loop (several clips stay in lockstep). Loops at the end."""

    def __init__(self, path, loop=True):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.loop = loop

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def close(self):
        self.cap.release()


class ImageSource:
    def __init__(self, path):
        self.img = cv2.imread(path)
        if self.img is None:
            raise SystemExit(f"cannot read {path}")

    def read(self):
        return True, self.img.copy()

    def close(self):
        pass


def open_sources(args):
    """One source per camera view."""
    if args.image:
        return [ImageSource(args.image)]
    if args.video:
        return [VideoSource(v) for v in args.video]
    return [CameraStream(i) for i in args.camera]


def pick_hand(hands, target, strict):
    for hd in hands:
        if target is None or hd.label == target:
            return hd
    # A camera that delivers mirrored frames reports the opposite label; accept its hand unless told not to.
    return None if strict or not hands else hands[0]


def draw_landmarks(img, xy):
    pts = [tuple(int(v) for v in pt) for pt in xy]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(img, pt, 4, (0, 140, 255), -1)


def put(img, text, y, color=(255, 255, 255)):
    (w, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    roi = img[max(y - th - 5, 0):y + 6, 4:min(w + 16, img.shape[1])]
    roi[:] = (roi * 0.35).astype(np.uint8)        # darken behind the text so it reads on any background
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def facing(hand):
    c = palm_cosine(hand)
    return "palm" if c > 0.3 else ("BACK" if c < -0.3 else "side")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, nargs="+", default=[0],
                     help="webcam index(es); give two, e.g. --camera 1 0, to fuse two views")
    src.add_argument("--video", nargs="+", help="recorded clip(s) instead of webcams (one per view, loops)")
    src.add_argument("--image", help="single image instead of the webcam")
    src.add_argument("--list-cameras", action="store_true", help="show a snapshot from each camera index and exit")
    ap.add_argument("--tracker", choices=("mediapipe", "wilor"), default="mediapipe",
                    help="hand tracker: mediapipe (fast) or wilor (robust to occlusion, needs requirements-wilor.txt)")
    ap.add_argument("--record", help="also save the raw camera video to this .mp4 (one file per camera)")
    ap.add_argument("--model", choices=tuple(ROBOTS), default="final",
                    help="final = the 5-digit hand AH_Right_5F_3S_Wrist (default); stock5 / stock4 = the stock hand "
                         "with / without the placeholder pinky from build_model.py")
    ap.add_argument("--hand", choices=("any", "right", "left"), default="any",
                    help="which of your hands to track (default: whichever the first camera sees)")
    ap.add_argument("--log", help="write motor commands (deg) to this CSV")
    ap.add_argument("--save", help="write the side-by-side view to this .mp4 or .png")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    ap.add_argument("--no-window", action="store_true", help="don't open a window (for testing)")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return
    robot = ROBOTS[args.model]
    if not robot.scene.exists():
        hint = "" if args.model == "final" else ": run `python build_model.py` first"
        raise SystemExit(f"{robot.scene} missing{hint}")
    digits = robot.digits

    sources = open_sources(args)
    n_views = len(sources)
    # MediaPipe tracks over time, so each view needs its own instance; WiLoR is per-frame and heavy, so share it.
    trackers = [make_tracker(args.tracker)]
    for _ in range(1, n_views):
        trackers.append(make_tracker(args.tracker) if args.tracker == "mediapipe" else trackers[0])
    filters = [OneEuro() for _ in sources]
    raw_writers = [None] * n_views
    calib = Calibration()
    sim = None
    writer = None
    log = csv.writer(open(args.log, "w", newline="")) if args.log else None
    if log:
        log.writerow(["t"] + [f"finger{k}_motor{j}" for k in range(1, len(digits) + 1) for j in (1, 2)])

    t_start = last = time.monotonic()
    fps = 0.0
    frame_i = 0
    angles = None
    motors = {}
    curls = {}
    while True:
        t_cap = time.monotonic()
        frames = [s.read() for s in sources]
        if not any(ok for ok, _ in frames):
            if args.video or args.image:
                break
            time.sleep(0.005)                      # cameras still warming up
            continue
        if sim is None:
            first = next(f for ok, f in frames if ok)
            panel_h = first.shape[0] if n_views == 1 else 360
            sim = Sim(robot, panel_h)

        target = None if args.hand == "any" else args.hand
        views, view_cams, panels, headers = [], [], [], []
        for i, (ok, frame) in enumerate(frames):
            if not ok:
                panels.append(np.zeros((panel_h, panel_h * 16 // 9, 3), np.uint8))
                headers.append((f"cam {i}: no frame", (80, 80, 255)))
                continue
            if args.record:
                if raw_writers[i] is None:
                    path = args.record if n_views == 1 else args.record.replace(".mp4", f"_cam{i}.mp4")
                    raw_writers[i] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                     (frame.shape[1], frame.shape[0]))
                raw_writers[i].write(frame)
            frame = cv2.flip(frame, 1)   # selfie view; MediaPipe handedness assumes mirrored input
            hand = pick_hand(trackers[i].detect(frame), target, strict=args.hand != "any")
            scale = panel_h / frame.shape[0]
            panel = cv2.resize(frame, (int(frame.shape[1] * scale), panel_h)) if scale != 1 else frame
            if hand is None:
                filters[i].x = None                # restart smoothing when the hand comes back
                headers.append((f"cam {i}: no hand", (80, 80, 255)))
            else:
                target = target or hand.label      # later cameras follow the first camera's hand
                draw_landmarks(panel, hand.xy * scale)
                hand.world = filters[i](hand.world.flatten(), t_cap).reshape(21, 3)
                views.append((hand_angles(hand.world), hand))
                view_cams.append(i)
                headers.append((f"cam {i}: {hand.label} hand, {facing(hand)}", (120, 255, 120)))
            panels.append(panel)

        if views:
            if len(views) > 1:
                angles, shares = fuse(views)
                for i, share in zip(view_cams, shares):
                    headers[i] = (f"{headers[i][0]}, trust {share * 100:.0f}%", headers[i][1])
            else:
                angles = views[0][0]
            motors, curls = retarget(angles, calib, robot)
            sim.command(motors)

        now = time.monotonic()
        # A still image gets time to settle; live, the sim follows the clock (at most 50 ms per frame).
        sim.advance(0.5 if args.image else min(now - last, 0.05))
        robot_view = sim.render()
        latency_ms = (time.monotonic() - t_cap) * 1000
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3) if frame_i else 0.0
        last = now

        for panel, (text, color) in zip(panels, headers):
            put(panel, text, 22, color)
        main_panel = panels[0]
        put(main_panel, f"{args.tracker}  {fps:4.1f} fps  {latency_ms:4.0f} ms" +
            ("" if views else "  (holding last pose)"), 44, (200, 200, 200))
        if angles:
            for i, name in enumerate(digits):
                curl, abd = angles[name]
                m1, m2 = motors[ROBOT_FINGER[name]]
                put(main_panel, f"{name:6s} curl {curl:4.0f} ({curls[name]*100:3.0f}%) abd {abd:+4.0f} "
                                f"servos {m1:+4.0f} {m2:+4.0f}", 70 + 22 * i)
        put(main_panel, "o=open  f=fist  r=reset  [ ]=rotate  v=view  q=quit", panel_h - 12, (200, 200, 200))
        view = np.hstack(panels + [robot_view])

        if log and motors:
            log.writerow([f"{now - t_start:.3f}"] + [f"{v:.1f}" for k in sorted(motors) for v in motors[k]])
        if args.save:
            if args.save.endswith(".png"):
                cv2.imwrite(args.save, view)
            else:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)

        frame_i += 1
        if args.frames and frame_i >= args.frames:
            break
        if args.no_window:
            continue
        cv2.imshow("AmazingHand teleop", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if angles and key == ord("o"):
            for name in digits:
                calib.curl_open[name], calib.abd_neutral[name] = angles[name]
            print("calibrated OPEN:", {n: round(angles[n][0]) for n in digits})
        elif angles and key == ord("f"):
            for name in digits:
                calib.curl_fist[name] = angles[name][0]
            print("calibrated FIST:", {n: round(angles[n][0]) for n in digits})
        elif key == ord("r"):
            calib = Calibration()
        elif key == ord("["):
            sim.cam.azimuth -= 10
        elif key == ord("]"):
            sim.cam.azimuth += 10
        elif key == ord("v"):
            sim.cam.azimuth = 0.0 if sim.cam.azimuth == 180.0 else 180.0

    for w in [writer] + raw_writers:
        if w:
            w.release()
    for s in sources:
        s.close()
    for t in {id(t): t for t in trackers}.values():
        t.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
