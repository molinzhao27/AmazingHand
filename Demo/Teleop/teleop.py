"""Live webcam hand tracking -> AmazingHand servo commands -> MuJoCo simulation.

Pipeline (one window, runs with plain `python` on macOS, no mjpython or dora needed):
  camera frame(s) -> hand tracker (MediaPipe, or WiLoR with --tracker wilor; 21 3D keypoints)
  -> One Euro filter -> per-digit curl + abduction angles
  -> with two cameras, per-finger fusion weighted by how well each camera sees the hand
  -> AmazingHand motor targets (2 servos per digit) -> MuJoCo position actuators
  -> offscreen render shown next to the camera image.

The motor targets are exactly what the real hand takes (degrees per servo), so the
same numbers can later go to the Feetech SCS0009 servos (see --log).

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

# Retargeting ranges, in degrees. Motor convention (matches the AmazingHand examples):
#   motor1 = flex + abd, motor2 = -flex + abd. flex -35 ~ straight, +90 ~ fully closed.
FLEX_OPEN, FLEX_CLOSED = -35.0, 90.0
THUMB_FLEX_OPEN, THUMB_FLEX_CLOSED = -20.0, 80.0
ABD_GAIN, ABD_MAX = 1.5, 35.0
THUMB_ABD_GAIN, THUMB_ABD_MAX = 1.0, 40.0
MOTOR_LIMIT = 90.0


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


def retarget(angles, calib, digits):
    """Human angles -> {robot_finger: (motor1_deg, motor2_deg)} plus normalised curl per digit."""
    motors, curls = {}, {}
    for name in digits:
        curl, abd = angles[name]
        lo, hi = calib.curl_open[name], calib.curl_fist[name]
        s = float(np.clip((curl - lo) / max(hi - lo, 1e-3), 0.0, 1.0))
        if name == "thumb":
            flex = THUMB_FLEX_OPEN + s * (THUMB_FLEX_CLOSED - THUMB_FLEX_OPEN)
            # Thumb moving in towards the index (smaller spread) = + abduction on the robot.
            a = np.clip(THUMB_ABD_GAIN * (calib.abd_neutral[name] - abd), -THUMB_ABD_MAX, THUMB_ABD_MAX)
        else:
            flex = FLEX_OPEN + s * (FLEX_CLOSED - FLEX_OPEN)
            # Abduction is unreliable on a curled finger, so fade it out as the finger closes.
            a = np.clip(ABD_GAIN * (abd - calib.abd_neutral[name]) * (1 - s), -ABD_MAX, ABD_MAX)
        m1 = float(np.clip(flex + a, -MOTOR_LIMIT, MOTOR_LIMIT))
        m2 = float(np.clip(-flex + a, -MOTOR_LIMIT, MOTOR_LIMIT))
        motors[ROBOT_FINGER[name]] = (m1, m2)
        curls[name] = s
    return motors, curls


class Sim:
    def __init__(self, scene: Path, size: int):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.n_fingers = self.m.nu // 2
        self.act = {k: (self.m.actuator(f"finger{k}_motor1").id, self.m.actuator(f"finger{k}_motor2").id)
                    for k in range(1, self.n_fingers + 1)}
        mujoco.mj_forward(self.m, self.d)
        self.renderer = mujoco.Renderer(self.m, size, size)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.02, -0.025 if self.n_fingers == 5 else -0.01, 0.085]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.30, 180.0, -12.0

    def command(self, motors_deg):
        for k, (m1, m2) in motors_deg.items():
            if k in self.act:
                self.d.ctrl[self.act[k][0]] = np.radians(m1)
                self.d.ctrl[self.act[k][1]] = np.radians(m2)

    def advance(self, seconds):
        for _ in range(max(1, int(min(seconds, 0.05) / self.m.opt.timestep))):
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
    ap.add_argument("--fingers", type=int, choices=(4, 5), default=5,
                    help="4 = stock AmazingHand, 5 = with added pinky (default)")
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
    scene = HERE / "models" / "AH_Right" / f"scene_{args.fingers}f.xml"
    if not scene.exists():
        raise SystemExit(f"{scene} missing: run `python build_model.py` first")
    digits = ["index", "middle", "ring", "thumb"] + (["pinky"] if args.fingers == 5 else [])

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
        log.writerow(["t"] + [f"finger{k}_motor{j}" for k in range(1, args.fingers + 1) for j in (1, 2)])

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
            sim = Sim(scene, panel_h)

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
            motors, curls = retarget(angles, calib, digits)
            sim.command(motors)

        now = time.monotonic()
        sim.advance(now - last)
        robot = sim.render()
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
        view = np.hstack(panels + [robot])

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
