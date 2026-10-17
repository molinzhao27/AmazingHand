"""Hand trackers with a common output: 21 keypoints in MediaPipe/OpenPose order.

  mediapipe  MediaPipe HandLandmarker. Fast on CPU, weak on self-occlusion and back-of-hand views.
  wilor      WiLoR (MANO-based 3D hand reconstruction, CVPR 2025) via the wilor-mini package.
             Much more robust to occlusion and unusual views, heavier. Uses the Apple GPU (MPS)
             when available. Install steps are in README.md ("WiLoR tracker").

Both trackers take a selfie-flipped BGR frame and return Hand objects whose `label` is the
person's real hand ("left"/"right") and whose `xy` is in pixel coords of that flipped frame.
"""

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task")

# Bones for drawing (MediaPipe / OpenPose ordering).
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


@dataclass
class Hand:
    label: str           # "left" or "right" (the person's real hand)
    score: float         # detector confidence 0..1
    xy: np.ndarray       # (21, 2) pixel coords in the (flipped) frame
    world: np.ndarray    # (21, 3) metric 3D keypoints, any origin, camera-aligned axes (x right, y down, z away)


def palm_facing_camera(hand: Hand) -> bool:
    """True if the palm (not the back of the hand) faces the camera."""
    p = hand.world
    # Only the x/y parts matter for n_z, so this is the 2D winding of wrist -> index MCP -> pinky MCP.
    # In a selfie-flipped image a right palm facing the camera has the index MCP left of the pinky MCP
    # (fingers up), which gives n_z > 0; a left hand is the mirror case.
    n = np.cross(p[5] - p[0], p[17] - p[0])
    if hand.label == "left":
        n = -n
    return bool(n[2] > 0)


class MediaPipeTracker:
    name = "mediapipe"

    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision
        import mediapipe as mp
        self._mp = mp
        path = HERE / "hand_landmarker.task"
        if not path.exists():
            print(f"Downloading MediaPipe hand model to {path} ...")
            urllib.request.urlretrieve(MP_MODEL_URL, path)
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        self.last_ts = -1
        self.t0 = time.monotonic()

    def detect(self, frame_bgr):
        import cv2
        h, w = frame_bgr.shape[:2]
        ts = max(int((time.monotonic() - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect_for_video(img, ts)
        hands = []
        for i, hd in enumerate(res.handedness):
            xy = np.array([[l.x * w, l.y * h] for l in res.hand_landmarks[i]])
            world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[i]])
            hands.append(Hand(hd[0].category_name.lower(), hd[0].score, xy, world))
        return hands

    def close(self):
        self.landmarker.close()     # avoids a noisy error from MediaPipe's destructor at exit


class WiLoRTracker:
    name = "wilor"

    def __init__(self, device=None):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        # float32 is the safe default; WILOR_FP16=1 roughly doubles speed on the GPU if outputs look sane.
        dtype = torch.float16 if device != "cpu" and os.environ.get("WILOR_FP16") == "1" else torch.float32
        print(f"Loading WiLoR on {device} ({dtype}); first run downloads ~1 GB of weights ...")
        self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device(device), dtype=dtype, verbose=False)

    def detect(self, frame_bgr):
        import cv2
        # WiLoR judges left/right from appearance, so give it the un-mirrored image and mirror back.
        raw = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)
        w = raw.shape[1]
        hands = []
        for det in self.pipe.predict(raw):
            preds = det.get("wilor_preds")
            if preds is None:
                continue
            xy = preds["pred_keypoints_2d"][0].copy()
            xy[:, 0] = w - 1 - xy[:, 0]
            world = preds["pred_keypoints_3d"][0].copy()
            world[:, 0] = -world[:, 0]          # mirror to match the flipped frame (like MediaPipe)
            label = "right" if det["is_right"] > 0.5 else "left"
            hands.append(Hand(label, 1.0, xy, world))
        return hands

    def close(self):
        pass


def make_tracker(name):
    if name == "mediapipe":
        return MediaPipeTracker()
    if name == "wilor":
        return WiLoRTracker()
    raise ValueError(name)
