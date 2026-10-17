"""Multi-camera helpers: threaded capture, a camera picker, and per-finger fusion of views.

Joint angles (curl, abduction) are measured in the hand's own frame, so they don't depend on
where a camera is. That means views can be merged per finger without calibrating the cameras:
each view's angles are averaged with weights that reflect how well that camera sees the hand.
"""

import threading

import cv2
import numpy as np

DIGITS = ["thumb", "index", "middle", "ring", "pinky"]


class CameraStream:
    """Reads a webcam on a background thread and always hands out the newest frame."""

    def __init__(self, index, width=960, height=540):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open camera {index} (run with --list-cameras; on macOS allow camera "
                             "access for your terminal app)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def close(self):
        self.running = False
        self.cap.release()


def list_cameras(max_index=6):
    """Show a snapshot from every camera OpenCV can open, labelled with its index."""
    tiles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, frame = False, None
        if cap.isOpened():
            for _ in range(10):                 # some cameras need a few frames to expose properly
                ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = frame.shape[:2]
        tile = cv2.resize(frame, (int(w * 270 / h), 270))
        cv2.putText(tile, f"--camera {i}   ({w}x{h})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(tile, f"--camera {i}   ({w}x{h})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
        print(f"camera {i}: {w}x{h}")
    if not tiles:
        print("no cameras found")
        return
    cv2.imwrite("cameras.png", np.hstack(tiles))
    print("saved cameras.png; press any key in the window to close")
    cv2.imshow("cameras", np.hstack(tiles))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def palm_cosine(hand):
    """+1 = palm faces the camera squarely, 0 = side view, -1 = back of the hand faces the camera."""
    p = hand.world
    n = np.cross(p[5] - p[0], p[17] - p[0])
    if hand.label == "left":
        n = -n
    return float(n[2] / (np.linalg.norm(n) + 1e-9))


def view_weights(hand):
    """How much to trust this view for (curl, abduction).

    Curl: fingers curl towards the palm, so a palm-on or side view sees them; from the back the
    fingertips are hidden behind the knuckles and the tracker is guessing.
    Abduction (spread) is seen best face-on, from either side, and poorly edge-on.
    """
    c = palm_cosine(hand)
    w_curl = hand.score * (0.15 + 0.85 * np.sqrt((c + 1) / 2))
    w_abd = hand.score * (0.1 + abs(c))
    return w_curl, w_abd


def fuse(views):
    """views: list of (angles, hand) from different cameras -> fused {digit: (curl, abd)} and curl weights."""
    fused, shares = {}, []
    weights = [view_weights(h) for _, h in views]
    for d in DIGITS:
        wc = np.array([w[0] for w in weights])
        # The thumb's spread angle behaves like a curl measurement (it is hidden from the back too).
        wa = wc if d == "thumb" else np.array([w[1] for w in weights])
        curl = sum(w * a[d][0] for w, (a, _) in zip(wc, views)) / wc.sum()
        abd = sum(w * a[d][1] for w, (a, _) in zip(wa, views)) / wa.sum()
        fused[d] = (float(curl), float(abd))
    total = sum(w[0] for w in weights)
    shares = [w[0] / total for w in weights]
    return fused, shares
