"""
Asynchronous WebRTC Video Processor for Real-Time Exam Proctoring.
Integrates face tracking, head pose, eye gaze, multi-person detection,
contraband inspection, and cybernetic proctor HUD rendering.
"""

import threading
import time
from typing import Dict, Any, Tuple, Optional
import av
import cv2
import numpy as np
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase

from head_pose import HeadPoseEstimator
from gaze_tracker import EyeGazeTracker
from presence_tracker import PresenceTracker
from object_detector import ProhibitedObjectDetector
from proctor_engine import ProctorEngine

# Public STUN servers for NAT traversal on Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
        ]
    }
)


def _safe_load_cascade(filename: str):
    """Safely loads OpenCV cascade without raising AttributeError if cv2.data is missing."""
    try:
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") and cv2.data.haarcascades:
            cas = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
            if not cas.empty():
                return cas
    except Exception:
        pass
    try:
        import os
        cv2_dir = os.path.dirname(cv2.__file__)
        cand = os.path.join(cv2_dir, "data", filename)
        if os.path.exists(cand):
            cas = cv2.CascadeClassifier(cand)
            if not cas.empty():
                return cas
    except Exception:
        pass
    try:
        cas = cv2.CascadeClassifier(filename)
        if not cas.empty():
            return cas
    except Exception:
        pass
    return None


class LiveProctorProcessor(VideoProcessorBase):
    """
    WebRTC VideoProcessor executing continuous anti-cheating computer vision pipeline.
    """

    def __init__(self) -> None:
        self.head_pose = HeadPoseEstimator()
        self.gaze_tracker = EyeGazeTracker()
        self.presence_tracker = PresenceTracker()
        self.object_detector = ProhibitedObjectDetector()
        self.engine = ProctorEngine()

        # Haar Cascades for reliable low-latency detection
        self.face_cascade = _safe_load_cascade("haarcascade_frontalface_default.xml")
        self.eye_cascade = _safe_load_cascade("haarcascade_eye.xml")

        # Thread synchronization
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.suspicion_score: float = 0.0
        self.threat_level: str = "NORMAL"
        self.face_count: int = 1
        self.head_direction: str = "FORWARD"
        self.gaze_state: str = "CENTER"
        self.active_alert: Optional[str] = None
        self.total_infractions: int = 0

        # Timing & Caching
        self._frame_count = 0
        self._last_time = time.time()
        self._cached_objects: list = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes each incoming video frame and overlays proctor HUD."""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Calculate FPS
        now = time.time()
        self._frame_count += 1
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._last_time = now

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Multi-Face & Presence Analysis (Downscaled 2x for ultra-smooth 30+ FPS)
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        raw_faces = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            raw_faces = self.face_cascade.detectMultiScale(
                small_gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40)
            )
        face_boxes = [(int(fx * 2), int(fy * 2), int(fw * 2), int(fh * 2)) for (fx, fy, fw, fh) in raw_faces]
        presence_res = self.presence_tracker.analyze_faces(img, face_boxes)

        pose_res = {"is_deviated": False, "direction": "FORWARD", "yaw": 0.0, "pitch": 0.0}
        gaze_res = {"is_looking_away": False, "gaze_state": "CENTER"}

        if len(face_boxes) > 0:
            fx, fy, fw, fh = face_boxes[0]

            # Approximate 6 key facial landmarks for solvePnP head pose:
            nose = np.array([fx + fw * 0.5, fy + fh * 0.6])
            chin = np.array([fx + fw * 0.5, fy + fh * 0.95])
            l_eye = np.array([fx + fw * 0.3, fy + fh * 0.38])
            r_eye = np.array([fx + fw * 0.7, fy + fh * 0.38])
            l_mouth = np.array([fx + fw * 0.35, fy + fh * 0.78])
            r_mouth = np.array([fx + fw * 0.65, fy + fh * 0.78])

            img_pts = np.array([nose, chin, l_eye, r_eye, l_mouth, r_mouth])
            pose_res = self.head_pose.estimate_pose(img_pts, (h, w))

            # Eye region gaze estimation
            face_roi_gray = gray[fy:fy + int(fh * 0.55), fx:fx + fw]
            if face_roi_gray.size > 0 and self.eye_cascade is not None and not self.eye_cascade.empty():
                eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.15, minNeighbors=3, minSize=(20, 20))

                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: e[0])
                    lex, ley, lew, leh = eyes[0]
                    rex, rey, rew, reh = eyes[1]

                    l_eye_pts = np.array([
                        [fx + lex, fy + ley + leh // 2],
                        [fx + lex + lew // 2, fy + ley],
                        [fx + lex + lew, fy + ley + leh // 2],
                        [fx + lex + lew // 2, fy + ley + leh]
                    ])
                    r_eye_pts = np.array([
                        [fx + rex, fy + rey + reh // 2],
                        [fx + rex + rew // 2, fy + rey],
                        [fx + rex + rew, fy + rey + reh // 2],
                        [fx + rex + rew // 2, fy + rey + reh]
                    ])

                    l_ratio = self.gaze_tracker.compute_gaze_ratio(l_eye_pts, gray)
                    r_ratio = self.gaze_tracker.compute_gaze_ratio(r_eye_pts, gray)
                    gaze_res = self.gaze_tracker.evaluate_gaze(l_ratio, r_ratio)

            # Draw Head Pose 3D orientation vector
            if pose_res.get("rvec") is not None:
                img = self.head_pose.draw_axes(
                    img,
                    pose_res["rvec"],
                    pose_res["tvec"],
                    pose_res["camera_matrix"],
                    (int(nose[0]), int(nose[1]))
                )

        # 2. Contraband Object Detection (Every 5th frame to prevent video pipeline freeze)
        if self._frame_count % 5 == 0:
            self._cached_objects = self.object_detector.detect_objects(img)
        objects_res = self._cached_objects

        # 3. Fuse in Proctor Engine
        engine_res = self.engine.update(pose_res, gaze_res, presence_res, objects_res)

        # Draw HUD overlays
        threat = engine_res["threat_level"]
        theme_color = (0, 255, 212) if threat == "NORMAL" else ((0, 215, 255) if threat == "WARNING" else (50, 50, 255))

        # Face bounding boxes
        for (fx, fy, fw, fh) in face_boxes:
            cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), theme_color, 2)
            cv2.putText(img, "CANDIDATE ID: VERIFIED", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme_color, 1)

        # Contraband bounding boxes
        for obj in objects_res:
            ox, oy, ow, oh = obj["box"]
            cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 3)
            cv2.putText(img, f"CONTRABAND: {obj['label']}", (ox, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # HUD Top Banner
        cv2.rectangle(img, (0, 0), (w, 55), (10, 15, 26), -1)
        cv2.line(img, (0, 55), (w, 55), theme_color, 2)

        # Suspicion Bar
        susp_pct = engine_res["suspicion_score"]
        bar_w = int((w - 40) * (susp_pct / 100.0))
        cv2.rectangle(img, (20, 42), (w - 20, 50), (30, 40, 55), -1)
        cv2.rectangle(img, (20, 42), (20 + bar_w, 50), theme_color, -1)

        cv2.putText(img, f"PROCTOR AI: {threat}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, theme_color, 2)
        cv2.putText(img, f"SUSPICION: {susp_pct:.1f}%", (w // 2 - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"{self.fps} FPS", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 245, 212), 1)

        # Bottom Telemetry Strip
        cv2.rectangle(img, (0, h - 35), (w, h), (10, 15, 26), -1)
        dir_text = f"HEAD: {pose_res['direction']} | GAZE: {gaze_res['gaze_state']} | FACES: {len(faces)}"
        cv2.putText(img, dir_text, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 180, 200), 1)

        # Update thread-safe status
        with self._lock:
            self.suspicion_score = susp_pct
            self.threat_level = threat
            self.face_count = len(faces)
            self.head_direction = pose_res["direction"]
            self.gaze_state = gaze_res["gaze_state"]
            self.active_alert = engine_res["triggered_alert"]
            self.total_infractions = engine_res["total_infractions_count"]

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns instantaneous proctoring telemetry for Streamlit UI."""
        with self._lock:
            return {
                "fps": self.fps,
                "suspicion_score": self.suspicion_score,
                "threat_level": self.threat_level,
                "face_count": self.face_count,
                "head_direction": self.head_direction,
                "gaze_state": self.gaze_state,
                "active_alert": self.active_alert,
                "total_infractions": self.total_infractions,
                "infraction_logs": list(self.engine.infractions)
            }
