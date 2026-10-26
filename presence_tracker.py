"""
Candidate Presence & Multi-Person Detector.
Verifies single-student presence and flags candidate absence or multiple unauthorized people in frame.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


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


class PresenceTracker:
    """
    Monitors candidate visual presence and detects unauthorized multiple individuals.
    """

    def __init__(self, absence_limit_frames: int = 45) -> None:
        self.absence_limit_frames = absence_limit_frames
        self.consecutive_absent_frames = 0
        self.consecutive_multi_person_frames = 0
        
        # Load Haar Cascade as high-speed lightweight fallback / validator
        self.face_cascade = _safe_load_cascade("haarcascade_frontalface_default.xml")

    def analyze_faces(
        self,
        frame: np.ndarray,
        detected_face_boxes: List[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Determines candidate count and presence status.
        face_boxes: list of (x, y, w, h)
        """
        if detected_face_boxes is None:
            # Fallback to Haar Cascade
            detected_face_boxes = []
            if self.face_cascade is not None and not self.face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                detected_face_boxes = [tuple(f) for f in faces]

        face_count = len(detected_face_boxes)

        status = "NORMAL"
        is_violation = False
        alert_msg = None

        if face_count == 0:
            self.consecutive_absent_frames += 1
            if self.consecutive_absent_frames >= 15:
                status = "CANDIDATE ABSENT"
                is_violation = True
                alert_msg = "Candidate left the camera frame!"
        elif face_count == 1:
            self.consecutive_absent_frames = 0
            self.consecutive_multi_person_frames = 0
            status = "CANDIDATE VERIFIED"
        else:
            self.consecutive_absent_frames = 0
            self.consecutive_multi_person_frames += 1
            status = f"MULTIPLE FACES ({face_count})"
            is_violation = True
            alert_msg = f"Multiple individuals ({face_count}) detected in testing zone!"

        return {
            "face_count": face_count,
            "status": status,
            "is_violation": is_violation,
            "alert_msg": alert_msg,
            "boxes": detected_face_boxes,
            "absent_frames": self.consecutive_absent_frames,
            "multi_frames": self.consecutive_multi_person_frames
        }
