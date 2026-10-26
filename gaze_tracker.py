"""
Eye Gaze Tracking & Iris Displacement Engine.
Calculates horizontal and vertical iris gaze ratios to detect off-screen glances.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class EyeGazeTracker:
    """
    Computes horizontal and vertical eye gaze ratios to verify whether
    the student is actively looking at their exam screen.
    """

    def __init__(self, horizontal_thresh: float = 0.35, vertical_thresh: float = 0.30) -> None:
        self.horizontal_thresh = horizontal_thresh
        self.vertical_thresh = vertical_thresh

    def compute_gaze_ratio(
        self,
        eye_points: np.ndarray,
        frame_gray: np.ndarray
    ) -> float:
        """
        Calculates iris displacement ratio across the eye bounding box.
        Ratio ~ 1.0 represents center gaze.
        Ratio < (1.0 - thresh) represents left gaze.
        Ratio > (1.0 + thresh) represents right gaze.
        """
        if eye_points is None or len(eye_points) < 4:
            return 1.0

        # Mask eye region
        h, w = frame_gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [eye_points.astype(np.int32)], 255)

        eye_isolated = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

        min_x = np.min(eye_points[:, 0])
        max_x = np.max(eye_points[:, 0])
        min_y = np.min(eye_points[:, 1])
        max_y = np.max(eye_points[:, 1])

        eye_crop = eye_isolated[int(min_y):int(max_y), int(min_x):int(max_x)]
        if eye_crop.size == 0:
            return 1.0

        # Threshold to segment dark iris
        _, thresh = cv2.threshold(eye_crop, 70, 255, cv2.THRESH_BINARY_INV)

        crop_h, crop_w = thresh.shape
        half_w = int(crop_w / 2)
        if half_w == 0:
            return 1.0

        left_side_white = cv2.countNonZero(thresh[:, :half_w])
        right_side_white = cv2.countNonZero(thresh[:, half_w:])

        if right_side_white == 0:
            return 2.5
        if left_side_white == 0:
            return 0.1

        return float(left_side_white / right_side_white)

    def evaluate_gaze(
        self,
        left_eye_ratio: float,
        right_eye_ratio: float
    ) -> Dict[str, Any]:
        """
        Combines left and right eye ratios to determine overall eye gaze direction.
        """
        avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        is_looking_away = False
        gaze_state = "CENTER (FOCUSED)"

        if avg_ratio < 0.65:
            gaze_state = "GAZE RIGHT (OFF-SCREEN)"
            is_looking_away = True
        elif avg_ratio > 1.65:
            gaze_state = "GAZE LEFT (OFF-SCREEN)"
            is_looking_away = True

        return {
            "gaze_state": gaze_state,
            "avg_ratio": round(avg_ratio, 2),
            "left_ratio": round(left_eye_ratio, 2),
            "right_ratio": round(right_eye_ratio, 2),
            "is_looking_away": is_looking_away
        }
