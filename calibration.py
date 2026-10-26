"""
Pre-Exam Baseline Calibration Wizard for ProctorAI.
Records candidate's natural resting head pose and neutral eye gaze ratio
to eliminate anatomical false positives.
"""

from typing import Dict, Any, List
import numpy as np


class ProctorCalibrator:
    """
    Computes candidate-specific baseline resting offsets for Yaw, Pitch, and Gaze.
    """

    def __init__(self, sample_size: int = 30) -> None:
        self.sample_size = sample_size
        self.yaw_samples: List[float] = []
        self.pitch_samples: List[float] = []
        self.gaze_samples: List[float] = []
        self.is_calibrated: bool = False
        self.baseline: Dict[str, float] = {"yaw": 0.0, "pitch": 0.0, "gaze": 1.0}

    def add_sample(self, yaw: float, pitch: float, gaze_ratio: float) -> bool:
        """
        Appends a baseline calibration sample. Returns True when calibration completes.
        """
        self.yaw_samples.append(yaw)
        self.pitch_samples.append(pitch)
        self.gaze_samples.append(gaze_ratio)

        if len(self.yaw_samples) >= self.sample_size:
            self.baseline = {
                "yaw": round(float(np.median(self.yaw_samples)), 2),
                "pitch": round(float(np.median(self.pitch_samples)), 2),
                "gaze": round(float(np.median(self.gaze_samples)), 2)
            }
            self.is_calibrated = True
            return True
        return False

    def get_calibrated_deviation(
        self,
        current_yaw: float,
        current_pitch: float,
        current_gaze: float
    ) -> Dict[str, float]:
        """
        Subtracts neutral resting baseline offsets from current instantaneous measurements.
        """
        if not self.is_calibrated:
            return {
                "delta_yaw": current_yaw,
                "delta_pitch": current_pitch,
                "delta_gaze": current_gaze
            }

        return {
            "delta_yaw": round(current_yaw - self.baseline["yaw"], 2),
            "delta_pitch": round(current_pitch - self.baseline["pitch"], 2),
            "delta_gaze": round(current_gaze / max(0.1, self.baseline["gaze"]), 2)
        }

    def reset(self) -> None:
        """Resets calibration buffer."""
        self.yaw_samples.clear()
        self.pitch_samples.clear()
        self.gaze_samples.clear()
        self.is_calibrated = False
        self.baseline = {"yaw": 0.0, "pitch": 0.0, "gaze": 1.0}
