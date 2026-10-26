"""
Core Proctoring State Machine & Composite Suspicion Scoring Engine.
Synthesizes head pose, eye gaze, multi-face presence, and contraband detection
into a dynamic Suspicion Index (0-100%) and timestamped audit log.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time


class ProctorEngine:
    """
    Evaluates multi-modal behavioral telemetry and manages the candidate's integrity score.
    """

    def __init__(
        self,
        decay_rate: float = 0.96,
        warning_thresh: float = 40.0,
        critical_thresh: float = 75.0
    ) -> None:
        self.decay_rate = decay_rate
        self.warning_thresh = warning_thresh
        self.critical_thresh = critical_thresh
        
        self.suspicion_score: float = 0.0
        self.infractions: List[Dict[str, Any]] = []
        self._last_alert_time: float = 0.0
        self.start_time: float = time.time()

    def update(
        self,
        pose_res: Dict[str, Any],
        gaze_res: Dict[str, Any],
        presence_res: Dict[str, Any],
        objects_res: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fuses sensory inputs, calculates incremental risk weights, updates decay,
        and logs infractions if thresholds are violated.
        """
        now = time.time()
        incremental_penalty = 0.0
        active_violations = []

        # 1. Multi-Face / Absence Check
        if presence_res.get("status") == "CANDIDATE ABSENT":
            incremental_penalty += 35.0
            active_violations.append({
                "type": "CANDIDATE_ABSENT",
                "severity": "CRITICAL",
                "description": "Candidate has left the proctored exam area."
            })
        elif "MULTIPLE FACES" in presence_res.get("status", ""):
            incremental_penalty += 45.0
            active_violations.append({
                "type": "MULTIPLE_PEOPLE",
                "severity": "CRITICAL",
                "description": f"Unauthorized people detected: {presence_res.get('status')}."
            })

        # 2. Contraband Object Check
        if objects_res:
            for obj in objects_res:
                incremental_penalty += 40.0
                active_violations.append({
                    "type": "PROHIBITED_DEVICE",
                    "severity": "CRITICAL",
                    "description": f"Prohibited {obj.get('label')} detected in candidate workspace."
                })

        # 3. Head Pose Deviation Check
        if pose_res.get("is_deviated", False):
            direction = pose_res.get("direction", "DEVIATED")
            penalty = 25.0 if "DOWN" in direction else 15.0
            incremental_penalty += penalty
            active_violations.append({
                "type": "HEAD_DEVIATION",
                "severity": "WARNING",
                "description": f"Head turned away from exam display: {direction}."
            })

        # 4. Gaze Direction Check
        if gaze_res.get("is_looking_away", False):
            incremental_penalty += 12.0
            active_violations.append({
                "type": "GAZE_OFF_SCREEN",
                "severity": "WARNING",
                "description": f"Eye gaze shifted off-screen: {gaze_res.get('gaze_state')}."
            })

        # Apply exponential decay to prior suspicion when no new infractions occur
        if incremental_penalty == 0.0:
            self.suspicion_score = max(0.0, self.suspicion_score * self.decay_rate)
        else:
            self.suspicion_score = min(100.0, self.suspicion_score + (incremental_penalty * 0.4))

        # Log active infractions (throttled to avoid duplicate flooding)
        triggered_alert = None
        if active_violations and (now - self._last_alert_time) > 2.5:
            self._last_alert_time = now
            primary_violation = active_violations[0]
            triggered_alert = primary_violation["description"]
            self.infractions.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "elapsed_seconds": int(now - self.start_time),
                "type": primary_violation["type"],
                "severity": primary_violation["severity"],
                "description": primary_violation["description"],
                "suspicion_index": round(self.suspicion_score, 1)
            })

        # Overall threat level classification
        if self.suspicion_score >= self.critical_thresh:
            threat_level = "CRITICAL"
        elif self.suspicion_score >= self.warning_thresh:
            threat_level = "WARNING"
        else:
            threat_level = "NORMAL"

        return {
            "suspicion_score": round(self.suspicion_score, 1),
            "threat_level": threat_level,
            "active_violations": active_violations,
            "triggered_alert": triggered_alert,
            "total_infractions_count": len(self.infractions)
        }

    def reset(self) -> None:
        """Reset proctoring session state."""
        self.suspicion_score = 0.0
        self.infractions = []
        self.start_time = time.time()
        self._last_alert_time = 0.0
