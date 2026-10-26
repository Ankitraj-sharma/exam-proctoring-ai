"""
Unit Test Suite for Online Exam Proctoring System.
Verifies solvePnP head pose, eye gaze ratio, multi-face presence,
suspicion decay, and LMS report generators.
"""

import numpy as np
from gaze_tracker import EyeGazeTracker
from proctor_engine import ProctorEngine
from calibration import ProctorCalibrator
from export_utils import generate_csv_audit_report, generate_json_audit_package

try:
    import cv2
    from head_pose import HeadPoseEstimator
    from presence_tracker import PresenceTracker
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def test_head_pose_forward():
    if not CV2_AVAILABLE:
        print("Skipping OpenCV solvePnP test (cv2 not in local path)")
        return
    estimator = HeadPoseEstimator()
    img_pts = np.array([
        [320.0, 240.0],
        [320.0, 360.0],
        [260.0, 180.0],
        [380.0, 180.0],
        [280.0, 300.0],
        [360.0, 300.0]
    ])
    res = estimator.estimate_pose(img_pts, (480, 640))
    assert res is not None
    assert "direction" in res
    assert "yaw" in res
    assert "pitch" in res


def test_eye_gaze_evaluator():
    tracker = EyeGazeTracker()
    
    # Normal center gaze
    res_center = tracker.evaluate_gaze(1.0, 1.0)
    assert not res_center["is_looking_away"]
    assert "CENTER" in res_center["gaze_state"]

    # Off-screen right gaze
    res_right = tracker.evaluate_gaze(0.3, 0.4)
    assert res_right["is_looking_away"]
    assert "RIGHT" in res_right["gaze_state"]


def test_presence_tracker():
    if not CV2_AVAILABLE:
        print("Skipping presence cascade test (cv2 not in local path)")
        return
    tracker = PresenceTracker()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Single face
    res_single = tracker.analyze_faces(dummy_frame, [(100, 100, 150, 150)])
    assert res_single["face_count"] == 1
    assert res_single["status"] == "CANDIDATE VERIFIED"
    assert not res_single["is_violation"]

    # Multiple faces (cheating infraction)
    res_multi = tracker.analyze_faces(dummy_frame, [(100, 100, 120, 120), (350, 100, 120, 120)])
    assert res_multi["face_count"] == 2
    assert res_multi["is_violation"]
    assert "MULTIPLE FACES" in res_multi["status"]


def test_proctor_engine_suspicion():
    engine = ProctorEngine()
    assert engine.suspicion_score == 0.0

    # Simulate head deviation
    pose_res = {"is_deviated": True, "direction": "LOOKING DOWN (POSSIBLE CHEAT SHEET)", "yaw": 0.0, "pitch": -25.0}
    gaze_res = {"is_looking_away": False, "gaze_state": "CENTER"}
    presence_res = {"status": "CANDIDATE VERIFIED", "is_violation": False}
    objects_res = []

    res = engine.update(pose_res, gaze_res, presence_res, objects_res)
    assert res["suspicion_score"] > 0.0

    # Test suspicion decay on clean frames
    pose_res["is_deviated"] = False
    clean_res = engine.update(pose_res, gaze_res, presence_res, objects_res)
    assert clean_res["suspicion_score"] < res["suspicion_score"]


def test_export_utilities():
    infractions = [
        {"timestamp": "10:15:30", "elapsed_seconds": 90, "type": "HEAD_DEVIATION", "severity": "WARNING", "suspicion_index": 45.0, "description": "Looking down"}
    ]
    csv_str = generate_csv_audit_report("CAND-001", "CS-101", infractions)
    assert "CAND-001" in csv_str
    assert "HEAD_DEVIATION" in csv_str

    json_str = generate_json_audit_package("CAND-001", "CS-101", 95.0, infractions)
    assert "ProctorAI" in json_str
    assert "CS-101" in json_str


def test_calibration_pipeline():
    cal = ProctorCalibrator(sample_size=3)
    for _ in range(3):
        cal.add_sample(yaw=3.0, pitch=-2.0, gaze_ratio=1.0)
    assert cal.is_calibrated
    dev = cal.get_calibrated_deviation(current_yaw=8.0, current_pitch=-2.0, current_gaze=1.0)
    assert dev["delta_yaw"] == 5.0
    assert dev["delta_pitch"] == 0.0


if __name__ == "__main__":
    test_head_pose_forward()
    test_eye_gaze_evaluator()
    test_presence_tracker()
    test_proctor_engine_suspicion()
    test_export_utilities()
    test_calibration_pipeline()
    print("All proctoring engine unit tests passed successfully!")
