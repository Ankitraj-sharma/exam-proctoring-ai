"""
Unit tests for candidate baseline calibration wizard.
"""

from calibration import ProctorCalibrator


def test_calibration_accumulation():
    calibrator = ProctorCalibrator(sample_size=5)
    assert not calibrator.is_calibrated

    # Add 4 samples (not yet calibrated)
    for _ in range(4):
        done = calibrator.add_sample(yaw=2.0, pitch=-1.0, gaze_ratio=1.0)
        assert not done

    # 5th sample triggers calibration
    done = calibrator.add_sample(yaw=2.0, pitch=-1.0, gaze_ratio=1.0)
    assert done
    assert calibrator.is_calibrated
    assert calibrator.baseline["yaw"] == 2.0
    assert calibrator.baseline["pitch"] == -1.0


def test_calibrated_deviation():
    calibrator = ProctorCalibrator(sample_size=3)
    calibrator.add_sample(yaw=5.0, pitch=-2.0, gaze_ratio=1.0)
    calibrator.add_sample(yaw=5.0, pitch=-2.0, gaze_ratio=1.0)
    calibrator.add_sample(yaw=5.0, pitch=-2.0, gaze_ratio=1.0)

    # Current instantaneous reading: yaw 15.0, pitch -2.0
    diff = calibrator.get_calibrated_deviation(15.0, -2.0, 1.0)
    assert diff["delta_yaw"] == 10.0
    assert diff["delta_pitch"] == 0.0


if __name__ == "__main__":
    test_calibration_accumulation()
    test_calibrated_deviation()
    print("All calibration unit tests passed successfully!")
