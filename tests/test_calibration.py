"""Verify tuple-based calibration and numerical coordinate mappings."""
import pytest
from src.logic.calibration import CalibrationConfig, HomographyCalibrator

@pytest.fixture
def calibrated():
    return HomographyCalibrator(CalibrationConfig(
        enabled=True, src_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        dst_points=[(0, 0), (2, 0), (2, 2), (0, 2)]))

@pytest.mark.parametrize('point,expected', [
    ((0, 0), (0, 0)), ((100, 100), (2, 2)), ((50, 50), (1, 1)),
    ((75, 25), (1.5, 0.5)), ((0, 100), (0, 2))])
def test_coordinate_mapping(calibrated, point, expected):
    assert calibrated.is_valid()
    assert calibrated.pixel_to_world(point) == pytest.approx(expected)
    assert calibrated.world_to_pixel(expected) == point

def test_invalid_or_disabled_calibration():
    disabled = HomographyCalibrator(CalibrationConfig(enabled=False))
    assert not disabled.is_valid()
    assert disabled.pixel_to_world((50, 50)) == (0.0, 0.0)
    invalid = HomographyCalibrator(CalibrationConfig(
        enabled=True, src_points=[(0, 0), (1, 1)], dst_points=[(0, 0), (1, 1)]))
    assert not invalid.is_valid()
