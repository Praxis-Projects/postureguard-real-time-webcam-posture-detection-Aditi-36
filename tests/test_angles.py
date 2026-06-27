import pytest
from anglecalc import compute_posture_metrics

@pytest.fixture
def perfect_posture_data():
    """Generates an ideal posture layout where values fall way below limits."""
    return {
        "nose": {"x": 0.5, "y": 0.15, "visibility": 1.0},
        "left_ear": {"x": 0.46, "y": 0.22, "visibility": 1.0},
        "right_ear": {"x": 0.54, "y": 0.22, "visibility": 1.0},
        "left_shoulder": {"x": 0.40, "y": 0.40, "visibility": 1.0},
        "right_shoulder": {"x": 0.60, "y": 0.40, "visibility": 1.0},
        "left_hip": {"x": 0.40, "y": 0.75, "visibility": 1.0},
        "right_hip": {"x": 0.60, "y": 0.75, "visibility": 1.0}
    }

def test_all_good_posture(perfect_posture_data):
    metrics = compute_posture_metrics(perfect_posture_data)
    assert metrics["verdict"] == "GOOD"
    assert metrics["neck_angle"] <= 15.0
    assert metrics["shoulder_slope"] <= 5.0
    assert metrics["back_curvature"] <= 0.15

def test_bad_neck_angle(perfect_posture_data):
    # Lean ears severely forward along X axis to force a bad angle
    perfect_posture_data["left_ear"]["x"] = 0.55
    perfect_posture_data["right_ear"]["x"] = 0.63
    
    metrics = compute_posture_metrics(perfect_posture_data)
    assert metrics["verdict"] == "POOR"
    assert "neck_angle" in metrics["reason"]

def test_bad_shoulder_slope(perfect_posture_data):
    # Drop one shoulder vertically significantly lower than the other
    perfect_posture_data["left_shoulder"]["y"] = 0.45 
    perfect_posture_data["right_shoulder"]["y"] = 0.38
    
    metrics = compute_posture_metrics(perfect_posture_data)
    assert metrics["verdict"] == "POOR"
    assert "shoulder_slope" in metrics["reason"]

def test_bad_back_curvature(perfect_posture_data):
    # Shift hips far off to the side relative to the shoulder midpoint line
    perfect_posture_data["left_hip"]["x"] = 0.65
    perfect_posture_data["right_hip"]["x"] = 0.85
    
    metrics = compute_posture_metrics(perfect_posture_data)
    assert metrics["verdict"] == "POOR"
    assert "back_curvature" in metrics["reason"]