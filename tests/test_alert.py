import pytest
import os
import csv
import time
from unittest.mock import MagicMock
from alerts import PostureAlertEngine

@pytest.fixture
def alert_engine(tmp_path):
    # Setup temporary configurations to avoid cluttering production file structures
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
neck_threshold_degrees: 25.0
shoulder_threshold_pct: 5.0
back_threshold_ratio: 0.15
alert_delay_seconds: 2
cooldown_minutes: 1
""")
    engine = PostureAlertEngine(config_path=str(cfg_file))
    engine.trigger_desktop_notification = MagicMock() # Mock OS notifications
    return engine

def test_alert_fires_after_sustained_poor_posture(alert_engine):
    bad_metrics = {"neck_angle": 26.0, "shoulder_slope": 2.0, "back_curvature": 0.05, "verdict": "POOR", "reason": "neck_angle fault"}
    
    # Simulating frames spanning a timeline
    alert_engine.process_frame_verdict(bad_metrics)
    time.sleep(2.5)
    alert_engine.process_frame_verdict(bad_metrics)
    
    assert alert_engine.trigger_desktop_notification.called
    assert os.path.exists(alert_engine.log_file)

def test_alert_does_not_fire_prematurely(alert_engine):
    bad_metrics = {"neck_angle": 26.0, "verdict": "POOR", "reason": "fault"}
    
    alert_engine.process_frame_verdict(bad_metrics)
    # Less than delay runtime configuration
    time.sleep(0.5) 
    alert_engine.process_frame_verdict(bad_metrics)
    
    assert not alert_engine.trigger_desktop_notification.called

def test_cooldown_suppression(alert_engine):
    bad_metrics = {"neck_angle": 26.0, "verdict": "POOR", "reason": "fault"}
    
    # Fire first alert
    alert_engine.process_frame_verdict(bad_metrics)
    time.sleep(2.5)
    alert_engine.process_frame_verdict(bad_metrics)
    assert alert_engine.trigger_desktop_notification.call_count == 1
    
    # Try immediate re-trigger during active cooldown
    alert_engine.process_frame_verdict(bad_metrics)
    assert alert_engine.trigger_desktop_notification.call_count == 1