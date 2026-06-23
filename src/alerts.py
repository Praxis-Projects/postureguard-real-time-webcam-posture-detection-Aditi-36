import os
import csv
import time
import yaml
from plyer import notification

class PostureAlertEngine:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.load_config()
        
        # State tracking variables
        self.poor_streak_start_time = None
        self.last_alert_time = 0
        self.in_cooldown = False
        self.good_posture_streak_start = None
        
        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/alert_log.csv"
        self.init_csv_log()

    def load_config(self):
        """Loads configuration configurations from the YAML file."""
        default_config = {
            "neck_threshold_degrees": 15.0,
            "shoulder_threshold_pct": 5.0,
            "back_threshold_ratio": 0.15,
            "alert_delay_seconds": 30,
            "cooldown_minutes": 5
        }
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f) or default_config
        else:
            self.config = default_config

    def init_csv_log(self):
        """Initializes the CSV logger with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "posture_type", "angle_value"])

    def log_alert_to_csv(self, posture_type, angle_value):
        """Appends a fired alert record into the log tracking system."""
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), posture_type, angle_value])

    def trigger_desktop_notification(self, reason):
        """Fires an OS-native desktop alert notification banner."""
        notification.notify(
            title="PostureGuard: Sit up straight!",
            message=f"Head too far forward. ({reason})",
            app_name="PostureGuard",
            timeout=5
        )

    def process_frame_verdict(self, metrics):
        """
        Main state machine handler processing per-frame analytics.
        metrics format: {'neck_angle': float, 'shoulder_slope': float, 'back_curvature': float, 'verdict': 'GOOD'|'POOR', 'reason': str}
        """
        current_time = time.time()
        cooldown_duration = self.config["cooldown_minutes"] * 60
        alert_delay = self.config["alert_delay_seconds"]

        # Check for active cooldown period
        if self.in_cooldown:
            if current_time - self.last_alert_time > cooldown_duration:
                self.in_cooldown = False
            else:
                # Early Reset Mechanism: If posture is GOOD for > 60 consecutive seconds, break cooldown early
                if metrics["verdict"] == "GOOD":
                    if self.good_posture_streak_start is None:
                        self.good_posture_streak_start = current_time
                    elif current_time - self.good_posture_streak_start >= 60:
                        self.in_cooldown = False  # Reset cooldown early!
                        self.good_posture_streak_start = None
                else:
                    self.good_posture_streak_start = None
                return  # Suppress alerts while locked down inside active cooldown window

        # Process standard alert logic tree when not in cooldown mode
        if metrics["verdict"] == "POOR":
            self.good_posture_streak_start = None
            if self.poor_streak_start_time is None:
                self.poor_streak_start_time = current_time
            
            # Check if user has maintained bad posture past the delay threshold
            elif current_time - self.poor_streak_start_time >= alert_delay:
                # Determine primary trigger factor for the log schema entry
                trigger_type = "Unknown"
                trigger_val = 0.0
                if metrics["neck_angle"] > self.config["neck_threshold_degrees"]:
                    trigger_type, trigger_val = "neck_angle", metrics["neck_angle"]
                elif metrics["shoulder_slope"] > self.config["shoulder_threshold_pct"]:
                    trigger_type, trigger_val = "shoulder_slope", metrics["shoulder_slope"]
                elif metrics["back_curvature"] > self.config["back_threshold_ratio"]:
                    trigger_type, trigger_val = "back_curvature", metrics["back_curvature"]

                # Fire alert notifications and transition states
                self.trigger_desktop_notification(metrics["reason"])
                self.log_alert_to_csv(trigger_type, trigger_val)
                
                self.last_alert_time = current_time
                self.in_cooldown = True
                self.poor_streak_start_time = None
        else:
            # Reset poor streak counter if user fixes posture
            self.poor_streak_start_time = None