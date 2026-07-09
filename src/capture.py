from __future__ import annotations
import threading
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

from anglecalc import compute_posture_metrics
from alerts import PostureAlertEngine

TARGET_LANDMARKS = {
    0: "nose",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip"
}

LEFT_HAND_INDICES = [15, 17, 19, 21]
RIGHT_HAND_INDICES = [16, 18, 20, 22]

HAND_CONNECTIONS = [
    (15, 17), (17, 19), (19, 21),
    (16, 18), (18, 20), (20, 22)
]


class PostureAnalyzer:
    """
    Wraps the MediaPipe pose pipeline in a background thread and exposes
    thread-safe state (latest JPEG frame + current metrics) so a Flask
    server (app.py) can read it while the capture loop keeps running.
    """

    def __init__(self, headless: bool = True, model_path: str = "pose_landmarker_lite.task"):
        self.headless = headless
        self.model_path = model_path

        self.lock = threading.Lock()

        self.latest_frame: bytes | None = None

        self.neck_angle = 0.0
        self.shoulder_slope = 0.0
        self.back_curvature = 0.0
        self.verdict = "GOOD"
        self.reason = ""

        self.alert_count = 0
        self.session_active = False
        self.session_start_time: float | None = None

        self._neck_samples: list[float] = []
        self._good_frames = 0
        self._total_frames = 0

        self.alert_system = PostureAlertEngine(alert_callback=self._on_alert)

        self._latest_result = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def start_session(self):
        with self.lock:
            self.session_active = True
            self.session_start_time = time.time()
            self.alert_count = 0
            self._neck_samples = []
            self._good_frames = 0
            self._total_frames = 0

    def stop_session(self):
        with self.lock:
            self.session_active = False

    def calibrate(self):
        pass

    def get_stats(self) -> dict:
        with self.lock:
            duration_s = 0
            if self.session_start_time:
                duration_s = int(time.time() - self.session_start_time)
            hrs, rem = divmod(duration_s, 3600)
            mins, secs = divmod(rem, 60)
            duration_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            accuracy = 0.0
            if self._total_frames > 0:
                accuracy = round((self._good_frames / self._total_frames) * 100, 1)

            avg_neck = round(sum(self._neck_samples) / len(self._neck_samples), 1) if self._neck_samples else 0.0

            return {
                "duration": duration_str,
                "accuracy": accuracy,
                "alerts": self.alert_count,
                "avg_neck": avg_neck,
            }

    def _on_alert(self):
        with self.lock:
            self.alert_count += 1

    def _save_result_callback(self, result, output_image, timestamp_ms):
        self._latest_result = result

    def _run(self):
        cap = cv2.VideoCapture(0)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._save_result_callback,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        with vision.PoseLandmarker.create_from_options(options) as detector:
            while cap.isOpened() and self._running:
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms = int(time.time() * 1000)
                detector.detect_async(mp_image, frame_timestamp_ms)

                if self._latest_result and self._latest_result.pose_landmarks:
                    pose_landmarks = self._latest_result.pose_landmarks[0]

                    for pose_landmarks_list in self._latest_result.pose_landmarks:
                        drawing_utils.draw_landmarks(
                            image=frame,
                            landmark_list=pose_landmarks_list,
                            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(204, 153, 255), thickness=2, circle_radius=2),
                            connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 153, 204), thickness=2)
                        )

                    landmarks_dict = {}
                    for idx, name in TARGET_LANDMARKS.items():
                        if idx < len(pose_landmarks):
                            landmark = pose_landmarks[idx]
                            landmarks_dict[name] = {
                                'x': landmark.x, 'y': landmark.y, 'visibility': landmark.visibility
                            }

                    metrics = compute_posture_metrics(landmarks_dict)

                    with self.lock:
                        self.neck_angle = metrics['neck_angle']
                        self.shoulder_slope = metrics['shoulder_slope']
                        self.back_curvature = metrics['back_curvature']
                        self.verdict = metrics['verdict']
                        self.reason = metrics['reason']

                        if self.session_active:
                            self._total_frames += 1
                            self._neck_samples.append(metrics['neck_angle'])
                            if metrics['verdict'] == "GOOD":
                                self._good_frames += 1

                    if self.session_active:
                        self.alert_system.process_frame_verdict(metrics)

                    h, w, _ = frame.shape
                    for hand_idx, hand_indices in enumerate([LEFT_HAND_INDICES, RIGHT_HAND_INDICES]):
                        hand_color = (255, 153, 0) if hand_idx == 0 else (0, 153, 255)
                        for idx in hand_indices:
                            if idx < len(pose_landmarks):
                                landmark = pose_landmarks[idx]
                                if landmark.visibility > 0.3:
                                    x, y = int(landmark.x * w), int(landmark.y * h)
                                    cv2.circle(frame, (x, y), 6, hand_color, -1)
                                    cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
                        for start_idx, end_idx in HAND_CONNECTIONS:
                            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                                sl, el = pose_landmarks[start_idx], pose_landmarks[end_idx]
                                if sl.visibility > 0.3 and el.visibility > 0.3:
                                    cv2.line(frame, (int(sl.x * w), int(sl.y * h)),
                                             (int(el.x * w), int(el.y * h)), hand_color, 2)

                ok, buf = cv2.imencode('.jpg', frame)
                if ok:
                    with self.lock:
                        self.latest_frame = buf.tobytes()

                if not self.headless:
                    cv2.imshow('PostureGuard', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        self._running = False


if __name__ == "__main__":
    print("Press 'q' in the video window to quit.\n")
    analyzer = PostureAnalyzer(headless=False)
    analyzer.start_session()
    analyzer._running = True
    analyzer._run()
