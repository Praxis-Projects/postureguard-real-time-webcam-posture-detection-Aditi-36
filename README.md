# PostureGuard — Real-Time Posture Analysis (Internship Project)

Lightweight Python project that uses MediaPipe Tasks and OpenCV to detect
pose landmarks from a webcam, compute posture metrics, and expose telemetry
via a local Flask dashboard.

## Quick overview
- Live webcam posture analysis (OpenCV + MediaPipe Tasks)
- Web dashboard served by `src/app.py` with a WebSocket telemetry feed
- Background posture engine in `src/capture.py` that exposes frames and metrics
- Unit tests located in `tests/` (run with `pytest`)

## Repository layout
```text
.
├── config.yaml
├── pose_landmarker_lite.task   # MediaPipe task model (required at runtime)
├── requirements.txt            # Python dependencies
├── src/
│   ├── app.py                  # Flask web server + WS broadcaster
│   ├── capture.py              # PostureAnalyzer: camera loop & ML integration
│   ├── anglecalc.py            # Posture metrics computation
│   ├── alerts.py               # Alerting engine used during sessions
│   └── templates/
│       └── index.html          # Web dashboard UI
├── tests/
│   ├── conftest.py
│   ├── test_alert.py
│   ├── test_angles.py
│   └── test_landmarks.py
└── README.md
```

## Requirements
- Python 3.10+ recommended
- See `requirements.txt` (installs `mediapipe`, `opencv-python`, `Flask`, `pytest`, etc.)

Prerequisite:
- Download and place `pose_landmarker_lite.task` in the project root before running the app.
  This model is required by the MediaPipe pose pipeline and must be available at the repository root.

Install dependencies:
```bash
pip install -r requirements.txt
```

Important: ensure `pose_landmarker_lite.task` is present in the project root —
the capture engine and `src/app.py` expect the model at that path.

## Running

1) Run the web dashboard (recommended):

```bash
python src/app.py
```

Then open http://127.0.0.1:5000 in your browser. The Flask app spawns a
background `PostureAnalyzer` (from `src/capture.py`) and serves a live MJPEG
video feed plus WebSocket telemetry to the dashboard.

2) Run the capture script standalone (desktop preview):

```bash
python src/capture.py
```

This opens an OpenCV window showing real-time overlays. Press `q` in the
window to quit.

## Tests

Run the full test suite with:

```bash
pytest -q
```

Or run an individual test file, for example:

```bash
pytest tests/test_angles.py -q
```

## File descriptions
- `src/capture.py`: Background MediaPipe PoseLandmarker wrapper exposing
   `PostureAnalyzer` with thread-safe fields (`latest_frame`, `neck_angle`, etc.).
- `src/anglecalc.py`: Functions to compute `neck_angle`, `shoulder_slope`,
   `back_curvature` and the per-frame `verdict`.
- `src/alerts.py`: Alerting logic used when sessions are active.
- `src/app.py`: Flask server and WebSocket broadcaster that integrates the
   background analyzer and serves `templates/index.html`.

## Troubleshooting
- If the camera is not detected, ensure no other application is using it and
   that OpenCV can access `/dev/video*` (or the correct device on Windows).
- If MediaPipe fails to load the model, verify `pose_landmarker_lite.task` is
   present and readable in the project root.




