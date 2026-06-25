# Internship Project: Pose Landmark Tracker & Posture Analysis
# postureguard-real-time-webcam-posture-detection-Aditi-36
Praxis internship project: PostureGuard — Real-Time Webcam Posture Detection

A Python-based computer vision application for real-time pose detection and posture analysis. The project captures pose landmarks from a live camera feed using MediaPipe and computes detailed posture metrics including neck inclination, shoulder slope, and back curvature.

## 🚀 Features
* **Real-Time Pose Detection:** Uses MediaPipe's pose landmarker for accurate, low-latency landmark detection
* **Live Posture Metrics:** Computes neck angle, shoulder slope, and back curvature in real-time
* **Verdict System:** Classifies posture as GOOD or POOR based on configurable thresholds
* **Landmark Visualization:** Draws pose skeleton and hand landmarks on the live feed
* **FPS Benchmarking:** Monitors and displays performance metrics
* **Modular Architecture:** Cleanly separated concerns with dedicated modules for capture and angle calculations

## 📂 Project Structure
```text
├── src/
│   ├── capture.py           # Main script: captures camera feed and displays pose landmarks
│   └── anglecalc.py         # Posture metrics engine: computes neck, shoulder, and back angles
├── tests/
│   ├── test_angles.py       # Unit tests for posture metrics
│   ├── test_landmarks.py    # Unit tests for landmark processing
│   └── conftest.py          # Test fixtures and configuration
├── pose_landmarker_lite.task # MediaPipe pose model
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 📋 Requirements
- Python 3.x
- MediaPipe Tasks
- OpenCV (cv2)
- pytest (for testing)

See `requirements.txt` for full dependency list.

## 🔧 Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the pose model is present:**
   - The `pose_landmarker_lite.task` file should be in the project root directory

## ▶️ Running the Application

Start the live pose detection and posture analysis:
```bash
python src/capture.py
```

**Controls:**
- Press `q` in the video window to quit

**Output:**
- Real-time landmark coordinates (nose, ears, shoulders, hips)
- Posture metrics: neck angle, shoulder slope, back curvature
- Verdict: GOOD (all metrics within thresholds) or POOR (metrics exceed thresholds)
- FPS performance metrics (updated every 30 frames)

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Individual test files:
```bash
pytest tests/test_angles.py        # Posture metrics tests
pytest tests/test_landmarks.py     # Landmark processing tests
```

## 📊 Posture Metrics

The application evaluates three key metrics:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Neck Angle** | ≤ 15° | Inclination of head relative to shoulders |
| **Shoulder Slope** | ≤ 5% | Asymmetry/slope between left and right shoulders |
| **Back Curvature** | ≤ 0.15 | Curvature of spine based on nose and hip positions |

Posture is classified as **GOOD** when all metrics are within thresholds, otherwise **POOR**.

## 🎯 Key Modules

### `capture.py`
- Initializes MediaPipe pose landmarker in LIVE_STREAM mode
- Processes camera frames asynchronously
- Extracts and converts pose landmarks to dictionary format
- Calls posture metrics computation
- Visualizes pose skeleton and hand landmarks

### `anglecalc.py`
- Core posture analysis engine
- Computes three posture metrics from landmark coordinates
- Provides per-frame verdict (GOOD/POOR)
- Handles edge cases and visibility thresholds

