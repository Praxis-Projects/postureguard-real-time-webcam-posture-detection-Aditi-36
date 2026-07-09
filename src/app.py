"""
PostureGuard Flask Backend Controller 
==================================================
Serves the web dashboard, manages dynamic path alignment, and coordinates
real-time telemetry between templates/index.html and the ML model in capture.py.

"""

import os
import sys
import time
import json
import logging
import threading
from flask import Flask, Response, render_template
from flask_sock import Sock

# 1. DYNAMIC PATH ALIGNMENT: Resolve Python imports and MediaPipe asset directories
# Get the absolute path of the directory containing this script (i.e., 'src')
src_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of 'src' (the project root)
project_root = os.path.dirname(src_dir)

# Add 'src' to the import search paths so capture, alerts, and anglecalc load cleanly
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Set the active working directory to the project root so MediaPipe finds 'pose_landmarker_lite.task'
os.chdir(project_root)

# Silence verbose Werkzeug logging in the console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Import the posture analysis ML engine from capture.py
from capture import PostureAnalyzer

# 2. INITIALIZE FLASK AND WEB SOCKETS
app = Flask(__name__, template_folder=os.path.join(src_dir, 'templates'))
sock = Sock(app)

# Initialize and start the asynchronous ML processing engine
analyzer = PostureAnalyzer(headless=True)
analyzer.start()

active_sockets = set()
socket_lock = threading.Lock()

def ws_telemetry_broadcaster():
    """Asynchronously broadcasts calculated telemetry values to active web clients at 10Hz."""
    last_alert_count = 0
    
    while True:
        time.sleep(0.10)  # ~10 updates per second
        if not active_sockets:
            continue

        with analyzer.lock:
            elapsed_time = "00:00:00"
            if analyzer.session_active and analyzer.session_start_time:
                elapsed_seconds = int(time.time() - analyzer.session_start_time)
                hrs = elapsed_seconds // 3600
                mins = (elapsed_seconds % 3600) // 60
                secs = elapsed_seconds % 60
                elapsed_time = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            # Detect if a new alert was fired in this interval
            alert_triggered = False
            current_alerts = analyzer.alert_count
            if analyzer.session_active and (current_alerts > last_alert_count):
                alert_triggered = True
                last_alert_count = current_alerts
            elif not analyzer.session_active:
                last_alert_count = 0

            payload = {
                "neck_angle": round(analyzer.neck_angle, 1),
                "shoulder_slope": round(analyzer.shoulder_slope, 1),
                "verdict": analyzer.verdict,
                "alert_count": analyzer.alert_count,
                "elapsed_time": elapsed_time,
                "alert_triggered": alert_triggered
            }

        # Send formatted packet over active WS routes
        message = json.dumps(payload)
        with socket_lock:
            broken_sockets = set()
            for ws in active_sockets:
                try:
                    ws.send(message)
                except Exception:
                    broken_sockets.add(ws)
            for ws in broken_sockets:
                active_sockets.remove(ws)

# Spawn broadcast worker daemon thread
threading.Thread(target=ws_telemetry_broadcaster, daemon=True).start()


@app.route('/')
def index_route():
    """Serves the posture analysis workspace dashboard HTML."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed_route():
    """Pipes the live MJPEG stream generated inside capture.py containing overlays directly to index.html."""
    def frame_generator():
        while True:
            if analyzer.latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + analyzer.latest_frame + b'\r\n')
            else:
                # Placeholder tiny black frame to prevent stream broken indicators during startup
                time.sleep(0.1)
                continue
            time.sleep(0.04)  # Sync around ~25 FPS
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@sock.route('/ws')
def ws_command_route(ws):
    """Establishes command signaling channel for start/stop and calibration controls."""
    with socket_lock:
        active_sockets.add(ws)
    
    try:
        while True:
            raw_command = ws.receive()
            if not raw_command:
                break
            payload = json.loads(raw_command)
            action = payload.get("action")

            if action == "start_session":
                analyzer.start_session()
            elif action == "stop_session":
                analyzer.stop_session()
                # Package and push stats immediately to trigger Summary Modal (Sprint 4 Task 2)
                stats = analyzer.get_stats()
                ws.send(json.dumps({
                    "type": "summary",
                    "duration": stats["duration"],
                    "accuracy": stats["accuracy"],
                    "alerts": stats["alerts"],
                    "avg_neck": stats["avg_neck"]
                }))
            elif action == "calibrate":
                analyzer.calibrate()
    except Exception:
        pass
    finally:
        with socket_lock:
            if ws in active_sockets:
                active_sockets.remove(ws)


if __name__ == "__main__":
    print("\n========================================================")
    print("      POSTUREGUARD WORKSPACE ACTIVE ON PORT 5000")
    print("      Navigate to: http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)