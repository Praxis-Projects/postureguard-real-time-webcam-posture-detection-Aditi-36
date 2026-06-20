from __future__ import annotations
from typing import TYPE_CHECKING
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import time
import cv2

if TYPE_CHECKING:
    pass


TARGET_LANDMARKS = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip"
}

# Hand landmark indices in the 33-point pose model
LEFT_HAND_INDICES = [15, 17, 19, 21]  # Left wrist, pinky, ring, middle, index
RIGHT_HAND_INDICES = [16, 18, 20, 22]  # Right wrist, pinky, ring, middle, index

# Hand connections for better visualization
HAND_CONNECTIONS = [
    (15, 17), (17, 19), (19, 21),  # Left hand chain
    (16, 18), (18, 20), (20, 22)   # Right hand chain
]

# Global variables for tracking results asynchronously in LIVE_STREAM mode
latest_result = None
frame_count = 0

def save_result_callback (result: 'vision.PoseLandmarkerResult' , output_image: 'mp.Image' , timestamp_ms: int): # type: ignore
    """Callback function executed whenever the landmarker finishes processing a frame."""
    global latest_result
    latest_result = result

# Initialize webcam
cap = cv2.VideoCapture(0)

# Configure the MediaPipe Task options for a LIVE_STREAM 
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result_callback,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Benchmarking states
start_time = time.time()
fps = 0.0

print("Press 'q' in the video window to quit.\n")

# Use context manager to initialize the new landmarker detector
with vision.PoseLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_count += 1
        
        # Mirror frame for selfie-view and switch BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV image matrix to MediaPipe's custom Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Provide monotonically increasing timestamps in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Send frame to the background pipeline (non-blocking)
        detector.detect_async(mp_image, frame_timestamp_ms)

        # Handle the latest available detection result safely
        if latest_result and latest_result.pose_landmarks:
            # MediaPipe Tasks outputs a list of poses; slice into the first detected pose
            pose_landmarks = latest_result.pose_landmarks[0]
            
            # Print requested landmark coordinates directly to console
            print(f"--- Frame {frame_count} Landmark Data ---")
            for idx, name in TARGET_LANDMARKS.items():
                if idx < len(pose_landmarks):
                    landmark = pose_landmarks[idx]
                    print(f"{name} (ID {idx}): x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}, visibility={landmark.visibility:.4f}")
            print("-" * 40)
            
            # Draw full pose skeleton
            for pose_landmarks_list in latest_result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=pose_landmarks_list,
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Enhanced hand visualization with explicit drawing
            h, w, c = frame.shape
            for hand_idx, hand_indices in enumerate([LEFT_HAND_INDICES, RIGHT_HAND_INDICES]):
                hand_name = "LEFT" if hand_idx == 0 else "RIGHT"
                hand_color = (255, 0, 0) if hand_idx == 0 else (0, 0, 255)  # Blue for left, Red for right
                
                # Draw hand landmarks as larger circles
                for idx in hand_indices:
                    if idx < len(pose_landmarks):
                        landmark = pose_landmarks[idx]
                        if landmark.visibility > 0.3:  # Only draw if visible enough
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 6, hand_color, -1)
                            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
                
                # Draw hand connections
                for start_idx, end_idx in HAND_CONNECTIONS:
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        start_landmark = pose_landmarks[start_idx]
                        end_landmark = pose_landmarks[end_idx]
                        
                        if start_landmark.visibility > 0.3 and end_landmark.visibility > 0.3:
                            start_x = int(start_landmark.x * w)
                            start_y = int(start_landmark.y * h)
                            end_x = int(end_landmark.x * w)
                            end_y = int(end_landmark.y * h)
                            cv2.line(frame, (start_x, start_y), (end_x, end_y), hand_color, 2)

        # Calculate and benchmark FPS every 30 frames
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            print(f"\n[BENCHMARK] Current Performance: {fps:.2f} FPS\n")
            start_time = time.time()

        # Display performance onto the OpenCV Window canvas
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Tasks API - Live Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()