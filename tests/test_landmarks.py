import pytest
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

def run_pose_inference(image_matrix: np.ndarray) -> vision.PoseLandmarkerResult: # type: ignore
    #Helper framework function to initialize the task detector in IMAGE mode and return the structured landing points
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, 
        num_poses=1
    )
    
    with vision.PoseLandmarker.create_from_options(options) as detector:
        # Convert the structural numpy matrix to MediaPipe custom Image container
        rgb_image = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return detector.detect(mp_image)


def test_person_image(person_fixture_image):
    #Verifies that an image with a visible person yields high confidence tracking.
    result = run_pose_inference(person_fixture_image)
    
    # Assert that a pose was actually detected
    assert result.pose_landmarks, "No human pose detected in the person fixture image."
    
    pose_landmarks = result.pose_landmarks[0]
    
    # Filter landmarks that meet the visibility criteria (> 0.5)
    high_visibility_landmarks = [
        lm for lm in pose_landmarks if lm.visibility > 0.5
    ]
    
    # Requirement Check: ≥ 25 landmarks must possess visibility > 0.5
    assert len(high_visibility_landmarks) >= 25, (
        f"Expected >= 25 high-visibility landmarks, but only found {len(high_visibility_landmarks)}."
    )


def test_blank_image_returns_no_false_detections(blank_fixture_image):
    #Verifies that a blank white image does not trigger false positive trackers.
    result = run_pose_inference(blank_fixture_image)
    
    # If MediaPipe completely ignores the frame (returns empty list), the test passes cleanly
    if not result.pose_landmarks:
        return  
        
    # If it does return structural values, ensure absolutely none of them have high confidence
    pose_landmarks = result.pose_landmarks[0]
    for lm in pose_landmarks:
        assert lm.visibility <= 0.5, (
            f"False positive detected! Landmark found with high confidence visibility: {lm.visibility}"
        )