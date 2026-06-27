import pytest
import cv2
import numpy as np
import os

@pytest.fixture
def person_fixture_image():
    # Provides an image matrix containing a clearly visible human target.
    path_to_image = "tests/person photo.jpg"
    if os.path.exists(path_to_image):
        img = cv2.imread(path_to_image)
        if img is not None:
            return img

    # Fallback: Generate a structural synthetic human shape if no file exists
    # A dark canvas with a light gray structural polygon representing a head and torso
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Head
    cv2.circle(canvas, (320, 140), 50, (200, 200, 200), -1)
    # Torso / Limbs
    poly_points = np.array([[220, 440], [420, 440], [360, 220], [280, 220]], np.int32)
    cv2.fillPoly(canvas, [poly_points], (200, 200, 200))
    return canvas


@pytest.fixture
def blank_fixture_image():
    # Loads a blank white image file
    path_to_image = "tests/blank_white.jpg"
    
    if os.path.exists(path_to_image):
        img = cv2.imread(path_to_image)
        if img is not None:
            return img
    
    # Fallback: Generate if file doesn't exist
    return np.full((480, 640, 3), 255, dtype=np.uint8)