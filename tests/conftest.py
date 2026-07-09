import pytest
import cv2
import numpy as np
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def person_fixture_image():
    # Provides an image matrix containing a clearly visible human target.
    path_to_image = TESTS_DIR / "person photo.jpg"
    if path_to_image.exists():
        img = cv2.imread(str(path_to_image))
        if img is not None:
            return img

    # Fallback: Generate a structural synthetic human shape if no file exists
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(canvas, (320, 140), 50, (200, 200, 200), -1)
    poly_points = np.array([[220, 440], [420, 440], [360, 220], [280, 220]], np.int32)
    cv2.fillPoly(canvas, [poly_points], (200, 200, 200))
    return canvas


@pytest.fixture
def blank_fixture_image():
    # Loads a blank white image file
    path_to_image = TESTS_DIR / "blank_white.jpg"
    if path_to_image.exists():
        img = cv2.imread(str(path_to_image))
        if img is not None:
            return img

    # Fallback: Generate if file doesn't exist
    return np.full((480, 640, 3), 255, dtype=np.uint8)
