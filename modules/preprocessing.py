"""
Image Preprocessing Module
Handles illumination normalization and quality checks.
"""
import cv2
import numpy as np
from typing import Tuple
from config import Config


def preprocess(gray: np.ndarray, gamma: float = 0.7, mode: str = "detect") -> np.ndarray:
    """
    mode:
      - "detect": for face/eye detection + recognition (CLAHE+gamma ok)
      - "liveness": for blink metric (keep it mild to avoid artificial edges)
    """
    gray = cv2.medianBlur(gray, 3)

    if mode == "liveness":
        # Keep it simple/stable for edge/gradient based blink features
        return gray

    # "detect" mode (your original pipeline)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    gray = cv2.LUT(gray, table)

    return gray


def blur_score(gray: np.ndarray) -> float:
    """
    Calculate blur score using Laplacian variance.
    Higher score = sharper image.
    
    Args:
        gray: Grayscale image
        
    Returns:
        Blur score (variance of Laplacian)
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def mean_brightness(gray: np.ndarray) -> float:
    """
    Calculate mean brightness of image.
    
    Args:
        gray: Grayscale image
        
    Returns:
        Mean pixel intensity (0-255)
    """
    return float(np.mean(gray))


def quality_check(gray: np.ndarray,
                  min_brightness: float = None,
                  max_brightness: float = None,
                  blur_threshold: float = None) -> Tuple[bool, dict]:
    """
    Validate frame quality for face recognition.
    
    Args:
        gray: Grayscale image
        min_brightness: Minimum acceptable mean brightness
        max_brightness: Maximum acceptable mean brightness
        blur_threshold: Minimum blur score (Laplacian variance)
        
    Returns:
        Tuple of (is_valid, details_dict)
    """
    if min_brightness is None:
        min_brightness = Config.MIN_BRIGHTNESS
    if max_brightness is None:
        max_brightness = Config.MAX_BRIGHTNESS
    if blur_threshold is None:
        blur_threshold = Config.BLUR_THRESHOLD
    
    brightness = mean_brightness(gray)
    blur = blur_score(gray)
    
    issues = []
    
    if brightness < min_brightness:
        issues.append("too_dark")
    if brightness > max_brightness:
        issues.append("too_bright")
    if blur < blur_threshold:
        issues.append("too_blurry")
    
    return len(issues) == 0, {
        "brightness": brightness,
        "blur_score": blur,
        "issues": issues
    }


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if needed.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
