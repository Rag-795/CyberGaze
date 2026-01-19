"""
Face Alignment Module
Eye-based face alignment for improved recognition.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from config import Config


def get_eye_centers(eyes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """
    Calculate center points of detected eyes.
    
    Args:
        eyes: List of eye bounding boxes (x, y, w, h) relative to face ROI
        
    Returns:
        List of eye center points (x, y)
    """
    centers = []
    for (ex, ey, ew, eh) in eyes:
        center_x = ex + ew // 2
        center_y = ey + eh // 2
        centers.append((center_x, center_y))
    return centers


def calculate_rotation_angle(left_eye: Tuple[int, int], 
                             right_eye: Tuple[int, int]) -> float:
    """
    Calculate rotation angle to make eyes horizontal.
    
    Args:
        left_eye: (x, y) center of left eye
        right_eye: (x, y) center of right eye
        
    Returns:
        Angle in degrees to rotate (positive = counterclockwise)
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def align_face(gray: np.ndarray,
               face: Tuple[int, int, int, int],
               eyes: Optional[List[Tuple[int, int, int, int]]] = None,
               target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Align face to have horizontal eyes and crop to standard size.
    
    Args:
        gray: Full grayscale image
        face: Face bounding box (x, y, w, h)
        eyes: Eye bounding boxes relative to face ROI (optional)
        target_size: Output size (width, height), default from config
        
    Returns:
        Aligned and cropped face image
    """
    if target_size is None:
        target_size = Config.ALIGNED_FACE_SIZE
    
    x, y, w, h = face
    face_roi = gray[y:y+h, x:x+w].copy()
    
    # If we have exactly 2 eyes, align them
    if eyes is not None and len(eyes) >= 2:
        centers = get_eye_centers(eyes[:2])
        
        # Sort by x coordinate (left eye first)
        centers.sort(key=lambda c: c[0])
        left_eye, right_eye = centers[0], centers[1]
        
        # Calculate rotation
        angle = calculate_rotation_angle(left_eye, right_eye)
        
        # Rotate around face center (must be floats for OpenCV)
        face_center = (float(w) / 2, float(h) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(face_center, float(angle), 1.0)
        face_roi = cv2.warpAffine(face_roi, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    
    # Resize to target size
    aligned = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
    
    return aligned


def extract_aligned_face(gray: np.ndarray,
                         face: Tuple[int, int, int, int],
                         eye_cascade=None) -> Tuple[np.ndarray, dict]:
    """
    Full pipeline: detect eyes, align face, crop.
    
    Args:
        gray: Full grayscale image
        face: Face bounding box
        eye_cascade: Optional eye cascade classifier
        
    Returns:
        Tuple of (aligned_face, info_dict)
    """
    from .detection import detect_eyes
    
    x, y, w, h = face
    info = {
        "eyes_detected": 0,
        "aligned": False,
        "rotation_angle": 0.0
    }
    
    # Detect eyes
    eyes = detect_eyes(gray, face)
    info["eyes_detected"] = len(eyes)
    
    if len(eyes) >= 2:
        # Get best two eyes
        centers = get_eye_centers(eyes[:2])
        centers.sort(key=lambda c: c[0])
        angle = calculate_rotation_angle(centers[0], centers[1])
        info["rotation_angle"] = angle
        info["aligned"] = True
    
    aligned = align_face(gray, face, eyes if len(eyes) >= 2 else None)
    
    return aligned, info
