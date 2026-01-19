"""
Face Detection Module
Haar cascade-based face and eye detection with validation.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import Config
from .calibration import get_calibration_matrix, distance_gate


# Global cascade objects (loaded once)
_face_cascade = None
_eye_cascade = None


def _load_cascades():
    """Load Haar cascade classifiers."""
    global _face_cascade, _eye_cascade
    
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(Config.FACE_CASCADE)
        if _face_cascade.empty():
            # Try OpenCV's built-in cascades
            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
    
    if _eye_cascade is None:
        _eye_cascade = cv2.CascadeClassifier(Config.EYE_CASCADE)
        if _eye_cascade.empty():
            _eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )


def detect_faces(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in grayscale image using Haar cascade.
    
    Args:
        gray: Preprocessed grayscale image
        
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    _load_cascades()
    
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=Config.SCALE_FACTOR,
        minNeighbors=Config.MIN_NEIGHBORS,
        minSize=Config.MIN_FACE_SIZE
    )
    
    return [tuple(face) for face in faces]


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def detect_eyes(gray: np.ndarray,
                face_roi: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    _load_cascades()

    x, y, w, h = face_roi
    H, W = gray.shape[:2]

    # Upper ~65% of face is safer for eyes (avoid mouth/nostrils)
    y1 = max(0, y)
    y2 = min(H, y + int(0.65 * h))
    x1 = max(0, x)
    x2 = min(W, x + w)

    face_gray = gray[y1:y2, x1:x2]
    if face_gray.size == 0:
        return []

    # Scale eye size to face size -> more stable than fixed (20,20)
    min_eye = max(15, w // 12)
    max_eye = max(20, w // 2)

    eyes = _eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=8,                 # higher => less jitter/false positives
        minSize=(min_eye, min_eye),
        maxSize=(max_eye, max_eye)
    )

    candidates = []
    for (ex, ey, ew, eh) in eyes:
        # Basic shape filter (kills eyebrow/nostril false positives)
        ar = ew / float(eh + 1e-6)
        if ar < 0.6 or ar > 2.5:
            continue
        # keep eyes in upper region of the search ROI
        if ey > 0.60 * face_gray.shape[0]:
            continue
        candidates.append((ex, ey, ew, eh))

    # Keep up to 2 best non-overlapping eyes by area
    candidates = sorted(candidates, key=lambda b: b[2]*b[3], reverse=True)
    chosen = []
    for c in candidates:
        if not chosen:
            chosen.append(c)
        else:
            if _iou(c, chosen[0]) < 0.2:
                chosen.append(c)
        if len(chosen) == 2:
            break

    # Stable ordering: left eye then right eye (important for temporal consistency)
    chosen = sorted(chosen, key=lambda b: b[0])
    return chosen


def is_face_centered(face: Tuple[int, int, int, int],
                     frame_shape: Tuple[int, int],
                     tolerance: float = 0.3) -> bool:
    """
    Check if face is reasonably centered in frame.
    
    Args:
        face: Face bounding box (x, y, w, h)
        frame_shape: (height, width) of frame
        tolerance: How far from center is acceptable (0-1)
        
    Returns:
        True if face is centered within tolerance
    """
    x, y, w, h = face
    frame_h, frame_w = frame_shape[:2]
    
    face_center_x = x + w / 2
    face_center_y = y + h / 2
    
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2
    
    # Calculate offset as fraction of frame size
    offset_x = abs(face_center_x - frame_center_x) / frame_w
    offset_y = abs(face_center_y - frame_center_y) / frame_h
    
    return offset_x <= tolerance and offset_y <= tolerance


def validate_face(gray: np.ndarray,
                  face: Tuple[int, int, int, int],
                  check_distance: bool = True,
                  check_centering: bool = True) -> Tuple[bool, dict]:
    """
    Validate detected face for distance and centering.
    
    Args:
        gray: Grayscale image
        face: Face bounding box (x, y, w, h)
        check_distance: Whether to check distance gating
        check_centering: Whether to check face centering
        
    Returns:
        Tuple of (is_valid, details_dict)
    """
    x, y, w, h = face
    details = {
        "face_box": face,
        "issues": []
    }
    
    # Distance check
    if check_distance:
        K = get_calibration_matrix(gray.shape)
        in_range, distance = distance_gate(K, w)
        details["distance"] = distance
        details["distance_valid"] = in_range
        if not in_range:
            if distance < Config.Z_MIN:
                details["issues"].append("too_close")
            else:
                details["issues"].append("too_far")
    
    # Centering check
    if check_centering:
        centered = is_face_centered(face, gray.shape)
        details["centered"] = centered
        if not centered:
            details["issues"].append("not_centered")
    
    is_valid = len(details["issues"]) == 0
    return is_valid, details


def get_largest_face(faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """
    Get the largest face from detected faces.
    
    Args:
        faces: List of face bounding boxes
        
    Returns:
        Largest face bounding box or None if no faces
    """
    if not faces:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def extract_face_roi(gray: np.ndarray, 
                     face: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract face region from image.
    
    Args:
        gray: Grayscale image
        face: Face bounding box (x, y, w, h)
        
    Returns:
        Cropped face region
    """
    x, y, w, h = face
    return gray[y:y+h, x:x+w].copy()
