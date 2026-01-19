"""
Liveness Detection Module - Improved Version
Challenge-response based liveness verification.
"""
import cv2
import numpy as np
import random
from typing import List, Tuple, Optional
from enum import Enum
from collections import deque


class Challenge(Enum):
    """Liveness challenge types."""
    BLINK = "BLINK"
    LOOK_LEFT = "LOOK_LEFT"
    LOOK_RIGHT = "LOOK_RIGHT"


def generate_challenge() -> Challenge:
    """Generate a random liveness challenge."""
    return random.choice(list(Challenge))


def get_challenge_instructions(challenge: Challenge) -> str:
    """Get human-readable instructions for a challenge."""
    instructions = {
        Challenge.BLINK: "Please BLINK TWICE slowly",
        Challenge.LOOK_LEFT: "Slowly turn your head LEFT then back to center",
        Challenge.LOOK_RIGHT: "Slowly turn your head RIGHT then back to center"
    }
    return instructions.get(challenge, "Complete the liveness check")


# =============================================================================
# EYE ASPECT RATIO (EAR) - More reliable than edge density alone
# =============================================================================

def compute_eye_openness(eye_roi: np.ndarray) -> float:
    """
    Compute eye openness score using multiple features.
    
    Combines:
    1. Edge density (original approach)
    2. Vertical gradient strength (eyelid creates horizontal edges)
    3. Intensity variance (closed eye = more uniform)
    
    Returns:
        Openness score (higher = more open)
    """
    if eye_roi is None or eye_roi.size == 0:
        return 0.0
    
    # Resize for consistency
    eye = cv2.resize(eye_roi, (60, 36))
    
    # Apply light blur to reduce noise
    eye = cv2.GaussianBlur(eye, (3, 3), 0)
    
    # Feature 1: Adaptive edge density
    # Use adaptive thresholding based on local contrast
    median_val = np.median(eye)
    low_thresh = max(10, int(median_val * 0.5))
    high_thresh = max(30, int(median_val * 1.2))
    edges = cv2.Canny(eye, low_thresh, high_thresh)
    edge_density = np.mean(edges) / 255.0
    
    # Feature 2: Horizontal edges (eyelids)
    # Sobel Y detects horizontal edges
    sobel_y = cv2.Sobel(eye, cv2.CV_64F, 0, 1, ksize=3)
    horiz_edge_strength = np.mean(np.abs(sobel_y)) / 255.0
    
    # Feature 3: Intensity variance (open eye has more texture/variance)
    intensity_var = np.std(eye.astype(float)) / 128.0  # Normalize
    
    # Combine features (weights can be tuned)
    openness = (0.4 * edge_density + 
                0.3 * horiz_edge_strength + 
                0.3 * intensity_var)
    
    return float(openness)


def smooth_signal(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Use 'same' mode and handle edges
    smoothed = np.convolve(values, kernel, mode='same')
    return smoothed


def _eye_score(eye: np.ndarray) -> float:
    eye = cv2.resize(eye, (60, 36))
    eye = cv2.GaussianBlur(eye, (3, 3), 0)
    edges = cv2.Canny(eye, 30, 90)
    return float((edges > 0).mean())  # edge density in [0,1]


def detect_blink(eye_frames: List[np.ndarray],
                 drop_ratio: float = 0.35,
                 min_closed_frames: int = 1,
                 max_closed_frames: int = 8,
                 required_blinks: int = 2) -> Tuple[bool, dict]:
    if len(eye_frames) < 10:
        return False, {"error": "Need at least 10 frames", "frames": len(eye_frames)}

    scores = np.array([_eye_score(e) for e in eye_frames], dtype=np.float32)

    # Smooth to reduce single-frame noise
    if len(scores) >= 3:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        scores_s = np.convolve(scores, kernel, mode="same")
    else:
        scores_s = scores

    # Baseline from early frames (assume eyes open at start)
    n0 = max(3, len(scores_s) // 5)
    baseline = float(np.median(scores_s[:n0]))
    if baseline < 1e-4:
        return False, {"error": "Baseline too small (bad eye ROI/lighting)", "baseline": baseline}

    thresh = baseline * (1.0 - drop_ratio)
    below = scores_s < thresh

    # Count blink events as below-runs with reasonable duration
    blinks = 0
    run = 0
    for b in below:
        if b:
            run += 1
        else:
            if min_closed_frames <= run <= max_closed_frames:
                blinks += 1
            run = 0
    # handle if it ends while "below"
    if min_closed_frames <= run <= max_closed_frames:
        blinks += 1

    passed = blinks >= required_blinks
    return passed, {
        "passed": passed,
        "blinks": blinks,
        "required": required_blinks,
        "baseline": baseline,
        "threshold": float(thresh),
        "scores": scores.tolist()
    }


# =============================================================================
# HEAD TURN DETECTION - Fixed direction + added return-to-center check
# =============================================================================

def detect_head_turn(face_positions: List[Tuple[int, int, int, int]],
                     direction: str,
                     min_turn_fraction: float = 0.20,
                     require_return: bool = True,
                     return_tolerance: float = 0.10,
                     is_mirrored: bool = True) -> Tuple[bool, dict]:
    """
    Detect head turn from sequence of face positions.
    
    For mirrored video (typical webcam):
    - User looks LEFT → face moves LEFT in image (x decreases)
    - User looks RIGHT → face moves RIGHT in image (x increases)
    
    Args:
        face_positions: List of face bboxes (x, y, w, h)
        direction: "left" or "right"
        min_turn_fraction: Minimum movement as fraction of face width
        require_return: Whether user must return to center
        return_tolerance: How close to start position counts as "returned"
        is_mirrored: Whether video is mirrored (cv2.flip)
        
    Returns:
        Tuple of (turn_detected, details)
    """
    MIN_FACES = 10
    
    if len(face_positions) < MIN_FACES:
        return False, {
            "error": f"Need at least {MIN_FACES} face positions",
            "faces_received": len(face_positions)
        }
    
    # Filter out None entries
    valid_positions = [p for p in face_positions if p is not None]
    if len(valid_positions) < MIN_FACES:
        return False, {
            "error": "Too many frames without face detection",
            "valid_faces": len(valid_positions)
        }
    
    # Calculate face center x positions and widths
    centers_x = []
    widths = []
    for (x, y, w, h) in valid_positions:
        centers_x.append(x + w / 2)
        widths.append(w)
    
    centers_x = np.array(centers_x)
    avg_width = np.mean(widths)
    
    # Normalize positions relative to start
    relative_x = (centers_x - centers_x[0]) / avg_width
    
    # Smooth the signal
    relative_x_smooth = smooth_signal(relative_x, window=3)
    
    # Find the extreme point
    if direction.lower() == "left":
        # For mirrored: looking left = face moves left = negative x change
        extreme_idx = np.argmin(relative_x_smooth)
        extreme_value = relative_x_smooth[extreme_idx]
        # Should be negative (moved left)
        turn_magnitude = -extreme_value  # Make positive for comparison
        expected_sign = -1
    else:  # right
        # For mirrored: looking right = face moves right = positive x change
        extreme_idx = np.argmax(relative_x_smooth)
        extreme_value = relative_x_smooth[extreme_idx]
        turn_magnitude = extreme_value
        expected_sign = 1
    
    # Check if turn was sufficient
    turn_sufficient = turn_magnitude >= min_turn_fraction
    
    # Check if user returned to center
    returned = True
    if require_return and extreme_idx < len(relative_x_smooth) - 3:
        # Check positions after the extreme
        after_extreme = relative_x_smooth[extreme_idx:]
        end_position = after_extreme[-1]
        returned = abs(end_position) < return_tolerance
    elif require_return:
        # Extreme was at the end, no return detected
        returned = False
    
    success = turn_sufficient and (returned or not require_return)
    
    details = {
        "success": success,
        "direction": direction,
        "turn_magnitude": float(turn_magnitude),
        "min_required": min_turn_fraction,
        "turn_sufficient": turn_sufficient,
        "returned_to_center": returned,
        "extreme_frame": int(extreme_idx),
        "is_mirrored": is_mirrored
    }
    
    return success, details


# =============================================================================
# TEXTURE/PLANARITY CHECK - Detect flat surfaces (photos, screens)
# =============================================================================

def check_face_texture(face_rois: List[np.ndarray],
                       min_variance: float = 15.0) -> Tuple[bool, dict]:
    """
    Check if face has natural 3D texture vs flat print/screen.
    
    Real faces have:
    - Higher local variance (skin texture, pores)
    - Varying gradients
    
    Flat images tend to have:
    - Smoother gradients
    - Lower high-frequency content
    
    Args:
        face_rois: List of face region images
        min_variance: Minimum local variance threshold
        
    Returns:
        Tuple of (is_real_texture, details)
    """
    if len(face_rois) < 3:
        return False, {"error": "Need at least 3 face samples"}
    
    variance_scores = []
    high_freq_scores = []
    
    for face in face_rois:
        if face is None or face.size == 0:
            continue
            
        # Resize for consistency
        face_resized = cv2.resize(face, (100, 100))
        
        # Local variance using a small window
        local_mean = cv2.blur(face_resized.astype(float), (5, 5))
        local_var = cv2.blur((face_resized.astype(float) - local_mean) ** 2, (5, 5))
        avg_local_var = np.mean(local_var)
        variance_scores.append(avg_local_var)
        
        # High-frequency content (Laplacian variance)
        laplacian = cv2.Laplacian(face_resized, cv2.CV_64F)
        high_freq = laplacian.var()
        high_freq_scores.append(high_freq)
    
    if not variance_scores:
        return False, {"error": "No valid face samples"}
    
    avg_variance = np.mean(variance_scores)
    avg_high_freq = np.mean(high_freq_scores)
    
    # Real faces typically have higher variance
    is_real = avg_variance > min_variance and avg_high_freq > 50
    
    return is_real, {
        "is_real_texture": is_real,
        "avg_local_variance": float(avg_variance),
        "avg_high_freq": float(avg_high_freq),
        "min_variance_threshold": min_variance
    }


# =============================================================================
# OPTICAL FLOW PLANARITY CHECK - Detect flat surfaces via motion
# =============================================================================

def check_motion_parallax(frames: List[np.ndarray],
                          face_boxes: List[Tuple[int, int, int, int]],
                          max_inlier_ratio: float = 0.85) -> Tuple[bool, dict]:
    """
    Check for 3D motion parallax to detect flat photos/screens.
    
    A flat surface (photo/screen) moves according to a homography.
    A real 3D face has non-planar motion that homography can't explain.
    
    Args:
        frames: List of grayscale frames
        face_boxes: Corresponding face bounding boxes
        max_inlier_ratio: If homography explains motion too well, it's flat
        
    Returns:
        Tuple of (is_3d_motion, details)
    """
    if len(frames) < 5:
        return True, {"warning": "Not enough frames for parallax check"}
    
    # Sample frames from sequence
    step = max(1, len(frames) // 5)
    sampled_indices = list(range(0, len(frames), step))[:5]
    
    if len(sampled_indices) < 2:
        return True, {"warning": "Not enough samples"}
    
    inlier_ratios = []
    
    for i in range(len(sampled_indices) - 1):
        idx1, idx2 = sampled_indices[i], sampled_indices[i + 1]
        
        frame1, frame2 = frames[idx1], frames[idx2]
        box1, box2 = face_boxes[idx1], face_boxes[idx2]
        
        if box1 is None or box2 is None:
            continue
        
        # Extract face ROIs
        x1, y1, w1, h1 = box1
        roi1 = frame1[y1:y1+h1, x1:x1+w1]
        
        x2, y2, w2, h2 = box2
        roi2 = frame2[y2:y2+h2, x2:x2+w2]
        
        if roi1.size == 0 or roi2.size == 0:
            continue
        
        # Resize to same size for comparison
        size = (100, 100)
        roi1 = cv2.resize(roi1, size)
        roi2 = cv2.resize(roi2, size)
        
        # Detect features
        orb = cv2.ORB_create(nfeatures=100)
        kp1, des1 = orb.detectAndCompute(roi1, None)
        kp2, des2 = orb.detectAndCompute(roi2, None)
        
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            continue
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 8:
            continue
        
        # Get matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Fit homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask)
            inlier_ratios.append(inlier_ratio)
    
    if not inlier_ratios:
        return True, {"warning": "Could not compute motion analysis"}
    
    avg_inlier_ratio = np.mean(inlier_ratios)
    
    # High inlier ratio = motion well-explained by homography = likely flat
    is_3d = avg_inlier_ratio < max_inlier_ratio
    
    return is_3d, {
        "is_3d_motion": is_3d,
        "avg_inlier_ratio": float(avg_inlier_ratio),
        "threshold": max_inlier_ratio,
        "interpretation": "Low ratio suggests 3D face, high ratio suggests flat surface"
    }


# =============================================================================
# COMBINED VERIFICATION
# =============================================================================

def verify_challenge(challenge: Challenge,
                     frames: List[np.ndarray],
                     faces: List[Tuple[int, int, int, int]],
                     eyes_per_frame: List[List[Tuple[int, int, int, int]]],
                     check_texture: bool = True,
                     check_parallax: bool = False) -> Tuple[bool, dict]:
    """
    Verify that user completed the liveness challenge.
    
    Args:
        challenge: The challenge type
        frames: List of grayscale frames
        faces: List of face bboxes per frame
        eyes_per_frame: List of eye bboxes per frame
        check_texture: Also verify face texture
        check_parallax: Also verify motion parallax
        
    Returns:
        Tuple of (challenge_passed, details)
    """
    if len(frames) < 15:
        return False, {"error": "Insufficient frames for verification", 
                      "frames": len(frames)}
    
    results = {"challenge": challenge.value}
    
    # Main challenge verification
    if challenge == Challenge.BLINK:
        # Collect eye ROIs across frames - use BOTH eyes for reliability
        all_eye_rois = []
        
        for frame, face, eyes in zip(frames, faces, eyes_per_frame):
            if face is None or len(eyes) < 1:
                continue
            
            x, y, w, h = face
            
            # Collect all detected eyes in this frame
            frame_eyes = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Up to 2 eyes
                roi = frame[y + ey:y + ey + eh, x + ex:x + ex + ew].copy()
                if roi.size > 0:
                    frame_eyes.append(roi)
            
            if frame_eyes:
                # Average openness across both eyes for this frame
                combined = np.hstack([cv2.resize(e, (30, 20)) for e in frame_eyes])
                all_eye_rois.append(combined)
        
        if len(all_eye_rois) < 15:
            return False, {"error": "Could not track eyes reliably",
                          "eye_frames": len(all_eye_rois)}
        
        challenge_passed, challenge_details = detect_blink(all_eye_rois, min_blinks=2)
        results["blink_check"] = challenge_details
        
    elif challenge in [Challenge.LOOK_LEFT, Challenge.LOOK_RIGHT]:
        direction = "left" if challenge == Challenge.LOOK_LEFT else "right"
        challenge_passed, challenge_details = detect_head_turn(
            faces, direction, 
            min_turn_fraction=0.20,
            require_return=True
        )
        results["head_turn_check"] = challenge_details
    else:
        return False, {"error": f"Unknown challenge: {challenge}"}
    
    results["challenge_passed"] = challenge_passed
    
    # Additional checks
    if check_texture:
        face_rois = []
        for frame, face in zip(frames, faces):
            if face is not None:
                x, y, w, h = face
                roi = frame[y:y+h, x:x+w].copy()
                face_rois.append(roi)
        
        if face_rois:
            texture_ok, texture_details = check_face_texture(face_rois)
            results["texture_check"] = texture_details
            
            if not texture_ok:
                results["texture_warning"] = "Face texture appears artificial"
                # Could fail here, or just warn
                # challenge_passed = False
    
    if check_parallax:
        parallax_ok, parallax_details = check_motion_parallax(frames, faces)
        results["parallax_check"] = parallax_details
        
        if not parallax_ok:
            results["parallax_warning"] = "Motion suggests flat surface"
    
    return challenge_passed, results


def extract_eye_rois(gray, face, eyes):
    x, y, w, h = face
    H, W = gray.shape[:2]
    rois = []

    for (ex, ey, ew, eh) in eyes:
        x1 = max(0, x + ex)
        y1 = max(0, y + ey)
        x2 = min(W, x + ex + ew)
        y2 = min(H, y + ey + eh)

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        roi = gray[y1:y2, x1:x2].copy()
        rois.append(roi)

    return rois