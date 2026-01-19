"""
Camera Calibration Module
Handles camera calibration and distance estimation using pinhole model.
"""
import cv2
import numpy as np
import os
from typing import Tuple, Optional, List
from config import Config


def calibrate_camera(images: List[np.ndarray], 
                     checkerboard: Tuple[int, int] = (9, 6),
                     square_size: float = 0.024) -> dict:
    """
    Calibrate camera using chessboard images.
    
    Args:
        images: List of calibration images (BGR format)
        checkerboard: Number of internal corners (cols, rows)
        square_size: Size of each square in meters
        
    Returns:
        Dictionary with calibration results
    """
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    img_shape = None
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        img_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
    
    if len(objpoints) < 5:
        return {
            "success": False,
            "error": f"Only {len(objpoints)} valid images found. Need at least 5.",
            "valid_images": len(objpoints)
        }
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    
    # Save calibration
    calib_path = os.path.join(Config.CALIBRATION_DIR, "camera_calib.npz")
    np.savez(calib_path, K=K, dist=dist, reprojection_error=mean_error)
    
    return {
        "success": True,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "reprojection_error": float(mean_error),
        "valid_images": len(objpoints)
    }


def load_calibration() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load saved calibration data.
    
    Returns:
        Tuple of (K matrix, distortion coefficients) or None if not found
    """
    calib_path = os.path.join(Config.CALIBRATION_DIR, "camera_calib.npz")
    if os.path.exists(calib_path):
        data = np.load(calib_path)
        return data["K"], data["dist"]
    return None


def get_default_calibration(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Get approximate calibration for common webcam resolutions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Approximate K matrix
    """
    # Approximate focal length based on typical webcam FOV (~60 degrees)
    # fx ≈ width / (2 * tan(fov/2))
    fov_horizontal = 60  # degrees, typical for webcams
    fx = width / (2 * np.tan(np.radians(fov_horizontal / 2)))
    fy = fx  # Assume square pixels
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def estimate_distance(K: np.ndarray, face_width_px: float, 
                      face_width_m: float = None) -> float:
    """
    Estimate distance from camera to face using pinhole model.
    
    Z = (fx * W) / w_px
    
    Args:
        K: Camera intrinsic matrix
        face_width_px: Detected face width in pixels
        face_width_m: Real face width in meters (default from config)
        
    Returns:
        Estimated distance in meters
    """
    if face_width_m is None:
        face_width_m = Config.ASSUMED_FACE_WIDTH
    
    fx = K[0, 0]
    return (fx * face_width_m) / max(face_width_px, 1)


def distance_gate(K: np.ndarray, face_width_px: float,
                  z_min: float = None, z_max: float = None) -> Tuple[bool, float]:
    """
    Check if face is within acceptable distance band.
    
    Args:
        K: Camera intrinsic matrix
        face_width_px: Detected face width in pixels
        z_min: Minimum acceptable distance (default from config)
        z_max: Maximum acceptable distance (default from config)
        
    Returns:
        Tuple of (is_valid, estimated_distance)
    """
    if z_min is None:
        z_min = Config.Z_MIN
    if z_max is None:
        z_max = Config.Z_MAX
        
    z = estimate_distance(K, face_width_px)
    is_valid = z_min <= z <= z_max
    
    return is_valid, z


def get_calibration_matrix(frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Get calibration matrix - uses saved calibration if available, else default.
    
    Args:
        frame_shape: (height, width) of frame
        
    Returns:
        K matrix
    """
    calib = load_calibration()
    if calib is not None:
        return calib[0]
    return get_default_calibration(frame_shape[1], frame_shape[0])
