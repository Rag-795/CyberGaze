"""
CyberGaze - Face Recognition Authentication Backend
Flask API for Electron.js integration
"""
import base64
import cv2
import numpy as np
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import Config
from modules.preprocessing import preprocess, quality_check, to_grayscale
from modules.detection import detect_faces, detect_eyes, validate_face, get_largest_face, extract_face_roi
from modules.alignment import extract_aligned_face
from modules.recognition import (
    train_recognizer, save_model, load_model, recognize,
    get_enrolled_users, save_user_metadata, get_user_metadata
)
from modules.liveness import (
    Challenge, generate_challenge, verify_challenge,
    get_challenge_instructions
)
from modules.security import (
    get_encryption_key, create_control_file, verify_control_file,
    update_control_file, hide_folder, show_folder,
    rate_limiter, log_event, get_logs
)
from modules.calibration import calibrate_camera, load_calibration, get_calibration_matrix


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize directories on startup
Config.init_directories()


# ==================== Helper Functions ====================

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to OpenCV image."""
    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def encode_image_base64(img: np.ndarray) -> str:
    """Encode OpenCV image to base64 string."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# ==================== Status Endpoint ====================

@app.route("/api/status", methods=["GET"])
def get_status():
    """Get system status."""
    calib = load_calibration()
    users = get_enrolled_users()
    
    return jsonify({
        "status": "ok",
        "calibrated": calib is not None,
        "enrolled_users": users,
        "demo_mode": Config.DEMO_MODE
    })


# ==================== Calibration Endpoints ====================

@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    """
    Camera calibration endpoint.
    
    Request body:
    {
        "images": ["base64_image1", "base64_image2", ...],
        "checkerboard": [9, 6],  // optional
        "square_size": 0.024    // optional, meters
    }
    """
    data = request.get_json()
    
    if "images" not in data or len(data["images"]) < 5:
        return jsonify({
            "success": False,
            "error": "Need at least 5 calibration images"
        }), 400
    
    # Decode images
    images = []
    for img_b64 in data["images"]:
        try:
            img = decode_base64_image(img_b64)
            images.append(img)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to decode image: {str(e)}"
            }), 400
    
    # Calibrate
    checkerboard = tuple(data.get("checkerboard", [9, 6]))
    square_size = data.get("square_size", 0.024)
    
    result = calibrate_camera(images, checkerboard, square_size)
    
    if result["success"]:
        log_event("calibration", "system", result, success=True)
    
    return jsonify(result)


# ==================== Enrollment Endpoints ====================

@app.route("/api/enroll", methods=["POST"])
def enroll():
    """
    Face enrollment endpoint.
    
    Request body:
    {
        "user_id": "john_doe",
        "images": ["base64_image1", "base64_image2", ...]
    }
    """
    data = request.get_json()
    
    user_id = data.get("user_id")
    images = data.get("images", [])
    
    if not user_id:
        return jsonify({"success": False, "error": "user_id required"}), 400
    
    if len(images) < 5:
        return jsonify({
            "success": False,
            "error": "Need at least 5 face images for enrollment"
        }), 400
    
    # Process images and extract aligned faces
    aligned_faces = []
    skipped = 0
    
    for img_b64 in images:
        try:
            img = decode_base64_image(img_b64)
            gray = to_grayscale(img)
            
            # Preprocess (detect/recognition)
            gray = preprocess(gray, mode="detect")
            
            # Detect faces
            faces = detect_faces(gray)
            if not faces:
                skipped += 1
                continue
            
            # Get largest face
            face = get_largest_face(faces)
            
            # Validate face
            valid, details = validate_face(gray, face)
            if not valid:
                skipped += 1
                continue
            
            # Extract aligned face
            aligned, info = extract_aligned_face(gray, face)
            aligned_faces.append(aligned)
            
        except Exception as e:
            skipped += 1
            continue
    
    if len(aligned_faces) < 3:
        return jsonify({
            "success": False,
            "error": f"Only {len(aligned_faces)} valid faces extracted. Need at least 3.",
            "skipped": skipped
        }), 400
    
    # Train recognizer
    try:
        recognizer = train_recognizer(aligned_faces, label=0)
        model_path = save_model(recognizer, user_id)
        
        # Save metadata
        metadata = {
            "user_id": user_id,
            "enrolled_at": datetime.now().isoformat(),
            "num_samples": len(aligned_faces)
        }
        save_user_metadata(user_id, metadata)
        
        log_event("enrollment", user_id, metadata, success=True)
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "samples_used": len(aligned_faces),
            "skipped": skipped
        })
        
    except Exception as e:
        log_event("enrollment", user_id, {"error": str(e)}, success=False)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== Challenge Endpoints ====================

@app.route("/api/challenge", methods=["GET"])
def get_challenge():
    """Get a random liveness challenge."""
    challenge = generate_challenge()
    
    return jsonify({
        "challenge": challenge.value,
        "instruction": get_challenge_instructions(challenge)
    })


# ==================== Authentication Endpoints ====================

@app.route("/api/authenticate", methods=["POST"])
def authenticate():
    """
    Face authentication with liveness check.
    
    Request body:
    {
        "user_id": "john_doe",
        "frames": ["base64_frame1", "base64_frame2", ...],
        "challenge": "BLINK"
    }
    """
    data = request.get_json()
    
    user_id = data.get("user_id")
    frames_b64 = data.get("frames", [])
    challenge_str = data.get("challenge")
    
    if not user_id:
        return jsonify({"success": False, "error": "user_id required"}), 400
    
    # Check rate limiting
    locked, remaining = rate_limiter.is_locked_out(user_id)
    if locked:
        return jsonify({
            "success": False,
            "error": "Too many failed attempts",
            "locked_out": True,
            "retry_after_seconds": remaining
        }), 429
    
    # Load user model
    recognizer = load_model(user_id)
    if recognizer is None:
        return jsonify({
            "success": False,
            "error": f"User {user_id} not enrolled"
        }), 404
    
    if len(frames_b64) < 5:
        return jsonify({
            "success": False,
            "error": "Need at least 5 frames for authentication"
        }), 400
    
    # Decode frames
    frames = []
    gray_frames = []
    gray_frames_liveness = []
    faces_list = []
    eyes_list = []
    quality_flags = []
    in_range_flags = []
    
    for frame_b64 in frames_b64:
        try:
            img = decode_base64_image(frame_b64)
            gray = to_grayscale(img)

            quality_ok, _ = quality_check(gray)

            gray_processed = preprocess(gray, mode="detect")
            gray_liveness = preprocess(gray, mode="liveness")
            
            frames.append(img)
            gray_frames.append(gray_processed)
            gray_frames_liveness.append(gray_liveness)
            
            # Detect face and eyes
            faces = detect_faces(gray_processed)
            face = get_largest_face(faces)
            faces_list.append(face)
            
            if face:
                eyes = detect_eyes(gray_processed, face)
                eyes_list.append(eyes)
            else:
                eyes_list.append([])

            # Liveness gating metadata (distance gate + quality)
            if face:
                _, face_details = validate_face(gray, face, check_centering=False)
                in_range = bool(face_details.get("distance_valid", False))
            else:
                in_range = False
            quality_flags.append(bool(quality_ok))
            in_range_flags.append(bool(in_range))
                
        except Exception:
            continue
    
    if len(gray_frames) < 5:
        rate_limiter.record_attempt(user_id, success=False)
        log_event("auth_attempt", user_id, {"error": "insufficient_frames"}, success=False)
        return jsonify({
            "success": False,
            "error": "Failed to process frames"
        }), 400
    
    # Find best frame for recognition (middle frame with valid face)
    best_frame_idx = len(gray_frames) // 2
    best_face = faces_list[best_frame_idx]
    
    if best_face is None:
        # Try to find any frame with a face
        for i, face in enumerate(faces_list):
            if face is not None:
                best_frame_idx = i
                best_face = face
                break
    
    if best_face is None:
        rate_limiter.record_attempt(user_id, success=False)
        log_event("auth_attempt", user_id, {"error": "no_face_detected"}, success=False)
        return jsonify({
            "success": False,
            "error": "No face detected in frames"
        }), 400
    
    # Validate face (distance, centering)
    gray = gray_frames[best_frame_idx]
    valid, details = validate_face(gray, best_face)
    
    if not valid:
        rate_limiter.record_attempt(user_id, success=False)
        log_event("auth_attempt", user_id, {"error": "face_validation_failed", "details": details}, success=False)
        return jsonify({
            "success": False,
            "error": "Face validation failed",
            "details": details
        }), 400
    
    # Verify liveness challenge
    if challenge_str:
        try:
            challenge = Challenge(challenge_str)
            live_frames = []
            live_faces = []
            live_eyes = []
            for f, fa, ey, q_ok, in_rng in zip(
                gray_frames_liveness, faces_list, eyes_list, quality_flags, in_range_flags
            ):
                if q_ok and in_rng:
                    live_frames.append(f)
                    live_faces.append(fa)
                    live_eyes.append(ey)

            liveness_passed, liveness_details = verify_challenge(
                challenge, live_frames, live_faces, live_eyes
            )
            
            if not liveness_passed:
                rate_limiter.record_attempt(user_id, success=False)
                log_event("auth_attempt", user_id, {
                    "error": "liveness_failed",
                    "challenge": challenge_str,
                    "details": liveness_details
                }, success=False)
                return jsonify({
                    "success": False,
                    "error": "Liveness check failed",
                    "liveness_details": liveness_details
                }), 401
                
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid challenge: {challenge_str}"
            }), 400
    
    # Extract aligned face for recognition
    aligned, align_info = extract_aligned_face(gray, best_face)
    
    # Recognize face
    is_match, label, confidence = recognize(recognizer, aligned)
    
    if not is_match:
        rate_limiter.record_attempt(user_id, success=False)
        log_event("auth_attempt", user_id, {
            "error": "recognition_failed",
            "confidence": confidence
        }, success=False)
        return jsonify({
            "success": False,
            "error": "Face recognition failed",
            "confidence": float(confidence)
        }), 401
    
    # Success!
    rate_limiter.record_attempt(user_id, success=True)
    log_event("auth_attempt", user_id, {
        "confidence": confidence,
        "distance": details.get("distance"),
        "liveness_challenge": challenge_str
    }, success=True)
    
    return jsonify({
        "success": True,
        "user_id": user_id,
        "confidence": float(confidence),
        "distance": details.get("distance"),
        "liveness_passed": True
    })


# ==================== Folder Protection Endpoints ====================

@app.route("/api/folder/register", methods=["POST"])
def register_folder():
    """
    Register a folder for protection.
    
    Request body:
    {
        "user_id": "john_doe",
        "folder_path": "C:/Users/john/SecretFolder",
        "password": "optional_password"  // only in production mode
    }
    """
    data = request.get_json()
    
    user_id = data.get("user_id")
    folder_path = data.get("folder_path")
    password = data.get("password")
    
    if not user_id or not folder_path:
        return jsonify({
            "success": False,
            "error": "user_id and folder_path required"
        }), 400
    
    if not os.path.exists(folder_path):
        return jsonify({
            "success": False,
            "error": "Folder does not exist"
        }), 404
    
    if not os.path.isdir(folder_path):
        return jsonify({
            "success": False,
            "error": "Path is not a directory"
        }), 400
    
    try:
        key, salt = get_encryption_key(password)
        control_path = create_control_file(folder_path, user_id, key)
        
        log_event("folder_register", user_id, {
            "folder": folder_path
        }, success=True)
        
        result = {
            "success": True,
            "folder_path": folder_path,
            "control_file": control_path
        }
        
        # Include salt if generated (production mode)
        if salt:
            result["salt"] = base64.b64encode(salt).decode()
        
        return jsonify(result)
        
    except Exception as e:
        log_event("folder_register", user_id, {
            "folder": folder_path,
            "error": str(e)
        }, success=False)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/folder/lock", methods=["POST"])
def lock_folder():
    """
    Lock a protected folder.
    
    Request body:
    {
        "user_id": "john_doe",
        "folder_path": "C:/Users/john/SecretFolder",
        "password": "optional_password"
    }
    """
    data = request.get_json()
    
    user_id = data.get("user_id")
    folder_path = data.get("folder_path")
    password = data.get("password")
    salt_b64 = data.get("salt")
    
    if not user_id or not folder_path:
        return jsonify({
            "success": False,
            "error": "user_id and folder_path required"
        }), 400
    
    try:
        salt = base64.b64decode(salt_b64) if salt_b64 else None
        key, _ = get_encryption_key(password, salt)
        
        # Verify control file
        valid, control_data = verify_control_file(folder_path, key)
        if not valid:
            return jsonify({
                "success": False,
                "error": "Invalid credentials or folder not registered"
            }), 401
        
        # Check ownership
        if control_data.get("user_id") != user_id:
            return jsonify({
                "success": False,
                "error": "Not authorized for this folder"
            }), 403
        
        # Hide folder
        hidden = hide_folder(folder_path)
        if hidden:
            update_control_file(folder_path, key, locked=True)
        
        log_event("folder_lock", user_id, {
            "folder": folder_path,
            "hidden": hidden
        }, success=True)
        
        return jsonify({
            "success": True,
            "folder_path": folder_path,
            "locked": True,
            "hidden": hidden
        })
        
    except Exception as e:
        log_event("folder_lock", user_id, {
            "folder": folder_path,
            "error": str(e)
        }, success=False)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/folder/unlock", methods=["POST"])
def unlock_folder():
    """
    Unlock a protected folder.
    
    Request body:
    {
        "user_id": "john_doe",
        "folder_path": "C:/Users/john/SecretFolder",
        "password": "optional_password"
    }
    """
    data = request.get_json()
    
    user_id = data.get("user_id")
    folder_path = data.get("folder_path")
    password = data.get("password")
    salt_b64 = data.get("salt")
    
    if not user_id or not folder_path:
        return jsonify({
            "success": False,
            "error": "user_id and folder_path required"
        }), 400
    
    try:
        salt = base64.b64decode(salt_b64) if salt_b64 else None
        key, _ = get_encryption_key(password, salt)
        
        # Verify control file
        valid, control_data = verify_control_file(folder_path, key)
        if not valid:
            return jsonify({
                "success": False,
                "error": "Invalid credentials or folder not registered"
            }), 401
        
        # Check ownership
        if control_data.get("user_id") != user_id:
            return jsonify({
                "success": False,
                "error": "Not authorized for this folder"
            }), 403
        
        # Show folder
        shown = show_folder(folder_path)
        if shown:
            update_control_file(folder_path, key, locked=False)
        
        log_event("folder_unlock", user_id, {
            "folder": folder_path,
            "shown": shown
        }, success=True)
        
        return jsonify({
            "success": True,
            "folder_path": folder_path,
            "locked": False,
            "visible": shown
        })
        
    except Exception as e:
        log_event("folder_unlock", user_id, {
            "folder": folder_path,
            "error": str(e)
        }, success=False)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== Logs Endpoint ====================

@app.route("/api/logs", methods=["GET"])
def get_audit_logs():
    """Get audit logs for a user."""
    user_id = request.args.get("user_id")
    limit = request.args.get("limit", 50, type=int)
    
    if not user_id:
        return jsonify({
            "success": False,
            "error": "user_id required"
        }), 400
    
    logs = get_logs(user_id, limit)
    
    return jsonify({
        "success": True,
        "user_id": user_id,
        "logs": logs
    })


# ==================== Main ====================

if __name__ == "__main__":
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                     CyberGaze Backend                     ║
║           Face Recognition Authentication System          ║
╠═══════════════════════════════════════════════════════════╣
║  Status: Running                                          ║
║  URL: http://{Config.HOST}:{Config.PORT}                          ║
║  Mode: {'DEMO' if Config.DEMO_MODE else 'PRODUCTION'}                                          ║
╚═══════════════════════════════════════════════════════════╝
    """)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
