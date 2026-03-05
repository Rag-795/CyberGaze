"""
CyberGaze - Liveness Detection Module
Multi-frame liveness verification using challenge-response system.
Uses dlib facial landmarks for blink detection and head pose estimation.
"""

import cv2
import numpy as np
import random
import uuid
import time
import os
import base64
import io
import urllib.request
import bz2
from PIL import Image


# Path for the dlib shape predictor model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PREDICTOR_PATH = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
PREDICTOR_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def download_landmark_model():
    """Auto-download the dlib 68-point facial landmark model if not present."""
    if os.path.exists(PREDICTOR_PATH):
        return True

    os.makedirs(MODEL_DIR, exist_ok=True)
    compressed_path = PREDICTOR_PATH + '.bz2'

    print(f"Downloading facial landmark model from {PREDICTOR_URL}...")
    print("This is a one-time download (~100 MB)...")

    try:
        urllib.request.urlretrieve(PREDICTOR_URL, compressed_path)
        print("Download complete. Extracting...")

        # Decompress .bz2 file
        with bz2.open(compressed_path, 'rb') as f_in:
            with open(PREDICTOR_PATH, 'wb') as f_out:
                f_out.write(f_in.read())

        # Remove compressed file
        os.remove(compressed_path)
        print("Landmark model ready!")
        return True

    except Exception as e:
        print(f"Failed to download landmark model: {e}")
        print(f"Please manually download from {PREDICTOR_URL}")
        print(f"Extract and place at: {PREDICTOR_PATH}")
        return False


# Try to import dlib
try:
    import dlib
    DLIB_AVAILABLE = True
    print("dlib loaded successfully!")
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Liveness detection will be limited.")


class LivenessDetector:
    """
    Multi-frame liveness detection using challenge-response.
    Detects blinks (EAR algorithm) and head pose (solvePnP).
    """

    # Eye indices in 68-point landmark model
    LEFT_EYE_INDICES = list(range(36, 42))   # Points 36-41
    RIGHT_EYE_INDICES = list(range(42, 48))  # Points 42-47

    # 3D model points for head pose estimation (standard face model)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (point 30)
        (0.0, -330.0, -65.0),     # Chin (point 8)
        (-225.0, 170.0, -135.0),  # Left eye left corner (point 36)
        (225.0, 170.0, -135.0),   # Right eye right corner (point 45)
        (-150.0, -150.0, -125.0), # Left mouth corner (point 48)
        (150.0, -150.0, -125.0)   # Right mouth corner (point 54)
    ], dtype=np.float64)

    # Available liveness challenges
    CHALLENGES = ['blink_twice', 'turn_left', 'turn_right']

    # Challenge display instructions
    CHALLENGE_INSTRUCTIONS = {
        'blink_twice': 'Please blink your eyes twice',
        'turn_left': 'Please slowly turn your head to the left',
        'turn_right': 'Please slowly turn your head to the right',
    }

    def __init__(self):
        self.detector = None
        self.predictor = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Active challenges with expiry
        self._active_challenges = {}

        self._initialize()

    def _initialize(self):
        """Initialize dlib detector and predictor."""
        if not DLIB_AVAILABLE:
            print("Liveness detection: dlib not available, using fallback mode")
            return

        # Download model if needed
        if not download_landmark_model():
            print("Liveness detection: landmark model not available")
            return

        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
            print("Liveness detector initialized successfully!")
        except Exception as e:
            print(f"Error initializing liveness detector: {e}")

    def _setup_camera_matrix(self, frame_width, frame_height):
        """Setup camera matrix based on frame dimensions."""
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def _shape_to_np(self, shape, dtype=np.float64):
        """Convert dlib shape to numpy array of (x, y) coordinates."""
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def eye_aspect_ratio(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        When the eye is open, EAR is roughly constant (~0.3).
        When the eye closes, EAR drops toward 0.
        """
        # Vertical eye distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal eye distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_blink_in_frame(self, frame, landmarks):
        """
        Detect if a blink is occurring in a single frame.
        
        Returns:
            True if eyes are closed (blink detected)
        """
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        # Threshold: below 0.25 indicates closed eyes
        return avg_ear < 0.25

    def detect_head_pose(self, landmarks, frame):
        """
        Estimate head pose using solvePnP.
        
        Returns:
            (pitch, yaw, roll) angles in degrees, or None if failed
        """
        if self.camera_matrix is None:
            h, w = frame.shape[:2]
            self._setup_camera_matrix(w, h)

        # 2D image points from landmarks
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left mouth corner
            landmarks[54]   # Right mouth corner
        ], dtype=np.float64)

        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.MODEL_POINTS, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            # Convert rotation vector to rotation matrix, then to Euler angles
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            return angles  # (pitch, yaw, roll) in degrees

        except Exception as e:
            print(f"Head pose estimation error: {e}")
            return None

    def _get_landmarks_from_frame(self, frame):
        """Extract 68-point facial landmarks from a frame."""
        if not self.detector or not self.predictor:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        if len(faces) == 0:
            return None

        # Use first detected face
        shape = self.predictor(gray, faces[0])
        return self._shape_to_np(shape)

    def _decode_frame(self, base64_frame):
        """Decode a base64-encoded frame to numpy array (BGR)."""
        if ',' in base64_frame:
            base64_frame = base64_frame.split(',')[1]

        image_bytes = base64.b64decode(base64_frame)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image.convert('RGB'))
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    def generate_challenge(self) -> dict:
        """
        Generate a random liveness challenge.
        
        Returns:
            dict with challenge_id, challenge type, and instruction text
        """
        challenge_type = random.choice(self.CHALLENGES)
        challenge_id = str(uuid.uuid4())

        # Store with 60-second expiry
        self._active_challenges[challenge_id] = {
            'type': challenge_type,
            'created_at': time.time(),
            'expires_at': time.time() + 60
        }

        # Clean up expired challenges
        self._cleanup_challenges()

        return {
            'challenge_id': challenge_id,
            'challenge': challenge_type,
            'instruction': self.CHALLENGE_INSTRUCTIONS.get(challenge_type, challenge_type),
            'duration_seconds': 3
        }

    def _cleanup_challenges(self):
        """Remove expired challenges."""
        now = time.time()
        expired = [cid for cid, c in self._active_challenges.items() if c['expires_at'] < now]
        for cid in expired:
            del self._active_challenges[cid]

    def verify_challenge(self, challenge_id: str, base64_frames: list) -> dict:
        """
        Verify that the user completed the liveness challenge.
        
        Args:
            challenge_id: The challenge ID from generate_challenge()
            base64_frames: List of base64-encoded frame images
            
        Returns:
            dict with verified status and details
        """
        # Check challenge exists and hasn't expired
        if challenge_id not in self._active_challenges:
            return {
                'verified': False,
                'error': 'Challenge expired or invalid. Please request a new challenge.'
            }

        challenge = self._active_challenges[challenge_id]

        if time.time() > challenge['expires_at']:
            del self._active_challenges[challenge_id]
            return {
                'verified': False,
                'error': 'Challenge expired. Please request a new challenge.'
            }

        challenge_type = challenge['type']

        # If dlib not available, pass liveness check with warning
        if not self.detector or not self.predictor:
            del self._active_challenges[challenge_id]
            return {
                'verified': True,
                'warning': 'Liveness detection limited - dlib not available',
                'challenge': challenge_type
            }

        # Decode all frames
        frames = []
        for b64_frame in base64_frames:
            try:
                frame = self._decode_frame(b64_frame)
                frames.append(frame)
            except Exception:
                continue

        if len(frames) < 5:
            return {
                'verified': False,
                'error': 'Not enough valid frames captured'
            }

        # Verify based on challenge type
        if challenge_type == 'blink_twice':
            result = self._verify_blink(frames, required_blinks=2)
        elif challenge_type == 'turn_left':
            result = self._verify_head_turn(frames, direction='left')
        elif challenge_type == 'turn_right':
            result = self._verify_head_turn(frames, direction='right')
        else:
            result = {'verified': False, 'error': 'Unknown challenge type'}

        # Remove used challenge
        del self._active_challenges[challenge_id]

        result['challenge'] = challenge_type
        return result

    def _verify_blink(self, frames, required_blinks=2) -> dict:
        """Verify blink challenge: detect N blinks in frame sequence."""
        blink_count = 0
        was_closed = False

        for frame in frames:
            landmarks = self._get_landmarks_from_frame(frame)
            if landmarks is None:
                continue

            is_closed = self.detect_blink_in_frame(frame, landmarks)

            # Detect blink: transition from closed → open
            if was_closed and not is_closed:
                blink_count += 1

            was_closed = is_closed

        return {
            'verified': blink_count >= required_blinks,
            'blinks_detected': blink_count,
            'blinks_required': required_blinks,
            'message': f'Detected {blink_count} blinks (need {required_blinks})'
        }

    def _verify_head_turn(self, frames, direction='left') -> dict:
        """Verify head turn challenge: detect significant yaw angle."""
        yaw_threshold = 15  # degrees

        for frame in frames:
            landmarks = self._get_landmarks_from_frame(frame)
            if landmarks is None:
                continue

            angles = self.detect_head_pose(landmarks, frame)
            if angles is None:
                continue

            yaw = angles[1]  # Y-axis rotation

            if direction == 'left' and yaw < -yaw_threshold:
                return {
                    'verified': True,
                    'max_yaw': float(yaw),
                    'message': f'Head turn {direction} detected (yaw: {yaw:.1f}°)'
                }
            elif direction == 'right' and yaw > yaw_threshold:
                return {
                    'verified': True,
                    'max_yaw': float(yaw),
                    'message': f'Head turn {direction} detected (yaw: {yaw:.1f}°)'
                }

        return {
            'verified': False,
            'message': f'Head turn {direction} not detected. Please turn more distinctly.'
        }

    @property
    def is_available(self):
        """Check if full liveness detection is available."""
        return self.detector is not None and self.predictor is not None


# Singleton instance
_liveness_detector = None


def get_liveness_detector() -> LivenessDetector:
    """Get singleton LivenessDetector instance."""
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector()
    return _liveness_detector
