"""
CyberGaze Configuration
Face Recognition Authentication System
"""
import os

class Config:
    """Application configuration settings."""
    
    # Base paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration")
    TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
    CONTROL_DIR = os.path.join(DATA_DIR, "control")
    LOGS_DIR = os.path.join(DATA_DIR, "logs")
    CASCADES_DIR = os.path.join(BASE_DIR, "cascades")
    
    # Cascade files
    FACE_CASCADE = os.path.join(CASCADES_DIR, "haarcascade_frontalface_default.xml")
    EYE_CASCADE = os.path.join(CASCADES_DIR, "haarcascade_eye.xml")
    
    # Distance gating (meters)
    Z_MIN = 0.35              # Was 0.45 - allow closer
    Z_MAX = 0.80              # Was 0.65 - allow farther
    ASSUMED_FACE_WIDTH = 0.16  # Average face width in meters
    
    # Face detection parameters
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 6
    MIN_FACE_SIZE = (80, 80)
    
    # Recognition threshold (lower = stricter matching)
    # LBPH: lower confidence = better match
    # 50-60 is strict, 70-80 is permissive
    LBPH_THRESHOLD = 55
    ALIGNED_FACE_SIZE = (160, 160)
    
    # Quality gates
    MIN_BRIGHTNESS = 30       # Was 40 - more lenient for darker conditions
    MAX_BRIGHTNESS = 240      # Was 220 - more lenient for bright conditions
    BLUR_THRESHOLD = 80       # Was 100 - more lenient for blur
    
    # Security settings
    MAX_ATTEMPTS = 5
    LOCKOUT_DURATION = 60  # seconds
    
    # Demo mode (set False in production)
    DEMO_MODE = True
    DEMO_KEY_PHRASE = "CyberGaze_Demo_Key_2024"
    
    # PBKDF2 settings (for production)
    PBKDF2_ITERATIONS = 480000
    SALT_LENGTH = 16
    
    # API settings
    HOST = "127.0.0.1"
    PORT = 5000
    DEBUG = True
    
    @classmethod
    def init_directories(cls):
        """Create required directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.CALIBRATION_DIR, cls.TEMPLATES_DIR,
                         cls.CONTROL_DIR, cls.LOGS_DIR, cls.CASCADES_DIR]:
            os.makedirs(dir_path, exist_ok=True)
