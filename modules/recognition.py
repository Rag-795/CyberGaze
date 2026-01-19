"""
Face Recognition Module
LBPH-based face recognition with training and prediction.
"""
import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Optional
from config import Config


def create_recognizer() -> cv2.face.LBPHFaceRecognizer:
    """
    Create a new LBPH face recognizer.
    
    Returns:
        LBPH face recognizer instance
    """
    return cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )


def train_recognizer(faces: List[np.ndarray], 
                     label: int) -> cv2.face.LBPHFaceRecognizer:
    """
    Train a new recognizer with face samples.
    
    Args:
        faces: List of aligned face images
        label: User label (integer ID)
        
    Returns:
        Trained recognizer
    """
    recognizer = create_recognizer()
    labels = np.array([label] * len(faces))
    recognizer.train(faces, labels)
    return recognizer


def update_recognizer(recognizer: cv2.face.LBPHFaceRecognizer,
                      faces: List[np.ndarray],
                      label: int) -> cv2.face.LBPHFaceRecognizer:
    """
    Update existing recognizer with new samples.
    
    Args:
        recognizer: Existing trained recognizer
        faces: New face images
        label: User label
        
    Returns:
        Updated recognizer
    """
    labels = np.array([label] * len(faces))
    recognizer.update(faces, labels)
    return recognizer


def save_model(recognizer: cv2.face.LBPHFaceRecognizer, 
               user_id: str) -> str:
    """
    Save trained model to disk.
    
    Args:
        recognizer: Trained recognizer
        user_id: User identifier
        
    Returns:
        Path to saved model
    """
    model_path = os.path.join(Config.TEMPLATES_DIR, f"{user_id}.yml")
    recognizer.write(model_path)
    return model_path


def load_model(user_id: str) -> Optional[cv2.face.LBPHFaceRecognizer]:
    """
    Load user's trained model.
    
    Args:
        user_id: User identifier
        
    Returns:
        Loaded recognizer or None if not found
    """
    model_path = os.path.join(Config.TEMPLATES_DIR, f"{user_id}.yml")
    if not os.path.exists(model_path):
        return None
    
    recognizer = create_recognizer()
    recognizer.read(model_path)
    return recognizer


def recognize(recognizer: cv2.face.LBPHFaceRecognizer,
              face: np.ndarray,
              threshold: float = None) -> Tuple[bool, int, float]:
    """
    Recognize a face against trained model.
    
    Args:
        recognizer: Trained recognizer
        face: Aligned face image
        threshold: Maximum confidence for match (lower = stricter)
        
    Returns:
        Tuple of (is_match, predicted_label, confidence)
        Note: In LBPH, lower confidence = better match
    """
    if threshold is None:
        threshold = Config.LBPH_THRESHOLD
    
    label, confidence = recognizer.predict(face)
    is_match = confidence < threshold
    
    return is_match, label, confidence


def get_enrolled_users() -> List[str]:
    """
    Get list of enrolled user IDs.
    
    Returns:
        List of user IDs with saved models
    """
    users = []
    if os.path.exists(Config.TEMPLATES_DIR):
        for filename in os.listdir(Config.TEMPLATES_DIR):
            if filename.endswith(".yml"):
                users.append(filename[:-4])
    return users


def delete_user_model(user_id: str) -> bool:
    """
    Delete a user's recognition model.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if deleted, False if not found
    """
    model_path = os.path.join(Config.TEMPLATES_DIR, f"{user_id}.yml")
    if os.path.exists(model_path):
        os.remove(model_path)
        return True
    return False


def get_user_metadata(user_id: str) -> Optional[dict]:
    """
    Get user enrollment metadata.
    
    Args:
        user_id: User identifier
        
    Returns:
        Metadata dict or None
    """
    meta_path = os.path.join(Config.TEMPLATES_DIR, f"{user_id}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def save_user_metadata(user_id: str, metadata: dict) -> None:
    """
    Save user enrollment metadata.
    
    Args:
        user_id: User identifier
        metadata: Metadata dictionary
    """
    meta_path = os.path.join(Config.TEMPLATES_DIR, f"{user_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
