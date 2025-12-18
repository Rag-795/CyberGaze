"""
CyberGaze - Face Recognition Module
Modular face recognition using FaceNet (PyTorch) for embedding extraction
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
import os

# Check for facenet-pytorch availability
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    FACENET_AVAILABLE = True
    print("FaceNet (PyTorch) loaded successfully!")
except ImportError:
    FACENET_AVAILABLE = False
    print("Warning: facenet-pytorch not available. Using OpenCV fallback.")


class FaceRecognition:
    """
    Face Recognition class using FaceNet (PyTorch) for embedding extraction.
    Designed to be modular and easily replaceable with custom models.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize face recognition module.
        
        Args:
            threshold: Cosine similarity threshold for face matching (0-1)
        """
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if FACENET_AVAILABLE else None
        self.detector = None
        self.embedder = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize FaceNet and MTCNN models"""
        if FACENET_AVAILABLE:
            try:
                print(f"Using device: {self.device}")
                print("Loading MTCNN face detector...")
                self.detector = MTCNN(
                    image_size=160,
                    margin=20,
                    keep_all=False,
                    device=self.device
                )
                print("Loading InceptionResnetV1 (FaceNet) model...")
                self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                print("Face recognition models loaded successfully!")
            except Exception as e:
                print(f"Error loading models: {e}")
                self._initialize_fallback()
        else:
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback OpenCV-based face detection"""
        print("Using OpenCV fallback for face detection...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade_detector = cv2.CascadeClassifier(cascade_path)
        self.embedder = None
        self.detector = None
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Decode base64 image string to numpy array.
        
        Args:
            base64_string: Base64 encoded image (may include data URI prefix)
            
        Returns:
            numpy array (BGR format)
        """
        # Remove data URI prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB numpy array
        image_array = np.array(image.convert('RGB'))
        
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        if self.detector and FACENET_AVAILABLE:
            # Use MTCNN for face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            boxes, probs = self.detector.detect(pil_image)
            
            if boxes is not None:
                result = []
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    result.append([x1, y1, x2 - x1, y2 - y1])
                return result
            return []
        else:
            # Fallback to OpenCV Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.cascade_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            return faces.tolist() if len(faces) > 0 else []
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from image.
        
        Args:
            image: BGR numpy array containing a face
            
        Returns:
            512-dimensional embedding vector or None if no face found
        """
        if not FACENET_AVAILABLE or not self.embedder:
            # Fallback: use resized face as simple embedding
            return self._fallback_embedding(image)
        
        try:
            # Convert BGR to RGB PIL Image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Detect and align face using MTCNN
            face_tensor = self.detector(pil_image)
            
            if face_tensor is not None:
                # Add batch dimension and move to device
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                # Get embedding
                with torch.no_grad():
                    embedding = self.embedder(face_tensor)
                
                return embedding.cpu().numpy().flatten()
            
            return None
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return self._fallback_embedding(image)
    
    def _fallback_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback embedding using simple image features.
        Not as accurate as FaceNet but works without dependencies.
        """
        if hasattr(self, 'cascade_detector'):
            faces = self.detect_faces(image)
        else:
            # Use basic face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            faces = faces.tolist() if len(faces) > 0 else []
        
        if not faces:
            return None
        
        x, y, w, h = faces[0]
        face_crop = image[y:y+h, x:x+w]
        
        # Resize to standard size and flatten
        face_resized = cv2.resize(face_crop, (64, 64))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Use simple histogram as embedding
        hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
        embedding = hist.flatten()
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1), higher is more similar
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Ensure similarity is between 0 and 1
        similarity = (similarity + 1) / 2  # Map from [-1, 1] to [0, 1]
        
        return float(similarity)
    
    def verify(self, embedding1: np.ndarray, embedding2: np.ndarray) -> dict:
        """
        Verify if two embeddings belong to the same person.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Dictionary with match result and similarity score
        """
        similarity = self.compare_embeddings(embedding1, embedding2)
        is_match = similarity >= self.threshold
        
        return {
            'match': is_match,
            'similarity': round(similarity, 4),
            'threshold': self.threshold
        }
    
    def extract_embedding_from_base64(self, base64_image: str) -> np.ndarray:
        """
        Extract face embedding from base64 encoded image.
        
        Args:
            base64_image: Base64 encoded image string
            
        Returns:
            Embedding vector or None if no face found
        """
        try:
            image = self.decode_base64_image(base64_image)
            return self.extract_embedding(image)
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None


# Singleton instance for the application
_face_recognition_instance = None


def get_face_recognition() -> FaceRecognition:
    """Get singleton FaceRecognition instance"""
    global _face_recognition_instance
    if _face_recognition_instance is None:
        _face_recognition_instance = FaceRecognition(threshold=0.7)
    return _face_recognition_instance
