"""
CyberGaze - Anti-Spoofing Module
Texture analysis for detecting printed photos, screen replays, and other spoofs.
Uses LBP (Local Binary Patterns) histogram analysis — no CNN model required.
"""

import cv2
import numpy as np


class SpoofDetector:
    """
    Detects spoofing attacks (printed photos, screen replays) using
    texture analysis with Local Binary Patterns (LBP).
    
    Real faces have more natural micro-texture patterns compared to
    printed/displayed images which show dot patterns, moiré effects, etc.
    """

    # Thresholds tuned for spoof detection
    SPOOF_THRESHOLD = 0.4       # Below this = likely spoofed
    HIGH_FREQ_THRESHOLD = 0.15  # Minimum high-frequency energy for real face

    def __init__(self):
        print("Anti-spoof detector initialized (LBP texture analysis)")

    def local_binary_pattern(self, image, radius=1, n_points=8):
        """
        Compute Local Binary Pattern for texture analysis.
        
        LBP compares each pixel with its neighbors to create a texture descriptor.
        Real faces have smoother, more varied LBP patterns than printed photos.
        
        Args:
            image: Grayscale image
            radius: Radius of circular neighborhood
            n_points: Number of neighbor points to sample
            
        Returns:
            LBP image (same dimensions as input)
        """
        rows, cols = image.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = 0

                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j - radius * np.sin(angle)))

                    if 0 <= x < rows and 0 <= y < cols:
                        if image[x, y] >= center:
                            binary_string |= (1 << k)

                lbp[i, j] = binary_string

        return lbp

    def analyze_texture(self, frame) -> dict:
        """
        Analyze frame texture for spoof indicators.
        
        Combines multiple texture features:
        1. LBP histogram variance (real faces more varied)
        2. High-frequency energy (real faces have more detail)
        3. Color space analysis (printed images differ in LAB space)
        
        Args:
            frame: BGR numpy array
            
        Returns:
            dict with spoof_score (0-1, higher = more likely real),
            is_real flag, and feature details
        """
        if frame is None or frame.size == 0:
            return {'is_real': False, 'spoof_score': 0.0, 'error': 'Empty frame'}

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize for consistent processing
        gray_resized = cv2.resize(gray, (128, 128))

        # Feature 1: LBP histogram analysis
        lbp_score = self._lbp_analysis(gray_resized)

        # Feature 2: High-frequency energy (Laplacian variance)
        hf_score = self._high_frequency_analysis(gray_resized)

        # Feature 3: Color space analysis
        color_score = self._color_analysis(frame)

        # Combined score (weighted average)
        spoof_score = (
            0.40 * lbp_score +
            0.35 * hf_score +
            0.25 * color_score
        )

        is_real = spoof_score >= self.SPOOF_THRESHOLD

        return {
            'is_real': is_real,
            'spoof_score': round(float(spoof_score), 4),
            'details': {
                'lbp_score': round(float(lbp_score), 4),
                'high_freq_score': round(float(hf_score), 4),
                'color_score': round(float(color_score), 4)
            }
        }

    def _lbp_analysis(self, gray_image) -> float:
        """
        LBP histogram analysis.
        Real faces produce more uniform LBP histograms with higher variance.
        Printed/screen images tend to have clustered histogram patterns.
        """
        lbp = self.local_binary_pattern(gray_image)

        # Compute normalized histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)

        # Normalize
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum

        # Compute histogram variance (real faces → higher variance)
        variance = np.var(hist)

        # Compute entropy (real faces → higher entropy)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

        # Normalize to [0, 1]
        # Typical entropy range for LBP: 3-8 bits
        normalized_entropy = min(entropy / 8.0, 1.0)

        return normalized_entropy

    def _high_frequency_analysis(self, gray_image) -> float:
        """
        Analyze high-frequency content using Laplacian variance.
        Real faces have natural skin texture with higher frequency detail.
        Printed photos/screens tend to have less high-frequency content.
        """
        # Laplacian computes second derivative (highlights edges/texture)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize (typical variance range: 10-1000)
        normalized = min(variance / 500.0, 1.0)

        return normalized

    def _color_analysis(self, frame) -> float:
        """
        Analyze color distribution in LAB color space.
        Printed/displayed images have different chrominance distributions
        compared to real faces under natural illumination.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Real faces have moderate variance in a,b channels
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)

        # Chrominance variance score
        chroma_var = (a_std + b_std) / 2.0

        # Normalize (typical range: 5-30)
        normalized = min(chroma_var / 25.0, 1.0)

        return normalized

    def is_real_face(self, frame) -> bool:
        """
        Simple boolean check: is this a real face?
        
        Args:
            frame: BGR numpy array
            
        Returns:
            True if the frame appears to contain a real face
        """
        result = self.analyze_texture(frame)
        return result['is_real']


# Singleton instance
_spoof_detector = None


def get_spoof_detector() -> SpoofDetector:
    """Get singleton SpoofDetector instance."""
    global _spoof_detector
    if _spoof_detector is None:
        _spoof_detector = SpoofDetector()
    return _spoof_detector
