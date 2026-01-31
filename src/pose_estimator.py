"""
Pose estimation module using MediaPipe for body keypoint detection.
Provides functionality to detect body landmarks for virtual try-on.
"""
import logging
from typing import Optional, List, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe not available. Pose estimation will use fallback mode.")
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None


class PoseEstimator:
    """
    Detects body keypoints using MediaPipe Pose.
    """
    
    # Standard pose landmarks indices
    LANDMARK_INDICES = {
        'nose': 0,
        'left_eye': 2,
        'right_eye': 5,
        'left_ear': 7,
        'right_ear': 8,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize PoseEstimator.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        if MEDIAPIPE_AVAILABLE and mp_pose is not None:
            self.mp_pose = mp_pose
            self.pose = mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.mp_pose = None
            self.pose = None
    
    def detect_keypoints(self, image: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Detect body keypoints in an image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Dictionary mapping landmark names to (x, y) coordinates,
            or None if detection fails
        """
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available. Using fallback keypoints.")
            return self._get_fallback_keypoints(image)
        
        try:
            # Process the image
            results = self.pose.process(image)
            
            if not results.pose_landmarks:
                logger.warning("No pose landmarks detected, using fallback keypoints")
                return self._get_fallback_keypoints(image)
            
            # Extract keypoints
            height, width = image.shape[:2]
            keypoints = {}
            
            for name, idx in self.LANDMARK_INDICES.items():
                landmark = results.pose_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                keypoints[name] = (x, y)
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Error detecting keypoints: {e}, using fallback")
            return self._get_fallback_keypoints(image)
    
    def _get_fallback_keypoints(self, image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Generate fallback keypoints based on image dimensions.
        Used when MediaPipe is not available.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of estimated keypoints
        """
        height, width = image.shape[:2]
        
        # Simple heuristic-based keypoint estimation
        keypoints = {
            'nose': (width // 2, int(height * 0.15)),
            'left_eye': (int(width * 0.45), int(height * 0.13)),
            'right_eye': (int(width * 0.55), int(height * 0.13)),
            'left_ear': (int(width * 0.42), int(height * 0.15)),
            'right_ear': (int(width * 0.58), int(height * 0.15)),
            'left_shoulder': (int(width * 0.35), int(height * 0.25)),
            'right_shoulder': (int(width * 0.65), int(height * 0.25)),
            'left_elbow': (int(width * 0.30), int(height * 0.40)),
            'right_elbow': (int(width * 0.70), int(height * 0.40)),
            'left_wrist': (int(width * 0.28), int(height * 0.50)),
            'right_wrist': (int(width * 0.72), int(height * 0.50)),
            'left_hip': (int(width * 0.40), int(height * 0.55)),
            'right_hip': (int(width * 0.60), int(height * 0.55)),
            'left_knee': (int(width * 0.38), int(height * 0.75)),
            'right_knee': (int(width * 0.62), int(height * 0.75)),
            'left_ankle': (int(width * 0.37), int(height * 0.95)),
            'right_ankle': (int(width * 0.63), int(height * 0.95)),
        }
        
        return keypoints
    
    def get_body_measurements(self, keypoints: Dict[str, Tuple[int, int]],
                             image_height: int) -> Dict[str, float]:
        """
        Calculate body measurements from keypoints.
        
        Args:
            keypoints: Dictionary of keypoints
            image_height: Height of the image (for scaling)
            
        Returns:
            Dictionary of measurements in pixels
        """
        measurements = {}
        
        try:
            # Shoulder width
            if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
                left_shoulder = keypoints['left_shoulder']
                right_shoulder = keypoints['right_shoulder']
                measurements['shoulder_width'] = abs(right_shoulder[0] - left_shoulder[0])
            
            # Torso length
            if 'left_shoulder' in keypoints and 'left_hip' in keypoints:
                shoulder = keypoints['left_shoulder']
                hip = keypoints['left_hip']
                measurements['torso_length'] = abs(hip[1] - shoulder[1])
            
            # Hip width
            if 'left_hip' in keypoints and 'right_hip' in keypoints:
                left_hip = keypoints['left_hip']
                right_hip = keypoints['right_hip']
                measurements['hip_width'] = abs(right_hip[0] - left_hip[0])
            
            # Leg length
            if 'left_hip' in keypoints and 'left_ankle' in keypoints:
                hip = keypoints['left_hip']
                ankle = keypoints['left_ankle']
                measurements['leg_length'] = abs(ankle[1] - hip[1])
            
            # Arm length
            if 'left_shoulder' in keypoints and 'left_wrist' in keypoints:
                shoulder = keypoints['left_shoulder']
                wrist = keypoints['left_wrist']
                dx = wrist[0] - shoulder[0]
                dy = wrist[1] - shoulder[1]
                measurements['arm_length'] = np.sqrt(dx**2 + dy**2)
            
        except Exception as e:
            logger.error(f"Error calculating measurements: {e}")
        
        return measurements
    
    def get_clothing_region(self, keypoints: Dict[str, Tuple[int, int]],
                           clothing_type: str) -> Optional[List[Tuple[int, int]]]:
        """
        Get the region of interest for a specific clothing type.
        
        Args:
            keypoints: Dictionary of keypoints
            clothing_type: Type of clothing ('shirt', 'pants', 'dress', etc.)
            
        Returns:
            List of corner points defining the clothing region, or None
        """
        try:
            if clothing_type.lower() in ['shirt', 'top', 'jacket']:
                # Upper body region
                if all(k in keypoints for k in ['left_shoulder', 'right_shoulder',
                                                'left_hip', 'right_hip']):
                    return [
                        keypoints['left_shoulder'],
                        keypoints['right_shoulder'],
                        keypoints['right_hip'],
                        keypoints['left_hip']
                    ]
            
            elif clothing_type.lower() == 'pants':
                # Lower body region
                if all(k in keypoints for k in ['left_hip', 'right_hip',
                                               'left_ankle', 'right_ankle']):
                    return [
                        keypoints['left_hip'],
                        keypoints['right_hip'],
                        keypoints['right_ankle'],
                        keypoints['left_ankle']
                    ]
            
            elif clothing_type.lower() in ['dress', 'skirt']:
                # Full body or lower body region
                if all(k in keypoints for k in ['left_shoulder', 'right_shoulder',
                                               'left_ankle', 'right_ankle']):
                    return [
                        keypoints['left_shoulder'],
                        keypoints['right_shoulder'],
                        keypoints['right_ankle'],
                        keypoints['left_ankle']
                    ]
            
        except Exception as e:
            logger.error(f"Error getting clothing region: {e}")
        
        return None
    
    def close(self):
        """Release resources."""
        if self.pose is not None:
            self.pose.close()
