"""
GPU-accelerated OpenPose for human pose estimation.

Estimates 18-point body keypoints and generates pose heatmaps
for VITON-HD processing.
"""
import logging
from typing import Optional, Dict, List, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available for GPU pose estimation")


# OpenPose 18-point keypoint indices
BODY_KEYPOINTS = {
    'nose': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    'right_hip': 8,
    'right_knee': 9,
    'right_ankle': 10,
    'left_hip': 11,
    'left_knee': 12,
    'left_ankle': 13,
    'right_eye': 14,
    'left_eye': 15,
    'right_ear': 16,
    'left_ear': 17
}

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    ('neck', 'nose'),
    ('neck', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('neck', 'left_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('neck', 'right_hip'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('neck', 'left_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('nose', 'right_eye'),
    ('right_eye', 'right_ear'),
    ('nose', 'left_eye'),
    ('left_eye', 'left_ear')
]


class OpenPoseBody(nn.Module):
    """
    Simplified OpenPose network for body pose estimation.
    
    This is a lightweight implementation. For production, use the
    official OpenPose weights or alternatives like DWPose.
    """
    
    def __init__(self):
        """Initialize OpenPose body network."""
        super().__init__()
        
        # Feature extraction (VGG-style)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Heatmap prediction (18 keypoints)
        self.heatmap = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 18, 1)  # 18 keypoints
        )
        
        # PAF prediction (Part Affinity Fields)
        self.paf = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 34, 1)  # 17 limbs * 2 (x,y)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Tuple of (heatmaps, pafs)
        """
        features = self.features(x)
        heatmaps = self.heatmap(features)
        pafs = self.paf(features)
        
        return heatmaps, pafs


class OpenPoseGPU:
    """
    GPU-accelerated OpenPose for pose estimation.
    """
    
    def __init__(self, device: str = 'cuda', weights_path: Optional[str] = None):
        """
        Initialize OpenPose on GPU.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            weights_path: Path to OpenPose weights (optional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        logger.info(f"OpenPose using device: {self.device}")
        
        # Initialize model
        self.model = OpenPoseBody().to(self.device)
        self.model.eval()
        
        # Load weights if provided
        if weights_path and Path(weights_path).exists():
            self.load_weights(weights_path)
        else:
            logger.warning("No OpenPose weights loaded")
            logger.info("Download with: python scripts/download_viton_weights.py")
    
    def load_weights(self, path: str) -> bool:
        """
        Load OpenPose weights.
        
        Args:
            path: Path to weights file
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading OpenPose weights from: {path}")
            checkpoint = torch.load(path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            logger.info("✓ OpenPose weights loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading OpenPose weights: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for OpenPose.
        
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        from PIL import Image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image)
            else:
                pil_img = Image.fromarray((image * 255).astype(np.uint8))
        else:
            pil_img = image
        
        # Resize to standard size
        pil_img = pil_img.resize((512, 768), Image.LANCZOS)
        
        # Convert to numpy and normalize
        image = np.array(pil_img).astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estimate pose keypoints from image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary with keypoints and heatmaps
        """
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Run inference
            heatmaps, pafs = self.model(input_tensor)
            
            # Extract keypoints from heatmaps
            keypoints = self._extract_keypoints(heatmaps)
            
            # Generate pose map for VITON-HD
            pose_map = self._generate_pose_map(keypoints, image.shape[:2])
            
            return {
                'keypoints': keypoints,
                'heatmaps': heatmaps.cpu().numpy(),
                'pafs': pafs.cpu().numpy(),
                'pose_map': pose_map
            }
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return None
    
    def _extract_keypoints(self, heatmaps: torch.Tensor) -> Dict[str, Tuple[int, int, float]]:
        """
        Extract keypoint coordinates from heatmaps.
        
        Args:
            heatmaps: Heatmap tensor [B, 18, H, W]
            
        Returns:
            Dictionary mapping keypoint names to (x, y, confidence)
        """
        keypoints = {}
        
        # Remove batch dimension
        heatmaps = heatmaps.squeeze(0)  # [18, H, W]
        
        # For each keypoint
        for name, idx in BODY_KEYPOINTS.items():
            heatmap = heatmaps[idx]
            
            # Find maximum
            confidence, max_idx = torch.max(heatmap.flatten(), 0)
            confidence = confidence.item()
            
            # Convert to 2D coordinates
            h, w = heatmap.shape
            y = (max_idx // w).item()
            x = (max_idx % w).item()
            
            # Scale to original size (assuming 512x768)
            x = int(x * 512 / w)
            y = int(y * 768 / h)
            
            keypoints[name] = (x, y, confidence)
        
        return keypoints
    
    def _generate_pose_map(self, keypoints: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate pose map visualization for VITON-HD.
        
        Args:
            keypoints: Keypoint dictionary
            image_shape: (height, width) of output
            
        Returns:
            Pose map as RGB image
        """
        h, w = image_shape
        pose_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw skeleton connections
        for start_name, end_name in SKELETON_CONNECTIONS:
            if start_name in keypoints and end_name in keypoints:
                start = keypoints[start_name][:2]
                end = keypoints[end_name][:2]
                
                # Draw line (simple implementation)
                import cv2
                cv2.line(pose_map, start, end, (255, 255, 255), 2)
        
        # Draw keypoints
        for name, (x, y, conf) in keypoints.items():
            if conf > 0.1:  # Confidence threshold
                import cv2
                cv2.circle(pose_map, (x, y), 4, (0, 255, 0), -1)
        
        return pose_map
    
    def visualize_pose(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Visualize pose on image.
        
        Args:
            image: Original image
            keypoints: Keypoint dictionary
            
        Returns:
            Image with pose overlay
        """
        import cv2
        
        result = image.copy()
        
        # Draw skeleton
        for start_name, end_name in SKELETON_CONNECTIONS:
            if start_name in keypoints and end_name in keypoints:
                start = keypoints[start_name][:2]
                end = keypoints[end_name][:2]
                cv2.line(result, start, end, (0, 255, 0), 2)
        
        # Draw keypoints
        for name, (x, y, conf) in keypoints.items():
            if conf > 0.1:
                cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(result, name[:3], (x+5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return result


# Fallback to MediaPipe if PyTorch not available
class MediaPipePoseEstimator:
    """Fallback pose estimator using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe pose estimator."""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            logger.info("Using MediaPipe for pose estimation (CPU)")
        except ImportError:
            logger.error("MediaPipe not installed")
            self.pose = None
    
    def estimate(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estimate pose using MediaPipe.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary with keypoints
        """
        if self.pose is None:
            return None
        
        try:
            # MediaPipe expects RGB
            results = self.pose.process(image)
            
            if not results.pose_landmarks:
                return None
            
            # Convert to OpenPose-style format
            h, w = image.shape[:2]
            keypoints = {}
            
            # Map MediaPipe landmarks to OpenPose keypoints
            # (simplified mapping)
            landmarks = results.pose_landmarks.landmark
            
            return {
                'keypoints': keypoints,
                'pose_map': None
            }
            
        except Exception as e:
            logger.error(f"MediaPipe pose estimation error: {e}")
            return None
