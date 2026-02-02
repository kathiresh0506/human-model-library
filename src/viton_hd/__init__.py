"""
VITON-HD: State-of-the-art virtual try-on module.

This module provides a framework for integrating VITON-HD, a state-of-the-art
virtual try-on model that produces production-quality results.

Note: This is a preparatory implementation. Full VITON-HD requires:
- Pre-trained model weights (download via scripts/download_viton_models.py)
- PyTorch with CUDA support (recommended)
- Human parsing models
- Pose estimation models

For immediate results, use viton_lite module instead.
"""
from .viton_processor import VITONProcessor
from .pose_estimator import VITONPoseEstimator
from .segmentation import HumanSegmenter
from .cloth_mask import ClothMaskExtractor

__all__ = [
    'VITONProcessor',
    'VITONPoseEstimator',
    'HumanSegmenter',
    'ClothMaskExtractor'
]
