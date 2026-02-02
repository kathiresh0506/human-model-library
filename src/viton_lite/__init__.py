"""
VITON-Lite: Lightweight virtual try-on module.

A simplified virtual try-on implementation that provides better results than basic overlay
without the full complexity of VITON-HD. Includes:
- Proper pose-based scaling
- Accurate chest positioning
- Perspective warping for body contours
- Alpha blending with feathered edges
- Basic shadow generation
"""
from .lite_tryon import VITONLiteFitter

__all__ = ['VITONLiteFitter']
