"""
IDM-VTON integration for virtual try-on.

This module provides integration with IDM-VTON (Image-based Deep Model Virtual Try-ON)
via Hugging Face Spaces API for state-of-the-art virtual try-on results.
"""

from .client import IDMVTONClient
from .tryon import virtual_tryon

__all__ = ['IDMVTONClient', 'virtual_tryon']
