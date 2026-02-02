"""
Clothing mask extraction for VITON-HD.

Extracts clothing masks from product images by removing backgrounds.
"""
import logging
from typing import Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ClothMaskExtractor:
    """
    Extract clothing masks from product images.
    
    Removes backgrounds and generates cloth masks required by VITON-HD.
    """
    
    def __init__(self):
        """Initialize cloth mask extractor."""
        logger.info("Cloth mask extractor initialized")
    
    def extract_mask(self,
                    clothing_image: np.ndarray,
                    method: str = 'auto') -> np.ndarray:
        """
        Extract clothing mask from product image.
        
        Args:
            clothing_image: Input clothing image (RGB or RGBA)
            method: Extraction method ('auto', 'alpha', 'threshold', 'grabcut')
            
        Returns:
            Binary mask (0-255)
        """
        try:
            if method == 'alpha' or (method == 'auto' and clothing_image.shape[2] == 4):
                # Use alpha channel if available
                return self._extract_from_alpha(clothing_image)
            
            elif method == 'grabcut':
                # Use GrabCut for intelligent background removal
                return self._extract_with_grabcut(clothing_image)
            
            else:
                # Use threshold-based extraction
                return self._extract_with_threshold(clothing_image)
                
        except Exception as e:
            logger.error(f"Error extracting cloth mask: {e}")
            # Return full mask as fallback
            return np.ones(clothing_image.shape[:2], dtype=np.uint8) * 255
    
    def _extract_from_alpha(self, image: np.ndarray) -> np.ndarray:
        """Extract mask from alpha channel."""
        if image.shape[2] != 4:
            logger.warning("No alpha channel found")
            return self._extract_with_threshold(image[:, :, :3])
        
        alpha = image[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        return mask
    
    def _extract_with_threshold(self, image: np.ndarray) -> np.ndarray:
        """Extract mask using threshold (assumes light background)."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold (assuming white background)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _extract_with_grabcut(self, image: np.ndarray) -> np.ndarray:
        """Extract mask using GrabCut algorithm."""
        try:
            # Ensure RGB
            if image.shape[2] == 4:
                rgb = image[:, :, :3]
            else:
                rgb = image
            
            # Initialize mask
            mask = np.zeros(rgb.shape[:2], np.uint8)
            
            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define rectangle around potential foreground
            height, width = rgb.shape[:2]
            rect = (10, 10, width-20, height-20)
            
            # Apply GrabCut
            cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Convert mask to binary
            mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            mask_binary = mask_binary * 255
            
            return mask_binary
            
        except Exception as e:
            logger.error(f"GrabCut failed: {e}, falling back to threshold")
            return self._extract_with_threshold(image)
    
    def refine_mask(self, mask: np.ndarray, iterations: int = 2) -> np.ndarray:
        """
        Refine mask with morphological operations.
        
        Args:
            mask: Input binary mask
            iterations: Number of refinement iterations
            
        Returns:
            Refined mask
        """
        kernel = np.ones((3, 3), np.uint8)
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
