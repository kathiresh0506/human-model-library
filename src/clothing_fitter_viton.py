"""
VITON-based clothing fitter with automatic fallback.

Provides a unified interface for virtual try-on using:
1. VITON-HD (if weights are available)
2. VITON-Lite (lightweight alternative)
3. Basic fitter (fallback)
"""
import logging
from typing import Optional
import numpy as np
from pathlib import Path

from .viton_lite import VITONLiteFitter
from .viton_hd import VITONProcessor, VITONPoseEstimator, HumanSegmenter
from .clothing_fitter import ClothingFitter

logger = logging.getLogger(__name__)


class VITONClothingFitter:
    """
    Unified clothing fitter with automatic fallback strategy.
    
    Tries methods in order:
    1. VITON-HD (best quality, requires weights)
    2. VITON-Lite (good quality, no weights needed)
    3. Basic fitter (fallback)
    """
    
    def __init__(self, prefer_viton_hd: bool = True, weights_dir: Optional[str] = None):
        """
        Initialize VITON clothing fitter.
        
        Args:
            prefer_viton_hd: Try VITON-HD first if weights available
            weights_dir: Directory containing VITON-HD weights
        """
        self.prefer_viton_hd = prefer_viton_hd
        self.weights_dir = Path(weights_dir) if weights_dir else Path("weights/viton_hd")
        
        # Initialize components
        self.viton_lite = VITONLiteFitter()
        self.basic_fitter = ClothingFitter()
        
        # Check if VITON-HD is available
        self.viton_hd_available = self._check_viton_hd_available()
        
        if self.viton_hd_available:
            self.viton_hd = VITONProcessor(str(self.weights_dir))
            self.viton_pose = VITONPoseEstimator()
            self.viton_segmenter = HumanSegmenter()
            logger.info("VITON-HD components loaded")
        else:
            self.viton_hd = None
            logger.info("VITON-HD not available, will use VITON-Lite")
        
        logger.info("VITON clothing fitter initialized")
    
    def _check_viton_hd_available(self) -> bool:
        """Check if VITON-HD weights are available."""
        if not self.weights_dir.exists():
            return False
        
        required_files = ['generator.pth']
        return all((self.weights_dir / f).exists() for f in required_files)
    
    def fit_clothing(self,
                    person_image: np.ndarray,
                    clothing_image: np.ndarray,
                    clothing_type: str = 'shirt',
                    method: str = 'auto') -> Optional[np.ndarray]:
        """
        Fit clothing onto person image using best available method.
        
        Args:
            person_image: Person/model image (RGB)
            clothing_image: Clothing image (RGB or RGBA)
            clothing_type: Type of clothing ('shirt', 'pants', 'dress', etc.)
            method: Method to use ('auto', 'viton_hd', 'viton_lite', 'basic')
            
        Returns:
            Result image with clothing fitted, or None if all methods fail
        """
        try:
            # Auto method selection
            if method == 'auto':
                if self.prefer_viton_hd and self.viton_hd_available:
                    method = 'viton_hd'
                else:
                    method = 'viton_lite'
            
            # Try requested method
            if method == 'viton_hd' and self.viton_hd_available:
                logger.info("Attempting VITON-HD try-on")
                result = self._try_viton_hd(person_image, clothing_image, clothing_type)
                if result is not None:
                    return result
                logger.warning("VITON-HD failed, falling back to VITON-Lite")
                method = 'viton_lite'
            
            if method == 'viton_lite':
                logger.info("Attempting VITON-Lite try-on")
                result = self.viton_lite.fit_clothing(
                    person_image, clothing_image, clothing_type
                )
                if result is not None:
                    return result
                logger.warning("VITON-Lite failed, falling back to basic fitter")
                method = 'basic'
            
            if method == 'basic':
                logger.info("Using basic clothing fitter")
                result = self.basic_fitter.fit_clothing(
                    person_image, clothing_image, clothing_type
                )
                return result
            
            logger.error(f"Unknown method: {method}")
            return None
            
        except Exception as e:
            logger.error(f"Error in clothing fitting: {e}")
            return None
    
    def _try_viton_hd(self,
                     person_image: np.ndarray,
                     clothing_image: np.ndarray,
                     clothing_type: str) -> Optional[np.ndarray]:
        """
        Try virtual try-on using VITON-HD.
        
        Args:
            person_image: Person image
            clothing_image: Clothing image
            clothing_type: Type of clothing
            
        Returns:
            Try-on result or None if VITON-HD fails
        """
        try:
            # Estimate pose
            pose = self.viton_pose.estimate(person_image)
            if pose is None:
                logger.warning("Pose estimation failed")
                return None
            
            # Segment person
            segmentation = self.viton_segmenter.segment(person_image)
            
            # Prepare inputs
            person_prep = self.viton_hd.preprocess_person(person_image)
            cloth_prep = self.viton_hd.preprocess_cloth(clothing_image)
            
            # Generate agnostic representation
            agnostic = self.viton_hd.generate_agnostic(person_prep, pose, segmentation)
            
            # Run VITON-HD
            result = self.viton_hd.try_on(agnostic, cloth_prep, pose)
            
            if result is not None:
                result = self.viton_hd.postprocess(result)
            
            return result
            
        except Exception as e:
            logger.error(f"VITON-HD processing error: {e}")
            return None
    
    def process_full_tryon(self,
                          model_path: str,
                          clothing_path: str,
                          clothing_type: str,
                          output_path: Optional[str] = None,
                          method: str = 'auto') -> Optional[np.ndarray]:
        """
        Complete try-on process from file paths.
        
        Args:
            model_path: Path to model image file
            clothing_path: Path to clothing image file
            clothing_type: Type of clothing
            output_path: Optional path to save result
            method: Method to use ('auto', 'viton_hd', 'viton_lite', 'basic')
            
        Returns:
            Result image, or None if processing fails
        """
        try:
            from .utils import ImageLoader
            
            loader = ImageLoader()
            
            # Load images
            logger.info(f"Loading model image from {model_path}")
            model_image = loader.load_image(model_path)
            if model_image is None:
                logger.error("Failed to load model image")
                return None
            
            logger.info(f"Loading clothing image from {clothing_path}")
            clothing_image = loader.load_image(clothing_path)
            if clothing_image is None:
                logger.error("Failed to load clothing image")
                return None
            
            # Perform try-on
            result = self.fit_clothing(model_image, clothing_image, clothing_type, method)
            
            if result is None:
                logger.error("Try-on failed")
                return None
            
            # Save result if output path is provided
            if output_path:
                logger.info(f"Saving result to {output_path}")
                loader.save_image(result, output_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in full try-on process: {e}")
            return None
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'viton_lite'):
                del self.viton_lite
            if hasattr(self, 'basic_fitter'):
                del self.basic_fitter
        except:
            pass
