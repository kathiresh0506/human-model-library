"""
Clothing fitting module for virtual try-on functionality.
Combines pose estimation, warping, and blending to fit clothes on models.
"""
import logging
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path

from .pose_estimator import PoseEstimator
from .warping import ImageWarper, BlendingUtility
from .utils import ImageLoader, Validator

logger = logging.getLogger(__name__)


class ClothingFitter:
    """
    Fits clothing onto human model images using virtual try-on techniques.
    """
    
    def __init__(self):
        """Initialize ClothingFitter with required components."""
        self.pose_estimator = PoseEstimator()
        self.image_warper = ImageWarper()
        self.blending_utility = BlendingUtility()
        self.image_loader = ImageLoader()
        self.validator = Validator()
    
    def fit_clothing(self,
                    model_image: np.ndarray,
                    clothing_image: np.ndarray,
                    clothing_type: str) -> Optional[np.ndarray]:
        """
        Fit clothing onto a model image.
        
        Args:
            model_image: Model image as numpy array (RGB)
            clothing_image: Clothing image as numpy array (RGB)
            clothing_type: Type of clothing ('shirt', 'pants', 'dress', etc.)
            
        Returns:
            Image with clothing fitted onto model, or None if fitting fails
        """
        try:
            # Validate inputs
            if not self.validator.validate_image_array(model_image):
                logger.error("Invalid model image")
                return None
            
            if not self.validator.validate_image_array(clothing_image):
                logger.error("Invalid clothing image")
                return None
            
            if not self.validator.validate_clothing_type(clothing_type):
                logger.warning(f"Unknown clothing type: {clothing_type}")
            
            # Step 1: Detect body keypoints on model
            logger.info("Detecting body keypoints...")
            body_keypoints = self.detect_body_keypoints(model_image)
            
            if body_keypoints is None:
                logger.error("Failed to detect body keypoints")
                return None
            
            # Step 2: Get clothing region based on type
            logger.info(f"Getting clothing region for {clothing_type}...")
            clothing_region = self.pose_estimator.get_clothing_region(
                body_keypoints, clothing_type
            )
            
            if clothing_region is None:
                logger.error("Failed to get clothing region")
                return None
            
            # Step 3: Warp clothing to fit body shape
            logger.info("Warping clothing to body shape...")
            warped_clothing = self.warp_clothing(
                clothing_image, body_keypoints, clothing_region, clothing_type
            )
            
            if warped_clothing is None:
                logger.error("Failed to warp clothing")
                return None
            
            # Step 4: Blend clothing onto model
            logger.info("Blending clothing onto model...")
            result = self.blend_images(model_image, warped_clothing)
            
            logger.info("Clothing fitting completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error fitting clothing: {e}")
            return None
    
    def detect_body_keypoints(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect body keypoints in an image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Dictionary of body keypoints, or None if detection fails
        """
        try:
            keypoints = self.pose_estimator.detect_keypoints(image)
            
            if keypoints is None:
                logger.warning("No keypoints detected")
                return None
            
            logger.info(f"Detected {len(keypoints)} keypoints")
            return keypoints
            
        except Exception as e:
            logger.error(f"Error detecting keypoints: {e}")
            return None
    
    def warp_clothing(self,
                     clothing_image: np.ndarray,
                     body_keypoints: dict,
                     clothing_region: List[Tuple[int, int]],
                     clothing_type: str) -> Optional[np.ndarray]:
        """
        Warp clothing image to fit body shape.
        
        Args:
            clothing_image: Clothing image
            body_keypoints: Body keypoints dictionary
            clothing_region: Region where clothing should be placed
            clothing_type: Type of clothing
            
        Returns:
            Warped clothing image, or None if warping fails
        """
        try:
            # Get dimensions of clothing region
            region_array = np.array(clothing_region)
            x_min, y_min = region_array.min(axis=0)
            x_max, y_max = region_array.max(axis=0)
            
            region_width = x_max - x_min
            region_height = y_max - y_min
            
            # Extract clothing mask (assuming white background)
            logger.info("Extracting clothing mask...")
            clothing_mask = self.image_warper.extract_clothing_mask(clothing_image)
            
            # Resize clothing to approximately fit the region
            logger.info("Resizing clothing...")
            resized_clothing = self.image_warper.resize_to_fit(
                clothing_image, region_width, region_height, maintain_aspect=True
            )
            resized_mask = self.image_warper.resize_to_fit(
                clothing_mask, region_width, region_height, maintain_aspect=True
            )
            
            # Create target image same size as model
            # Get actual image dimensions from keypoints
            max_y = max([kp[1] for kp in body_keypoints.values()])
            max_x = max([kp[0] for kp in body_keypoints.values()])
            
            # Add padding and create canvas
            canvas_height = max_y + 100
            canvas_width = max_x + 100
            
            # Create a canvas to place the clothing
            warped_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Calculate position to place resized clothing
            clothing_h, clothing_w = resized_clothing.shape[:2]
            
            # Center clothing in the region
            start_y = int(y_min)
            start_x = int(x_min + (region_width - clothing_w) / 2)
            
            # Ensure boundaries
            end_y = min(start_y + clothing_h, warped_canvas.shape[0])
            end_x = min(start_x + clothing_w, warped_canvas.shape[1])
            
            # Adjust if clothing is too large
            actual_h = end_y - start_y
            actual_w = end_x - start_x
            
            if actual_h < clothing_h or actual_w < clothing_w:
                resized_clothing = resized_clothing[:actual_h, :actual_w]
            
            # Place clothing on canvas
            warped_canvas[start_y:end_y, start_x:end_x] = resized_clothing
            
            logger.info("Clothing warped successfully")
            return warped_canvas
            
        except Exception as e:
            logger.error(f"Error warping clothing: {e}")
            return None
    
    def blend_images(self,
                    model_image: np.ndarray,
                    warped_clothing: np.ndarray) -> np.ndarray:
        """
        Blend warped clothing onto model image.
        
        Args:
            model_image: Base model image
            warped_clothing: Warped clothing image
            
        Returns:
            Blended image
        """
        try:
            # Ensure both images have the same dimensions
            if warped_clothing.shape[:2] != model_image.shape[:2]:
                import cv2
                warped_clothing = cv2.resize(
                    warped_clothing,
                    (model_image.shape[1], model_image.shape[0])
                )
            
            # Create mask from warped clothing (non-zero pixels)
            mask = np.any(warped_clothing != 0, axis=2).astype(np.uint8) * 255
            
            # Feather mask edges for smooth blending
            mask = self.blending_utility.feather_edges(mask, feather_amount=10)
            
            # Alpha blend
            blended = self.blending_utility.alpha_blend(
                model_image, warped_clothing, mask, alpha=0.95
            )
            
            logger.info("Blending completed successfully")
            return blended
            
        except Exception as e:
            logger.error(f"Error blending images: {e}")
            return model_image
    
    def process_full_tryon(self,
                          model_path: str,
                          clothing_path: str,
                          clothing_type: str,
                          output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Complete try-on process from file paths.
        
        Args:
            model_path: Path to model image file
            clothing_path: Path to clothing image file
            clothing_type: Type of clothing
            output_path: Optional path to save result
            
        Returns:
            Result image, or None if processing fails
        """
        try:
            # Load images
            logger.info(f"Loading model image from {model_path}")
            model_image = self.image_loader.load_image(model_path)
            
            if model_image is None:
                logger.error("Failed to load model image")
                return None
            
            logger.info(f"Loading clothing image from {clothing_path}")
            clothing_image = self.image_loader.load_image(clothing_path)
            
            if clothing_image is None:
                logger.error("Failed to load clothing image")
                return None
            
            # Perform try-on
            result = self.fit_clothing(model_image, clothing_image, clothing_type)
            
            if result is None:
                logger.error("Try-on failed")
                return None
            
            # Save result if output path is provided
            if output_path:
                logger.info(f"Saving result to {output_path}")
                self.image_loader.save_image(result, output_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in full try-on process: {e}")
            return None
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.pose_estimator.close()
        except:
            pass
