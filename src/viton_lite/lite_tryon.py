"""
VITON-Lite implementation for improved virtual try-on.

This module provides a lightweight alternative to full VITON-HD that significantly
improves upon basic overlay methods by:
1. Proper pose estimation for accurate body measurements
2. Clothing scaling to match body width (shoulder to shoulder)
3. Correct positioning on chest area (not stomach)
4. Perspective warping to follow body contours
5. Alpha blending with feathered edges for natural look
6. Basic shadow generation for realism
"""
import logging
from typing import Optional, Tuple, List
import numpy as np
import cv2

from ..pose_estimator import PoseEstimator
from ..warping import ImageWarper, BlendingUtility

logger = logging.getLogger(__name__)


class VITONLiteFitter:
    """
    Lightweight virtual try-on fitter that provides improved results over basic overlay.
    """
    
    def __init__(self):
        """Initialize VITON-Lite fitter with required components."""
        self.pose_estimator = PoseEstimator()
        self.image_warper = ImageWarper()
        self.blending_utility = BlendingUtility()
        logger.info("VITON-Lite fitter initialized")
    
    def fit_clothing(self,
                    person_image: np.ndarray,
                    clothing_image: np.ndarray,
                    clothing_type: str = 'shirt') -> Optional[np.ndarray]:
        """
        Fit clothing onto person image with improved scaling and positioning.
        
        Args:
            person_image: Person/model image as numpy array (RGB)
            clothing_image: Clothing image as numpy array (RGB or RGBA)
            clothing_type: Type of clothing ('shirt', 'top', 'jacket', 'pants', 'dress')
            
        Returns:
            Image with clothing fitted onto person, or None if fitting fails
        """
        try:
            logger.info(f"Starting VITON-Lite fitting for {clothing_type}")
            
            # Step 1: Detect body keypoints
            keypoints = self.pose_estimator.detect_keypoints(person_image)
            if keypoints is None:
                logger.error("Failed to detect body keypoints")
                return None
            
            # Step 2: Calculate body measurements
            measurements = self.pose_estimator.get_body_measurements(
                keypoints, person_image.shape[0]
            )
            
            # Step 3: Get clothing region based on type
            clothing_region = self.pose_estimator.get_clothing_region(
                keypoints, clothing_type
            )
            
            if clothing_region is None:
                logger.error("Failed to determine clothing region")
                return None
            
            # Step 4: Process clothing with proper scaling
            processed_clothing = self._process_clothing(
                clothing_image, measurements, clothing_region, clothing_type
            )
            
            if processed_clothing is None:
                logger.error("Failed to process clothing")
                return None
            
            # Step 5: Warp clothing to follow body contours
            warped_clothing = self._warp_to_body_contours(
                processed_clothing, keypoints, clothing_region, clothing_type
            )
            
            # Step 6: Add shadows for depth
            clothing_with_shadows = self._add_shadows(
                warped_clothing, keypoints, clothing_type
            )
            
            # Step 7: Blend onto person with feathered edges
            result = self._blend_with_feathering(
                person_image, clothing_with_shadows
            )
            
            logger.info("VITON-Lite fitting completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in VITON-Lite fitting: {e}")
            return None
    
    def _process_clothing(self,
                         clothing_image: np.ndarray,
                         measurements: dict,
                         clothing_region: List[Tuple[int, int]],
                         clothing_type: str) -> Optional[np.ndarray]:
        """
        Process clothing with proper scaling based on body measurements.
        
        Args:
            clothing_image: Original clothing image
            measurements: Body measurements from pose estimation
            clothing_region: Target region for clothing
            clothing_type: Type of clothing
            
        Returns:
            Processed clothing image with alpha channel
        """
        try:
            # Convert to RGBA if needed
            if clothing_image.shape[2] == 3:
                # Extract mask
                mask = self.image_warper.extract_clothing_mask(clothing_image)
                clothing_rgba = np.dstack([clothing_image, mask])
            else:
                clothing_rgba = clothing_image.copy()
            
            # Calculate target dimensions based on body measurements
            region_array = np.array(clothing_region)
            target_width = int(np.max(region_array[:, 0]) - np.min(region_array[:, 0]))
            target_height = int(np.max(region_array[:, 1]) - np.min(region_array[:, 1]))
            
            # Apply scaling factor based on clothing type
            if clothing_type.lower() in ['shirt', 'top', 'jacket']:
                # For tops, use shoulder width for scaling
                if 'shoulder_width' in measurements:
                    # Scale to 95% of shoulder width for natural fit
                    target_width = int(measurements['shoulder_width'] * 0.95)
                    
                # Height should cover torso
                if 'torso_length' in measurements:
                    # Extend a bit beyond torso for natural draping
                    target_height = int(measurements['torso_length'] * 1.1)
            
            # Resize clothing to match body dimensions
            resized_clothing = self.image_warper.resize_to_fit(
                clothing_rgba, target_width, target_height, maintain_aspect=True
            )
            
            logger.info(f"Processed clothing to {resized_clothing.shape[:2]} size")
            return resized_clothing
            
        except Exception as e:
            logger.error(f"Error processing clothing: {e}")
            return None
    
    def _warp_to_body_contours(self,
                               clothing: np.ndarray,
                               keypoints: dict,
                               clothing_region: List[Tuple[int, int]],
                               clothing_type: str) -> np.ndarray:
        """
        Warp clothing to follow body contours using perspective transformation.
        
        Args:
            clothing: Processed clothing image with alpha
            keypoints: Body keypoints
            clothing_region: Target region for clothing
            clothing_type: Type of clothing
            
        Returns:
            Warped clothing image
        """
        try:
            height, width = clothing.shape[:2]
            
            # Define source points (corners of clothing)
            src_points = np.array([
                [0, 0],                    # Top-left
                [width - 1, 0],            # Top-right
                [width - 1, height - 1],   # Bottom-right
                [0, height - 1]            # Bottom-left
            ], dtype=np.float32)
            
            # Calculate center position based on chest/shoulder area
            region_array = np.array(clothing_region)
            center_x = int(np.mean(region_array[:, 0]))
            center_y = int(np.min(region_array[:, 1]))  # Start at top (chest)
            
            # Define target points following body shape
            if clothing_type.lower() in ['shirt', 'top', 'jacket']:
                # For tops, follow shoulder and hip widths
                left_shoulder = keypoints.get('left_shoulder', (center_x - width//2, center_y))
                right_shoulder = keypoints.get('right_shoulder', (center_x + width//2, center_y))
                left_hip = keypoints.get('left_hip', (center_x - width//2 + 20, center_y + height))
                right_hip = keypoints.get('right_hip', (center_x + width//2 - 20, center_y + height))
                
                dst_points = np.array([
                    [left_shoulder[0], left_shoulder[1]],
                    [right_shoulder[0], right_shoulder[1]],
                    [right_hip[0], right_hip[1]],
                    [left_hip[0], left_hip[1]]
                ], dtype=np.float32)
            else:
                # For other types, use rectangular region
                dst_points = np.array(clothing_region[:4], dtype=np.float32)
            
            # Apply perspective transformation
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Calculate output size to contain warped image
            max_x = int(np.max([kp[0] for kp in keypoints.values()]) + 100)
            max_y = int(np.max([kp[1] for kp in keypoints.values()]) + 100)
            
            warped = cv2.warpPerspective(
                clothing, matrix, (max_x, max_y),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            
            logger.info("Clothing warped to body contours")
            return warped
            
        except Exception as e:
            logger.error(f"Error warping to body contours: {e}")
            return clothing
    
    def _add_shadows(self,
                    clothing: np.ndarray,
                    keypoints: dict,
                    clothing_type: str) -> np.ndarray:
        """
        Add subtle shadows to clothing for depth and realism.
        
        Args:
            clothing: Warped clothing image with alpha
            keypoints: Body keypoints for shadow positioning
            clothing_type: Type of clothing
            
        Returns:
            Clothing with shadows added
        """
        try:
            if clothing.shape[2] != 4:
                logger.warning("Cannot add shadows to non-RGBA image")
                return clothing
            
            result = clothing.copy()
            height, width = result.shape[:2]
            
            # Extract RGB and alpha
            rgb = result[:, :, :3]
            alpha = result[:, :, 3]
            
            # Create shadow mask from alpha
            shadow_mask = alpha.astype(float) / 255.0
            
            # Apply vertical gradient for natural lighting (darker at bottom)
            gradient = np.linspace(1.0, 0.85, height)
            gradient = np.tile(gradient[:, np.newaxis], (1, width))
            
            # Apply shadow to RGB channels
            for c in range(3):
                channel = rgb[:, :, c].astype(float)
                channel = channel * gradient * (0.95 + 0.05 * shadow_mask)
                rgb[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
            
            result[:, :, :3] = rgb
            
            logger.info("Added shadows to clothing")
            return result
            
        except Exception as e:
            logger.error(f"Error adding shadows: {e}")
            return clothing
    
    def _blend_with_feathering(self,
                              person_image: np.ndarray,
                              clothing: np.ndarray) -> np.ndarray:
        """
        Blend clothing onto person with feathered edges for natural look.
        
        Args:
            person_image: Base person image (RGB)
            clothing: Warped clothing with alpha channel (RGBA)
            
        Returns:
            Blended result image
        """
        try:
            # Ensure person image is RGB
            if person_image.shape[2] == 4:
                person_rgb = person_image[:, :, :3]
            else:
                person_rgb = person_image.copy()
            
            # Resize clothing if needed
            if clothing.shape[:2] != person_rgb.shape[:2]:
                clothing = cv2.resize(
                    clothing,
                    (person_rgb.shape[1], person_rgb.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            # Extract alpha channel
            if clothing.shape[2] == 4:
                clothing_rgb = clothing[:, :, :3]
                alpha = clothing[:, :, 3].astype(float) / 255.0
            else:
                clothing_rgb = clothing
                # Create alpha from non-zero pixels
                alpha = (np.any(clothing != 0, axis=2)).astype(float)
            
            # Feather edges for smooth blending
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            feathered_alpha = self.blending_utility.feather_edges(alpha_uint8, feather_amount=15)
            alpha = feathered_alpha.astype(float) / 255.0
            
            # Expand alpha to 3 channels
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            
            # Alpha blend
            blended = (clothing_rgb * alpha_3d + person_rgb * (1 - alpha_3d)).astype(np.uint8)
            
            logger.info("Blended clothing with feathered edges")
            return blended
            
        except Exception as e:
            logger.error(f"Error blending with feathering: {e}")
            return person_rgb
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'pose_estimator'):
                self.pose_estimator.close()
        except:
            pass
