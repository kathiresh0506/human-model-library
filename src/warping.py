"""
Image warping module for clothing deformation and transformation.
Implements thin-plate spline warping and perspective transformation.
"""
import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ImageWarper:
    """
    Handles image warping operations for clothing fitting.
    """
    
    @staticmethod
    def thin_plate_spline_warp(image: np.ndarray,
                               source_points: np.ndarray,
                               target_points: np.ndarray) -> np.ndarray:
        """
        Apply thin-plate spline warping to an image.
        
        Args:
            image: Input image
            source_points: Source control points (N x 2)
            target_points: Target control points (N x 2)
            
        Returns:
            Warped image
        """
        try:
            height, width = image.shape[:2]
            
            # Use OpenCV's thin plate spline transformation if available
            # For production, consider using scikit-image or custom implementation
            matches = []
            for i in range(len(source_points)):
                matches.append(cv2.DMatch(i, i, 0))
            
            # Create a TPS transformer
            tps = cv2.createThinPlateSplineShapeTransformer()
            
            # Reshape points for OpenCV
            source_shaped = source_points.reshape(1, -1, 2).astype(np.float32)
            target_shaped = target_points.reshape(1, -1, 2).astype(np.float32)
            
            tps.estimateTransformation(target_shaped, source_shaped, matches)
            
            # Apply transformation
            warped = tps.warpImage(image)
            
            return warped
            
        except Exception as e:
            logger.error(f"TPS warping failed: {e}. Using affine transform fallback.")
            return ImageWarper.affine_warp(image, source_points, target_points)
    
    @staticmethod
    def affine_warp(image: np.ndarray,
                   source_points: np.ndarray,
                   target_points: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation as a fallback method.
        
        Args:
            image: Input image
            source_points: Source control points (at least 3 points)
            target_points: Target control points (at least 3 points)
            
        Returns:
            Warped image
        """
        try:
            # Use first 3 points for affine transformation
            src_pts = source_points[:3].astype(np.float32)
            dst_pts = target_points[:3].astype(np.float32)
            
            # Calculate affine transformation matrix
            matrix = cv2.getAffineTransform(src_pts, dst_pts)
            
            # Apply transformation
            height, width = image.shape[:2]
            warped = cv2.warpAffine(image, matrix, (width, height))
            
            return warped
            
        except Exception as e:
            logger.error(f"Affine warping failed: {e}")
            return image
    
    @staticmethod
    def perspective_transform(image: np.ndarray,
                            source_corners: List[Tuple[int, int]],
                            target_corners: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply perspective transformation to an image.
        
        Args:
            image: Input image
            source_corners: Four corner points in source image
            target_corners: Four corner points in target position
            
        Returns:
            Transformed image
        """
        try:
            # Convert to numpy arrays
            src_pts = np.array(source_corners, dtype=np.float32)
            dst_pts = np.array(target_corners, dtype=np.float32)
            
            # Calculate perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Apply transformation
            height, width = image.shape[:2]
            warped = cv2.warpPerspective(image, matrix, (width, height))
            
            return warped
            
        except Exception as e:
            logger.error(f"Perspective transform failed: {e}")
            return image
    
    @staticmethod
    def resize_to_fit(clothing_image: np.ndarray,
                     target_width: int,
                     target_height: int,
                     maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize clothing image to fit target dimensions.
        
        Args:
            clothing_image: Input clothing image
            target_width: Target width
            target_height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        try:
            if maintain_aspect:
                # Calculate scaling factor
                height, width = clothing_image.shape[:2]
                scale = min(target_width / width, target_height / height)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = cv2.resize(clothing_image, (new_width, new_height),
                                   interpolation=cv2.INTER_LANCZOS4)
            else:
                resized = cv2.resize(clothing_image, (target_width, target_height),
                                   interpolation=cv2.INTER_LANCZOS4)
            
            return resized
            
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            return clothing_image
    
    @staticmethod
    def extract_clothing_mask(clothing_image: np.ndarray,
                            threshold: int = 240) -> np.ndarray:
        """
        Extract clothing from background supporting both RGB and RGBA.
        
        Args:
            clothing_image: Input clothing image (RGB or RGBA)
            threshold: Threshold for background detection
            
        Returns:
            Binary mask (1 channel, 0-255)
        """
        try:
            # Check if image has alpha channel
            if clothing_image.shape[2] == 4:
                # Use alpha channel as mask
                alpha = clothing_image[:, :, 3]
                # Threshold alpha to create binary mask
                _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
                return mask
            
            # For RGB images, detect by brightness (assume white background)
            # Convert to grayscale
            gray = cv2.cvtColor(clothing_image, cv2.COLOR_RGB2GRAY)
            
            # Invert (assuming light background)
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            logger.error(f"Mask extraction failed: {e}")
            # Return full mask if extraction fails
            return np.ones(clothing_image.shape[:2], dtype=np.uint8) * 255
    
    @staticmethod
    def warp_clothing_to_body(clothing_image: np.ndarray,
                             clothing_keypoints: List[Tuple[int, int]],
                             body_keypoints: List[Tuple[int, int]]) -> np.ndarray:
        """
        Warp clothing image to fit body shape using keypoints.
        
        Args:
            clothing_image: Input clothing image
            clothing_keypoints: Key points on the clothing
            body_keypoints: Corresponding key points on the body
            
        Returns:
            Warped clothing image
        """
        try:
            # Convert lists to numpy arrays
            src_points = np.array(clothing_keypoints, dtype=np.float32)
            dst_points = np.array(body_keypoints, dtype=np.float32)
            
            # Use TPS warping for smooth deformation
            warped = ImageWarper.thin_plate_spline_warp(
                clothing_image, src_points, dst_points
            )
            
            return warped
            
        except Exception as e:
            logger.error(f"Clothing warping failed: {e}")
            return clothing_image


class BlendingUtility:
    """
    Utilities for blending clothing onto model images.
    """
    
    @staticmethod
    def alpha_blend(model_image: np.ndarray,
                   clothing_image: np.ndarray,
                   mask: np.ndarray,
                   alpha: float = 0.95) -> np.ndarray:
        """
        Alpha blend clothing onto model using a mask.
        
        Args:
            model_image: Base model image (RGB)
            clothing_image: Clothing image to blend (RGB)
            mask: Binary mask (1 channel, 0-255)
            alpha: Blending factor (0-1)
            
        Returns:
            Blended image
        """
        try:
            # Ensure images have same dimensions
            if model_image.shape[:2] != clothing_image.shape[:2]:
                clothing_image = cv2.resize(clothing_image, 
                                          (model_image.shape[1], model_image.shape[0]))
            
            if mask.shape[:2] != model_image.shape[:2]:
                mask = cv2.resize(mask, (model_image.shape[1], model_image.shape[0]))
            
            # Normalize mask to 0-1
            mask_norm = mask.astype(float) / 255.0
            
            # Expand mask to 3 channels
            mask_3d = np.stack([mask_norm] * 3, axis=2)
            
            # Apply alpha blending
            blended = (clothing_image * mask_3d * alpha + 
                      model_image * (1 - mask_3d * alpha))
            
            return blended.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Alpha blending failed: {e}")
            return model_image
    
    @staticmethod
    def poisson_blend(model_image: np.ndarray,
                     clothing_image: np.ndarray,
                     mask: np.ndarray,
                     center: Tuple[int, int]) -> np.ndarray:
        """
        Poisson blending for seamless integration.
        
        Args:
            model_image: Base model image
            clothing_image: Clothing image
            mask: Binary mask
            center: Center point for blending
            
        Returns:
            Blended image
        """
        try:
            # Use OpenCV's seamless cloning
            result = cv2.seamlessClone(
                clothing_image,
                model_image,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
            return result
            
        except Exception as e:
            logger.error(f"Poisson blending failed: {e}. Using alpha blend.")
            return BlendingUtility.alpha_blend(model_image, clothing_image, mask)
    
    @staticmethod
    def feather_edges(mask: np.ndarray, feather_amount: int = 5) -> np.ndarray:
        """
        Feather the edges of a mask for smoother blending.
        
        Args:
            mask: Binary mask
            feather_amount: Amount of feathering in pixels
            
        Returns:
            Feathered mask
        """
        try:
            # Apply Gaussian blur to feather edges
            feathered = cv2.GaussianBlur(mask, 
                                        (feather_amount * 2 + 1, feather_amount * 2 + 1),
                                        0)
            return feathered
            
        except Exception as e:
            logger.error(f"Edge feathering failed: {e}")
            return mask
