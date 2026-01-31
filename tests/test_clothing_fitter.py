"""
Unit tests for ClothingFitter class.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clothing_fitter import ClothingFitter
from src.pose_estimator import PoseEstimator
from src.utils import ImageLoader, Validator


class TestClothingFitter:
    """Test cases for ClothingFitter."""
    
    @pytest.fixture
    def fitter(self):
        """Create a ClothingFitter instance for testing."""
        return ClothingFitter()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple 512x512 RGB image
        return np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    @pytest.fixture
    def sample_keypoints(self):
        """Create sample body keypoints for testing."""
        return {
            'nose': (256, 100),
            'left_shoulder': (200, 200),
            'right_shoulder': (312, 200),
            'left_hip': (220, 350),
            'right_hip': (292, 350),
            'left_knee': (215, 450),
            'right_knee': (297, 450),
        }
    
    def test_initialization(self, fitter):
        """Test ClothingFitter initialization."""
        assert fitter is not None
        assert fitter.pose_estimator is not None
        assert fitter.image_warper is not None
        assert fitter.blending_utility is not None
        assert fitter.validator is not None
    
    def test_detect_body_keypoints(self, fitter, sample_image):
        """Test body keypoint detection."""
        keypoints = fitter.detect_body_keypoints(sample_image)
        
        # Keypoints may be None or a dict depending on image content
        assert keypoints is None or isinstance(keypoints, dict)
        
        if keypoints is not None:
            # Check that keypoints have expected structure
            assert isinstance(keypoints, dict)
            # Should have x, y coordinates as tuples
            for name, coords in keypoints.items():
                assert isinstance(coords, tuple)
                assert len(coords) == 2
    
    def test_fit_clothing_invalid_model_image(self, fitter, sample_image):
        """Test fit_clothing with invalid model image."""
        # Pass invalid image (not RGB)
        invalid_image = np.ones((512, 512), dtype=np.uint8)  # Grayscale
        
        result = fitter.fit_clothing(
            model_image=invalid_image,
            clothing_image=sample_image,
            clothing_type='shirt'
        )
        
        assert result is None
    
    def test_fit_clothing_invalid_clothing_image(self, fitter, sample_image):
        """Test fit_clothing with invalid clothing image."""
        # Pass invalid clothing image
        invalid_image = np.ones((512, 512), dtype=np.uint8)  # Grayscale
        
        result = fitter.fit_clothing(
            model_image=sample_image,
            clothing_image=invalid_image,
            clothing_type='shirt'
        )
        
        assert result is None
    
    def test_fit_clothing_valid_images(self, fitter, sample_image):
        """Test fit_clothing with valid images."""
        result = fitter.fit_clothing(
            model_image=sample_image,
            clothing_image=sample_image,
            clothing_type='shirt'
        )
        
        # Result may be None or array depending on keypoint detection
        assert result is None or isinstance(result, np.ndarray)
        
        if result is not None:
            # Check result has same shape as model image
            assert result.shape == sample_image.shape
    
    def test_fit_clothing_different_types(self, fitter, sample_image):
        """Test fit_clothing with different clothing types."""
        clothing_types = ['shirt', 'pants', 'dress', 'jacket']
        
        for clothing_type in clothing_types:
            result = fitter.fit_clothing(
                model_image=sample_image,
                clothing_image=sample_image,
                clothing_type=clothing_type
            )
            
            # Should handle all clothing types without errors
            assert result is None or isinstance(result, np.ndarray)
    
    def test_warp_clothing(self, fitter, sample_image, sample_keypoints):
        """Test clothing warping."""
        clothing_region = [
            (200, 200),  # top-left
            (312, 200),  # top-right
            (292, 350),  # bottom-right
            (220, 350),  # bottom-left
        ]
        
        result = fitter.warp_clothing(
            clothing_image=sample_image,
            body_keypoints=sample_keypoints,
            clothing_region=clothing_region,
            clothing_type='shirt'
        )
        
        # Should return an array or None
        assert result is None or isinstance(result, np.ndarray)
    
    def test_blend_images(self, fitter, sample_image):
        """Test image blending."""
        # Create slightly different images
        model_image = sample_image.copy()
        clothing_image = sample_image.copy() * 0.8
        
        result = fitter.blend_images(model_image, clothing_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == model_image.shape
    
    def test_blend_images_different_sizes(self, fitter):
        """Test blending images of different sizes."""
        model_image = np.ones((512, 512, 3), dtype=np.uint8) * 100
        clothing_image = np.ones((256, 256, 3), dtype=np.uint8) * 150
        
        # Should handle different sizes
        result = fitter.blend_images(model_image, clothing_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == model_image.shape


class TestPoseEstimator:
    """Test cases for PoseEstimator."""
    
    @pytest.fixture
    def estimator(self):
        """Create a PoseEstimator instance for testing."""
        return PoseEstimator()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    def test_initialization(self, estimator):
        """Test PoseEstimator initialization."""
        assert estimator is not None
    
    def test_detect_keypoints(self, estimator, sample_image):
        """Test keypoint detection."""
        keypoints = estimator.detect_keypoints(sample_image)
        
        # Should return None or dict
        assert keypoints is None or isinstance(keypoints, dict)
    
    def test_get_body_measurements(self, estimator):
        """Test body measurements calculation."""
        sample_keypoints = {
            'left_shoulder': (200, 200),
            'right_shoulder': (312, 200),
            'left_hip': (220, 350),
            'right_hip': (292, 350),
            'left_ankle': (215, 500),
        }
        
        measurements = estimator.get_body_measurements(sample_keypoints, 512)
        
        assert isinstance(measurements, dict)
        
        if 'shoulder_width' in measurements:
            assert measurements['shoulder_width'] > 0
    
    def test_get_clothing_region(self, estimator):
        """Test getting clothing region."""
        sample_keypoints = {
            'left_shoulder': (200, 200),
            'right_shoulder': (312, 200),
            'left_hip': (220, 350),
            'right_hip': (292, 350),
        }
        
        region = estimator.get_clothing_region(sample_keypoints, 'shirt')
        
        # Should return list of points or None
        assert region is None or isinstance(region, list)
        
        if region is not None:
            assert len(region) == 4  # Four corners


class TestValidator:
    """Test cases for Validator."""
    
    def test_validate_gender(self):
        """Test gender validation."""
        assert Validator.validate_gender('male') == True
        assert Validator.validate_gender('female') == True
        assert Validator.validate_gender('MALE') == True
        assert Validator.validate_gender('invalid') == False
    
    def test_validate_size(self):
        """Test size validation."""
        assert Validator.validate_size('S') == True
        assert Validator.validate_size('M') == True
        assert Validator.validate_size('L') == True
        assert Validator.validate_size('XL') == True
        assert Validator.validate_size('s') == True
        assert Validator.validate_size('XXL') == False
    
    def test_validate_age_group(self):
        """Test age group validation."""
        assert Validator.validate_age_group('young') == True
        assert Validator.validate_age_group('middle') == True
        assert Validator.validate_age_group('senior') == True
        assert Validator.validate_age_group('YOUNG') == True
        assert Validator.validate_age_group('child') == False
    
    def test_validate_ethnicity(self):
        """Test ethnicity validation."""
        assert Validator.validate_ethnicity('asian') == True
        assert Validator.validate_ethnicity('african') == True
        assert Validator.validate_ethnicity('caucasian') == True
        assert Validator.validate_ethnicity('hispanic') == True
        assert Validator.validate_ethnicity('middle_eastern') == True
        assert Validator.validate_ethnicity('invalid') == False
    
    def test_validate_clothing_type(self):
        """Test clothing type validation."""
        assert Validator.validate_clothing_type('shirt') == True
        assert Validator.validate_clothing_type('pants') == True
        assert Validator.validate_clothing_type('dress') == True
        assert Validator.validate_clothing_type('invalid') == False
    
    def test_validate_image_array(self):
        """Test image array validation."""
        # Valid RGB image
        valid_image = np.ones((512, 512, 3), dtype=np.uint8)
        assert Validator.validate_image_array(valid_image) == True
        
        # Valid RGBA image
        valid_image_rgba = np.ones((512, 512, 4), dtype=np.uint8)
        assert Validator.validate_image_array(valid_image_rgba) == True
        
        # Invalid: grayscale
        invalid_image = np.ones((512, 512), dtype=np.uint8)
        assert Validator.validate_image_array(invalid_image) == False
        
        # Invalid: not numpy array
        assert Validator.validate_image_array([1, 2, 3]) == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
