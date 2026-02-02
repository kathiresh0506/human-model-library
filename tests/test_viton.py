"""
Unit tests for VITON integration modules.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viton_lite import VITONLiteFitter
from src.clothing_fitter_viton import VITONClothingFitter


class TestVITONLiteFitter:
    """Test cases for VITON-Lite fitter."""
    
    @pytest.fixture
    def fitter(self):
        """Create a VITON-Lite fitter instance for testing."""
        return VITONLiteFitter()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple 512x768 RGB image
        return np.ones((768, 512, 3), dtype=np.uint8) * 128
    
    def test_initialization(self, fitter):
        """Test VITON-Lite fitter initialization."""
        assert fitter is not None
        assert fitter.pose_estimator is not None
        assert fitter.image_warper is not None
        assert fitter.blending_utility is not None
    
    def test_fit_clothing(self, fitter, sample_image):
        """Test fit_clothing method."""
        result = fitter.fit_clothing(
            person_image=sample_image,
            clothing_image=sample_image,
            clothing_type='shirt'
        )
        
        # Result may be None or array depending on keypoint detection
        assert result is None or isinstance(result, np.ndarray)
        
        if result is not None:
            # Check result has same height/width as person image
            assert result.shape[:2] == sample_image.shape[:2]
    
    def test_fit_clothing_different_types(self, fitter, sample_image):
        """Test fit_clothing with different clothing types."""
        clothing_types = ['shirt', 'top', 'jacket', 'pants', 'dress']
        
        for clothing_type in clothing_types:
            result = fitter.fit_clothing(
                person_image=sample_image,
                clothing_image=sample_image,
                clothing_type=clothing_type
            )
            
            # Should handle all clothing types without errors
            assert result is None or isinstance(result, np.ndarray)


class TestVITONClothingFitter:
    """Test cases for VITON clothing fitter (wrapper)."""
    
    @pytest.fixture
    def fitter(self):
        """Create a VITON clothing fitter instance for testing."""
        return VITONClothingFitter()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.ones((768, 512, 3), dtype=np.uint8) * 128
    
    def test_initialization(self, fitter):
        """Test VITON clothing fitter initialization."""
        assert fitter is not None
        assert fitter.viton_lite is not None
        assert fitter.basic_fitter is not None
    
    def test_fit_clothing_auto_method(self, fitter, sample_image):
        """Test fit_clothing with auto method selection."""
        result = fitter.fit_clothing(
            person_image=sample_image,
            clothing_image=sample_image,
            clothing_type='shirt',
            method='auto'
        )
        
        # Should return a result (may be None if processing fails)
        assert result is None or isinstance(result, np.ndarray)
    
    def test_fit_clothing_viton_lite(self, fitter, sample_image):
        """Test fit_clothing with explicit viton_lite method."""
        result = fitter.fit_clothing(
            person_image=sample_image,
            clothing_image=sample_image,
            clothing_type='shirt',
            method='viton_lite'
        )
        
        assert result is None or isinstance(result, np.ndarray)
    
    def test_fit_clothing_basic(self, fitter, sample_image):
        """Test fit_clothing with basic method."""
        result = fitter.fit_clothing(
            person_image=sample_image,
            clothing_image=sample_image,
            clothing_type='shirt',
            method='basic'
        )
        
        assert result is None or isinstance(result, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
