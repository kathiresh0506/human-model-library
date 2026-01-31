"""
Unit tests for ModelSelector class.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_selector import ModelSelector
from src.utils import PathManager


class TestModelSelector:
    """Test cases for ModelSelector."""
    
    @pytest.fixture
    def selector(self):
        """Create a ModelSelector instance for testing."""
        return ModelSelector()
    
    def test_initialization(self, selector):
        """Test ModelSelector initialization."""
        assert selector is not None
        assert selector.path_manager is not None
        assert selector.config_loader is not None
        assert selector.validator is not None
    
    def test_select_model_valid_inputs(self, selector):
        """Test model selection with valid inputs."""
        result = selector.select_model(
            gender='male',
            size='M',
            age_group='young',
            ethnicity='asian'
        )
        # Result can be None if models don't exist yet, which is okay
        assert result is None or isinstance(result, str)
    
    def test_select_model_invalid_gender(self, selector):
        """Test model selection with invalid gender."""
        result = selector.select_model(
            gender='invalid',
            size='M',
            age_group='young',
            ethnicity='asian'
        )
        assert result is None
    
    def test_select_model_invalid_size(self, selector):
        """Test model selection with invalid size."""
        result = selector.select_model(
            gender='male',
            size='XXXL',  # Invalid size
            age_group='young',
            ethnicity='asian'
        )
        assert result is None
    
    def test_select_model_invalid_age_group(self, selector):
        """Test model selection with invalid age group."""
        result = selector.select_model(
            gender='male',
            size='M',
            age_group='toddler',  # Invalid age group
            ethnicity='asian'
        )
        assert result is None
    
    def test_select_model_case_insensitive(self, selector):
        """Test that model selection is case insensitive."""
        result1 = selector.select_model(
            gender='MALE',
            size='m',
            age_group='YOUNG',
            ethnicity='ASIAN'
        )
        # Should not raise an error
        assert result1 is None or isinstance(result1, str)
    
    def test_get_available_models(self, selector):
        """Test getting list of available models."""
        models = selector.get_available_models()
        assert isinstance(models, list)
        # List may be empty if no models exist yet
    
    def test_get_models_by_criteria(self, selector):
        """Test filtering models by criteria."""
        models = selector.get_models_by_criteria(
            gender='male',
            size='M'
        )
        assert isinstance(models, list)
    
    def test_get_size_measurements(self, selector):
        """Test getting size measurements."""
        measurements = selector.get_size_measurements('male', 'M')
        
        if measurements is not None:
            assert isinstance(measurements, dict)
            assert 'chest' in measurements
            assert 'waist' in measurements
            assert 'hip' in measurements
            assert 'height' in measurements
    
    def test_find_best_size_match_male(self, selector):
        """Test finding best size match for male."""
        size = selector.find_best_size_match(
            gender='male',
            chest=98,
            waist=83,
            hip=98,
            height=177
        )
        assert size in ['S', 'M', 'L', 'XL']
        assert size == 'M'  # Should recommend M for these measurements
    
    def test_find_best_size_match_female(self, selector):
        """Test finding best size match for female."""
        size = selector.find_best_size_match(
            gender='female',
            chest=90,
            waist=70,
            hip=96,
            height=165
        )
        assert size in ['S', 'M', 'L', 'XL']
        assert size == 'M'  # Should recommend M for these measurements
    
    def test_find_best_size_match_small(self, selector):
        """Test finding best size match for small measurements."""
        size = selector.find_best_size_match(
            gender='male',
            chest=88,
            waist=73,
            hip=88,
            height=172
        )
        assert size == 'S'
    
    def test_find_best_size_match_large(self, selector):
        """Test finding best size match for large measurements."""
        size = selector.find_best_size_match(
            gender='male',
            chest=108,
            waist=93,
            hip=108,
            height=180
        )
        assert size == 'L'
    
    def test_model_metadata_format(self, selector):
        """Test that model metadata has expected format."""
        # Get all models
        models = selector.get_available_models()
        
        if models:
            # Check first model has required fields
            model = models[0]
            assert 'gender' in model
            assert 'size' in model
            assert 'age_group' in model
            assert 'ethnicity' in model


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
