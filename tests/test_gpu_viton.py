"""
Unit tests for RealModelSelector and GPU VITON-HD components.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_selector_real import RealModelSelector


class TestRealModelSelector:
    """Test cases for RealModelSelector."""
    
    @pytest.fixture
    def selector(self):
        """Create a RealModelSelector instance for testing."""
        return RealModelSelector()
    
    def test_initialization(self, selector):
        """Test RealModelSelector initialization."""
        assert selector is not None
        assert selector.base_dir.exists()
    
    def test_get_summary(self, selector):
        """Test getting model library summary."""
        summary = selector.get_summary()
        
        assert 'male' in summary
        assert 'female' in summary
        assert 'total' in summary
        
        # Check all sizes are present
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                assert size in summary[gender]
                assert isinstance(summary[gender][size], int)
    
    def test_count_models(self, selector):
        """Test model counting."""
        total = selector.count_models()
        assert isinstance(total, int)
        assert total >= 0
    
    def test_get_models_valid_inputs(self, selector):
        """Test getting models with valid inputs."""
        models = selector.get_models('male', 'M')
        assert isinstance(models, list)
        
        # If models exist, check they are Path objects
        for model in models:
            assert isinstance(model, Path)
            assert model.exists()
    
    def test_get_models_invalid_gender(self, selector):
        """Test getting models with invalid gender."""
        models = selector.get_models('invalid', 'M')
        assert len(models) == 0
    
    def test_get_all_sizes(self, selector):
        """Test getting all sizes for a gender."""
        result = selector.get_all_sizes('male')
        
        assert isinstance(result, dict)
        assert len(result) == 4  # S, M, L, XL
        
        for size in ['S', 'M', 'L', 'XL']:
            assert size in result
            assert isinstance(result[size], list)


class TestGPUVITON:
    """Test cases for GPU VITON-HD components."""
    
    def test_viton_gpu_import(self):
        """Test that VITON GPU module can be imported."""
        try:
            from src.viton_hd.viton_gpu import VITONHDModel, VITONHDGenerator
            assert VITONHDModel is not None
            assert VITONHDGenerator is not None
        except ImportError as e:
            pytest.skip(f"PyTorch not installed: {e}")
    
    def test_openpose_gpu_import(self):
        """Test that OpenPose GPU module can be imported."""
        try:
            from src.viton_hd.openpose_gpu import OpenPoseGPU
            assert OpenPoseGPU is not None
        except ImportError as e:
            pytest.skip(f"PyTorch not installed: {e}")
    
    def test_human_parsing_gpu_import(self):
        """Test that human parsing GPU module can be imported."""
        try:
            from src.viton_hd.human_parsing_gpu import HumanParsingGPU
            assert HumanParsingGPU is not None
        except ImportError as e:
            pytest.skip(f"PyTorch not installed: {e}")
    
    def test_geometric_matching_import(self):
        """Test that geometric matching module can be imported."""
        try:
            from src.viton_hd.geometric_matching import ClothWarpingGPU
            assert ClothWarpingGPU is not None
        except ImportError as e:
            pytest.skip(f"PyTorch not installed: {e}")


class TestDownloadScripts:
    """Test cases for download scripts."""
    
    def test_download_real_models_script_exists(self):
        """Test that download_real_models.py exists."""
        script_path = Path('scripts/download_real_models.py')
        assert script_path.exists()
    
    def test_download_viton_weights_script_exists(self):
        """Test that download_viton_weights.py exists."""
        script_path = Path('scripts/download_viton_weights.py')
        assert script_path.exists()
    
    def test_demo_viton_gpu_script_exists(self):
        """Test that demo_viton_gpu.py exists."""
        script_path = Path('scripts/demo_viton_gpu.py')
        assert script_path.exists()


class TestMetadata:
    """Test cases for metadata."""
    
    def test_metadata_exists(self):
        """Test that metadata.json exists."""
        metadata_path = Path('models/realistic/metadata.json')
        assert metadata_path.exists()
    
    def test_metadata_valid_json(self):
        """Test that metadata.json is valid JSON."""
        import json
        
        metadata_path = Path('models/realistic/metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert 'version' in metadata
        assert 'models' in metadata
        assert 'male' in metadata['models']
        assert 'female' in metadata['models']
