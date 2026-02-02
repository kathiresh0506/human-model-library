"""
Tests for IDM-VTON integration.

Simple tests to verify the module loads and basic functionality works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_module_imports():
    """Test that the idm_vton module can be imported."""
    from idm_vton import IDMVTONClient, virtual_tryon
    assert IDMVTONClient is not None
    assert virtual_tryon is not None


def test_client_initialization():
    """Test that IDMVTONClient can be initialized."""
    from idm_vton import IDMVTONClient
    
    client = IDMVTONClient()
    assert client.space_id == "yisol/IDM-VTON"
    assert client.client is None  # Not connected yet
    
    # Test with custom space_id
    custom_client = IDMVTONClient(space_id="custom/space")
    assert custom_client.space_id == "custom/space"


def test_seed_parameter_default_value():
    """Test that the default seed parameter is within valid range (<= 40)."""
    import inspect
    from idm_vton import IDMVTONClient
    
    # Get the default value of seed parameter from try_on method signature
    sig = inspect.signature(IDMVTONClient.try_on)
    seed_param = sig.parameters['seed']
    seed_default = seed_param.default
    
    # Verify seed default is within valid range (API max is 40)
    assert seed_default <= 40, f"Default seed value {seed_default} exceeds maximum allowed value of 40"
    assert seed_default >= 0, f"Default seed value {seed_default} should be non-negative"


def test_model_selector_real_with_real_humans_dir():
    """Test that RealModelSelector works with real_humans directory."""
    from model_selector_real import RealModelSelector
    
    # Test with realistic directory
    selector1 = RealModelSelector(base_dir='models/realistic')
    assert selector1.base_dir.name == 'realistic'
    
    # Test with real_humans directory  
    selector2 = RealModelSelector(base_dir='models/real_humans')
    assert selector2.base_dir.name == 'real_humans'


def test_model_selector_list_available():
    """Test that list_available method works."""
    from model_selector_real import RealModelSelector
    
    selector = RealModelSelector(base_dir='models/realistic')
    available = selector.list_available()
    
    assert 'male' in available
    assert 'female' in available
    assert 'S' in available['male']
    assert 'M' in available['male']
    assert 'L' in available['male']
    assert 'XL' in available['male']


if __name__ == '__main__':
    # Run tests manually
    print("Running IDM-VTON tests...")
    
    try:
        test_module_imports()
        print("✓ Module imports test passed")
    except Exception as e:
        print(f"✗ Module imports test failed: {e}")
    
    try:
        test_client_initialization()
        print("✓ Client initialization test passed")
    except Exception as e:
        print(f"✗ Client initialization test failed: {e}")
    
    try:
        test_seed_parameter_default_value()
        print("✓ Seed parameter default value test passed")
    except Exception as e:
        print(f"✗ Seed parameter default value test failed: {e}")
    
    try:
        test_model_selector_real_with_real_humans_dir()
        print("✓ Model selector test passed")
    except Exception as e:
        print(f"✗ Model selector test failed: {e}")
    
    try:
        test_model_selector_list_available()
        print("✓ List available test passed")
    except Exception as e:
        print(f"✗ List available test failed: {e}")
    
    print("\nAll tests completed!")
