#!/usr/bin/env python3
"""
One-command setup for VITON-HD integration.

This script:
1. Checks system requirements
2. Installs dependencies
3. Downloads model weights (optional)
4. Downloads sample models (optional)
5. Runs test to verify everything works
"""
import argparse
import logging
import sys
import subprocess
from pathlib import Path
import platform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VITONSetup:
    """Setup and configuration for VITON system."""
    
    def __init__(self, base_dir: str = '.'):
        """
        Initialize setup.
        
        Args:
            base_dir: Base directory for project
        """
        self.base_dir = Path(base_dir)
    
    def check_system_requirements(self) -> dict:
        """
        Check system requirements.
        
        Returns:
            Dictionary with requirement status
        """
        logger.info("Checking system requirements...")
        
        requirements = {
            'python_version': sys.version_info >= (3, 8),
            'platform': platform.system(),
            'gpu_available': False,
        }
        
        # Check for CUDA/GPU
        try:
            import torch
            requirements['pytorch_available'] = True
            requirements['gpu_available'] = torch.cuda.is_available()
            requirements['pytorch_version'] = torch.__version__
        except ImportError:
            requirements['pytorch_available'] = False
            requirements['pytorch_version'] = None
        
        # Check for required packages
        required_packages = ['numpy', 'opencv-python', 'Pillow', 'mediapipe']
        requirements['packages'] = {}
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                requirements['packages'][package] = True
            except ImportError:
                requirements['packages'][package] = False
        
        return requirements
    
    def print_requirements(self, requirements: dict):
        """Print requirements check results."""
        logger.info("=" * 60)
        logger.info("System Requirements Check")
        logger.info("=" * 60)
        
        status = "✓" if requirements['python_version'] else "✗"
        logger.info(f"{status} Python version: {sys.version.split()[0]}")
        
        logger.info(f"  Platform: {requirements['platform']}")
        
        status = "✓" if requirements.get('pytorch_available') else "○"
        pytorch_info = requirements.get('pytorch_version', 'Not installed')
        logger.info(f"{status} PyTorch: {pytorch_info}")
        
        if requirements.get('gpu_available'):
            logger.info("  ✓ GPU/CUDA available - VITON-HD will run faster")
        else:
            logger.info("  ○ No GPU detected - will use CPU (slower)")
        
        logger.info("\nRequired packages:")
        for package, available in requirements['packages'].items():
            status = "✓" if available else "✗"
            logger.info(f"{status} {package}")
        
        logger.info("=" * 60)
    
    def install_dependencies(self, include_pytorch: bool = False) -> bool:
        """
        Install required dependencies.
        
        Args:
            include_pytorch: Whether to install PyTorch (for full VITON-HD)
            
        Returns:
            True if installation successful
        """
        logger.info("Installing dependencies...")
        
        try:
            # Install basic requirements
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                check=True,
                cwd=self.base_dir
            )
            
            # Install PyTorch if requested
            if include_pytorch:
                logger.info("Installing PyTorch (this may take a while)...")
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'],
                    check=True
                )
            
            logger.info("✓ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def download_weights(self, include_optional: bool = False) -> bool:
        """
        Download model weights.
        
        Args:
            include_optional: Download optional weights too
            
        Returns:
            True if download successful
        """
        logger.info("Downloading model weights...")
        
        try:
            cmd = [sys.executable, 'scripts/download_viton_models.py']
            if include_optional:
                cmd.append('--all')
            
            subprocess.run(cmd, check=False, cwd=self.base_dir)
            
            logger.info("✓ Weight download attempted (may need manual setup)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            return False
    
    def download_sample_models(self) -> bool:
        """
        Download sample model photos.
        
        Returns:
            True if download successful
        """
        logger.info("Generating sample model photos...")
        
        try:
            subprocess.run(
                [sys.executable, 'scripts/download_sample_models.py'],
                check=True,
                cwd=self.base_dir
            )
            
            logger.info("✓ Sample models generated")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate sample models: {e}")
            return False
    
    def run_test(self) -> bool:
        """
        Run a test to verify setup.
        
        Returns:
            True if test passes
        """
        logger.info("Running verification test...")
        
        try:
            # Import to check modules load
            sys.path.insert(0, str(self.base_dir))
            
            from src.viton_lite import VITONLiteFitter
            from src.clothing_fitter_viton import VITONClothingFitter
            
            # Create instances
            lite_fitter = VITONLiteFitter()
            viton_fitter = VITONClothingFitter()
            
            logger.info("✓ All modules loaded successfully")
            logger.info("✓ System is ready for virtual try-on!")
            return True
            
        except Exception as e:
            logger.error(f"Verification test failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Setup VITON-HD integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick setup (VITON-Lite only, no PyTorch needed)
  python scripts/setup_viton.py
  
  # Full setup with PyTorch for VITON-HD
  python scripts/setup_viton.py --full
  
  # Just check requirements
  python scripts/setup_viton.py --check-only
  
  # Skip weight download
  python scripts/setup_viton.py --no-weights

Recommendation:
- For immediate use: Quick setup (VITON-Lite works without weights)
- For production quality: Full setup with VITON-HD weights
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full setup including PyTorch for VITON-HD'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check requirements, don\'t install anything'
    )
    parser.add_argument(
        '--no-weights',
        action='store_true',
        help='Skip model weight download'
    )
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Skip sample model generation'
    )
    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    setup = VITONSetup(args.base_dir)
    
    # Check requirements
    requirements = setup.check_system_requirements()
    setup.print_requirements(requirements)
    
    if args.check_only:
        return 0
    
    # Verify Python version
    if not requirements['python_version']:
        logger.error("Python 3.8 or higher is required")
        return 1
    
    # Install dependencies
    if not setup.install_dependencies(include_pytorch=args.full):
        logger.error("Failed to install dependencies")
        return 1
    
    # Download weights (if not skipped)
    if not args.no_weights:
        setup.download_weights(include_optional=args.full)
    
    # Download sample models (if not skipped)
    if not args.no_samples:
        setup.download_sample_models()
    
    # Run verification test
    if not setup.run_test():
        logger.warning("Verification test failed, but setup may still work")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Setup Complete!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Test with: python scripts/demo_viton_tryon.py")
    logger.info("2. Read documentation: docs/VITON_SETUP.md")
    logger.info("3. Check API: uvicorn api.app:app")
    
    if not args.full:
        logger.info("\nNote: Using VITON-Lite (lightweight mode)")
        logger.info("For full VITON-HD: run with --full flag")
    
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
