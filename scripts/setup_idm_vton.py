#!/usr/bin/env python3
"""
One-command setup for IDM-VTON integration.

Sets up everything needed for Myntra-quality virtual try-on:
1. Installs gradio_client dependency
2. Downloads/creates directory structure for real human photos
3. Tests connection to IDM-VTON on Hugging Face

Usage:
    python scripts/setup_idm_vton.py
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print a formatted header."""
    logger.info("\n" + "=" * 60)
    logger.info(text)
    logger.info("=" * 60)


def main():
    """Main setup function."""
    print_header("Setting up IDM-VTON for Myntra-Quality Virtual Try-On")
    
    # Step 1: Install gradio_client
    print_header("Step 1/3: Installing gradio_client")
    logger.info("Installing gradio_client>=0.10.0...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gradio_client>=0.10.0"],
            check=True
        )
        logger.info("✅ gradio_client installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install gradio_client: {e}")
        logger.info("\nTry manually installing:")
        logger.info("  pip install gradio_client>=0.10.0")
        sys.exit(1)
    
    # Step 2: Download/create real human photos directory structure
    print_header("Step 2/3: Setting up real human photos directory")
    
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_real_human_photos.py"
    
    if download_script.exists():
        logger.info("Running download_real_human_photos.py...")
        try:
            subprocess.run(
                [sys.executable, str(download_script)],
                check=True
            )
            logger.info("✅ Directory structure created!")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠ Script execution had issues: {e}")
            logger.info("Continuing with setup...")
    else:
        logger.warning(f"⚠ Download script not found: {download_script}")
        logger.info("Creating basic directory structure...")
        
        # Create basic structure manually
        base_dir = Path("models/real_humans")
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                size_dir = base_dir / gender / size
                size_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Basic directory structure created!")
    
    # Step 3: Test IDM-VTON connection
    print_header("Step 3/3: Testing IDM-VTON connection")
    logger.info("Attempting to connect to Hugging Face Space...")
    logger.info("(This may take a moment on first connection)")
    
    try:
        from gradio_client import Client
        
        logger.info("Connecting to yisol/IDM-VTON...")
        client = Client("yisol/IDM-VTON")
        logger.info("✅ Successfully connected to IDM-VTON!")
        
    except ImportError as e:
        logger.error("❌ gradio_client not available")
        logger.info("\nPlease install it manually:")
        logger.info("  pip install gradio_client>=0.10.0")
        sys.exit(1)
        
    except Exception as e:
        logger.warning(f"⚠ Connection test failed: {e}")
        logger.info("\nThis might be temporary. Possible reasons:")
        logger.info("  - Hugging Face Spaces is busy")
        logger.info("  - Network connectivity issues")
        logger.info("  - Space is loading/initializing")
        logger.info("\nYou can try running the demo later:")
        logger.info("  python scripts/demo_idm_vton.py --clothing your_image.jpg")
    
    # Final summary
    print_header("Setup Complete!")
    
    logger.info("\n📋 Next Steps:")
    logger.info("\n1. Add real human photos:")
    logger.info("   - Navigate to: models/real_humans/")
    logger.info("   - Add photos to appropriate gender/size directories")
    logger.info("   - Follow README.md instructions in each directory")
    
    logger.info("\n2. (Optional) Use existing realistic models:")
    logger.info("   - Photos in models/realistic/ can also be used")
    logger.info("   - Specify --base-dir models/realistic in demo")
    
    logger.info("\n3. Run a demo:")
    logger.info("   python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png")
    
    logger.info("\n4. With custom images:")
    logger.info("   python scripts/demo_idm_vton.py --person your_model.jpg --clothing your_shirt.jpg")
    
    logger.info("\n5. Check the output:")
    logger.info("   - Results will be saved to: output/idm_vton_result.png")
    logger.info("   - Expect Myntra-quality virtual try-on!")
    
    print_header("🎉 Ready to use IDM-VTON!")


if __name__ == "__main__":
    main()
