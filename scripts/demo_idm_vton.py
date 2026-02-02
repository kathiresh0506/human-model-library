#!/usr/bin/env python3
"""
Demo script for IDM-VTON virtual try-on with real human photos.

Produces Myntra-quality results using state-of-the-art IDM-VTON model
hosted on Hugging Face Spaces.

Usage:
    python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png
    python scripts/demo_idm_vton.py --person path/to/person.jpg --clothing path/to/shirt.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from idm_vton import IDMVTONClient, virtual_tryon
from model_selector_real import RealModelSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="IDM-VTON Virtual Try-On Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use auto-selected model
  python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png
  
  # Specify gender and size
  python scripts/demo_idm_vton.py --clothing shirt.jpg --gender male --size M
  
  # Use specific person image
  python scripts/demo_idm_vton.py --person model.jpg --clothing shirt.jpg
  
  # Specify clothing type
  python scripts/demo_idm_vton.py --clothing dress.jpg --clothing-type dresses
        """
    )
    
    parser.add_argument(
        "--person",
        type=str,
        help="Path to person image (optional, will auto-select if not provided)"
    )
    parser.add_argument(
        "--clothing",
        type=str,
        required=True,
        help="Path to clothing image (required)"
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female"],
        default="male",
        help="Gender for model selection (default: male)"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["S", "M", "L", "XL"],
        default="M",
        help="Size for model selection (default: M)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/idm_vton_result.png",
        help="Output path for result (default: output/idm_vton_result.png)"
    )
    parser.add_argument(
        "--clothing-type",
        type=str,
        choices=["upper_body", "lower_body", "dresses"],
        default="upper_body",
        help="Type of clothing (default: upper_body)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="models/real_humans",
        help="Base directory for real human photos (default: models/real_humans)"
    )
    
    args = parser.parse_args()
    
    # Validate clothing image exists
    if not Path(args.clothing).exists():
        logger.error(f"Clothing image not found: {args.clothing}")
        sys.exit(1)
    
    # Select or validate person image
    if args.person:
        # Use provided person image
        if not Path(args.person).exists():
            logger.error(f"Person image not found: {args.person}")
            sys.exit(1)
        
        person_image = args.person
        logger.info(f"Using provided person image: {person_image}")
    else:
        # Auto-select from model library
        logger.info(f"Auto-selecting model: {args.gender}/{args.size}")
        
        try:
            # Try real_humans directory first
            selector = RealModelSelector(base_dir=args.base_dir)
            person_image = selector.select_model(args.gender, args.size)
            
            if not person_image:
                # Fallback to realistic directory
                logger.warning(f"No models found in {args.base_dir}, trying models/realistic")
                selector = RealModelSelector(base_dir="models/realistic")
                person_image = selector.select_model(args.gender, args.size)
            
            if not person_image:
                logger.error(f"No models available for {args.gender}/{args.size}")
                logger.info("\nTo add models:")
                logger.info("1. Run: python scripts/download_real_human_photos.py")
                logger.info("2. Or manually add photos to the appropriate directories")
                sys.exit(1)
            
            logger.info(f"✓ Selected model: {person_image}")
            
        except FileNotFoundError as e:
            logger.error(f"Model directory not found: {e}")
            logger.info("\nTo set up models:")
            logger.info("1. Run: python scripts/download_real_human_photos.py")
            logger.info("2. Add real human photos to the directories")
            sys.exit(1)
    
    # Display configuration
    logger.info("\n" + "=" * 60)
    logger.info("IDM-VTON Virtual Try-On Configuration")
    logger.info("=" * 60)
    logger.info(f"Person Image:    {person_image}")
    logger.info(f"Clothing Image:  {args.clothing}")
    logger.info(f"Clothing Type:   {args.clothing_type}")
    logger.info(f"Output Path:     {args.output}")
    logger.info("=" * 60)
    
    # Run virtual try-on
    logger.info("\n🚀 Starting IDM-VTON virtual try-on...")
    logger.info("⏳ This may take 30-60 seconds (first run may be slower)...")
    
    try:
        result = virtual_tryon(
            person_image=person_image,
            clothing_image=args.clothing,
            output_path=args.output,
            clothing_type=args.clothing_type
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS!")
        logger.info("=" * 60)
        logger.info(f"Result saved to: {result}")
        logger.info(f"\nOpen {result} to see your Myntra-quality try-on result!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("❌ Try-on failed")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Check your internet connection")
        logger.info("2. Ensure images are valid and accessible")
        logger.info("3. Try again (Hugging Face Spaces may be temporarily busy)")
        logger.info("4. Check if gradio_client is installed: pip install gradio_client>=0.10.0")
        sys.exit(1)


if __name__ == "__main__":
    main()
