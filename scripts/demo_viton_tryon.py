#!/usr/bin/env python3
"""
Demo script for VITON-based virtual try-on.

Demonstrates VITON-HD/VITON-Lite functionality with side-by-side comparison.
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clothing_fitter_viton import VITONClothingFitter
from src.utils import ImageLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_comparison_image(person_image: np.ndarray,
                           clothing_image: np.ndarray,
                           result_image: np.ndarray) -> np.ndarray:
    """
    Create side-by-side comparison image.
    
    Args:
        person_image: Original person image
        clothing_image: Clothing image used
        result_image: Try-on result
        
    Returns:
        Combined comparison image
    """
    # Resize all to same height
    target_height = 768
    
    def resize_to_height(img, height):
        aspect = img.shape[1] / img.shape[0]
        new_width = int(height * aspect)
        return np.array(Image.fromarray(img).resize((new_width, height), Image.LANCZOS))
    
    person_resized = resize_to_height(person_image, target_height)
    clothing_resized = resize_to_height(clothing_image, target_height)
    result_resized = resize_to_height(result_image, target_height)
    
    # Make clothing image same width as others (pad if needed)
    max_width = max(person_resized.shape[1], result_resized.shape[1])
    
    def pad_to_width(img, width):
        if img.shape[1] >= width:
            return img
        pad_width = width - img.shape[1]
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=255)
    
    person_padded = pad_to_width(person_resized, max_width)
    clothing_padded = pad_to_width(clothing_resized, max_width)
    result_padded = pad_to_width(result_resized, max_width)
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    
    def add_label(img, text):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font = None
        try:
            # Try to load a system font (cross-platform)
            import platform
            system = platform.system()
            if system == 'Windows':
                font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 24)
            elif system == 'Darwin':  # macOS
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            else:  # Linux
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            # Fallback to default font
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background rectangle
        x = (pil_img.width - text_width) // 2
        y = 10
        draw.rectangle([x - 10, y - 5, x + text_width + 10, y + text_height + 5], fill=(0, 0, 0, 200))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        return np.array(pil_img)
    
    person_labeled = add_label(person_padded, "Person")
    clothing_labeled = add_label(clothing_padded, "Clothing")
    result_labeled = add_label(result_padded, "Result")
    
    # Concatenate horizontally
    comparison = np.hstack([person_labeled, clothing_labeled, result_labeled])
    
    return comparison


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='VITON virtual try-on demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demo with default model and clothing
  python scripts/demo_viton_tryon.py
  
  # Use specific person and clothing images
  python scripts/demo_viton_tryon.py \\
    --person models/male/M/young/asian/front.png \\
    --clothing samples/clothing/tshirt_blue.png
  
  # Try VITON-HD if weights are available
  python scripts/demo_viton_tryon.py --method viton_hd
  
  # Use VITON-Lite for fast results
  python scripts/demo_viton_tryon.py --method viton_lite
  
  # Save to specific output
  python scripts/demo_viton_tryon.py --output my_result.jpg
  
  # Try on multiple clothing items
  python scripts/demo_viton_tryon.py \\
    --person models/male/M/young/asian/front.png \\
    --clothing samples/clothing/tshirt_blue.png samples/clothing/tshirt_red.png
        """
    )
    
    parser.add_argument(
        '--person',
        help='Path to person/model image'
    )
    parser.add_argument(
        '--clothing',
        nargs='+',
        help='Path(s) to clothing image(s)'
    )
    parser.add_argument(
        '--clothing-type',
        default='shirt',
        choices=['shirt', 'top', 'jacket', 'pants', 'dress'],
        help='Type of clothing (default: shirt)'
    )
    parser.add_argument(
        '--method',
        default='auto',
        choices=['auto', 'viton_hd', 'viton_lite', 'basic'],
        help='Try-on method (default: auto)'
    )
    parser.add_argument(
        '--output',
        default='output/viton_result.jpg',
        help='Output path for result'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Also save side-by-side comparison image'
    )
    
    args = parser.parse_args()
    
    # Use defaults if not specified
    if not args.person:
        # Try to find a model
        model_paths = list(Path('models').glob('*/M/*/*/front.png'))
        if not model_paths:
            logger.error("No model images found. Run: python scripts/create_sample_models.py")
            return 1
        args.person = str(model_paths[0])
        logger.info(f"Using default person: {args.person}")
    
    if not args.clothing:
        # Try to find clothing
        clothing_paths = list(Path('samples/clothing').glob('*.png'))
        if not clothing_paths:
            logger.error("No clothing images found. Run: python scripts/create_sample_clothing.py")
            return 1
        args.clothing = [str(clothing_paths[0])]
        logger.info(f"Using default clothing: {args.clothing[0]}")
    
    # Load images
    loader = ImageLoader()
    
    logger.info(f"Loading person image: {args.person}")
    person_image = loader.load_image(args.person)
    if person_image is None:
        logger.error(f"Failed to load person image: {args.person}")
        return 1
    
    # Initialize fitter
    logger.info(f"Initializing VITON fitter with method: {args.method}")
    fitter = VITONClothingFitter()
    
    # Process each clothing item
    for i, clothing_path in enumerate(args.clothing):
        logger.info(f"Loading clothing image: {clothing_path}")
        clothing_image = loader.load_image(clothing_path)
        if clothing_image is None:
            logger.error(f"Failed to load clothing image: {clothing_path}")
            continue
        
        # Perform try-on
        logger.info(f"Performing virtual try-on...")
        result = fitter.fit_clothing(
            person_image,
            clothing_image,
            args.clothing_type,
            args.method
        )
        
        if result is None:
            logger.error("Virtual try-on failed")
            continue
        
        # Determine output path
        if len(args.clothing) > 1:
            output_path = Path(args.output)
            output_path = output_path.parent / f"{output_path.stem}_{i+1}{output_path.suffix}"
        else:
            output_path = Path(args.output)
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save result
        logger.info(f"Saving result to: {output_path}")
        loader.save_image(result, str(output_path))
        
        # Create comparison if requested
        if args.comparison:
            comparison_path = output_path.parent / f"comparison_{output_path.name}"
            logger.info(f"Creating comparison image: {comparison_path}")
            
            comparison = create_comparison_image(person_image, clothing_image, result)
            loader.save_image(comparison, str(comparison_path))
        
        logger.info(f"✓ Try-on completed successfully: {output_path}")
    
    logger.info("All done! Check the output folder for results.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
