"""
Demo script for virtual try-on functionality.
Tests the complete pipeline with sample models and clothing.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_selector import ModelSelector
from src.clothing_fitter import ClothingFitter
from src.utils import ImageLoader


def create_comparison_image(model_img, clothing_img, result_img):
    """
    Create a side-by-side comparison image.
    
    Args:
        model_img: Original model image (numpy array)
        clothing_img: Clothing image (numpy array)
        result_img: Result image (numpy array)
    
    Returns:
        Combined comparison image
    """
    # Convert to PIL for easier manipulation
    model_pil = Image.fromarray(model_img.astype('uint8'))
    result_pil = Image.fromarray(result_img.astype('uint8'))
    
    # Handle clothing image with transparency
    if len(clothing_img.shape) == 3 and clothing_img.shape[2] == 4:
        # RGBA image
        clothing_pil = Image.fromarray(clothing_img.astype('uint8'), 'RGBA')
        # Create white background
        bg = Image.new('RGB', clothing_pil.size, (255, 255, 255))
        bg.paste(clothing_pil, mask=clothing_pil.split()[3])
        clothing_pil = bg
    else:
        clothing_pil = Image.fromarray(clothing_img.astype('uint8'))
    
    # Resize all to same height for comparison
    target_height = 600
    
    model_aspect = model_pil.width / model_pil.height
    model_width = int(target_height * model_aspect)
    model_pil = model_pil.resize((model_width, target_height), Image.LANCZOS)
    
    clothing_aspect = clothing_pil.width / clothing_pil.height
    clothing_width = int(target_height * clothing_aspect)
    clothing_pil = clothing_pil.resize((clothing_width, target_height), Image.LANCZOS)
    
    result_aspect = result_pil.width / result_pil.height
    result_width = int(target_height * result_aspect)
    result_pil = result_pil.resize((result_width, target_height), Image.LANCZOS)
    
    # Create combined image
    total_width = model_width + clothing_width + result_width + 60  # 20px spacing between
    combined = Image.new('RGB', (total_width, target_height + 80), (255, 255, 255))
    
    # Paste images
    combined.paste(model_pil, (20, 60))
    combined.paste(clothing_pil, (model_width + 40, 60))
    combined.paste(result_pil, (model_width + clothing_width + 60, 60))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    
    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((20, 20), "Original Model", fill=(0, 0, 0), font=font)
    draw.text((model_width + 40, 20), "Clothing", fill=(0, 0, 0), font=font)
    draw.text((model_width + clothing_width + 60, 20), "Result", fill=(0, 0, 0), font=font)
    
    return np.array(combined)


def main():
    parser = argparse.ArgumentParser(description='Demo virtual try-on')
    parser.add_argument('--gender', type=str, default='male', 
                       choices=['male', 'female'],
                       help='Gender of the model')
    parser.add_argument('--size', type=str, default='M',
                       choices=['S', 'M', 'L', 'XL'],
                       help='Size of the model')
    parser.add_argument('--age_group', type=str, default='young',
                       choices=['young', 'middle', 'senior'],
                       help='Age group of the model')
    parser.add_argument('--ethnicity', type=str, default='asian',
                       choices=['asian', 'african', 'caucasian', 'hispanic', 'middle_eastern'],
                       help='Ethnicity of the model')
    parser.add_argument('--clothing', type=str,
                       default='samples/clothing/tshirt_blue.png',
                       help='Path to clothing image')
    parser.add_argument('--clothing_type', type=str, default='shirt',
                       help='Type of clothing (shirt, pants, dress, etc.)')
    parser.add_argument('--output', type=str, default='output/demo_result.png',
                       help='Output path for result image')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Virtual Try-On Demo")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    selector = ModelSelector()
    fitter = ClothingFitter()
    loader = ImageLoader()
    
    # Select model
    print(f"\n2. Selecting model: {args.gender}/{args.size}/{args.age_group}/{args.ethnicity}")
    try:
        model_path = selector.select_model(
            gender=args.gender,
            size=args.size,
            age_group=args.age_group,
            ethnicity=args.ethnicity
        )
        model_image_path = f"{model_path}/front.png"
        print(f"   Model path: {model_image_path}")
    except Exception as e:
        print(f"   ❌ Error selecting model: {e}")
        return
    
    # Load images
    print(f"\n3. Loading images...")
    model_img = loader.load_image(model_image_path)
    if model_img is None:
        print(f"   ❌ Failed to load model image from {model_image_path}")
        return
    print(f"   ✓ Model image loaded: {model_img.shape}")
    
    # Handle clothing image
    base_dir = Path(__file__).parent.parent
    clothing_path = args.clothing
    if not os.path.isabs(clothing_path):
        clothing_path = base_dir / clothing_path
    
    # Load clothing with transparency support
    try:
        clothing_pil = Image.open(clothing_path)
        if clothing_pil.mode == 'RGBA':
            clothing_img = np.array(clothing_pil)
        else:
            clothing_img = np.array(clothing_pil.convert('RGB'))
    except Exception as e:
        print(f"   ❌ Failed to load clothing image: {e}")
        return
    print(f"   ✓ Clothing image loaded: {clothing_img.shape}")
    
    # Convert RGBA to RGB for processing if needed
    if len(clothing_img.shape) == 3 and clothing_img.shape[2] == 4:
        # Create white background and blend
        rgb_clothing = np.zeros((clothing_img.shape[0], clothing_img.shape[1], 3), dtype=np.uint8)
        rgb_clothing[:, :] = 255  # White background
        alpha = clothing_img[:, :, 3:4] / 255.0
        rgb_clothing = (clothing_img[:, :, :3] * alpha + rgb_clothing * (1 - alpha)).astype(np.uint8)
        clothing_img_for_fitting = rgb_clothing
    else:
        clothing_img_for_fitting = clothing_img[:, :, :3] if len(clothing_img.shape) == 3 else clothing_img
    
    # Perform try-on
    print(f"\n4. Performing virtual try-on...")
    result_img = fitter.fit_clothing(
        model_image=model_img,
        clothing_image=clothing_img_for_fitting,
        clothing_type=args.clothing_type
    )
    
    if result_img is None:
        print("   ❌ Try-on failed")
        return
    print("   ✓ Try-on completed successfully")
    
    # Save result
    print(f"\n5. Saving results to {args.output}...")
    output_path = base_dir / args.output if not os.path.isabs(args.output) else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    loader.save_image(result_img, str(output_path))
    print(f"   ✓ Result saved to {output_path}")
    
    # Create and save comparison
    comparison_path = output_path.parent / f"comparison_{output_path.name}"
    comparison_img = create_comparison_image(model_img, clothing_img, result_img)
    loader.save_image(comparison_img, str(comparison_path))
    print(f"   ✓ Comparison saved to {comparison_path}")
    
    print("\n" + "=" * 60)
    print("✨ Demo completed successfully!")
    print("=" * 60)
    print(f"\nView results:")
    print(f"  - Result: {output_path}")
    print(f"  - Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
