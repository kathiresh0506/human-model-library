#!/usr/bin/env python3
"""
Demo script for GPU-accelerated VITON-HD virtual try-on.

Tests VITON-HD with real human photos on GPU (RTX 3050, A100, etc).
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        
        logger.info("=" * 60)
        logger.info("GPU Check")
        logger.info("=" * 60)
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.version.cuda}")
            logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            logger.info(f"✓ Memory: {props.total_memory / 1e9:.1f} GB")
            logger.info(f"✓ Compute Capability: {props.major}.{props.minor}")
            
            return True
        else:
            logger.warning("✗ CUDA not available, will use CPU")
            return False
            
    except ImportError:
        logger.error("PyTorch not installed")
        logger.info("Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False


def run_demo(args):
    """
    Run VITON-HD demo.
    
    Args:
        args: Command line arguments
    """
    from src.viton_hd.viton_gpu import VITONHDModel
    from src.model_selector_real import RealModelSelector
    from src.utils import ImageLoader
    
    logger.info("=" * 60)
    logger.info("VITON-HD GPU Demo")
    logger.info("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu()
    device = args.device if has_gpu else 'cpu'
    
    # Initialize VITON-HD model
    logger.info("\nInitializing VITON-HD model...")
    weights_path = Path('weights/viton_hd/generator.pth')
    
    if not weights_path.exists():
        logger.warning(f"Weights not found at {weights_path}")
        logger.info("Download with: python scripts/download_viton_weights.py --all")
        logger.info("Continuing with random initialization (for testing only)...")
        weights_path = None
    
    try:
        viton = VITONHDModel(device=device, weights_path=str(weights_path) if weights_path else None)
    except ImportError as e:
        logger.error(f"Error: {e}")
        logger.info("\nInstall PyTorch with:")
        logger.info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # Print device info
    device_info = viton.get_device_info()
    logger.info("\nDevice Information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")
    
    # Load or select person image
    if args.person:
        person_path = args.person
        logger.info(f"\nUsing person image: {person_path}")
    else:
        # Select from library
        logger.info("\nNo person image specified, selecting from library...")
        selector = RealModelSelector()
        
        if args.gender and args.size:
            person_path = selector.select_model(args.gender, args.size)
        else:
            person_path = selector.select_random_model()
        
        if not person_path:
            logger.error("No models found in library")
            logger.info("Download models with: python scripts/download_real_models.py")
            return
        
        logger.info(f"Selected: {person_path}")
    
    # Load person image
    loader = ImageLoader()
    person_image = loader.load_image(person_path)
    
    if person_image is None:
        logger.error(f"Failed to load person image: {person_path}")
        return
    
    logger.info(f"Person image size: {person_image.shape}")
    
    # Load clothing image
    if not args.clothing:
        logger.error("No clothing image specified")
        logger.info("Usage: --clothing path/to/clothing.png")
        return
    
    clothing_path = args.clothing
    logger.info(f"\nLoading clothing: {clothing_path}")
    
    clothing_image = loader.load_image(clothing_path)
    
    if clothing_image is None:
        logger.error(f"Failed to load clothing image: {clothing_path}")
        return
    
    logger.info(f"Clothing image size: {clothing_image.shape}")
    
    # Perform try-on
    logger.info("\n" + "=" * 60)
    logger.info("Performing Virtual Try-On")
    logger.info("=" * 60)
    
    if args.batch_size > 1:
        logger.info(f"Batch mode: {args.batch_size} copies")
        
        # Create batches
        person_batch = [person_image] * args.batch_size
        clothing_batch = [clothing_image] * args.batch_size
        
        # Run batch inference
        results = viton.try_on_batch(person_batch, clothing_batch)
        
        if results and results[0] is not None:
            result_image = results[0]
            logger.info("✓ Batch try-on completed")
        else:
            logger.error("Batch try-on failed")
            return
    else:
        # Single inference
        result_image = viton.try_on(person_image, clothing_image)
        
        if result_image is None:
            logger.error("Try-on failed")
            return
        
        logger.info("✓ Try-on completed")
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving result to: {output_path}")
    loader.save_image(result_image, str(output_path))
    
    # Create comparison if requested
    if args.comparison:
        logger.info("Creating comparison image...")
        
        # Resize all to same height
        target_h = 768
        
        def resize_h(img, h):
            pil = Image.fromarray(img)
            aspect = pil.width / pil.height
            new_w = int(h * aspect)
            pil = pil.resize((new_w, h), Image.LANCZOS)
            return np.array(pil)
        
        person_resized = resize_h(person_image, target_h)
        clothing_resized = resize_h(clothing_image, target_h)
        result_resized = resize_h(result_image, target_h)
        
        # Concatenate horizontally
        comparison = np.concatenate([person_resized, clothing_resized, result_resized], axis=1)
        
        comparison_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
        loader.save_image(comparison, str(comparison_path))
        logger.info(f"Comparison saved to: {comparison_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"\nResult: {output_path}")
    
    if args.high_res:
        logger.info("\nNote: High-resolution mode requested but not yet implemented")
        logger.info("Current output: 512x768")


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated VITON-HD virtual try-on demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use specific person and clothing
  python scripts/demo_viton_gpu.py --person models/realistic/male/M/front_001.jpg --clothing samples/clothing/tshirt.png
  
  # Auto-select person from library
  python scripts/demo_viton_gpu.py --gender male --size M --clothing samples/clothing/tshirt.png
  
  # Batch processing (for benchmarking)
  python scripts/demo_viton_gpu.py --person models/realistic/female/L/front_001.jpg --clothing samples/clothing/dress.png --batch-size 8
  
  # With comparison output
  python scripts/demo_viton_gpu.py --person models/realistic/male/M/front_001.jpg --clothing samples/clothing/tshirt.png --comparison
  
  # Force CPU
  python scripts/demo_viton_gpu.py --device cpu --person ... --clothing ...

GPU Requirements:
  - NVIDIA GPU with CUDA support (RTX 3050, A100, etc.)
  - CUDA 11.8 or higher
  - 4GB+ VRAM recommended
        """
    )
    
    parser.add_argument('--person', help='Path to person/model image')
    parser.add_argument('--clothing', help='Path to clothing image')
    parser.add_argument('--gender', choices=['male', 'female'],
                       help='Gender for model selection (if --person not specified)')
    parser.add_argument('--size', choices=['S', 'M', 'L', 'XL'],
                       help='Size for model selection (if --person not specified)')
    parser.add_argument('--output', default='output/viton_result.jpg',
                       help='Output path for result image')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing (for benchmarking)')
    parser.add_argument('--high-res', action='store_true',
                       help='[EXPERIMENTAL] Enable high-resolution output (not yet implemented)')
    parser.add_argument('--comparison', action='store_true',
                       help='Create side-by-side comparison image')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.person and not (args.gender or args.size):
        logger.info("No person specified, will select random model from library")
    
    if not args.clothing:
        parser.error("--clothing is required")
    
    try:
        run_demo(args)
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
