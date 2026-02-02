"""
Create realistic human models using MakeHuman + Blender or fallback to enhanced generator.

This script provides two options:
Option A - If MakeHuman/Blender are installed:
  - Generate .mhm files with MakeHuman
  - Render with Blender

Option B - If MakeHuman not installed:
  - Use the enhanced realistic model generator
  - Creates high-quality 2D models with shading and features
"""
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.makehuman_generator import MakeHumanGenerator
from generator.blender_renderer import BlenderRenderer
from generator.realistic_model_generator import RealisticModelGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if MakeHuman and Blender are available."""
    mh_gen = MakeHumanGenerator()
    renderer = BlenderRenderer()
    
    has_makehuman = mh_gen.has_makehuman
    has_blender = renderer.has_blender
    
    logger.info("=" * 60)
    logger.info("Dependency Check:")
    logger.info(f"  MakeHuman: {'✓ Available' if has_makehuman else '✗ Not found'}")
    logger.info(f"  Blender:   {'✓ Available' if has_blender else '✗ Not found'}")
    logger.info("=" * 60)
    
    return has_makehuman, has_blender


def create_with_makehuman_blender(gender, size, age_group, ethnicity, output_dir):
    """
    Create model using MakeHuman + Blender pipeline.
    
    Args:
        gender: Model gender
        size: Model size
        age_group: Age group
        ethnicity: Ethnicity
        output_dir: Output directory
        
    Returns:
        True if successful
    """
    logger.info("Creating model with MakeHuman + Blender...")
    
    # Initialize generators
    mh_gen = MakeHumanGenerator()
    renderer = BlenderRenderer()
    
    # Sample measurements (should be loaded from config)
    measurements = {
        'chest': 98,
        'waist': 83,
        'hip': 98,
        'height': 177
    }
    
    # Generate MakeHuman configuration
    model_name = f"{gender}_{size}_{age_group}_{ethnicity}"
    mhm_path = Path(output_dir) / f"{model_name}.mhm"
    
    success = mh_gen.generate_model(
        gender=gender,
        size_params=measurements,
        age=25 if age_group == 'young' else 40,
        ethnicity=ethnicity,
        output_path=str(mhm_path)
    )
    
    if not success:
        logger.error("Failed to generate MakeHuman configuration")
        return False
    
    logger.info(f"MakeHuman configuration created: {mhm_path}")
    logger.info("Note: To complete the pipeline:")
    logger.info("1. Open MakeHuman and load the .mhm file")
    logger.info("2. Export as FBX or Collada format")
    logger.info("3. Use Blender to render the exported model")
    
    return True


def create_with_enhanced_generator(gender, size, age_group, ethnicity, output_dir):
    """
    Create model using enhanced 2D generator (fallback).
    
    Args:
        gender: Model gender
        size: Model size
        age_group: Age group
        ethnicity: Ethnicity
        output_dir: Output directory
        
    Returns:
        True if successful
    """
    logger.info("Creating model with Enhanced 2D Generator...")
    
    # Initialize generator
    generator = RealisticModelGenerator()
    
    # Load measurements based on size
    measurements = get_measurements_for_size(gender, size)
    
    # Generate model
    output_path = Path(output_dir) / f"{gender}_{size}_{age_group}_{ethnicity}_front.png"
    
    success = generator.generate_model(
        gender=gender,
        size=size,
        age_group=age_group,
        ethnicity=ethnicity,
        measurements=measurements,
        output_path=str(output_path)
    )
    
    if success:
        logger.info(f"✓ Model created: {output_path}")
    
    return success


def get_measurements_for_size(gender, size):
    """Get body measurements for a given size."""
    # These should ideally be loaded from config/sizes.yaml
    measurements_male = {
        'S': {'chest': 88, 'waist': 73, 'hip': 88, 'height': 172},
        'M': {'chest': 98, 'waist': 83, 'hip': 98, 'height': 177},
        'L': {'chest': 108, 'waist': 93, 'hip': 108, 'height': 180},
        'XL': {'chest': 118, 'waist': 103, 'hip': 118, 'height': 182},
    }
    
    measurements_female = {
        'S': {'bust': 84, 'waist': 64, 'hip': 90, 'height': 162},
        'M': {'bust': 90, 'waist': 70, 'hip': 96, 'height': 165},
        'L': {'bust': 98, 'waist': 78, 'hip': 104, 'height': 167},
        'XL': {'bust': 106, 'waist': 86, 'hip': 112, 'height': 169},
    }
    
    if gender.lower() == 'male':
        return measurements_male.get(size.upper(), measurements_male['M'])
    else:
        return measurements_female.get(size.upper(), measurements_female['M'])


def main():
    parser = argparse.ArgumentParser(
        description="Create realistic human models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check available tools
  python scripts/create_realistic_models.py --check

  # Create a male model (will auto-detect available tools)
  python scripts/create_realistic_models.py --gender male --size M --age young --ethnicity asian --output models/male/M/young/asian

  # Force use of enhanced generator
  python scripts/create_realistic_models.py --gender female --size S --force-enhanced
        """
    )
    
    parser.add_argument('--check', action='store_true',
                       help='Check if MakeHuman and Blender are available')
    parser.add_argument('--gender', choices=['male', 'female'],
                       help='Model gender')
    parser.add_argument('--size', choices=['S', 'M', 'L', 'XL'],
                       help='Model size')
    parser.add_argument('--age', dest='age_group', 
                       choices=['young', 'middle', 'senior'],
                       default='young', help='Age group')
    parser.add_argument('--ethnicity', 
                       choices=['asian', 'african', 'caucasian', 'hispanic', 'middle_eastern'],
                       default='asian', help='Ethnicity')
    parser.add_argument('--output', default='output',
                       help='Output directory')
    parser.add_argument('--force-enhanced', action='store_true',
                       help='Force use of enhanced 2D generator')
    
    args = parser.parse_args()
    
    # Check dependencies
    has_makehuman, has_blender = check_dependencies()
    
    if args.check:
        logger.info("\nRecommendations:")
        if not has_makehuman:
            logger.info("  Install MakeHuman: http://www.makehumancommunity.org/")
        if not has_blender:
            logger.info("  Install Blender: https://www.blender.org/download/")
        if not has_makehuman or not has_blender:
            logger.info("  Or use enhanced 2D generator (no installation needed)")
        return
    
    # Validate required arguments
    if not args.gender or not args.size:
        parser.error("--gender and --size are required")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose generation method
    if args.force_enhanced or not (has_makehuman and has_blender):
        if not args.force_enhanced:
            logger.info("MakeHuman/Blender not available, using enhanced 2D generator...")
        success = create_with_enhanced_generator(
            args.gender, args.size, args.age_group, args.ethnicity, output_dir
        )
    else:
        logger.info("Using MakeHuman + Blender pipeline...")
        success = create_with_makehuman_blender(
            args.gender, args.size, args.age_group, args.ethnicity, output_dir
        )
    
    if success:
        logger.info("\n✨ Model creation completed!")
    else:
        logger.error("\n❌ Model creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
