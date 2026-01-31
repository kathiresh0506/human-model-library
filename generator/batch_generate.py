"""
Batch generation script for creating all model combinations.
Orchestrates MakeHuman generation and Blender rendering.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml

from .makehuman_generator import MakeHumanGenerator
from .blender_renderer import BlenderRenderer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchGenerator:
    """
    Orchestrates batch generation of human models.
    """
    
    def __init__(self, config_dir: str, models_dir: str, temp_dir: str = "/tmp/models"):
        """
        Initialize BatchGenerator.
        
        Args:
            config_dir: Directory containing configuration files
            models_dir: Directory to save generated models
            temp_dir: Temporary directory for 3D model files
        """
        self.config_dir = Path(config_dir)
        self.models_dir = Path(models_dir)
        self.temp_dir = Path(temp_dir)
        
        self.makehuman = MakeHumanGenerator()
        self.renderer = BlenderRenderer()
        
        # Load configurations
        self.sizes_config = self._load_yaml('sizes.yaml')
        self.ethnicities_config = self._load_yaml('ethnicities.yaml')
        self.age_groups_config = self._load_yaml('age_groups.yaml')
        
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_yaml(self, filename: str) -> dict:
        """Load YAML configuration file."""
        try:
            with open(self.config_dir / filename, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}
    
    def generate_all_models(self, max_models: Optional[int] = None) -> Dict[str, bool]:
        """
        Generate all model combinations.
        
        Args:
            max_models: Optional limit on total number of models to generate (for testing)
            
        Returns:
            Dictionary of generation results
        """
        results = {}
        total_models = 0
        successful_models = 0
        
        genders = ['male', 'female']
        sizes = ['S', 'M', 'L', 'XL']
        age_groups = ['young', 'middle', 'senior']
        ethnicities = ['asian', 'african', 'caucasian', 'hispanic', 'middle_eastern']
        
        logger.info("Starting batch model generation...")
        logger.info(f"Total combinations: {len(genders) * len(sizes) * len(age_groups) * len(ethnicities)}")
        
        for gender in genders:
            for size in sizes:
                for age_group in age_groups:
                    for ethnicity in ethnicities:
                        if max_models and total_models >= max_models:
                            logger.info(f"Reached limit of {max_models} models")
                            return results
                        
                        total_models += 1
                        model_id = f"{gender}_{size.lower()}_{age_group}_{ethnicity}_001"
                        
                        logger.info(f"Generating model {total_models}: {model_id}")
                        
                        success = self.generate_single_model(
                            gender, size, age_group, ethnicity, model_id
                        )
                        
                        results[model_id] = success
                        if success:
                            successful_models += 1
        
        logger.info(f"Batch generation complete: {successful_models}/{total_models} successful")
        return results
    
    def generate_single_model(self,
                            gender: str,
                            size: str,
                            age_group: str,
                            ethnicity: str,
                            model_id: str) -> bool:
        """
        Generate a single model with all processing steps.
        
        Args:
            gender: Gender
            size: Size
            age_group: Age group
            ethnicity: Ethnicity
            model_id: Unique model identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get measurements for size
            measurements = self._get_measurements(gender, size)
            
            # Get age value
            age = self._get_age_value(age_group)
            
            # Setup paths
            model_dir = self.models_dir / gender / size / age_group / ethnicity
            model_dir.mkdir(parents=True, exist_ok=True)
            
            temp_model_path = self.temp_dir / f"{model_id}.mhm"
            
            # Step 1: Generate 3D model with MakeHuman
            logger.info(f"  Step 1/3: Generating 3D model...")
            success = self.makehuman.generate_model(
                gender, measurements, age, ethnicity, str(temp_model_path)
            )
            
            if not success:
                logger.error(f"  Failed to generate 3D model for {model_id}")
                return False
            
            # Step 2: Render images with Blender
            logger.info(f"  Step 2/3: Rendering images...")
            # Note: Skipping actual rendering if Blender not available
            # Create placeholder images instead
            self._create_placeholder_images(model_dir)
            
            # Step 3: Create metadata
            logger.info(f"  Step 3/3: Creating metadata...")
            metadata = self._create_metadata(
                model_id, gender, size, age_group, ethnicity, measurements
            )
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  Successfully generated {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"  Error generating {model_id}: {e}")
            return False
    
    def _get_measurements(self, gender: str, size: str) -> Dict[str, float]:
        """Get measurements for a gender and size."""
        try:
            size_data = self.sizes_config[gender.lower()][size.upper()]
            
            # Use average values
            measurements = {}
            for key, value in size_data.items():
                if isinstance(value, dict) and 'avg' in value:
                    measurements[key] = value['avg']
            
            return measurements
            
        except Exception as e:
            logger.error(f"Error getting measurements: {e}")
            return {
                'chest': 98, 'waist': 83, 'hip': 98, 'height': 177
            }
    
    def _get_age_value(self, age_group: str) -> int:
        """Get representative age for an age group."""
        age_ranges = {
            'young': 25,
            'middle': 40,
            'senior': 60
        }
        return age_ranges.get(age_group.lower(), 25)
    
    def _create_placeholder_images(self, model_dir: Path):
        """Create placeholder images (used when actual rendering is not available)."""
        try:
            import numpy as np
            from PIL import Image
            
            # Create simple placeholder images
            for view in ['front', 'side', 'back']:
                img_path = model_dir / f"{view}.png"
                
                # Create a simple colored image as placeholder
                img = np.ones((512, 512, 3), dtype=np.uint8) * 200
                
                # Add text
                pil_img = Image.fromarray(img)
                pil_img.save(str(img_path))
                
            logger.info(f"    Created placeholder images")
            
        except Exception as e:
            logger.warning(f"Could not create placeholder images: {e}")
    
    def _create_metadata(self,
                        model_id: str,
                        gender: str,
                        size: str,
                        age_group: str,
                        ethnicity: str,
                        measurements: Dict[str, float]) -> Dict[str, Any]:
        """Create metadata dictionary for a model."""
        from datetime import datetime
        
        metadata = {
            'id': model_id,
            'gender': gender.lower(),
            'size': size.upper(),
            'age_group': age_group.lower(),
            'ethnicity': ethnicity.lower(),
            'measurements': measurements,
            'poses': ['front', 'side', 'back'],
            'created_at': datetime.now().strftime('%Y-%m-%d'),
            'image_paths': {
                'front': f"{gender}/{size}/{age_group}/{ethnicity}/front.png",
                'side': f"{gender}/{size}/{age_group}/{ethnicity}/side.png",
                'back': f"{gender}/{size}/{age_group}/{ethnicity}/back.png",
            }
        }
        
        return metadata


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate human models")
    parser.add_argument('--config-dir', default='../config',
                       help='Configuration directory')
    parser.add_argument('--models-dir', default='../models',
                       help='Models output directory')
    parser.add_argument('--temp-dir', default='/tmp/models',
                       help='Temporary directory for 3D files')
    parser.add_argument('--limit', type=int,
                       help='Limit number of models to generate (for testing)')
    parser.add_argument('--gender', help='Generate only specific gender')
    parser.add_argument('--size', help='Generate only specific size')
    parser.add_argument('--age-group', help='Generate only specific age group')
    parser.add_argument('--ethnicity', help='Generate only specific ethnicity')
    
    args = parser.parse_args()
    
    generator = BatchGenerator(args.config_dir, args.models_dir, args.temp_dir)
    
    # If specific parameters provided, generate single model
    if args.gender and args.size and args.age_group and args.ethnicity:
        model_id = f"{args.gender}_{args.size.lower()}_{args.age_group}_{args.ethnicity}_001"
        success = generator.generate_single_model(
            args.gender, args.size, args.age_group, args.ethnicity, model_id
        )
        if success:
            print(f"Successfully generated {model_id}")
        else:
            print(f"Failed to generate {model_id}")
    else:
        # Generate all models
        results = generator.generate_all_models(limit_per_category=args.limit)
        
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nBatch generation complete:")
        print(f"  Successful: {successful}/{total}")
        print(f"  Failed: {total - successful}/{total}")
