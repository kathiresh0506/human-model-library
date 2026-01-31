"""
Generate realistic human model images for testing.
Creates anatomically correct figures with shading, facial features, and hair using Pillow.
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import PathManager
from generator.realistic_model_generator import RealisticModelGenerator


def create_sample_models():
    """Create all sample model images with realistic features."""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    # Initialize realistic generator
    generator = RealisticModelGenerator()
    
    # Model specifications with proper measurements
    models = [
        {
            'gender': 'male',
            'size': 'M',
            'age_group': 'young',
            'ethnicity': 'asian',
            'measurements': {
                'chest': 98,
                'waist': 83,
                'hip': 98,
                'height': 177
            },
            'description': 'Male M Young Asian - Athletic build with light tan skin'
        },
        {
            'gender': 'female',
            'size': 'S',
            'age_group': 'young',
            'ethnicity': 'caucasian',
            'measurements': {
                'bust': 86,
                'waist': 64,
                'hip': 90,
                'height': 162
            },
            'description': 'Female S Young Caucasian - Slim build with fair skin'
        },
        {
            'gender': 'male',
            'size': 'L',
            'age_group': 'middle',
            'ethnicity': 'african',
            'measurements': {
                'chest': 108,
                'waist': 93,
                'hip': 108,
                'height': 180
            },
            'description': 'Male L Middle African - Larger build with brown skin'
        }
    ]
    
    created_models = []
    
    for model_spec in models:
        # Create directory
        model_dir = models_dir / model_spec['gender'] / model_spec['size'] / model_spec['age_group'] / model_spec['ethnicity']
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate realistic model image
        output_path = model_dir / "front.png"
        
        success = generator.generate_model(
            gender=model_spec['gender'],
            size=model_spec['size'],
            age_group=model_spec['age_group'],
            ethnicity=model_spec['ethnicity'],
            measurements=model_spec['measurements'],
            output_path=str(output_path)
        )
        
        if success:
            model_path = f"models/{model_spec['gender']}/{model_spec['size']}/{model_spec['age_group']}/{model_spec['ethnicity']}/front.png"
            created_models.append(model_path)
            print(f"✅ Created realistic model: {model_path}")
            print(f"   {model_spec['description']}")
        else:
            print(f"❌ Failed to create model: {model_spec}")
    
    return created_models


if __name__ == "__main__":
    print("Generating realistic sample model images...")
    models = create_sample_models()
    print(f"\n✨ Successfully created {len(models)} realistic sample models with:")
    print("   - Facial features (eyes, nose, mouth)")
    print("   - Hair styling")
    print("   - 3D shading and highlights")
    print("   - Natural body proportions")
    print("   - Transparent background (RGBA)")
