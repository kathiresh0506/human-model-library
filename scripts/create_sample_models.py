"""
Generate placeholder human model images for testing.
Creates simple silhouette/mannequin style images using Pillow.
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import PathManager


def create_human_silhouette(width, height, skin_tone, shoulder_width, chest_width, waist_width, hip_width):
    """
    Create a simple human body silhouette.
    
    Args:
        width: Image width
        height: Image height
        skin_tone: RGB tuple for skin color
        shoulder_width: Width at shoulders in pixels
        chest_width: Width at chest in pixels
        waist_width: Width at waist in pixels
        hip_width: Width at hips in pixels
    
    Returns:
        PIL Image with human silhouette
    """
    # Create image with white background
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Calculate vertical positions (as fractions of height)
    head_top = int(height * 0.05)
    head_bottom = int(height * 0.15)
    shoulder_y = int(height * 0.18)
    chest_y = int(height * 0.30)
    waist_y = int(height * 0.45)
    hip_y = int(height * 0.55)
    legs_end = int(height * 0.95)
    
    center_x = width // 2
    
    # Draw head (ellipse)
    head_radius = int(width * 0.08)
    head_center_y = (head_top + head_bottom) // 2
    draw.ellipse(
        [center_x - head_radius, head_center_y - head_radius,
         center_x + head_radius, head_center_y + head_radius],
        fill=skin_tone
    )
    
    # Draw neck
    neck_width = int(width * 0.06)
    draw.rectangle(
        [center_x - neck_width // 2, head_bottom,
         center_x + neck_width // 2, shoulder_y],
        fill=skin_tone
    )
    
    # Draw torso (polygon connecting shoulder, chest, waist, hip)
    torso_points = [
        (center_x - shoulder_width // 2, shoulder_y),
        (center_x + shoulder_width // 2, shoulder_y),
        (center_x + chest_width // 2, chest_y),
        (center_x + waist_width // 2, waist_y),
        (center_x + hip_width // 2, hip_y),
        (center_x - hip_width // 2, hip_y),
        (center_x - waist_width // 2, waist_y),
        (center_x - chest_width // 2, chest_y),
    ]
    draw.polygon(torso_points, fill=skin_tone)
    
    # Draw arms
    arm_width = int(width * 0.04)
    arm_length_y = waist_y
    # Left arm
    draw.rectangle(
        [center_x - shoulder_width // 2 - arm_width, shoulder_y,
         center_x - shoulder_width // 2, arm_length_y],
        fill=skin_tone
    )
    # Right arm
    draw.rectangle(
        [center_x + shoulder_width // 2, shoulder_y,
         center_x + shoulder_width // 2 + arm_width, arm_length_y],
        fill=skin_tone
    )
    
    # Draw legs
    leg_width = int(hip_width * 0.45)
    leg_gap = int(width * 0.02)
    # Left leg
    draw.polygon(
        [
            (center_x - leg_gap // 2, hip_y),
            (center_x - leg_gap // 2 - leg_width, hip_y),
            (center_x - leg_gap // 2 - leg_width // 2, legs_end),
            (center_x - leg_gap // 2, legs_end),
        ],
        fill=skin_tone
    )
    # Right leg
    draw.polygon(
        [
            (center_x + leg_gap // 2, hip_y),
            (center_x + leg_gap // 2 + leg_width, hip_y),
            (center_x + leg_gap // 2 + leg_width // 2, legs_end),
            (center_x + leg_gap // 2, legs_end),
        ],
        fill=skin_tone
    )
    
    return img


def create_sample_models():
    """Create all sample model images."""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    # Model specifications
    models = [
        {
            'gender': 'male',
            'size': 'M',
            'age_group': 'young',
            'ethnicity': 'asian',
            'height': 177,
            'shoulder_width': 260,
            'chest_width': 180,
            'waist_width': 150,
            'hip_width': 180,
            'skin_tone': (255, 224, 189)  # Light tan
        },
        {
            'gender': 'female',
            'size': 'S',
            'age_group': 'young',
            'ethnicity': 'caucasian',
            'height': 162,
            'shoulder_width': 220,
            'chest_width': 150,
            'waist_width': 115,
            'hip_width': 160,
            'skin_tone': (255, 228, 206)  # Fair
        },
        {
            'gender': 'male',
            'size': 'L',
            'age_group': 'middle',
            'ethnicity': 'african',
            'height': 180,
            'shoulder_width': 275,
            'chest_width': 195,
            'waist_width': 170,
            'hip_width': 195,
            'skin_tone': (139, 90, 43)  # Brown
        }
    ]
    
    created_models = []
    
    for model_spec in models:
        # Create directory
        model_dir = models_dir / model_spec['gender'] / model_spec['size'] / model_spec['age_group'] / model_spec['ethnicity']
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate image
        img = create_human_silhouette(
            width=512,
            height=1024,
            skin_tone=model_spec['skin_tone'],
            shoulder_width=model_spec['shoulder_width'],
            chest_width=model_spec['chest_width'],
            waist_width=model_spec['waist_width'],
            hip_width=model_spec['hip_width']
        )
        
        # Save image
        output_path = model_dir / "front.png"
        img.save(output_path)
        
        model_path = f"models/{model_spec['gender']}/{model_spec['size']}/{model_spec['age_group']}/{model_spec['ethnicity']}/front.png"
        created_models.append(model_path)
        print(f"✅ Created model: {model_path}")
    
    return created_models


if __name__ == "__main__":
    print("Generating sample model images...")
    models = create_sample_models()
    print(f"\n✨ Successfully created {len(models)} sample models!")
