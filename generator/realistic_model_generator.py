"""
Enhanced realistic model generator using Pillow.
Creates more anatomically correct human figures with shading, features, and proper proportions.
This is a fallback generator for when MakeHuman is not available.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RealisticModelGenerator:
    """
    Generates realistic human model images with proper anatomy, shading, and features.
    """
    
    def __init__(self):
        """Initialize the generator."""
        self.default_size = (1024, 2048)  # Width x Height for full body portrait
    
    def generate_model(self,
                      gender: str,
                      size: str,
                      age_group: str,
                      ethnicity: str,
                      measurements: Dict[str, float],
                      output_path: str) -> bool:
        """
        Generate a realistic human model image.
        
        Args:
            gender: 'male' or 'female'
            size: 'S', 'M', 'L', or 'XL'
            age_group: 'young', 'middle', or 'senior'
            ethnicity: Ethnicity identifier
            measurements: Body measurements dict
            output_path: Path to save the generated image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Generating {gender} {size} {age_group} {ethnicity} model")
            
            # Get skin tone based on ethnicity
            skin_base, skin_shadow = self._get_skin_tones(ethnicity)
            
            # Get body proportions based on measurements
            proportions = self._calculate_proportions(measurements, gender, size)
            
            # Create the model image
            img = self._create_realistic_figure(
                gender, age_group, skin_base, skin_shadow, proportions
            )
            
            # Save the image
            img.save(output_path)
            logger.info(f"Model saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating model: {e}")
            return False
    
    def _get_skin_tones(self, ethnicity: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get skin tone colors based on ethnicity.
        
        Args:
            ethnicity: Ethnicity identifier
            
        Returns:
            Tuple of (base_color, shadow_color) as RGB tuples
        """
        skin_tones = {
            'asian': ((255, 224, 189), (220, 190, 160)),
            'caucasian': ((255, 228, 206), (230, 200, 180)),
            'african': ((139, 90, 43), (100, 65, 30)),
            'hispanic': ((210, 160, 120), (180, 130, 90)),
            'middle_eastern': ((200, 165, 130), (170, 135, 100)),
        }
        
        return skin_tones.get(ethnicity.lower(), ((255, 224, 189), (220, 190, 160)))
    
    def _calculate_proportions(self, measurements: Dict[str, float], 
                              gender: str, size: str) -> Dict[str, int]:
        """
        Calculate pixel proportions from measurements.
        
        Args:
            measurements: Body measurements
            gender: Gender
            size: Size
            
        Returns:
            Dictionary of proportions in pixels
        """
        # Get height-based scaling
        height_cm = measurements.get('height', 170)
        scale = self.default_size[1] / height_cm  # pixels per cm
        
        # Get body measurements
        chest_key = 'bust' if gender.lower() == 'female' else 'chest'
        chest = measurements.get(chest_key, 90)
        waist = measurements.get('waist', 75)
        hip = measurements.get('hip', 95)
        
        # Convert to pixel widths (rough approximation)
        proportions = {
            'head_radius': int(scale * 10),  # ~10cm head radius
            'neck_width': int(scale * 6),
            'shoulder_width': int(chest * scale * 0.45),
            'chest_width': int(chest * scale * 0.35),
            'waist_width': int(waist * scale * 0.35),
            'hip_width': int(hip * scale * 0.35),
            'arm_width': int(scale * 5),
            'leg_width': int(scale * 8),
        }
        
        return proportions
    
    def _create_realistic_figure(self,
                                 gender: str,
                                 age_group: str,
                                 skin_base: Tuple[int, int, int],
                                 skin_shadow: Tuple[int, int, int],
                                 proportions: Dict[str, int]) -> Image.Image:
        """
        Create a realistic human figure with shading and features.
        
        Args:
            gender: Gender
            age_group: Age group
            skin_base: Base skin color
            skin_shadow: Shadow skin color
            proportions: Body proportions
            
        Returns:
            PIL Image with realistic human figure
        """
        width, height = self.default_size
        
        # Create image with transparent background
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Create a separate layer for the body
        body_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(body_layer)
        
        center_x = width // 2
        
        # Calculate vertical positions (head-to-toe proportions)
        head_top = int(height * 0.03)
        head_bottom = int(height * 0.11)
        neck_bottom = int(height * 0.13)
        shoulder_y = int(height * 0.15)
        chest_y = int(height * 0.23)
        waist_y = int(height * 0.35)
        hip_y = int(height * 0.43)
        crotch_y = int(height * 0.48)
        knee_y = int(height * 0.70)
        ankle_y = int(height * 0.93)
        
        # Draw head with gradient shading
        head_center_y = (head_top + head_bottom) // 2
        self._draw_shaded_ellipse(
            draw, 
            center_x - proportions['head_radius'], head_center_y - proportions['head_radius'],
            center_x + proportions['head_radius'], head_center_y + proportions['head_radius'],
            skin_base, skin_shadow
        )
        
        # Draw simple facial features
        self._draw_facial_features(draw, center_x, head_center_y, proportions['head_radius'], age_group)
        
        # Draw hair
        self._draw_hair(draw, center_x, head_top, head_center_y, proportions['head_radius'], gender, age_group)
        
        # Draw neck
        self._draw_shaded_rectangle(
            draw,
            center_x - proportions['neck_width'] // 2, head_bottom,
            center_x + proportions['neck_width'] // 2, neck_bottom,
            skin_base, skin_shadow
        )
        
        # Draw torso with natural curves
        self._draw_torso(
            draw, center_x, shoulder_y, chest_y, waist_y, hip_y,
            proportions, skin_base, skin_shadow, gender
        )
        
        # Draw arms
        self._draw_arms(
            draw, center_x, shoulder_y, waist_y,
            proportions, skin_base, skin_shadow
        )
        
        # Draw legs
        self._draw_legs(
            draw, center_x, hip_y, crotch_y, knee_y, ankle_y,
            proportions, skin_base, skin_shadow
        )
        
        # Apply slight blur for softer edges
        body_layer = body_layer.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Composite onto main image
        img = Image.alpha_composite(img, body_layer)
        
        return img
    
    def _draw_shaded_ellipse(self, draw, x1, y1, x2, y2, base_color, shadow_color):
        """Draw an ellipse with gradient shading."""
        # Draw shadow (left side)
        draw.ellipse([x1, y1, x2, y2], fill=shadow_color + (255,))
        
        # Draw highlight (right side) - create a gradient effect
        cx = (x1 + x2) // 2
        width = x2 - x1
        
        # Overlay lighter color on right side
        highlight_x = cx + width // 6
        draw.ellipse(
            [highlight_x - width // 3, y1, x2, y2],
            fill=base_color + (200,)
        )
    
    def _draw_shaded_rectangle(self, draw, x1, y1, x2, y2, base_color, shadow_color):
        """Draw a rectangle with gradient shading."""
        cx = (x1 + x2) // 2
        
        # Left side (shadow)
        draw.rectangle([x1, y1, cx, y2], fill=shadow_color + (255,))
        
        # Right side (lighter)
        draw.rectangle([cx, y1, x2, y2], fill=base_color + (255,))
    
    def _draw_facial_features(self, draw, center_x, center_y, head_radius, age_group):
        """Draw simple facial features."""
        # Eyes
        eye_y = center_y - head_radius // 4
        eye_offset = head_radius // 3
        eye_size = head_radius // 8
        
        # Left eye
        draw.ellipse(
            [center_x - eye_offset - eye_size, eye_y - eye_size // 2,
             center_x - eye_offset + eye_size, eye_y + eye_size // 2],
            fill=(50, 40, 30, 255)
        )
        
        # Right eye
        draw.ellipse(
            [center_x + eye_offset - eye_size, eye_y - eye_size // 2,
             center_x + eye_offset + eye_size, eye_y + eye_size // 2],
            fill=(50, 40, 30, 255)
        )
        
        # Nose (simple line)
        nose_y = center_y + head_radius // 6
        draw.ellipse(
            [center_x - head_radius // 12, nose_y - head_radius // 12,
             center_x + head_radius // 12, nose_y + head_radius // 12],
            fill=(120, 80, 60, 150)
        )
        
        # Mouth (simple line)
        mouth_y = center_y + head_radius // 2
        mouth_width = head_radius // 3
        draw.line(
            [center_x - mouth_width, mouth_y, center_x + mouth_width, mouth_y],
            fill=(150, 100, 80, 200), width=2
        )
    
    def _draw_hair(self, draw, center_x, head_top, head_center_y, head_radius, gender, age_group):
        """Draw hair shape on head."""
        hair_colors = {
            'young': (40, 30, 20, 255),  # Dark brown/black
            'middle': (60, 50, 40, 255),  # Brown with slight gray
            'senior': (150, 150, 150, 255),  # Gray
        }
        
        hair_color = hair_colors.get(age_group, (40, 30, 20, 255))
        
        if gender.lower() == 'male':
            # Short hair
            draw.ellipse(
                [center_x - head_radius, head_top - head_radius // 4,
                 center_x + head_radius, head_center_y],
                fill=hair_color
            )
        else:
            # Long hair
            draw.ellipse(
                [center_x - head_radius, head_top - head_radius // 4,
                 center_x + head_radius, head_center_y + head_radius],
                fill=hair_color
            )
            # Side hair
            draw.rectangle(
                [center_x - head_radius - head_radius // 4, head_center_y,
                 center_x - head_radius, head_center_y + head_radius * 2],
                fill=hair_color
            )
            draw.rectangle(
                [center_x + head_radius, head_center_y,
                 center_x + head_radius + head_radius // 4, head_center_y + head_radius * 2],
                fill=hair_color
            )
    
    def _draw_torso(self, draw, center_x, shoulder_y, chest_y, waist_y, hip_y,
                   proportions, skin_base, skin_shadow, gender):
        """Draw torso with natural curves."""
        # Create smooth body outline
        points = [
            (center_x - proportions['shoulder_width'] // 2, shoulder_y),
            (center_x + proportions['shoulder_width'] // 2, shoulder_y),
            (center_x + proportions['chest_width'] // 2, chest_y),
            (center_x + proportions['waist_width'] // 2, waist_y),
            (center_x + proportions['hip_width'] // 2, hip_y),
            (center_x - proportions['hip_width'] // 2, hip_y),
            (center_x - proportions['waist_width'] // 2, waist_y),
            (center_x - proportions['chest_width'] // 2, chest_y),
        ]
        
        # Draw with shading
        draw.polygon(points, fill=skin_shadow + (255,))
        
        # Add highlight on right side
        right_points = [
            (center_x, shoulder_y),
            (center_x + proportions['shoulder_width'] // 2, shoulder_y),
            (center_x + proportions['chest_width'] // 2, chest_y),
            (center_x + proportions['waist_width'] // 2, waist_y),
            (center_x + proportions['hip_width'] // 2, hip_y),
            (center_x, hip_y),
            (center_x, waist_y),
            (center_x, chest_y),
        ]
        draw.polygon(right_points, fill=skin_base + (255,))
    
    def _draw_arms(self, draw, center_x, shoulder_y, waist_y,
                  proportions, skin_base, skin_shadow):
        """Draw arms with natural shape."""
        arm_length = waist_y - shoulder_y
        elbow_y = shoulder_y + arm_length // 2
        
        # Left arm
        left_shoulder_x = center_x - proportions['shoulder_width'] // 2
        left_arm_x = left_shoulder_x - proportions['arm_width']
        
        # Upper left arm
        draw.polygon([
            (left_shoulder_x, shoulder_y),
            (left_shoulder_x - proportions['arm_width'], shoulder_y + proportions['arm_width']),
            (left_arm_x - proportions['arm_width'] // 4, elbow_y),
            (left_shoulder_x - proportions['arm_width'] // 2, elbow_y),
        ], fill=skin_shadow + (255,))
        
        # Lower left arm
        draw.polygon([
            (left_shoulder_x - proportions['arm_width'] // 2, elbow_y),
            (left_arm_x - proportions['arm_width'] // 4, elbow_y),
            (left_arm_x, waist_y),
            (left_shoulder_x - proportions['arm_width'] // 3, waist_y),
        ], fill=skin_shadow + (255,))
        
        # Right arm (lighter)
        right_shoulder_x = center_x + proportions['shoulder_width'] // 2
        right_arm_x = right_shoulder_x + proportions['arm_width']
        
        # Upper right arm
        draw.polygon([
            (right_shoulder_x, shoulder_y),
            (right_shoulder_x + proportions['arm_width'], shoulder_y + proportions['arm_width']),
            (right_arm_x + proportions['arm_width'] // 4, elbow_y),
            (right_shoulder_x + proportions['arm_width'] // 2, elbow_y),
        ], fill=skin_base + (255,))
        
        # Lower right arm
        draw.polygon([
            (right_shoulder_x + proportions['arm_width'] // 2, elbow_y),
            (right_arm_x + proportions['arm_width'] // 4, elbow_y),
            (right_arm_x, waist_y),
            (right_shoulder_x + proportions['arm_width'] // 3, waist_y),
        ], fill=skin_base + (255,))
    
    def _draw_legs(self, draw, center_x, hip_y, crotch_y, knee_y, ankle_y,
                  proportions, skin_base, skin_shadow):
        """Draw legs with natural shape."""
        leg_width = proportions['leg_width']
        leg_gap = proportions['hip_width'] // 8
        
        # Left leg (shadow side)
        draw.polygon([
            (center_x - leg_gap, crotch_y),
            (center_x - leg_gap - leg_width, crotch_y),
            (center_x - leg_gap - leg_width * 0.9, knee_y),
            (center_x - leg_gap - leg_width * 0.7, ankle_y),
            (center_x - leg_gap, ankle_y),
            (center_x - leg_gap, knee_y),
        ], fill=skin_shadow + (255,))
        
        # Right leg (lighter side)
        draw.polygon([
            (center_x + leg_gap, crotch_y),
            (center_x + leg_gap + leg_width, crotch_y),
            (center_x + leg_gap + leg_width * 0.9, knee_y),
            (center_x + leg_gap + leg_width * 0.7, ankle_y),
            (center_x + leg_gap, ankle_y),
            (center_x + leg_gap, knee_y),
        ], fill=skin_base + (255,))


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate realistic human model images")
    parser.add_argument('--gender', required=True, choices=['male', 'female'])
    parser.add_argument('--size', required=True, choices=['S', 'M', 'L', 'XL'])
    parser.add_argument('--age-group', required=True, choices=['young', 'middle', 'senior'])
    parser.add_argument('--ethnicity', required=True)
    parser.add_argument('--output', required=True, help='Output image path')
    
    args = parser.parse_args()
    
    # Example measurements
    measurements = {
        'chest': 98,
        'waist': 83,
        'hip': 98,
        'height': 177
    }
    
    generator = RealisticModelGenerator()
    success = generator.generate_model(
        args.gender,
        args.size,
        args.age_group,
        args.ethnicity,
        measurements,
        args.output
    )
    
    if success:
        print(f"Model generated successfully: {args.output}")
    else:
        print("Model generation failed")
