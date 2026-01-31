"""
Generate sample clothing images for testing.
Creates simple clothing shapes with transparent backgrounds using Pillow.
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw


def create_tshirt(width, height, color):
    """
    Create a simple t-shirt shape with transparent background.
    
    Args:
        width: Image width
        height: Image height
        color: RGB tuple for t-shirt color
    
    Returns:
        PIL Image with t-shirt (RGBA with transparency)
    """
    # Create image with transparent background
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Calculate t-shirt dimensions
    center_x = width // 2
    top_y = int(height * 0.1)
    bottom_y = int(height * 0.9)
    
    # Shoulders
    shoulder_left = int(width * 0.1)
    shoulder_right = int(width * 0.9)
    shoulder_bottom = int(height * 0.25)
    
    # Body
    body_left = int(width * 0.25)
    body_right = int(width * 0.75)
    
    # Create t-shirt shape (polygon)
    tshirt_points = [
        # Left shoulder
        (shoulder_left, top_y),
        (shoulder_left, shoulder_bottom),
        (body_left, shoulder_bottom),
        # Body
        (body_left, bottom_y),
        (body_right, bottom_y),
        (body_right, shoulder_bottom),
        # Right shoulder
        (shoulder_right, shoulder_bottom),
        (shoulder_right, top_y),
        # Neck (slight curve using straight line approximation)
        (center_x + int(width * 0.1), top_y),
        (center_x + int(width * 0.1), int(height * 0.15)),
        (center_x - int(width * 0.1), int(height * 0.15)),
        (center_x - int(width * 0.1), top_y),
    ]
    
    # Draw t-shirt
    draw.polygon(tshirt_points, fill=color + (255,))  # Add alpha channel
    
    return img


def create_pants(width, height, color):
    """
    Create simple pants shape with transparent background.
    
    Args:
        width: Image width
        height: Image height
        color: RGB tuple for pants color
    
    Returns:
        PIL Image with pants (RGBA with transparency)
    """
    # Create image with transparent background
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Calculate pants dimensions
    center_x = width // 2
    top_y = int(height * 0.05)
    bottom_y = int(height * 0.95)
    
    waist_width = int(width * 0.8)
    leg_width = int(width * 0.35)
    leg_gap = int(width * 0.1)
    
    # Create left leg
    left_leg = [
        (center_x - waist_width // 2, top_y),
        (center_x - leg_gap // 2, top_y),
        (center_x - leg_gap // 2, bottom_y),
        (center_x - leg_gap // 2 - leg_width, bottom_y),
    ]
    draw.polygon(left_leg, fill=color + (255,))
    
    # Create right leg
    right_leg = [
        (center_x + leg_gap // 2, top_y),
        (center_x + waist_width // 2, top_y),
        (center_x + leg_gap // 2 + leg_width, bottom_y),
        (center_x + leg_gap // 2, bottom_y),
    ]
    draw.polygon(right_leg, fill=color + (255,))
    
    return img


def create_sample_clothing():
    """Create all sample clothing images."""
    base_dir = Path(__file__).parent.parent
    clothing_dir = base_dir / "samples" / "clothing"
    clothing_dir.mkdir(parents=True, exist_ok=True)
    
    # Clothing specifications
    clothing_items = [
        {
            'name': 'tshirt_blue.png',
            'type': 'tshirt',
            'color': (65, 105, 225),  # Royal blue
            'width': 200,
            'height': 250
        },
        {
            'name': 'tshirt_red.png',
            'type': 'tshirt',
            'color': (220, 20, 60),  # Crimson
            'width': 200,
            'height': 250
        },
        {
            'name': 'pants_black.png',
            'type': 'pants',
            'color': (30, 30, 30),  # Near black
            'width': 180,
            'height': 500
        }
    ]
    
    created_items = []
    
    for item_spec in clothing_items:
        # Generate image
        if item_spec['type'] == 'tshirt':
            img = create_tshirt(
                item_spec['width'],
                item_spec['height'],
                item_spec['color']
            )
        elif item_spec['type'] == 'pants':
            img = create_pants(
                item_spec['width'],
                item_spec['height'],
                item_spec['color']
            )
        
        # Save image
        output_path = clothing_dir / item_spec['name']
        img.save(output_path)
        
        item_path = f"samples/clothing/{item_spec['name']}"
        created_items.append(item_path)
        print(f"✅ Created clothing: {item_path}")
    
    return created_items


if __name__ == "__main__":
    print("Generating sample clothing images...")
    items = create_sample_clothing()
    print(f"\n✨ Successfully created {len(items)} sample clothing items!")
