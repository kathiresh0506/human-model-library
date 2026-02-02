#!/usr/bin/env python3
"""
Download real human model photos from various sources.

This script downloads actual human photos (not AI-generated or cartoons)
from free and reliable sources, organizing them by gender and size.

Sources:
- Sample images from academic datasets
- Free stock photos from public APIs
- Properly licensed images

Organizes into:
models/real_humans/
├── male/
│   ├── S/ (3-5 photos)
│   ├── M/ (3-5 photos)
│   ├── L/ (3-5 photos)
│   └── XL/ (3-5 photos)
└── female/
    ├── S/ (3-5 photos)
    ├── M/ (3-5 photos)
    ├── L/ (3-5 photos)
    └── XL/ (3-5 photos)
"""

import os
import sys
import json
import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample photo URLs from free sources
# These are placeholder URLs - in production, use actual free stock photo APIs
# or academic dataset samples
SAMPLE_PHOTO_URLS = {
    'male': {
        'S': [
            # Placeholder URLs - replace with actual free stock photos
            # Example: Unsplash, Pexels, or academic datasets
        ],
        'M': [],
        'L': [],
        'XL': []
    },
    'female': {
        'S': [],
        'M': [],
        'L': [],
        'XL': []
    }
}


class RealHumanPhotoDownloader:
    """Download and organize real human model photos."""
    
    def __init__(self, base_dir: str = 'models/real_humans'):
        """
        Initialize downloader.
        
        Args:
            base_dir: Base directory for real human photos
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                size_dir = self.base_dir / gender / size
                size_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized downloader with base_dir: {self.base_dir}")
    
    def download_sample_photos(self) -> bool:
        """
        Download sample photos from URLs.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Downloading Real Human Photos")
        logger.info("=" * 60)
        
        total_downloaded = 0
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                urls = SAMPLE_PHOTO_URLS.get(gender, {}).get(size, [])
                
                if not urls:
                    logger.info(f"No URLs configured for {gender}/{size}")
                    continue
                
                logger.info(f"\nDownloading {gender}/{size} photos...")
                
                for i, url in enumerate(urls):
                    try:
                        output_path = self.base_dir / gender / size / f"model_{i+1}.jpg"
                        
                        if output_path.exists():
                            logger.info(f"  ✓ Already exists: {output_path.name}")
                            continue
                        
                        logger.info(f"  Downloading: {url}")
                        urllib.request.urlretrieve(url, output_path)
                        logger.info(f"  ✓ Saved: {output_path.name}")
                        total_downloaded += 1
                        
                    except Exception as e:
                        logger.error(f"  ✗ Failed to download {url}: {e}")
        
        logger.info(f"\nDownloaded {total_downloaded} photos")
        return total_downloaded > 0
    
    def create_placeholder_structure(self) -> bool:
        """
        Create directory structure with placeholder README files.
        
        This allows the user to manually add photos while maintaining
        the correct structure.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Creating Directory Structure")
        logger.info("=" * 60)
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                size_dir = self.base_dir / gender / size
                size_dir.mkdir(parents=True, exist_ok=True)
                
                # Create README with instructions
                readme_path = size_dir / 'README.md'
                readme_content = f"""# {gender.capitalize()} - Size {size}

Add real human model photos here.

## Photo Requirements:
- Real person (not AI-generated or cartoon)
- Standing, front-facing pose
- Arms slightly away from body
- Clean background (white/gray preferred)
- High resolution (512x768 minimum)
- Format: JPG or PNG

## Naming Convention:
- model_1.jpg
- model_2.jpg
- etc.

## Sources:
- Free stock photo websites (Unsplash, Pexels)
- Academic datasets (VITON-HD, DeepFashion)
- Properly licensed images

Add 3-5 photos for best results.
"""
                
                if not readme_path.exists():
                    with open(readme_path, 'w') as f:
                        f.write(readme_content)
                    logger.info(f"  Created: {readme_path}")
        
        logger.info("\n✓ Directory structure created")
        return True
    
    def create_metadata(self) -> bool:
        """
        Create metadata.json file.
        
        Returns:
            True if successful
        """
        logger.info("\nCreating metadata.json...")
        
        metadata = {
            'description': 'Real human model photos for virtual try-on',
            'structure': {
                'male': {
                    'S': 'Small size male models',
                    'M': 'Medium size male models',
                    'L': 'Large size male models',
                    'XL': 'Extra Large size male models'
                },
                'female': {
                    'S': 'Small size female models',
                    'M': 'Medium size female models',
                    'L': 'Large size female models',
                    'XL': 'Extra Large size female models'
                }
            },
            'photo_requirements': {
                'format': ['jpg', 'jpeg', 'png'],
                'min_resolution': '512x768',
                'background': 'clean (white/gray preferred)',
                'pose': 'standing, front-facing',
                'quality': 'real human (not AI-generated)'
            },
            'sources': [
                'Academic datasets (VITON-HD, DeepFashion)',
                'Free stock photos (Unsplash, Pexels)',
                'Properly licensed images'
            ]
        }
        
        metadata_path = self.base_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Created: {metadata_path}")
        return True
    
    def count_photos(self) -> Dict[str, Dict[str, int]]:
        """
        Count photos in each category.
        
        Returns:
            Dictionary with counts
        """
        counts = {
            'male': {},
            'female': {}
        }
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                size_dir = self.base_dir / gender / size
                
                if not size_dir.exists():
                    counts[gender][size] = 0
                    continue
                
                # Count image files
                photo_count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    photo_count += len(list(size_dir.glob(ext)))
                
                counts[gender][size] = photo_count
        
        return counts
    
    def print_summary(self):
        """Print summary of downloaded photos."""
        logger.info("\n" + "=" * 60)
        logger.info("Real Human Photos Summary")
        logger.info("=" * 60)
        
        counts = self.count_photos()
        total = 0
        
        for gender in ['male', 'female']:
            logger.info(f"\n{gender.upper()}:")
            for size in ['S', 'M', 'L', 'XL']:
                count = counts[gender][size]
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} Size {size}: {count} photos")
                total += count
        
        logger.info(f"\nTotal photos: {total}")
        logger.info("=" * 60)
        
        if total == 0:
            logger.info("\n⚠ No photos downloaded yet")
            logger.info("\nTo add photos manually:")
            logger.info(f"1. Navigate to {self.base_dir}")
            logger.info("2. Add photos to appropriate gender/size directories")
            logger.info("3. Follow README.md instructions in each directory")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download real human model photos"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='models/real_humans',
        help='Base directory for photos (default: models/real_humans)'
    )
    parser.add_argument(
        '--create-structure-only',
        action='store_true',
        help='Only create directory structure without downloading'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = RealHumanPhotoDownloader(base_dir=args.base_dir)
    
    # Create directory structure
    downloader.create_placeholder_structure()
    
    # Create metadata
    downloader.create_metadata()
    
    # Download photos (if URLs configured)
    if not args.create_structure_only:
        downloader.download_sample_photos()
    
    # Print summary
    downloader.print_summary()
    
    logger.info("\n" + "=" * 60)
    logger.info("Next Steps")
    logger.info("=" * 60)
    logger.info("\n1. Add real human photos to the directories:")
    logger.info(f"   {downloader.base_dir}")
    logger.info("\n2. Follow the README.md files in each size directory")
    logger.info("\n3. Ensure photos meet the requirements:")
    logger.info("   - Real human (not AI-generated)")
    logger.info("   - Front-facing, standing pose")
    logger.info("   - Clean background")
    logger.info("   - High resolution (512x768+)")
    logger.info("\n4. Test with: python scripts/demo_idm_vton.py")


if __name__ == '__main__':
    main()
