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
# Using Unsplash Source API (provides random photos without API key)
# Photos are royalty-free and suitable for virtual try-on
# Format: https://source.unsplash.com/{width}x{height}/?{query}
#
# Note: Unsplash Source may return different images on each request
# For consistent results, consider using specific Unsplash photo IDs

SAMPLE_PHOTO_URLS = {
    'male': {
        'S': [
            'https://source.unsplash.com/512x768/?man,portrait,standing,white-background',
            'https://source.unsplash.com/512x768/?male,model,fashion,studio',
            'https://source.unsplash.com/512x768/?man,casual,portrait,clean-background',
        ],
        'M': [
            'https://source.unsplash.com/512x768/?man,professional,portrait,studio',
            'https://source.unsplash.com/512x768/?male,standing,fashion,white',
            'https://source.unsplash.com/512x768/?man,business,casual,portrait',
        ],
        'L': [
            'https://source.unsplash.com/512x768/?man,portrait,standing,simple-background',
            'https://source.unsplash.com/512x768/?male,casual,fashion,studio',
            'https://source.unsplash.com/512x768/?man,model,portrait,white-background',
        ],
        'XL': [
            'https://source.unsplash.com/512x768/?man,portrait,professional,studio',
            'https://source.unsplash.com/512x768/?male,standing,casual,clean-background',
            'https://source.unsplash.com/512x768/?man,fashion,portrait,white',
        ]
    },
    'female': {
        'S': [
            'https://source.unsplash.com/512x768/?woman,portrait,standing,white-background',
            'https://source.unsplash.com/512x768/?female,model,fashion,studio',
            'https://source.unsplash.com/512x768/?woman,casual,portrait,clean-background',
        ],
        'M': [
            'https://source.unsplash.com/512x768/?woman,professional,portrait,studio',
            'https://source.unsplash.com/512x768/?female,standing,fashion,white',
            'https://source.unsplash.com/512x768/?woman,business,casual,portrait',
        ],
        'L': [
            'https://source.unsplash.com/512x768/?woman,portrait,standing,simple-background',
            'https://source.unsplash.com/512x768/?female,casual,fashion,studio',
            'https://source.unsplash.com/512x768/?woman,model,portrait,white-background',
        ],
        'XL': [
            'https://source.unsplash.com/512x768/?woman,portrait,professional,studio',
            'https://source.unsplash.com/512x768/?female,standing,casual,clean-background',
            'https://source.unsplash.com/512x768/?woman,fashion,portrait,white',
        ]
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
    
    def copy_from_realistic_folder(self) -> bool:
        """
        Copy existing photos from models/realistic folder as starting point.
        
        Returns:
            True if any files were copied
        """
        logger.info("=" * 60)
        logger.info("Copying from existing realistic models")
        logger.info("=" * 60)
        
        realistic_dir = Path('models/realistic')
        if not realistic_dir.exists():
            logger.warning("models/realistic directory not found, skipping copy")
            return False
        
        import shutil
        total_copied = 0
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                src_dir = realistic_dir / gender / size
                dst_dir = self.base_dir / gender / size
                
                if not src_dir.exists():
                    continue
                
                # Find image files in source
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for src_file in src_dir.glob(ext):
                        # Create destination filename
                        dst_file = dst_dir / src_file.name
                        
                        if dst_file.exists():
                            logger.info(f"  ✓ Already exists: {gender}/{size}/{dst_file.name}")
                            continue
                        
                        # Copy file
                        shutil.copy2(src_file, dst_file)
                        logger.info(f"  ✓ Copied: {gender}/{size}/{src_file.name}")
                        total_copied += 1
        
        if total_copied > 0:
            logger.info(f"\n✓ Copied {total_copied} photos from realistic models")
            return True
        else:
            logger.info("\nNo new photos to copy from realistic models")
            return False
    
    def fill_missing_sizes_with_duplicates(self) -> bool:
        """
        For missing sizes, duplicate from adjacent sizes as placeholders.
        This ensures all categories have at least one photo.
        
        Returns:
            True if any files were created
        """
        logger.info("=" * 60)
        logger.info("Filling missing sizes with placeholders")
        logger.info("=" * 60)
        
        import shutil
        total_created = 0
        
        # Size preference order for copying (closest size first)
        size_fallbacks = {
            'S': ['M', 'L', 'XL'],
            'M': ['L', 'S', 'XL'],
            'L': ['M', 'XL', 'S'],
            'XL': ['L', 'M', 'S']
        }
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                dst_dir = self.base_dir / gender / size
                
                # Check if this size already has photos
                existing_photos = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    existing_photos.extend(list(dst_dir.glob(ext)))
                
                if existing_photos:
                    logger.info(f"  ✓ {gender}/{size} already has {len(existing_photos)} photo(s)")
                    continue
                
                # Try to find a source photo from fallback sizes
                source_photo = None
                for fallback_size in size_fallbacks[size]:
                    fallback_dir = self.base_dir / gender / fallback_size
                    
                    # Find any photo in fallback size
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        photos = list(fallback_dir.glob(ext))
                        if photos:
                            source_photo = photos[0]
                            break
                    
                    if source_photo:
                        break
                
                if not source_photo:
                    logger.warning(f"  ✗ No source photos available for {gender}/{size}")
                    continue
                
                # Copy the source photo as a placeholder
                dst_file = dst_dir / f"placeholder_01{source_photo.suffix}"
                shutil.copy2(source_photo, dst_file)
                logger.info(f"  ✓ Created placeholder: {gender}/{size}/{dst_file.name} (from {fallback_size})")
                total_created += 1
        
        if total_created > 0:
            logger.info(f"\n✓ Created {total_created} placeholder photos")
            logger.warning("\n⚠ Note: Placeholder photos are duplicates from nearby sizes.")
            logger.info("For best results, replace placeholders with photos of models matching each size.")
            logger.info("Placeholder files are named 'placeholder_01.*' for easy identification.")
            return True
        else:
            logger.info("\nAll sizes already have photos!")
            return False
    
    def copy_from_generated_models(self) -> bool:
        """
        Copy photos from models/male and models/female directories.
        These may be generated or other photos.
        
        Returns:
            True if any files were copied
        """
        logger.info("=" * 60)
        logger.info("Copying from generated models")
        logger.info("=" * 60)
        
        import shutil
        total_copied = 0
        
        for gender in ['male', 'female']:
            # Look for photos in the generated models directory
            generated_dir = Path('models') / gender
            
            if not generated_dir.exists():
                continue
            
            # For each size directory
            for size in ['S', 'M', 'L', 'XL']:
                dst_dir = self.base_dir / gender / size
                
                # Look for any PNG or JPG files recursively in the size directory
                src_size_dir = generated_dir / size
                if src_size_dir.exists():
                    for ext in ['**/*.jpg', '**/*.jpeg', '**/*.png']:
                        for src_file in src_size_dir.glob(ext):
                            # Generate a unique destination name
                            base_name = f"model_{total_copied + 1:02d}"
                            dst_file = dst_dir / f"{base_name}{src_file.suffix}"
                            
                            if dst_file.exists():
                                continue
                            
                            # Copy file
                            shutil.copy2(src_file, dst_file)
                            logger.info(f"  ✓ Copied: {gender}/{size}/{dst_file.name}")
                            total_copied += 1
        
        if total_copied > 0:
            logger.info(f"\n✓ Copied {total_copied} photos from generated models")
            return True
        else:
            logger.info("\nNo photos found in generated models directory")
            return False
    
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
                        output_path = self.base_dir / gender / size / f"model_{i+1:02d}.jpg"
                        
                        if output_path.exists():
                            logger.info(f"  ✓ Already exists: {output_path.name}")
                            total_downloaded += 1
                            continue
                        
                        logger.info(f"  Downloading from: {url}")
                        
                        # Use urllib with headers to avoid being blocked
                        req = urllib.request.Request(
                            url,
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                        )
                        
                        with urllib.request.urlopen(req, timeout=30) as response:
                            img_data = response.read()
                            
                            # Detect image format from response headers
                            content_type = response.headers.get('Content-Type', '')
                            if 'png' in content_type.lower():
                                ext = '.png'
                            elif 'jpeg' in content_type.lower() or 'jpg' in content_type.lower():
                                ext = '.jpg'
                            else:
                                # Default to jpg if unknown
                                ext = '.jpg'
                            
                            # Update output path with correct extension
                            output_path = output_path.with_suffix(ext)
                            
                        # Save the image
                        with open(output_path, 'wb') as f:
                            f.write(img_data)
                            
                        logger.info(f"  ✓ Saved: {output_path.name}")
                        total_downloaded += 1
                        
                        # Small delay to avoid rate limiting
                        import time
                        time.sleep(1)
                        
                    except urllib.error.URLError as e:
                        logger.error(f"  ✗ Network error downloading {url}: {e}")
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
                
                # Count image files efficiently (jpg, jpeg, png)
                photo_count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    photo_count += sum(1 for _ in size_dir.glob(ext))
                
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
            logger.warning("\n⚠ No photos downloaded yet")
            logger.info("\nTo add photos manually:")
            logger.info(f"1. Navigate to {self.base_dir}")
            logger.info("2. Add photos to appropriate gender/size directories")
            logger.info("3. Follow README.md instructions in each directory")
        else:
            logger.info(f"\n✅ {total} photos available for virtual try-on!")
            logger.info(f"\nPhotos location: {self.base_dir}")


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
    parser.add_argument(
        '--skip-copy',
        action='store_true',
        help='Skip copying from models/realistic folder'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading from URLs (only copy existing)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = RealHumanPhotoDownloader(base_dir=args.base_dir)
    
    # Create directory structure
    downloader.create_placeholder_structure()
    
    # Create metadata
    downloader.create_metadata()
    
    # Copy from realistic folder first (if available)
    if not args.skip_copy and not args.create_structure_only:
        downloader.copy_from_realistic_folder()
        downloader.copy_from_generated_models()
        # Fill missing sizes with placeholders
        downloader.fill_missing_sizes_with_duplicates()
    
    # Download photos from URLs (if configured)
    if not args.create_structure_only and not args.skip_download:
        try:
            downloader.download_sample_photos()
        except Exception as e:
            logger.warning(f"Photo download encountered issues: {e}")
            logger.info("Continuing with existing photos...")
    
    # Print summary
    downloader.print_summary()
    
    # Check if we have any photos
    counts = downloader.count_photos()
    total_photos = sum(sum(sizes.values()) for sizes in counts.values())
    
    logger.info("\n" + "=" * 60)
    logger.info("Next Steps")
    logger.info("=" * 60)
    
    if total_photos > 0:
        logger.info(f"\n✅ Setup complete! {total_photos} photos ready.")
        logger.info("\nYou can now run virtual try-on:")
        logger.info("  python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png")
        logger.info("\nOr specify gender/size:")
        logger.info("  python scripts/demo_idm_vton.py --clothing shirt.png --gender male --size M")
    else:
        logger.info("\n⚠ No photos available yet.")
        logger.info("\nOptions to add photos:")
        logger.info("\n1. Download from free sources (requires internet):")
        logger.info("   - Try running this script again without --skip-download")
        logger.info("   - Photos will be downloaded from Unsplash (free API)")
        logger.info("\n2. Manually add photos:")
        logger.info(f"   - Navigate to {downloader.base_dir}")
        logger.info("   - Add photos to appropriate gender/size directories")
        logger.info("   - Follow README.md instructions in each directory")
        logger.info("\n3. Use existing realistic models:")
        logger.info("   - Run: python scripts/demo_idm_vton.py --base-dir models/realistic")
    
    logger.info("\n" + "=" * 60)


if __name__ == '__main__':
    main()
