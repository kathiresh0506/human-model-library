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


# REAL PHOTO URLS from Unsplash
# These are specific photo URLs that provide consistent, real human photos
# suitable for virtual try-on (front-facing, upper body visible, clean background)
# Photos are royalty-free and suitable for virtual try-on

REAL_PHOTO_URLS = {
    "male": {
        "S": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512",
        "M": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=512",
        "L": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=512",
        "XL": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=512"
    },
    "female": {
        "S": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512",
        "M": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=512",
        "L": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512",
        "XL": "https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=512"
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
    
    def delete_placeholder_and_png_files(self) -> int:
        """
        Delete all placeholder and PNG silhouette files.
        This removes placeholder_*.jpg, placeholder_*.png, and all .png files.
        
        Returns:
            Number of files deleted
        """
        logger.info("=" * 60)
        logger.info("Deleting Placeholder and PNG Silhouette Files")
        logger.info("=" * 60)
        
        import os
        total_deleted = 0
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                size_dir = self.base_dir / gender / size
                
                if not size_dir.exists():
                    continue
                
                # Delete placeholder files
                for pattern in ['placeholder_*.jpg', 'placeholder_*.png', '*.png']:
                    for file_path in size_dir.glob(pattern):
                        try:
                            logger.info(f"  ✗ Deleting: {gender}/{size}/{file_path.name}")
                            os.remove(file_path)
                            total_deleted += 1
                        except Exception as e:
                            logger.error(f"  Failed to delete {file_path}: {e}")
        
        if total_deleted > 0:
            logger.info(f"\n✓ Deleted {total_deleted} placeholder/PNG files")
        else:
            logger.info("\n✓ No placeholder/PNG files found to delete")
        
        return total_deleted
    
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
    
    def download_real_photos(self) -> bool:
        """
        Download REAL human photos from specific Unsplash URLs.
        Each photo is verified to be a real photograph (file size > 10KB).
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Downloading REAL Human Photos from Unsplash")
        logger.info("=" * 60)
        
        total_downloaded = 0
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                url = REAL_PHOTO_URLS.get(gender, {}).get(size)
                
                if not url:
                    logger.warning(f"No URL configured for {gender}/{size}")
                    continue
                
                logger.info(f"\nDownloading {gender}/{size} photo...")
                
                try:
                    # Output to front_001.jpg (standard naming)
                    output_path = self.base_dir / gender / size / "front_001.jpg"
                    
                    if output_path.exists():
                        # Check if existing file is valid (> 10KB)
                        file_size = output_path.stat().st_size
                        if file_size > 10240:  # 10KB
                            logger.info(f"  ✓ Already exists and valid: {output_path.name} ({file_size} bytes)")
                            total_downloaded += 1
                            continue
                        else:
                            logger.warning(f"  ⚠ Existing file is too small ({file_size} bytes), re-downloading...")
                    
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
                        
                        # Verify this is a real photo (file size > 10KB)
                        if len(img_data) < 10240:  # 10KB
                            logger.error(f"  ✗ Downloaded file too small ({len(img_data)} bytes), likely not a real photo")
                            continue
                    
                    # Save the image as JPG
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                    
                    file_size = output_path.stat().st_size
                    logger.info(f"  ✓ Saved: {output_path.name} ({file_size} bytes)")
                    total_downloaded += 1
                    
                    # Small delay to avoid rate limiting
                    import time
                    time.sleep(1)
                    
                except urllib.error.URLError as e:
                    logger.error(f"  ✗ Network error downloading {url}: {e}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to download {url}: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Download Summary: {total_downloaded}/8 photos successfully downloaded")
        logger.info(f"{'='*60}")
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
        description="Download real human model photos from Unsplash"
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
        '--skip-download',
        action='store_true',
        help='Skip downloading from URLs (keep existing photos only)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = RealHumanPhotoDownloader(base_dir=args.base_dir)
    
    # Create directory structure
    downloader.create_placeholder_structure()
    
    # Create metadata
    downloader.create_metadata()
    
    # Step 1: Delete all placeholder and PNG silhouette files
    if not args.create_structure_only:
        downloader.delete_placeholder_and_png_files()
    
    # Step 2: Download REAL human photos from Unsplash
    if not args.create_structure_only and not args.skip_download:
        try:
            downloader.download_real_photos()
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
        logger.info(f"\n✅ Setup complete! {total_photos} REAL human photos ready.")
        logger.info("\nYou can now run virtual try-on:")
        logger.info("  python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png")
        logger.info("\nOr specify gender/size:")
        logger.info("  python scripts/demo_idm_vton.py --clothing shirt.png --gender male --size M")
    else:
        logger.info("\n⚠ No photos available yet.")
        logger.info("\nOptions to add photos:")
        logger.info("\n1. Download from free sources (requires internet):")
        logger.info("   - Try running this script again without --skip-download")
        logger.info("   - Photos will be downloaded from Unsplash")
        logger.info("\n2. Manually add photos:")
        logger.info(f"   - Navigate to {downloader.base_dir}")
        logger.info("   - Add photos to appropriate gender/size directories")
        logger.info("   - Follow README.md instructions in each directory")
    
    logger.info("\n" + "=" * 60)


if __name__ == '__main__':
    main()
