#!/usr/bin/env python3
"""
Download real human model photos from various sources.

Sources:
1. VITON-HD test dataset (real person images)
2. Unsplash API (free stock photos)
3. Pexels API (free stock photos)
4. DeepFashion dataset (academic use)

Organizes photos by gender and size (S, M, L, XL).
"""
import os
import sys
import json
import argparse
import logging
import urllib.request
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# VITON-HD test dataset URLs (Google Drive)
VITON_HD_URLS = {
    'test_images': {
        'url': 'https://drive.google.com/uc?export=download&id=1Uc0DTTkSfCPXDhd4CMx2TQlzlC6QpgqJ',
        'filename': 'viton_hd_test.zip',
        'md5': None  # To be filled after download
    }
}

# Sample Unsplash/Pexels query parameters
PHOTO_SEARCH_QUERIES = {
    'male': [
        'male fashion model standing white background',
        'man full body standing pose',
        'male model front view standing'
    ],
    'female': [
        'female fashion model standing white background',
        'woman full body standing pose',
        'female model front view standing'
    ]
}


class RealModelDownloader:
    """Download and organize real human model photos."""
    
    def __init__(self, base_dir: str = 'models/realistic'):
        """
        Initialize downloader.
        
        Args:
            base_dir: Base directory for models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create size directories
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                size_dir = self.base_dir / gender / size
                size_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized downloader with base_dir: {self.base_dir}")
    
    def download_viton_dataset(self) -> bool:
        """
        Download VITON-HD test dataset (real person images).
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Downloading VITON-HD Test Dataset")
        logger.info("=" * 60)
        
        try:
            # Note: VITON-HD dataset requires manual download from official sources
            # Due to Google Drive restrictions, direct download may not work
            logger.info("VITON-HD dataset download requires manual steps:")
            logger.info("1. Visit: https://github.com/shadow2496/VITON-HD")
            logger.info("2. Download test images from the official repository")
            logger.info("3. Extract to: models/viton_hd/test/image/")
            logger.info("4. Run this script with --organize-viton to organize by size")
            
            return False
            
        except Exception as e:
            logger.error(f"Error downloading VITON dataset: {e}")
            return False
    
    def download_from_unsplash(self, api_key: Optional[str] = None, count: int = 5) -> bool:
        """
        Download fashion model photos from Unsplash.
        
        Args:
            api_key: Unsplash API key (optional)
            count: Number of photos to download per category
            
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Downloading from Unsplash")
        logger.info("=" * 60)
        
        if not api_key:
            logger.warning("No Unsplash API key provided")
            logger.info("To download from Unsplash:")
            logger.info("1. Get free API key from: https://unsplash.com/developers")
            logger.info("2. Run: python scripts/download_real_models.py --unsplash-key YOUR_KEY")
            return False
        
        try:
            import requests
            
            base_url = "https://api.unsplash.com/search/photos"
            headers = {"Authorization": f"Client-ID {api_key}"}
            
            downloaded = 0
            
            for gender, queries in PHOTO_SEARCH_QUERIES.items():
                for query in queries:
                    logger.info(f"Searching for: {query}")
                    
                    params = {
                        'query': query,
                        'per_page': count,
                        'orientation': 'portrait'
                    }
                    
                    response = requests.get(base_url, headers=headers, params=params)
                    
                    if response.status_code != 200:
                        logger.error(f"Unsplash API error: {response.status_code}")
                        continue
                    
                    results = response.json().get('results', [])
                    
                    for idx, photo in enumerate(results):
                        # Download full resolution image
                        image_url = photo['urls']['full']
                        photo_id = photo['id']
                        
                        # Estimate size based on image properties (simple heuristic)
                        # In production, would use body measurement estimation
                        size = self._estimate_size_from_index(idx, len(results))
                        
                        # Download image
                        output_path = self.base_dir / gender / size / f"unsplash_{photo_id}.jpg"
                        
                        if output_path.exists():
                            logger.info(f"Already exists: {output_path}")
                            continue
                        
                        logger.info(f"Downloading: {image_url} -> {output_path}")
                        urllib.request.urlretrieve(image_url, str(output_path))
                        downloaded += 1
            
            logger.info(f"✓ Downloaded {downloaded} photos from Unsplash")
            return True
            
        except ImportError:
            logger.error("requests library not installed. Install with: pip install requests")
            return False
        except Exception as e:
            logger.error(f"Error downloading from Unsplash: {e}")
            return False
    
    def download_from_pexels(self, api_key: Optional[str] = None, count: int = 5) -> bool:
        """
        Download fashion model photos from Pexels.
        
        Args:
            api_key: Pexels API key (optional)
            count: Number of photos to download per category
            
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Downloading from Pexels")
        logger.info("=" * 60)
        
        if not api_key:
            logger.warning("No Pexels API key provided")
            logger.info("To download from Pexels:")
            logger.info("1. Get free API key from: https://www.pexels.com/api/")
            logger.info("2. Run: python scripts/download_real_models.py --pexels-key YOUR_KEY")
            return False
        
        try:
            import requests
            
            base_url = "https://api.pexels.com/v1/search"
            headers = {"Authorization": api_key}
            
            downloaded = 0
            
            for gender, queries in PHOTO_SEARCH_QUERIES.items():
                for query in queries:
                    logger.info(f"Searching for: {query}")
                    
                    params = {
                        'query': query,
                        'per_page': count,
                        'orientation': 'portrait'
                    }
                    
                    response = requests.get(base_url, headers=headers, params=params)
                    
                    if response.status_code != 200:
                        logger.error(f"Pexels API error: {response.status_code}")
                        continue
                    
                    results = response.json().get('photos', [])
                    
                    for idx, photo in enumerate(results):
                        # Download large image
                        image_url = photo['src']['large']
                        photo_id = photo['id']
                        
                        # Estimate size
                        size = self._estimate_size_from_index(idx, len(results))
                        
                        # Download image
                        output_path = self.base_dir / gender / size / f"pexels_{photo_id}.jpg"
                        
                        if output_path.exists():
                            logger.info(f"Already exists: {output_path}")
                            continue
                        
                        logger.info(f"Downloading: {image_url} -> {output_path}")
                        urllib.request.urlretrieve(image_url, str(output_path))
                        downloaded += 1
            
            logger.info(f"✓ Downloaded {downloaded} photos from Pexels")
            return True
            
        except ImportError:
            logger.error("requests library not installed. Install with: pip install requests")
            return False
        except Exception as e:
            logger.error(f"Error downloading from Pexels: {e}")
            return False
    
    def _estimate_size_from_index(self, idx: int, total: int) -> str:
        """
        Estimate size category from index (simple distribution).
        
        Args:
            idx: Photo index
            total: Total photos
            
        Returns:
            Size category ('S', 'M', 'L', or 'XL')
        """
        # Simple distribution: S=25%, M=35%, L=25%, XL=15%
        ratio = idx / max(total - 1, 1)
        
        if ratio < 0.25:
            return 'S'
        elif ratio < 0.60:
            return 'M'
        elif ratio < 0.85:
            return 'L'
        else:
            return 'XL'
    
    def generate_metadata(self) -> bool:
        """
        Generate metadata.json for the photo library.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Generating Metadata")
        logger.info("=" * 60)
        
        try:
            metadata = {
                'version': '1.0',
                'description': 'Real human model photo library',
                'sources': [
                    'VITON-HD test dataset',
                    'Unsplash API',
                    'Pexels API'
                ],
                'organization': {
                    'structure': 'models/realistic/{gender}/{size}/',
                    'sizes': ['S', 'M', 'L', 'XL'],
                    'genders': ['male', 'female']
                },
                'requirements': {
                    'resolution': '≥512x768',
                    'format': 'JPEG',
                    'background': 'clean/white/gray preferred',
                    'pose': 'front-facing, standing, arms slightly away'
                },
                'models': {}
            }
            
            # Count models per category
            for gender in ['male', 'female']:
                metadata['models'][gender] = {}
                for size in ['S', 'M', 'L', 'XL']:
                    size_dir = self.base_dir / gender / size
                    if size_dir.exists():
                        photos = list(size_dir.glob('*.jpg')) + list(size_dir.glob('*.jpeg'))
                        metadata['models'][gender][size] = {
                            'count': len(photos),
                            'files': [p.name for p in photos]
                        }
            
            # Save metadata
            metadata_path = self.base_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Metadata saved to: {metadata_path}")
            
            # Print summary
            logger.info("\nModel Library Summary:")
            for gender in ['male', 'female']:
                logger.info(f"\n  {gender.upper()}:")
                for size in ['S', 'M', 'L', 'XL']:
                    count = metadata['models'][gender][size]['count']
                    logger.info(f"    {size}: {count} photos")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return False
    
    def validate_photos(self) -> bool:
        """
        Validate downloaded photos meet requirements.
        
        Returns:
            True if validation passes
        """
        logger.info("=" * 60)
        logger.info("Validating Photos")
        logger.info("=" * 60)
        
        try:
            from PIL import Image
            
            min_width = 512
            min_height = 768
            issues = []
            
            for gender in ['male', 'female']:
                for size in ['S', 'M', 'L', 'XL']:
                    size_dir = self.base_dir / gender / size
                    if not size_dir.exists():
                        continue
                    
                    photos = list(size_dir.glob('*.jpg')) + list(size_dir.glob('*.jpeg'))
                    
                    for photo_path in photos:
                        try:
                            with Image.open(photo_path) as img:
                                width, height = img.size
                                
                                if width < min_width or height < min_height:
                                    issues.append(
                                        f"{photo_path.name}: {width}x{height} "
                                        f"(min: {min_width}x{min_height})"
                                    )
                        except Exception as e:
                            issues.append(f"{photo_path.name}: Error opening - {e}")
            
            if issues:
                logger.warning(f"Found {len(issues)} validation issues:")
                for issue in issues[:10]:  # Show first 10
                    logger.warning(f"  - {issue}")
                return False
            else:
                logger.info("✓ All photos validated successfully")
                return True
                
        except ImportError:
            logger.warning("PIL not installed, skipping validation")
            return True
        except Exception as e:
            logger.error(f"Error validating photos: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Download real human model photos from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Unsplash (requires API key)
  python scripts/download_real_models.py --unsplash-key YOUR_KEY --count 10
  
  # Download from Pexels (requires API key)
  python scripts/download_real_models.py --pexels-key YOUR_KEY --count 10
  
  # Download from both sources
  python scripts/download_real_models.py --unsplash-key KEY1 --pexels-key KEY2
  
  # Generate metadata for existing photos
  python scripts/download_real_models.py --metadata-only
  
  # Validate photos
  python scripts/download_real_models.py --validate

API Keys (Free):
  Unsplash: https://unsplash.com/developers
  Pexels: https://www.pexels.com/api/
  
Note: VITON-HD dataset requires manual download from:
  https://github.com/shadow2496/VITON-HD
        """
    )
    
    parser.add_argument('--base-dir', default='models/realistic',
                       help='Base directory for models')
    parser.add_argument('--unsplash-key', help='Unsplash API key')
    parser.add_argument('--pexels-key', help='Pexels API key')
    parser.add_argument('--count', type=int, default=5,
                       help='Number of photos to download per category')
    parser.add_argument('--metadata-only', action='store_true',
                       help='Only generate metadata for existing photos')
    parser.add_argument('--validate', action='store_true',
                       help='Validate downloaded photos')
    parser.add_argument('--download-viton', action='store_true',
                       help='Show instructions for VITON-HD dataset')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = RealModelDownloader(args.base_dir)
    
    # Execute requested operations
    if args.download_viton:
        downloader.download_viton_dataset()
    
    if args.metadata_only:
        downloader.generate_metadata()
        return
    
    if args.validate:
        downloader.validate_photos()
        return
    
    # Download from sources
    if args.unsplash_key:
        downloader.download_from_unsplash(args.unsplash_key, args.count)
    
    if args.pexels_key:
        downloader.download_from_pexels(args.pexels_key, args.count)
    
    # Generate metadata
    downloader.generate_metadata()
    
    # Validate
    downloader.validate_photos()
    
    logger.info("\n" + "=" * 60)
    logger.info("Download Complete!")
    logger.info("=" * 60)
    logger.info("\nNext Steps:")
    logger.info("1. Review downloaded photos in: models/realistic/")
    logger.info("2. Download VITON-HD weights: python scripts/download_viton_weights.py")
    logger.info("3. Test virtual try-on: python scripts/demo_viton_gpu.py")


if __name__ == "__main__":
    main()
