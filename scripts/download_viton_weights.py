#!/usr/bin/env python3
"""
Download pre-trained weights for VITON-HD and related models.

Downloads from:
- Hugging Face model hub
- Google Drive (official VITON-HD releases)
- GitHub releases
"""
import os
import sys
import json
import argparse
import logging
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model weights configuration
WEIGHTS_CONFIG = {
    'viton_hd': {
        'generator': {
            'description': 'VITON-HD Generator Network (~170MB)',
            'url': 'https://huggingface.co/spaces/shadow2496/VITON-HD/resolve/main/weights/generator.pth',
            'filename': 'generator.pth',
            'size_mb': 170,
            'md5': None,  # Will be computed after download
            'required': True
        }
    },
    'openpose': {
        'body_pose': {
            'description': 'OpenPose Body 25 Model (~200MB)',
            'url': 'https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth',
            'filename': 'body_pose.pth',
            'size_mb': 200,
            'md5': None,
            'required': True
        }
    },
    'human_parsing': {
        'lip_parsing': {
            'description': 'LIP Human Parsing Model (~50MB)',
            'url': 'https://huggingface.co/mattresnick/human-parsing/resolve/main/lip_final.pth',
            'filename': 'lip_parsing.pth',
            'size_mb': 50,
            'md5': None,
            'required': True
        }
    }
}


class WeightsDownloader:
    """Download and manage model weights."""
    
    def __init__(self, base_dir: str = 'weights'):
        """
        Initialize downloader.
        
        Args:
            base_dir: Base directory for weights
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for model_type in WEIGHTS_CONFIG.keys():
            (self.base_dir / model_type).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized downloader with base_dir: {self.base_dir}")
    
    def download_file(self, url: str, output_path: Path, 
                     expected_size_mb: Optional[int] = None) -> bool:
        """
        Download a file with progress tracking.
        
        Args:
            url: URL to download from
            output_path: Path to save file
            expected_size_mb: Expected file size in MB (for validation)
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Downloading: {url}")
            logger.info(f"Saving to: {output_path}")
            
            # Download with progress
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100.0 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    
                    if block_num % 100 == 0:  # Update every 100 blocks
                        logger.info(f"Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            
            urllib.request.urlretrieve(url, str(output_path), reporthook)
            
            # Validate size
            actual_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Downloaded: {actual_size_mb:.1f} MB")
            
            if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 10:
                logger.warning(
                    f"File size mismatch: expected ~{expected_size_mb}MB, "
                    f"got {actual_size_mb:.1f}MB"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def compute_md5(self, file_path: Path) -> str:
        """
        Compute MD5 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        md5 = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        
        return md5.hexdigest()
    
    def download_model_weights(self, model_type: str, force: bool = False) -> bool:
        """
        Download all weights for a model type.
        
        Args:
            model_type: Type of model ('viton_hd', 'openpose', 'human_parsing')
            force: Force re-download even if file exists
            
        Returns:
            True if all downloads successful
        """
        if model_type not in WEIGHTS_CONFIG:
            logger.error(f"Unknown model type: {model_type}")
            return False
        
        logger.info("=" * 60)
        logger.info(f"Downloading {model_type.upper()} Weights")
        logger.info("=" * 60)
        
        model_dir = self.base_dir / model_type
        weights_config = WEIGHTS_CONFIG[model_type]
        
        all_success = True
        
        for weight_name, weight_info in weights_config.items():
            logger.info(f"\n{weight_info['description']}")
            
            output_path = model_dir / weight_info['filename']
            
            # Check if already exists
            if output_path.exists() and not force:
                logger.info(f"✓ Already exists: {output_path}")
                logger.info(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
                continue
            
            # Download
            success = self.download_file(
                weight_info['url'],
                output_path,
                weight_info.get('size_mb')
            )
            
            if success:
                # Compute and save MD5
                md5 = self.compute_md5(output_path)
                logger.info(f"  MD5: {md5}")
                
                # Update config with MD5
                weight_info['md5'] = md5
            else:
                all_success = False
                logger.error(f"✗ Failed to download: {weight_name}")
        
        return all_success
    
    def download_all(self, force: bool = False) -> bool:
        """
        Download all model weights.
        
        Args:
            force: Force re-download even if files exist
            
        Returns:
            True if all downloads successful
        """
        logger.info("=" * 60)
        logger.info("DOWNLOADING ALL MODEL WEIGHTS")
        logger.info("=" * 60)
        
        all_success = True
        
        for model_type in WEIGHTS_CONFIG.keys():
            success = self.download_model_weights(model_type, force)
            if not success:
                all_success = False
        
        return all_success
    
    def verify_weights(self) -> Dict[str, bool]:
        """
        Verify all weights are present and valid.
        
        Returns:
            Dictionary mapping model types to verification status
        """
        logger.info("=" * 60)
        logger.info("Verifying Model Weights")
        logger.info("=" * 60)
        
        results = {}
        
        for model_type, weights_config in WEIGHTS_CONFIG.items():
            logger.info(f"\n{model_type.upper()}:")
            
            model_dir = self.base_dir / model_type
            all_present = True
            
            for weight_name, weight_info in weights_config.items():
                weight_path = model_dir / weight_info['filename']
                
                if weight_path.exists():
                    size_mb = weight_path.stat().st_size / (1024 * 1024)
                    logger.info(f"  ✓ {weight_name}: {size_mb:.1f} MB")
                else:
                    logger.warning(f"  ✗ {weight_name}: MISSING")
                    all_present = False
            
            results[model_type] = all_present
        
        return results
    
    def list_weights(self):
        """List all available weights and their status."""
        logger.info("=" * 60)
        logger.info("Available Model Weights")
        logger.info("=" * 60)
        
        for model_type, weights_config in WEIGHTS_CONFIG.items():
            logger.info(f"\n{model_type.upper()}:")
            
            for weight_name, weight_info in weights_config.items():
                logger.info(f"\n  {weight_name}:")
                logger.info(f"    Description: {weight_info['description']}")
                logger.info(f"    Size: ~{weight_info['size_mb']} MB")
                logger.info(f"    Required: {weight_info['required']}")
                
                # Check if exists
                weight_path = self.base_dir / model_type / weight_info['filename']
                if weight_path.exists():
                    logger.info(f"    Status: ✓ Downloaded")
                else:
                    logger.info(f"    Status: ✗ Not downloaded")
    
    def get_manual_download_instructions(self):
        """Print manual download instructions for problematic weights."""
        logger.info("\n" + "=" * 60)
        logger.info("Manual Download Instructions")
        logger.info("=" * 60)
        
        logger.info("\nIf automatic download fails, manually download:")
        
        logger.info("\n1. VITON-HD Generator:")
        logger.info("   URL: https://github.com/shadow2496/VITON-HD")
        logger.info("   Or: https://huggingface.co/spaces/shadow2496/VITON-HD")
        logger.info(f"   Save to: {self.base_dir}/viton_hd/generator.pth")
        
        logger.info("\n2. OpenPose Body Pose:")
        logger.info("   URL: https://huggingface.co/lllyasviel/Annotators")
        logger.info(f"   Save to: {self.base_dir}/openpose/body_pose.pth")
        
        logger.info("\n3. Human Parsing (LIP):")
        logger.info("   URL: https://huggingface.co/mattresnick/human-parsing")
        logger.info(f"   Save to: {self.base_dir}/human_parsing/lip_parsing.pth")
        
        logger.info("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained weights for VITON-HD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all weights
  python scripts/download_viton_weights.py --all
  
  # Download specific model
  python scripts/download_viton_weights.py --model viton_hd
  
  # Force re-download
  python scripts/download_viton_weights.py --all --force
  
  # List available weights
  python scripts/download_viton_weights.py --list
  
  # Verify downloaded weights
  python scripts/download_viton_weights.py --verify

Note: Some downloads may be large (up to 200MB each).
Total download size: ~420MB
        """
    )
    
    parser.add_argument('--base-dir', default='weights',
                       help='Base directory for weights')
    parser.add_argument('--all', action='store_true',
                       help='Download all model weights')
    parser.add_argument('--model', choices=['viton_hd', 'openpose', 'human_parsing'],
                       help='Download specific model weights')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')
    parser.add_argument('--list', action='store_true',
                       help='List available weights and their status')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded weights')
    parser.add_argument('--manual-instructions', action='store_true',
                       help='Show manual download instructions')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = WeightsDownloader(args.base_dir)
    
    # Execute requested action
    if args.list:
        downloader.list_weights()
        return
    
    if args.verify:
        results = downloader.verify_weights()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Verification Summary:")
        all_ok = all(results.values())
        for model_type, ok in results.items():
            status = "✓ Complete" if ok else "✗ Incomplete"
            logger.info(f"  {model_type}: {status}")
        
        if all_ok:
            logger.info("\n✓ All weights are present!")
        else:
            logger.warning("\n✗ Some weights are missing. Run with --all to download.")
        
        return
    
    if args.manual_instructions:
        downloader.get_manual_download_instructions()
        return
    
    # Download weights
    if args.all:
        success = downloader.download_all(args.force)
    elif args.model:
        success = downloader.download_model_weights(args.model, args.force)
    else:
        parser.print_help()
        return
    
    # Verify after download
    logger.info("\n")
    results = downloader.verify_weights()
    
    if success and all(results.values()):
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL WEIGHTS DOWNLOADED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("\nNext Steps:")
        logger.info("1. Test VITON-HD: python scripts/demo_viton_gpu.py")
        logger.info("2. Download real models: python scripts/download_real_models.py")
    else:
        logger.warning("\n" + "=" * 60)
        logger.warning("⚠ SOME DOWNLOADS MAY HAVE FAILED")
        logger.warning("=" * 60)
        logger.info("\nTry:")
        logger.info("1. Run with --force to retry")
        logger.info("2. Check network connection")
        logger.info("3. Use --manual-instructions for alternative download methods")


if __name__ == "__main__":
    main()
