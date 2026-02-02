"""
Real model selector for selecting actual human photos.

Selects real human photos from the organized library based on gender and size.
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class RealModelSelector:
    """
    Select real human photos based on gender and size.
    
    Uses the organized photo library in models/realistic/
    """
    
    def __init__(self, base_dir: str = 'models/realistic'):
        """
        Initialize real model selector.
        
        Args:
            base_dir: Base directory for realistic models.
                     Can be 'models/realistic' or 'models/real_humans'
        """
        self.base_dir = Path(base_dir)
        
        if not self.base_dir.exists():
            logger.error(f"Models directory not found: {self.base_dir}")
            raise FileNotFoundError(f"Models directory not found: {self.base_dir}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"Real model selector initialized with base_dir: {self.base_dir}")
    
    def _load_metadata(self) -> Optional[Dict]:
        """
        Load metadata.json if available.
        
        Returns:
            Metadata dictionary or None
        """
        metadata_path = self.base_dir / 'metadata.json'
        
        if not metadata_path.exists():
            logger.warning("metadata.json not found, will scan directory")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info("Loaded metadata.json")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None
    
    def select_model(self, gender: str, size: str) -> Optional[str]:
        """
        Select a real human photo based on gender and size.
        
        Args:
            gender: Gender ('male' or 'female')
            size: Size ('S', 'M', 'L', or 'XL')
            
        Returns:
            Path to selected model photo, or None if not found
        """
        gender = gender.lower()
        size = size.upper()
        
        # Validate inputs
        if gender not in ['male', 'female']:
            logger.error(f"Invalid gender: {gender}. Must be 'male' or 'female'")
            return None
        
        if size not in ['S', 'M', 'L', 'XL']:
            logger.error(f"Invalid size: {size}. Must be S, M, L, or XL")
            return None
        
        # Get available models
        models = self.get_models(gender, size)
        
        if not models:
            logger.warning(f"No models found for {gender}/{size}")
            return None
        
        # Randomly select one
        selected = random.choice(models)
        logger.info(f"Selected model: {selected}")
        
        return str(selected)
    
    def get_models(self, gender: str, size: str) -> List[Path]:
        """
        Get all available models for a gender and size.
        
        Args:
            gender: Gender ('male' or 'female')
            size: Size ('S', 'M', 'L', or 'XL')
            
        Returns:
            List of model photo paths
        """
        gender = gender.lower()
        size = size.upper()
        
        size_dir = self.base_dir / gender / size
        
        if not size_dir.exists():
            logger.warning(f"Directory not found: {size_dir}")
            return []
        
        # Find all image files
        models = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            models.extend(size_dir.glob(ext))
        
        return sorted(models)
    
    def get_all_sizes(self, gender: str) -> Dict[str, List[Path]]:
        """
        Get available models for all sizes of a gender.
        
        Args:
            gender: Gender ('male' or 'female')
            
        Returns:
            Dictionary mapping sizes to lists of model paths
        """
        gender = gender.lower()
        
        result = {}
        for size in ['S', 'M', 'L', 'XL']:
            models = self.get_models(gender, size)
            result[size] = models
        
        return result
    
    def count_models(self, gender: Optional[str] = None, 
                    size: Optional[str] = None) -> int:
        """
        Count available models.
        
        Args:
            gender: Optional gender filter
            size: Optional size filter
            
        Returns:
            Number of models matching criteria
        """
        if gender and size:
            return len(self.get_models(gender, size))
        elif gender:
            total = 0
            for size in ['S', 'M', 'L', 'XL']:
                total += len(self.get_models(gender, size))
            return total
        else:
            total = 0
            for gender in ['male', 'female']:
                for size in ['S', 'M', 'L', 'XL']:
                    total += len(self.get_models(gender, size))
            return total
    
    def get_summary(self) -> Dict:
        """
        Get summary of available models.
        
        Returns:
            Summary dictionary with counts per category
        """
        summary = {
            'male': {},
            'female': {},
            'total': 0
        }
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                count = len(self.get_models(gender, size))
                summary[gender][size] = count
                summary['total'] += count
        
        return summary
    
    def print_summary(self):
        """Print summary of available models."""
        summary = self.get_summary()
        
        logger.info("=" * 60)
        logger.info("Real Model Library Summary")
        logger.info("=" * 60)
        
        for gender in ['male', 'female']:
            logger.info(f"\n{gender.upper()}:")
            for size in ['S', 'M', 'L', 'XL']:
                count = summary[gender][size]
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} Size {size}: {count} models")
        
        logger.info(f"\nTotal models: {summary['total']}")
        logger.info("=" * 60)
    
    def validate_library(self) -> bool:
        """
        Validate the model library.
        
        Returns:
            True if library has models for all categories
        """
        logger.info("Validating model library...")
        
        all_valid = True
        missing = []
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                models = self.get_models(gender, size)
                if not models:
                    missing.append(f"{gender}/{size}")
                    all_valid = False
        
        if all_valid:
            logger.info("✓ Library validation passed")
            return True
        else:
            logger.warning("✗ Library validation failed")
            logger.warning(f"Missing models for: {', '.join(missing)}")
            logger.info("\nTo add models:")
            logger.info("1. Download photos: python scripts/download_real_models.py --unsplash-key KEY")
            logger.info("2. Or manually add photos to appropriate directories")
            return False
    
    def load_model_image(self, model_path: str):
        """
        Load model image from path.
        
        Args:
            model_path: Path to model image
            
        Returns:
            Image array or None
        """
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(model_path)
            img = img.convert('RGB')
            
            return np.array(img)
            
        except Exception as e:
            logger.error(f"Error loading model image: {e}")
            return None
    
    def list_available(self) -> dict:
        """
        List all available real human models.
        
        Returns:
            Dictionary with counts per gender/size
        """
        available = {}
        for gender in ["male", "female"]:
            available[gender] = {}
            for size in ["S", "M", "L", "XL"]:
                model_dir = self.base_dir / gender / size
                if model_dir.exists():
                    photos = list(model_dir.glob("*.jpg")) + list(model_dir.glob("*.png")) + list(model_dir.glob("*.jpeg"))
                    available[gender][size] = len(photos)
                else:
                    available[gender][size] = 0
        return available

    def select_random_model(self) -> Optional[str]:
        """
        Select a completely random model from entire library.
        
        Returns:
            Path to randomly selected model
        """
        all_models = []
        
        for gender in ['male', 'female']:
            for size in ['S', 'M', 'L', 'XL']:
                models = self.get_models(gender, size)
                all_models.extend(models)
        
        if not all_models:
            logger.error("No models found in library")
            return None
        
        selected = random.choice(all_models)
        logger.info(f"Randomly selected: {selected}")
        
        return str(selected)
    
    # Standard chest/bust measurement ranges for sizing (in cm)
    # Based on common US/EU sizing charts
    SIZE_THRESHOLDS_MALE = {
        'S': (0, 92),      # Small: <92cm chest
        'M': (92, 102),    # Medium: 92-102cm chest
        'L': (102, 112),   # Large: 102-112cm chest
        'XL': (112, 999)   # Extra Large: >112cm chest
    }
    
    SIZE_THRESHOLDS_FEMALE = {
        'S': (0, 87),      # Small: <87cm bust
        'M': (87, 94),     # Medium: 87-94cm bust
        'L': (94, 102),    # Large: 94-102cm bust
        'XL': (102, 999)   # Extra Large: >102cm bust
    }
    
    def find_best_match(self, target_height_cm: float, 
                       target_chest_cm: float,
                       gender: str) -> Optional[str]:
        """
        Find best matching model based on body measurements.
        
        Uses standard sizing chart thresholds to estimate appropriate size
        based on chest/bust measurement.
        
        Args:
            target_height_cm: Target height in cm
            target_chest_cm: Target chest/bust in cm
            gender: Gender
            
        Returns:
            Path to best matching model
        """
        gender = gender.lower()
        
        # Select appropriate size thresholds
        thresholds = self.SIZE_THRESHOLDS_MALE if gender == 'male' else self.SIZE_THRESHOLDS_FEMALE
        
        # Find matching size based on chest measurement
        size = 'M'  # Default
        for size_name, (min_cm, max_cm) in thresholds.items():
            if min_cm <= target_chest_cm < max_cm:
                size = size_name
                break
        
        logger.info(f"Estimated size: {size} (chest: {target_chest_cm}cm)")
        
        return self.select_model(gender, size)


def main():
    """Demo usage of RealModelSelector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real model selector")
    parser.add_argument('--gender', choices=['male', 'female'],
                       help='Gender to select')
    parser.add_argument('--size', choices=['S', 'M', 'L', 'XL'],
                       help='Size to select')
    parser.add_argument('--summary', action='store_true',
                       help='Print library summary')
    parser.add_argument('--validate', action='store_true',
                       help='Validate library')
    parser.add_argument('--random', action='store_true',
                       help='Select random model')
    
    args = parser.parse_args()
    
    # Initialize selector
    selector = RealModelSelector()
    
    if args.summary:
        selector.print_summary()
    elif args.validate:
        selector.validate_library()
    elif args.random:
        model_path = selector.select_random_model()
        if model_path:
            print(f"Selected: {model_path}")
    elif args.gender and args.size:
        model_path = selector.select_model(args.gender, args.size)
        if model_path:
            print(f"Selected: {model_path}")
    else:
        parser.print_help()
        print("\n")
        selector.print_summary()


if __name__ == "__main__":
    main()
