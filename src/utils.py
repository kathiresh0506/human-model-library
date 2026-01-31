"""
Utility functions for the Human Model Library system.
Provides image handling, path management, and validation helpers.
"""
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathManager:
    """Manages paths for the Human Model Library system."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize PathManager.
        
        Args:
            base_dir: Base directory for the project. Defaults to project root.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.config_dir = self.base_dir / "config"
        self.models_dir = self.base_dir / "models"
        self.src_dir = self.base_dir / "src"
        
    def get_config_path(self, config_name: str) -> Path:
        """Get path to a configuration file."""
        return self.config_dir / config_name
    
    def get_model_dir(self, gender: str, size: str, age_group: str, 
                      ethnicity: str) -> Path:
        """Get path to a specific model directory."""
        return self.models_dir / gender / size / age_group / ethnicity
    
    def get_model_metadata_path(self, gender: str, size: str, age_group: str,
                                ethnicity: str) -> Path:
        """Get path to model metadata file."""
        return self.get_model_dir(gender, size, age_group, ethnicity) / "metadata.json"
    
    def ensure_dir_exists(self, path: Path) -> None:
        """Ensure a directory exists, create if it doesn't."""
        path.mkdir(parents=True, exist_ok=True)


class ImageLoader:
    """Handles image loading and saving operations."""
    
    @staticmethod
    def load_image(path: str) -> Optional[np.ndarray]:
        """
        Load an image from file.
        
        Args:
            path: Path to the image file
            
        Returns:
            Image as numpy array (RGB) or None if loading fails
        """
        try:
            img = Image.open(path)
            img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.error(f"Failed to load image from {path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, path: str) -> bool:
        """
        Save an image to file.
        
        Args:
            image: Image as numpy array
            path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Convert to PIL Image and save
            img = Image.fromarray(image.astype('uint8'))
            img.save(path)
            logger.info(f"Image saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Resize an image to specified dimensions.
        
        Args:
            image: Input image as numpy array
            width: Target width
            height: Target height
            
        Returns:
            Resized image as numpy array
        """
        img = Image.fromarray(image.astype('uint8'))
        img = img.resize((width, height), Image.LANCZOS)
        return np.array(img)


class ConfigLoader:
    """Loads and manages configuration files."""
    
    def __init__(self, path_manager: PathManager):
        """
        Initialize ConfigLoader.
        
        Args:
            path_manager: PathManager instance
        """
        self.path_manager = path_manager
        self._sizes_config = None
        self._ethnicities_config = None
        self._age_groups_config = None
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: Name of the YAML file
            
        Returns:
            Configuration dictionary
        """
        try:
            path = self.path_manager.get_config_path(filename)
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML from {filename}: {e}")
            return {}
    
    def load_json(self, path: str) -> Dict[str, Any]:
        """
        Load a JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            JSON data as dictionary
        """
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {path}: {e}")
            return {}
    
    @property
    def sizes_config(self) -> Dict[str, Any]:
        """Get sizes configuration (cached)."""
        if self._sizes_config is None:
            self._sizes_config = self.load_yaml('sizes.yaml')
        return self._sizes_config
    
    @property
    def ethnicities_config(self) -> Dict[str, Any]:
        """Get ethnicities configuration (cached)."""
        if self._ethnicities_config is None:
            self._ethnicities_config = self.load_yaml('ethnicities.yaml')
        return self._ethnicities_config
    
    @property
    def age_groups_config(self) -> Dict[str, Any]:
        """Get age groups configuration (cached)."""
        if self._age_groups_config is None:
            self._age_groups_config = self.load_yaml('age_groups.yaml')
        return self._age_groups_config


class Validator:
    """Validates input parameters."""
    
    VALID_GENDERS = ['male', 'female']
    VALID_SIZES = ['S', 'M', 'L', 'XL']
    VALID_AGE_GROUPS = ['young', 'middle', 'senior']
    VALID_ETHNICITIES = ['asian', 'african', 'caucasian', 'hispanic', 'middle_eastern']
    VALID_CLOTHING_TYPES = ['shirt', 'pants', 'dress', 'jacket', 'skirt', 'top']
    
    @classmethod
    def validate_gender(cls, gender: str) -> bool:
        """Validate gender parameter."""
        return gender.lower() in cls.VALID_GENDERS
    
    @classmethod
    def validate_size(cls, size: str) -> bool:
        """Validate size parameter."""
        return size.upper() in cls.VALID_SIZES
    
    @classmethod
    def validate_age_group(cls, age_group: str) -> bool:
        """Validate age group parameter."""
        return age_group.lower() in cls.VALID_AGE_GROUPS
    
    @classmethod
    def validate_ethnicity(cls, ethnicity: str) -> bool:
        """Validate ethnicity parameter."""
        return ethnicity.lower() in cls.VALID_ETHNICITIES
    
    @classmethod
    def validate_clothing_type(cls, clothing_type: str) -> bool:
        """Validate clothing type parameter."""
        return clothing_type.lower() in cls.VALID_CLOTHING_TYPES
    
    @classmethod
    def validate_image_array(cls, image: np.ndarray) -> bool:
        """Validate image array."""
        if not isinstance(image, np.ndarray):
            return False
        if len(image.shape) != 3:
            return False
        if image.shape[2] not in [3, 4]:  # RGB or RGBA
            return False
        return True


def normalize_string(s: str) -> str:
    """
    Normalize a string for comparison (lowercase, stripped).
    
    Args:
        s: Input string
        
    Returns:
        Normalized string
    """
    return s.strip().lower()
