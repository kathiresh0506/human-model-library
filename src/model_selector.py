"""
Model selection module for the Human Model Library.
Selects appropriate human models based on gender, size, age, and ethnicity.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .utils import PathManager, ConfigLoader, Validator, normalize_string

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Selects appropriate human models based on user criteria.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize ModelSelector.
        
        Args:
            base_dir: Base directory for the project
        """
        self.path_manager = PathManager(base_dir)
        self.config_loader = ConfigLoader(self.path_manager)
        self.validator = Validator()
        
        # Cache for model metadata
        self._model_cache: Dict[str, Dict[str, Any]] = {}
    
    def select_model(self,
                    gender: str,
                    size: str,
                    age_group: str,
                    ethnicity: Optional[str] = None) -> Optional[str]:
        """
        Select the best matching model based on criteria.
        
        Args:
            gender: Gender ('male' or 'female')
            size: Size ('S', 'M', 'L', or 'XL')
            age_group: Age group ('young', 'middle', or 'senior')
            ethnicity: Optional ethnicity preference
            
        Returns:
            Path to the selected model directory, or None if not found
        """
        # Normalize inputs
        gender = normalize_string(gender)
        size = size.upper()
        age_group = normalize_string(age_group)
        
        # Validate inputs
        if not self.validator.validate_gender(gender):
            logger.error(f"Invalid gender: {gender}")
            return None
        
        if not self.validator.validate_size(size):
            logger.error(f"Invalid size: {size}")
            return None
        
        if not self.validator.validate_age_group(age_group):
            logger.error(f"Invalid age group: {age_group}")
            return None
        
        # If ethnicity is provided, try to find exact match
        if ethnicity:
            ethnicity = normalize_string(ethnicity)
            if self.validator.validate_ethnicity(ethnicity):
                model_dir = self.path_manager.get_model_dir(
                    gender, size, age_group, ethnicity
                )
                if model_dir.exists():
                    logger.info(f"Found exact match: {model_dir}")
                    return str(model_dir)
        
        # Try to find any model matching gender, size, and age group
        base_path = self.path_manager.models_dir / gender / size / age_group
        
        if not base_path.exists():
            logger.warning(f"No models found for {gender}/{size}/{age_group}")
            return None
        
        # Get first available ethnicity
        try:
            ethnicities = [d for d in base_path.iterdir() if d.is_dir()]
            if ethnicities:
                selected = ethnicities[0]
                logger.info(f"Selected model: {selected}")
                return str(selected)
        except Exception as e:
            logger.error(f"Error searching for models: {e}")
        
        return None
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of all available models with their metadata.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        models_dir = self.path_manager.models_dir
        
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return models
        
        try:
            # Traverse directory structure
            for gender_dir in models_dir.iterdir():
                if not gender_dir.is_dir():
                    continue
                
                gender = gender_dir.name
                
                for size_dir in gender_dir.iterdir():
                    if not size_dir.is_dir():
                        continue
                    
                    size = size_dir.name
                    
                    for age_dir in size_dir.iterdir():
                        if not age_dir.is_dir():
                            continue
                        
                        age_group = age_dir.name
                        
                        for ethnicity_dir in age_dir.iterdir():
                            if not ethnicity_dir.is_dir():
                                continue
                            
                            ethnicity = ethnicity_dir.name
                            
                            # Try to load metadata
                            metadata = self.get_model_metadata(str(ethnicity_dir))
                            if metadata:
                                models.append(metadata)
                            else:
                                # Create basic info if no metadata file
                                models.append({
                                    'path': str(ethnicity_dir),
                                    'gender': gender,
                                    'size': size,
                                    'age_group': age_group,
                                    'ethnicity': ethnicity
                                })
        
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models
    
    def get_model_metadata(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Model metadata dictionary, or None if not found
        """
        # Check cache first
        if model_path in self._model_cache:
            return self._model_cache[model_path]
        
        metadata_path = Path(model_path) / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return None
        
        try:
            metadata = self.config_loader.load_json(str(metadata_path))
            
            # Add path to metadata
            metadata['path'] = model_path
            
            # Cache the metadata
            self._model_cache[model_path] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {e}")
            return None
    
    def get_models_by_criteria(self,
                              gender: Optional[str] = None,
                              size: Optional[str] = None,
                              age_group: Optional[str] = None,
                              ethnicity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all models matching the given criteria.
        
        Args:
            gender: Optional gender filter
            size: Optional size filter
            age_group: Optional age group filter
            ethnicity: Optional ethnicity filter
            
        Returns:
            List of matching model metadata
        """
        all_models = self.get_available_models()
        
        # Filter based on criteria
        filtered_models = []
        
        for model in all_models:
            match = True
            
            if gender and normalize_string(model.get('gender', '')) != normalize_string(gender):
                match = False
            
            if size and model.get('size', '').upper() != size.upper():
                match = False
            
            if age_group and normalize_string(model.get('age_group', '')) != normalize_string(age_group):
                match = False
            
            if ethnicity and normalize_string(model.get('ethnicity', '')) != normalize_string(ethnicity):
                match = False
            
            if match:
                filtered_models.append(model)
        
        return filtered_models
    
    def get_size_measurements(self, gender: str, size: str) -> Optional[Dict[str, Any]]:
        """
        Get standard measurements for a given gender and size.
        
        Args:
            gender: Gender ('male' or 'female')
            size: Size ('S', 'M', 'L', or 'XL')
            
        Returns:
            Dictionary of measurements, or None if not found
        """
        gender = normalize_string(gender)
        size = size.upper()
        
        sizes_config = self.config_loader.sizes_config
        
        try:
            return sizes_config.get(gender, {}).get(size)
        except Exception as e:
            logger.error(f"Error getting size measurements: {e}")
            return None
    
    def find_best_size_match(self,
                            gender: str,
                            chest: float,
                            waist: float,
                            hip: float,
                            height: float) -> str:
        """
        Find the best size match based on body measurements.
        
        Args:
            gender: Gender ('male' or 'female')
            chest: Chest/bust measurement in cm
            waist: Waist measurement in cm
            hip: Hip measurement in cm
            height: Height in cm
            
        Returns:
            Best matching size ('S', 'M', 'L', or 'XL')
        """
        gender = normalize_string(gender)
        sizes_config = self.config_loader.sizes_config
        
        if gender not in sizes_config:
            logger.warning(f"Unknown gender: {gender}. Defaulting to 'M'")
            return 'M'
        
        best_size = 'M'
        min_diff = float('inf')
        
        # Determine which measurement key to use (chest vs bust)
        chest_key = 'bust' if gender == 'female' else 'chest'
        
        for size, measurements in sizes_config[gender].items():
            # Calculate difference from measurements
            chest_avg = measurements.get(chest_key, {}).get('avg', 0)
            waist_avg = measurements.get('waist', {}).get('avg', 0)
            hip_avg = measurements.get('hip', {}).get('avg', 0)
            height_avg = measurements.get('height', {}).get('avg', 0)
            
            # Calculate normalized difference
            diff = (
                abs(chest - chest_avg) / chest_avg +
                abs(waist - waist_avg) / waist_avg +
                abs(hip - hip_avg) / hip_avg +
                abs(height - height_avg) / height_avg
            )
            
            if diff < min_diff:
                min_diff = diff
                best_size = size
        
        logger.info(f"Best size match for measurements: {best_size}")
        return best_size
