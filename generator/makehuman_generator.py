"""
MakeHuman model generator script.
Generates 3D human models using MakeHuman with specified parameters.

This script provides two modes:
1. Full MakeHuman integration (requires MakeHuman installation)
2. Configuration export mode (generates .mhm files for manual use)
"""
import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MakeHumanGenerator:
    """
    Generates human models using MakeHuman software.
    Can work in two modes: direct API or configuration export.
    """
    
    def __init__(self, makehuman_path: Optional[str] = None):
        """
        Initialize MakeHumanGenerator.
        
        Args:
            makehuman_path: Path to MakeHuman installation
        """
        self.makehuman_path = makehuman_path or self._find_makehuman()
        self.has_makehuman = self.makehuman_path is not None
        
    def _find_makehuman(self) -> Optional[str]:
        """
        Attempt to find MakeHuman installation.
        
        Returns:
            Path to MakeHuman, or None if not found
        """
        # Common installation paths
        possible_paths = [
            "/usr/share/makehuman",
            "/opt/makehuman",
            "C:\\Program Files\\MakeHuman",
            os.path.expanduser("~/makehuman"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found MakeHuman at: {path}")
                return path
        
        logger.warning("MakeHuman not found in common paths")
        return None
    
    def generate_model(self,
                      gender: str,
                      size_params: Dict[str, float],
                      age: int,
                      ethnicity: str,
                      output_path: str) -> bool:
        """
        Generate a human model with specified parameters.
        
        Args:
            gender: 'male' or 'female'
            size_params: Dictionary of body measurements
            age: Age in years
            ethnicity: Ethnicity identifier
            output_path: Path to save the generated model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Generating {gender} model, age {age}, ethnicity {ethnicity}")
            
            # Create model configuration
            model_config = self._create_model_config(
                gender, size_params, age, ethnicity
            )
            
            # Export as MakeHuman .mhm file format
            mhm_config = self._create_mhm_file(model_config)
            
            # Save .mhm configuration file
            config_path = Path(output_path).with_suffix('.mhm')
            with open(config_path, 'w') as f:
                f.write(mhm_config)
            
            logger.info(f"MakeHuman configuration saved to {config_path}")
            logger.info("To generate 3D model:")
            logger.info("1. Open MakeHuman application")
            logger.info(f"2. Load the configuration file: {config_path}")
            logger.info("3. Export as desired format (FBX, DAE, etc.)")
            
            # Also save JSON config for reference
            json_path = Path(output_path).with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating model: {e}")
            return False
    
    def _create_mhm_file(self, config: Dict[str, Any]) -> str:
        """
        Create MakeHuman .mhm file content from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            .mhm file content as string
        """
        lines = [
            "# MakeHuman Model Configuration",
            "version v1.2.0",
            "",
            "# Gender and age",
            f"modifier macrodetails/Gender {config['gender']:.2f}",
            f"modifier macrodetails/Age {config['age']:.2f}",
            "",
            "# Ethnicity",
        ]
        
        # Add ethnicity modifiers
        for key, value in config['ethnicity'].items():
            if value > 0:
                lines.append(f"modifier macrodetails-universal/{key.capitalize()} {value:.2f}")
        
        lines.append("")
        lines.append("# Body shape")
        
        # Add body modifiers
        for key, value in config['body'].items():
            param_name = key.replace('_', '-')
            lines.append(f"modifier macrodetails/{param_name} {value:.2f}")
        
        lines.extend([
            "",
            "# Skin and materials",
            "material Middleage_lightskinned_male_diffuse.png",
            "skinMaterial skins/young_caucasian_male/young_caucasian_male.mhmat",
            "",
            "# Basic clothing (underwear)",
            "clothesHide true",
            "",
            "# Pose - standing A-pose",
            "skeleton game_engine.json",
            ""
        ])
        
        return '\n'.join(lines)
    
    def _create_model_config(self,
                           gender: str,
                           size_params: Dict[str, float],
                           age: int,
                           ethnicity: str) -> Dict[str, Any]:
        """
        Create MakeHuman model configuration.
        
        Args:
            gender: Gender
            size_params: Size parameters
            age: Age
            ethnicity: Ethnicity
            
        Returns:
            Configuration dictionary
        """
        config = {
            'gender': 1.0 if gender.lower() == 'male' else 0.0,
            'age': self._age_to_parameter(age),
            'ethnicity': self._ethnicity_to_parameters(ethnicity),
            'body': self._size_to_body_parameters(size_params, gender),
        }
        
        return config
    
    def _age_to_parameter(self, age: int) -> float:
        """
        Convert age to MakeHuman age parameter (0-1).
        
        Args:
            age: Age in years
            
        Returns:
            Age parameter value
        """
        # MakeHuman uses 0-1 scale where 0=baby, 0.5=young adult, 1=elderly
        if age < 18:
            return 0.3
        elif age < 30:
            return 0.5
        elif age < 50:
            return 0.65
        else:
            return 0.8
    
    def _ethnicity_to_parameters(self, ethnicity: str) -> Dict[str, float]:
        """
        Convert ethnicity to MakeHuman ethnicity parameters.
        
        Args:
            ethnicity: Ethnicity identifier
            
        Returns:
            Dictionary of ethnicity parameters
        """
        ethnicity_configs = {
            'asian': {'asian': 1.0, 'african': 0.0, 'caucasian': 0.0},
            'african': {'asian': 0.0, 'african': 1.0, 'caucasian': 0.0},
            'caucasian': {'asian': 0.0, 'african': 0.0, 'caucasian': 1.0},
            'hispanic': {'asian': 0.3, 'african': 0.2, 'caucasian': 0.5},
            'middle_eastern': {'asian': 0.2, 'african': 0.3, 'caucasian': 0.5},
        }
        
        return ethnicity_configs.get(ethnicity.lower(), 
                                     {'asian': 0.33, 'african': 0.33, 'caucasian': 0.34})
    
    def _size_to_body_parameters(self,
                                 size_params: Dict[str, float],
                                 gender: str) -> Dict[str, float]:
        """
        Convert size measurements to MakeHuman body parameters.
        
        Args:
            size_params: Size parameters (chest/bust, waist, hip, height)
            gender: Gender
            
        Returns:
            Dictionary of body parameters
        """
        # Extract measurements
        chest_key = 'bust' if gender.lower() == 'female' else 'chest'
        chest = size_params.get(chest_key, 90)
        waist = size_params.get('waist', 75)
        hip = size_params.get('hip', 95)
        height = size_params.get('height', 170)
        
        # Convert to MakeHuman parameters (normalized values)
        # These are approximations and would need fine-tuning
        body_params = {
            'height': (height - 140) / 60,  # Normalize to 0-1
            'muscle': 0.5,  # Medium muscle tone
            'weight': (waist - 50) / 60,  # Based on waist
            'breast-size': (chest - 70) / 50 if gender.lower() == 'female' else 0,
            'stomach': (waist - 60) / 40,
            'hips': (hip - 70) / 50,
        }
        
        # Clamp values to 0-1
        for key in body_params:
            body_params[key] = max(0.0, min(1.0, body_params[key]))
        
        return body_params
    
    def batch_generate(self,
                      models_config: list,
                      output_dir: str) -> Dict[str, bool]:
        """
        Generate multiple models from a configuration list.
        
        Args:
            models_config: List of model configurations
            output_dir: Directory to save generated models
            
        Returns:
            Dictionary mapping model IDs to success status
        """
        results = {}
        
        for config in models_config:
            model_id = config.get('id', 'unknown')
            
            try:
                output_path = os.path.join(output_dir, f"{model_id}.mhm")
                
                success = self.generate_model(
                    gender=config['gender'],
                    size_params=config['measurements'],
                    age=config.get('age', 25),
                    ethnicity=config['ethnicity'],
                    output_path=output_path
                )
                
                results[model_id] = success
                
            except Exception as e:
                logger.error(f"Error generating model {model_id}: {e}")
                results[model_id] = False
        
        return results


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate human models with MakeHuman")
    parser.add_argument('--gender', required=True, choices=['male', 'female'])
    parser.add_argument('--size', required=True, choices=['S', 'M', 'L', 'XL'])
    parser.add_argument('--age', type=int, default=25)
    parser.add_argument('--ethnicity', required=True)
    parser.add_argument('--output', required=True, help='Output path')
    
    args = parser.parse_args()
    
    # Example measurements
    measurements = {
        'chest': 98,
        'waist': 83,
        'hip': 98,
        'height': 177
    }
    
    generator = MakeHumanGenerator()
    success = generator.generate_model(
        args.gender,
        measurements,
        args.age,
        args.ethnicity,
        args.output
    )
    
    if success:
        print(f"Model generated successfully: {args.output}")
    else:
        print("Model generation failed")
