"""
Simple virtual try-on function using IDM-VTON.

Provides a high-level interface for performing virtual try-on
with real human photos and clothing images.
"""

from pathlib import Path
from PIL import Image
import logging
from .client import IDMVTONClient

logger = logging.getLogger(__name__)


def virtual_tryon(
    person_image: str,
    clothing_image: str,
    output_path: str = None,
    clothing_type: str = "upper_body"
) -> str:
    """
    Simple function for virtual try-on.
    
    Args:
        person_image: Path to real human photo
        clothing_image: Path to clothing image
        output_path: Where to save result (optional)
        clothing_type: upper_body, lower_body, or dresses
        
    Returns:
        Path to result image
    """
    logger.info("=" * 60)
    logger.info("IDM-VTON Virtual Try-On")
    logger.info("=" * 60)
    
    # Validate inputs
    if not Path(person_image).exists():
        raise FileNotFoundError(f"Person image not found: {person_image}")
    
    if not Path(clothing_image).exists():
        raise FileNotFoundError(f"Clothing image not found: {clothing_image}")
    
    # Perform try-on
    client = IDMVTONClient()
    result = client.try_on(
        person_image=person_image,
        clothing_image=clothing_image,
        clothing_type=clothing_type
    )
    
    # Save to custom location if specified
    if output_path:
        logger.info(f"Copying result to: {output_path}")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy result to desired location
        img = Image.open(result)
        img.save(output_path)
        
        logger.info(f"✅ Result saved to: {output_path}")
        return output_path
    
    logger.info(f"✅ Result saved to: {result}")
    return result
