"""
API routes for virtual try-on functionality.
"""
import logging
import base64
import io
from typing import Optional
from fastapi import APIRouter, HTTPException
from pathlib import Path
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model_selector import ModelSelector
from src.clothing_fitter import ClothingFitter
from src.utils import ImageLoader
from api.schemas import TryOnRequest, TryOnResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
try:
    model_selector = ModelSelector()
    clothing_fitter = ClothingFitter()
    image_loader = ImageLoader()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    model_selector = None
    clothing_fitter = None
    image_loader = None


def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
    """
    Decode base64 encoded image.
    
    Args:
        image_data: Base64 encoded image string (with or without data URI prefix)
        
    Returns:
        Image as numpy array, or None if decoding fails
    """
    try:
        # Remove data URI prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Convert to numpy array
        return np.array(image)
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


def encode_image_to_base64(image: np.ndarray) -> Optional[str]:
    """
    Encode numpy array image to base64.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded image string with data URI prefix, or None if encoding fails
    """
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Add data URI prefix
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None


@router.post("/tryon", response_model=TryOnResponse)
async def virtual_tryon(request: TryOnRequest):
    """
    Perform virtual try-on of clothing on a model.
    
    The clothing_image should be base64 encoded or a URL.
    Returns the result image as base64 encoded data.
    """
    if model_selector is None or clothing_fitter is None:
        raise HTTPException(
            status_code=500,
            detail="Service not properly initialized"
        )
    
    try:
        # Step 1: Select appropriate model
        logger.info(f"Selecting model: gender={request.gender}, size={request.size}, "
                   f"age_group={request.age_group}, ethnicity={request.ethnicity}")
        
        model_path = model_selector.select_model(
            gender=request.gender,
            size=request.size,
            age_group=request.age_group,
            ethnicity=request.ethnicity
        )
        
        if model_path is None:
            return TryOnResponse(
                success=False,
                message="No matching model found for the specified criteria",
                result_image=None,
                selected_model=None
            )
        
        # Get model metadata
        model_metadata = model_selector.get_model_metadata(model_path)
        logger.info(f"Selected model: {model_metadata.get('id', 'unknown')}")
        
        # Step 2: Load model image (front view)
        model_image_path = Path(model_path) / "front.png"
        
        if not model_image_path.exists():
            return TryOnResponse(
                success=False,
                message="Model image not found",
                result_image=None,
                selected_model=model_metadata
            )
        
        model_image = image_loader.load_image(str(model_image_path))
        
        if model_image is None:
            return TryOnResponse(
                success=False,
                message="Failed to load model image",
                result_image=None,
                selected_model=model_metadata
            )
        
        # Step 3: Decode clothing image
        logger.info("Decoding clothing image...")
        
        # Check if it's a URL or base64
        if request.clothing_image.startswith('http'):
            # TODO: Implement URL fetching
            return TryOnResponse(
                success=False,
                message="URL-based clothing images not yet supported. Please use base64 encoded images.",
                result_image=None,
                selected_model=model_metadata
            )
        else:
            clothing_image = decode_base64_image(request.clothing_image)
        
        if clothing_image is None:
            return TryOnResponse(
                success=False,
                message="Failed to decode clothing image",
                result_image=None,
                selected_model=model_metadata
            )
        
        # Step 4: Perform try-on
        logger.info(f"Performing virtual try-on with clothing type: {request.clothing_type}")
        
        result_image = clothing_fitter.fit_clothing(
            model_image=model_image,
            clothing_image=clothing_image,
            clothing_type=request.clothing_type
        )
        
        if result_image is None:
            return TryOnResponse(
                success=False,
                message="Virtual try-on failed",
                result_image=None,
                selected_model=model_metadata
            )
        
        # Step 5: Encode result image
        logger.info("Encoding result image...")
        result_base64 = encode_image_to_base64(result_image)
        
        if result_base64 is None:
            return TryOnResponse(
                success=False,
                message="Failed to encode result image",
                result_image=None,
                selected_model=model_metadata
            )
        
        # Success!
        logger.info("Virtual try-on completed successfully")
        
        return TryOnResponse(
            success=True,
            message="Virtual try-on completed successfully",
            result_image=result_base64,
            selected_model=model_metadata
        )
        
    except Exception as e:
        logger.error(f"Error in virtual try-on: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
