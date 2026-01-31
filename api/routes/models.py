"""
API routes for model management and selection.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model_selector import ModelSelector
from api.schemas import (
    ModelSelectRequest,
    ModelInfo,
    ModelListResponse,
    SizeMeasurementsRequest,
    SizeMeasurementsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize model selector
try:
    model_selector = ModelSelector()
except Exception as e:
    logger.error(f"Failed to initialize ModelSelector: {e}")
    model_selector = None


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    gender: Optional[str] = Query(None, description="Filter by gender"),
    size: Optional[str] = Query(None, description="Filter by size"),
    age_group: Optional[str] = Query(None, description="Filter by age group"),
    ethnicity: Optional[str] = Query(None, description="Filter by ethnicity")
):
    """
    List all available models with optional filters.
    """
    if model_selector is None:
        raise HTTPException(status_code=500, detail="Model selector not initialized")
    
    try:
        # Get models with filters
        if any([gender, size, age_group, ethnicity]):
            models = model_selector.get_models_by_criteria(
                gender=gender,
                size=size,
                age_group=age_group,
                ethnicity=ethnicity
            )
        else:
            models = model_selector.get_available_models()
        
        return {
            "models": models,
            "total": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{gender}/{size}", response_model=ModelListResponse)
async def get_models_by_gender_size(
    gender: str,
    size: str,
    age_group: Optional[str] = Query(None, description="Filter by age group"),
    ethnicity: Optional[str] = Query(None, description="Filter by ethnicity")
):
    """
    Get models by gender and size with optional filters.
    """
    if model_selector is None:
        raise HTTPException(status_code=500, detail="Model selector not initialized")
    
    try:
        models = model_selector.get_models_by_criteria(
            gender=gender,
            size=size,
            age_group=age_group,
            ethnicity=ethnicity
        )
        
        if not models:
            raise HTTPException(
                status_code=404,
                detail=f"No models found for gender={gender}, size={size}"
            )
        
        return {
            "models": models,
            "total": len(models)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/select")
async def select_model(request: ModelSelectRequest):
    """
    Select the best matching model based on criteria.
    """
    if model_selector is None:
        raise HTTPException(status_code=500, detail="Model selector not initialized")
    
    try:
        model_path = model_selector.select_model(
            gender=request.gender,
            size=request.size,
            age_group=request.age_group,
            ethnicity=request.ethnicity
        )
        
        if model_path is None:
            raise HTTPException(
                status_code=404,
                detail="No matching model found"
            )
        
        # Get model metadata
        metadata = model_selector.get_model_metadata(model_path)
        
        if metadata is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model metadata"
            )
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/size-recommendation", response_model=SizeMeasurementsResponse)
async def get_size_recommendation(request: SizeMeasurementsRequest):
    """
    Get size recommendation based on body measurements.
    """
    if model_selector is None:
        raise HTTPException(status_code=500, detail="Model selector not initialized")
    
    try:
        # Find best size match
        recommended_size = model_selector.find_best_size_match(
            gender=request.gender,
            chest=request.chest,
            waist=request.waist,
            hip=request.hip,
            height=request.height
        )
        
        # Get measurements for recommended size
        measurements = model_selector.get_size_measurements(
            gender=request.gender,
            size=recommended_size
        )
        
        return {
            "recommended_size": recommended_size,
            "measurements": measurements
        }
        
    except Exception as e:
        logger.error(f"Error getting size recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
