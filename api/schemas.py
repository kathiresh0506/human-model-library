"""
Pydantic schemas for API request and response validation.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ModelSelectRequest(BaseModel):
    """Request schema for model selection."""
    gender: str = Field(..., description="Gender: male or female")
    size: str = Field(..., description="Size: S, M, L, or XL")
    age_group: str = Field(..., description="Age group: young, middle, or senior")
    ethnicity: Optional[str] = Field(None, description="Optional ethnicity preference")
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "male",
                "size": "M",
                "age_group": "young",
                "ethnicity": "asian"
            }
        }


class ModelInfo(BaseModel):
    """Model information schema."""
    id: str
    gender: str
    size: str
    age_group: str
    ethnicity: str
    measurements: Dict[str, float]
    poses: List[str]
    created_at: str
    path: Optional[str] = None
    image_paths: Optional[Dict[str, str]] = None


class ModelListResponse(BaseModel):
    """Response schema for model list."""
    models: List[ModelInfo]
    total: int
    
    class Config:
        schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "male_m_young_asian_001",
                        "gender": "male",
                        "size": "M",
                        "age_group": "young",
                        "ethnicity": "asian",
                        "measurements": {
                            "chest": 98,
                            "waist": 83,
                            "hip": 98,
                            "height": 177
                        },
                        "poses": ["front", "side", "back"],
                        "created_at": "2024-01-01"
                    }
                ],
                "total": 1
            }
        }


class TryOnRequest(BaseModel):
    """Request schema for virtual try-on."""
    clothing_image: str = Field(..., description="Base64 encoded clothing image or URL")
    clothing_type: str = Field(..., description="Type of clothing (shirt, pants, dress, etc.)")
    gender: str = Field(..., description="Gender: male or female")
    size: str = Field(..., description="Size: S, M, L, or XL")
    age_group: str = Field(default="young", description="Age group: young, middle, or senior")
    ethnicity: Optional[str] = Field(None, description="Optional ethnicity preference")
    
    class Config:
        schema_extra = {
            "example": {
                "clothing_image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
                "clothing_type": "shirt",
                "gender": "male",
                "size": "M",
                "age_group": "young",
                "ethnicity": "asian"
            }
        }


class TryOnResponse(BaseModel):
    """Response schema for virtual try-on."""
    success: bool
    message: str
    result_image: Optional[str] = Field(None, description="Base64 encoded result image")
    model_used: Optional[ModelInfo] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Try-on completed successfully",
                "result_image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
                "model_used": {
                    "id": "male_m_young_asian_001",
                    "gender": "male",
                    "size": "M",
                    "age_group": "young",
                    "ethnicity": "asian"
                }
            }
        }


class SizeMeasurementsRequest(BaseModel):
    """Request schema for size measurements."""
    gender: str
    chest: float = Field(..., description="Chest/bust measurement in cm")
    waist: float = Field(..., description="Waist measurement in cm")
    hip: float = Field(..., description="Hip measurement in cm")
    height: float = Field(..., description="Height in cm")
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "male",
                "chest": 98,
                "waist": 83,
                "hip": 98,
                "height": 177
            }
        }


class SizeMeasurementsResponse(BaseModel):
    """Response schema for size measurements."""
    recommended_size: str
    measurements: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "recommended_size": "M",
                "measurements": {
                    "chest": {"min": 96, "max": 101, "avg": 98.5},
                    "waist": {"min": 81, "max": 86, "avg": 83.5},
                    "hip": {"min": 96, "max": 101, "avg": 98.5},
                    "height": {"min": 175, "max": 180, "avg": 177.5}
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
