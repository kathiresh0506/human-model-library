"""
IDM-VTON client for Hugging Face Spaces API.

Provides a client interface to interact with the IDM-VTON model
hosted on Hugging Face Spaces for virtual try-on functionality.
"""

from gradio_client import Client, handle_file
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class IDMVTONClient:
    """Client for IDM-VTON on Hugging Face Spaces"""
    
    def __init__(self, space_id: str = "yisol/IDM-VTON"):
        """
        Initialize the IDM-VTON client.
        
        Args:
            space_id: Hugging Face Space ID for IDM-VTON
        """
        self.space_id = space_id
        self.client = None
        logger.info(f"IDM-VTON client initialized with space: {space_id}")
        
    def connect(self):
        """
        Connect to the Hugging Face Space.
        
        Returns:
            self for chaining
        """
        logger.info(f"Connecting to {self.space_id}...")
        try:
            self.client = Client(self.space_id)
            logger.info("Connected successfully!")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
        return self
        
    def try_on(
        self,
        person_image: str,
        clothing_image: str,
        clothing_type: str = "upper_body",  # upper_body, lower_body, dresses
        num_steps: int = 30,
        guidance_scale: float = 2.0,
        seed: int = 30
    ) -> str:
        """
        Perform virtual try-on using IDM-VTON.
        
        Args:
            person_image: Path to person/model image
            clothing_image: Path to clothing image
            clothing_type: Type of clothing (upper_body, lower_body, dresses)
            num_steps: Number of diffusion steps (default: 30)
            guidance_scale: Guidance scale for generation (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            
        Returns:
            Path to result image
        """
        if self.client is None:
            self.connect()
            
        logger.info(f"Processing virtual try-on...")
        logger.info(f"  Person: {person_image}")
        logger.info(f"  Clothing: {clothing_image}")
        logger.info(f"  Type: {clothing_type}")
        
        try:
            # IDM-VTON API parameters:
            # - auto-mask: Automatically detect and mask the person's clothing area
            # - auto-crop: Automatically crop and align the garment for better fitting
            result = self.client.predict(
                dict(
                    background=handle_file(person_image),
                    layers=[],
                    composite=None
                ),
                handle_file(clothing_image),
                clothing_type,
                num_steps,
                guidance_scale,
                seed,
                True,  # auto-mask: enable automatic person masking
                True,  # auto-crop: enable automatic garment cropping
                api_name="/tryon"
            )
            
            logger.info(f"Result saved to: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Try-on failed: {e}")
            raise
