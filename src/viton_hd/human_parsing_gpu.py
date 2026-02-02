"""
GPU-accelerated human parsing/segmentation for VITON-HD.

Segments body parts using neural networks for accurate clothing placement.
Supports LIP (Look Into Person) and SCHP models.
"""
import logging
from typing import Optional, Dict, List
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available for GPU human parsing")


# LIP (Look Into Person) body part labels
LIP_LABELS = {
    0: 'background',
    1: 'hat',
    2: 'hair',
    3: 'glove',
    4: 'sunglasses',
    5: 'upper_clothes',
    6: 'dress',
    7: 'coat',
    8: 'socks',
    9: 'pants',
    10: 'jumpsuits',
    11: 'scarf',
    12: 'skirt',
    13: 'face',
    14: 'left_arm',
    15: 'right_arm',
    16: 'left_leg',
    17: 'right_leg',
    18: 'left_shoe',
    19: 'right_shoe'
}


class ResNetBlock(nn.Module):
    """ResNet-style block for human parsing."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize ResNet block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for convolution
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """Forward pass with residual connection."""
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.skip is not None:
            identity = self.skip(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class HumanParsingNet(nn.Module):
    """
    Simplified human parsing network.
    
    Based on ResNet-style encoder-decoder architecture.
    For production, use LIP or SCHP pre-trained weights.
    """
    
    def __init__(self, num_classes: int = 20):
        """
        Initialize human parsing network.
        
        Args:
            num_classes: Number of body part classes (20 for LIP)
        """
        super().__init__()
        
        # Encoder (ResNet-style)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        
        # Final classification layer
        self.classifier = nn.Conv2d(16, num_classes, 1)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int = 1):
        """Create a ResNet layer."""
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Decoder
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))
        x = F.relu(self.up5(x))
        
        # Classification
        x = self.classifier(x)
        
        return x


class HumanParsingGPU:
    """
    GPU-accelerated human parsing/segmentation.
    """
    
    def __init__(self, device: str = 'cuda', weights_path: Optional[str] = None):
        """
        Initialize human parsing on GPU.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            weights_path: Path to parsing model weights (optional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        logger.info(f"Human parsing using device: {self.device}")
        
        # Initialize model
        self.model = HumanParsingNet(num_classes=20).to(self.device)
        self.model.eval()
        
        # Load weights if provided
        if weights_path and Path(weights_path).exists():
            self.load_weights(weights_path)
        else:
            logger.warning("No parsing weights loaded")
            logger.info("Download with: python scripts/download_viton_weights.py")
    
    def load_weights(self, path: str) -> bool:
        """
        Load human parsing weights.
        
        Args:
            path: Path to weights file
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading human parsing weights from: {path}")
            checkpoint = torch.load(path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            logger.info("✓ Human parsing weights loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading parsing weights: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for human parsing.
        
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        from PIL import Image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image)
            else:
                pil_img = Image.fromarray((image * 255).astype(np.uint8))
        else:
            pil_img = image
        
        # Resize to standard size
        pil_img = pil_img.resize((512, 768), Image.LANCZOS)
        
        # Convert to numpy and normalize
        image = np.array(pil_img).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def segment(self, image: np.ndarray) -> Optional[Dict]:
        """
        Segment body parts from image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary with segmentation map and masks
        """
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Run inference
            logits = self.model(input_tensor)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Resize to original size
            from PIL import Image
            h, w = image.shape[:2]
            preds_pil = Image.fromarray(preds.astype(np.uint8))
            preds_pil = preds_pil.resize((w, h), Image.NEAREST)
            preds = np.array(preds_pil)
            
            # Generate individual masks
            masks = self._generate_masks(preds)
            
            # Generate visualization
            vis = self._visualize_segmentation(preds)
            
            return {
                'segmentation': preds,
                'masks': masks,
                'visualization': vis
            }
            
        except Exception as e:
            logger.error(f"Error in human parsing: {e}")
            return None
    
    def _generate_masks(self, segmentation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate binary masks for each body part.
        
        Args:
            segmentation: Segmentation map [H, W]
            
        Returns:
            Dictionary of binary masks
        """
        masks = {}
        
        for label_id, label_name in LIP_LABELS.items():
            mask = (segmentation == label_id).astype(np.uint8)
            masks[label_name] = mask
        
        return masks
    
    def _visualize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Create colored visualization of segmentation.
        
        Args:
            segmentation: Segmentation map [H, W]
            
        Returns:
            RGB visualization [H, W, 3]
        """
        # Color palette for visualization
        colors = [
            [0, 0, 0],      # background
            [128, 0, 0],    # hat
            [255, 0, 0],    # hair
            [0, 85, 0],     # glove
            [170, 0, 51],   # sunglasses
            [255, 85, 0],   # upper_clothes
            [0, 0, 85],     # dress
            [0, 119, 221],  # coat
            [85, 85, 0],    # socks
            [0, 85, 85],    # pants
            [85, 51, 0],    # jumpsuits
            [52, 86, 128],  # scarf
            [0, 128, 0],    # skirt
            [0, 0, 255],    # face
            [51, 170, 221], # left_arm
            [0, 255, 255],  # right_arm
            [85, 255, 170], # left_leg
            [170, 255, 85], # right_leg
            [255, 255, 0],  # left_shoe
            [255, 170, 0]   # right_shoe
        ]
        
        h, w = segmentation.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label_id in range(len(colors)):
            mask = (segmentation == label_id)
            vis[mask] = colors[label_id]
        
        return vis
    
    def get_cloth_agnostic_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Generate cloth-agnostic mask for VITON-HD.
        
        Args:
            segmentation: Segmentation map
            
        Returns:
            Binary mask [H, W] with 1 for visible body parts
        """
        # Parts to keep visible (not replaced by clothing)
        visible_parts = [
            1,  # hat
            2,  # hair
            13, # face
            14, # left_arm (partially)
            15, # right_arm (partially)
        ]
        
        mask = np.zeros_like(segmentation, dtype=np.uint8)
        for part_id in visible_parts:
            mask[segmentation == part_id] = 1
        
        return mask
