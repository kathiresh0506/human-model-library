# VITON-HD Setup Guide

## Overview

This guide explains how to set up and use VITON-HD (Virtual Try-On High Definition) for professional-quality virtual clothing try-on in the Human Model Library.

## What is VITON-HD?

VITON-HD is a state-of-the-art virtual try-on AI model that:
- Uses real human photos (not cartoons)
- Warps clothing realistically to match body shape
- Handles occlusion and various body poses
- Produces production-quality results similar to Myntra

## Quick Start (VITON-Lite)

For immediate results without heavy model weights, use **VITON-Lite**:

```bash
# One-command setup
python scripts/setup_viton.py

# Run demo
python scripts/demo_viton_tryon.py
```

VITON-Lite provides:
- ✓ Proper clothing scaling based on body measurements
- ✓ Correct positioning on chest (not stomach)
- ✓ Perspective warping to follow body contours
- ✓ Feathered edges for natural blending
- ✓ Subtle shadows for depth
- ✓ No model weights required

## Full VITON-HD Setup (Advanced)

For production-quality results, use full VITON-HD:

### System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 20GB free disk space

**Recommended:**
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.0+
- 50GB free disk space

### Installation Steps

#### 1. Install PyTorch

```bash
# For GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower)
pip install torch torchvision
```

#### 2. Run Full Setup

```bash
python scripts/setup_viton.py --full
```

This will:
- Install all dependencies
- Check for GPU availability
- Download model weights (placeholders for now)
- Generate sample models
- Run verification tests

#### 3. Download VITON-HD Weights

**Option A: Official Weights (Recommended)**

1. Visit: https://github.com/shadow2496/VITON-HD
2. Follow their instructions to download pre-trained weights
3. Place weights in `weights/viton_hd/generator.pth`

**Option B: Train Your Own**

If you have a dataset of clothing and person images:
1. Follow VITON-HD training instructions
2. Train your own model
3. Place trained weights in `weights/viton_hd/`

**Option C: Use Placeholder Script**

```bash
# This creates placeholders for testing the framework
python scripts/download_viton_models.py
```

### Directory Structure

After setup, you should have:

```
weights/
├── viton_hd/
│   ├── generator.pth           # VITON-HD generator weights
│   └── discriminator.pth       # (optional) for training
├── openpose/
│   └── body_pose_model.pth     # (optional) for better pose estimation
├── human_parsing/
│   └── lip_parsing.pth         # (optional) for human segmentation
└── cloth_segmentation/
    └── cloth_segm.pth          # (optional) for clothing mask extraction

models/realistic/
├── male_m_front_asian.jpg      # Sample model photos
├── female_s_front_caucasian.jpg
└── ...
```

## Usage

### Command Line

```bash
# Using auto-select (prefers VITON-HD if available, falls back to Lite)
python scripts/demo_viton_tryon.py \
    --person models/realistic/male_m_front_asian.jpg \
    --clothing samples/clothing/tshirt_blue.png \
    --output output/viton_result.jpg

# Force VITON-Lite
python scripts/demo_viton_tryon.py \
    --person models/realistic/female_s_front_caucasian.jpg \
    --clothing samples/clothing/tshirt_red.png \
    --method viton_lite

# Try multiple clothing items
python scripts/demo_viton_tryon.py \
    --person models/realistic/male_m_front_asian.jpg \
    --clothing samples/clothing/*.png \
    --comparison
```

### Python API

```python
from src.clothing_fitter_viton import VITONClothingFitter
from src.utils import ImageLoader

# Initialize fitter
fitter = VITONClothingFitter()

# Load images
loader = ImageLoader()
person = loader.load_image('models/realistic/male_m_front.jpg')
clothing = loader.load_image('samples/clothing/tshirt_blue.png')

# Perform try-on
result = fitter.fit_clothing(person, clothing, clothing_type='shirt')

# Save result
loader.save_image(result, 'output/result.jpg')
```

### REST API

Start the API server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Use the VITON endpoint:

```bash
curl -X POST "http://localhost:8000/api/tryon/viton" \
  -F "person_image=@models/realistic/male_m_front.jpg" \
  -F "clothing_image=@samples/clothing/tshirt_blue.png" \
  -F "clothing_type=shirt"
```

## Performance

### VITON-Lite (Lightweight)

- **Speed**: ~1-2 seconds per image (CPU)
- **Quality**: Good - significantly better than basic overlay
- **Memory**: ~500MB RAM
- **GPU**: Not required
- **Weights**: None required

### VITON-HD (Full)

- **Speed**: ~5-10 seconds per image (GPU), 30-60s (CPU)
- **Quality**: Excellent - production-quality results
- **Memory**: ~4GB VRAM (GPU) or ~8GB RAM (CPU)
- **GPU**: Highly recommended
- **Weights**: ~250MB download

## Troubleshooting

### "No module named 'torch'"

VITON-HD requires PyTorch. Install it:

```bash
pip install torch torchvision
```

Or use VITON-Lite which doesn't require PyTorch:

```bash
python scripts/demo_viton_tryon.py --method viton_lite
```

### "VITON-HD model not loaded"

Model weights are not available. Either:
1. Download official weights (see Installation Steps above)
2. Use VITON-Lite instead: `--method viton_lite`

### "CUDA out of memory"

GPU doesn't have enough memory. Try:
1. Use CPU instead: `export CUDA_VISIBLE_DEVICES=""`
2. Reduce image resolution
3. Use VITON-Lite: `--method viton_lite`

### "Clothing is still tiny or mispositioned"

This suggests VITON-Lite needs tuning. Try:
1. Check that input images are high quality
2. Ensure person is standing upright, front-facing
3. Verify clothing image has transparent background or white background

### Poor Results

For best results:
- Use high-resolution images (512x768 or higher)
- Person should be standing, front-facing, arms slightly away from body
- Good lighting with minimal shadows
- Clothing should have clean background
- For VITON-HD: ensure all model weights are properly loaded

## Model Selection

The system automatically selects the best available method:

1. **VITON-HD** (if weights available) → Production quality
2. **VITON-Lite** (always available) → Good quality, fast
3. **Basic Fitter** (fallback) → Basic overlay

You can force a specific method:

```python
fitter = VITONClothingFitter()

# Force VITON-Lite
result = fitter.fit_clothing(person, clothing, method='viton_lite')

# Try VITON-HD (falls back if not available)
result = fitter.fit_clothing(person, clothing, method='viton_hd')
```

## Comparison: Before vs After

### Before (Old System)
- Cartoon-like models
- Clothing scaled incorrectly (too small)
- Positioned on stomach instead of chest
- Looks like a sticker, not natural

### After (VITON-Lite)
- Works with real photos
- Proper scaling based on body measurements
- Correctly positioned on chest
- Perspective warping follows body shape
- Feathered edges for natural look

### After (VITON-HD)
- Production-quality results
- Realistic cloth warping and draping
- Handles complex poses and occlusion
- Indistinguishable from real photos

## Limitations

### VITON-Lite
- Doesn't handle complex poses well
- Limited cloth deformation
- Best with front-facing, upright poses
- May not handle occlusion perfectly

### VITON-HD
- Requires large model weights download
- Slower without GPU
- Needs high-quality input images
- Best results require proper preprocessing

## Advanced Configuration

### Custom Model Weights

To use your own trained VITON-HD model:

```python
fitter = VITONClothingFitter(weights_dir='path/to/your/weights')
```

### Preprocessing Options

For better results, preprocess images:

```python
from src.viton_hd.cloth_mask import ClothMaskExtractor

# Extract clean clothing mask
mask_extractor = ClothMaskExtractor()
clothing_mask = mask_extractor.extract_mask(clothing_image, method='grabcut')
```

## API Integration

Add VITON support to your API:

```python
# In api/routes/tryon.py
from src.clothing_fitter_viton import VITONClothingFitter

fitter = VITONClothingFitter()

@router.post("/tryon/viton")
async def viton_tryon(
    person_image: UploadFile,
    clothing_image: UploadFile,
    method: str = "auto"
):
    # Load images
    person = await load_image(person_image)
    clothing = await load_image(clothing_image)
    
    # Perform try-on
    result = fitter.fit_clothing(person, clothing, method=method)
    
    return {"result_image": encode_image(result)}
```

## Contributing

To contribute to VITON integration:

1. Test with various person and clothing combinations
2. Report issues with sample images
3. Suggest improvements to VITON-Lite algorithm
4. Help integrate additional models

## References

- VITON-HD Paper: https://arxiv.org/abs/2103.16874
- VITON-HD GitHub: https://github.com/shadow2496/VITON-HD
- MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review demo script examples
3. Open an issue on GitHub
4. Check API documentation at `/docs`

---

**Note**: This implementation provides a framework for VITON-HD integration. Full functionality requires obtaining or training model weights. VITON-Lite works immediately without weights and provides significantly improved results over basic overlay methods.
