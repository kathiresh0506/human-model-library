# IDM-VTON Integration Guide

## Overview

This guide explains how to use the new IDM-VTON integration for Myntra-quality virtual try-on results.

## What is IDM-VTON?

IDM-VTON (Image-based Deep Model Virtual Try-ON) is a state-of-the-art virtual try-on model that produces professional-quality results comparable to what you see on e-commerce platforms like Myntra.

Unlike the previous solutions, IDM-VTON:
- Uses deep learning for realistic clothing fit
- Handles complex clothing patterns and textures
- Produces natural-looking results with proper shadows and folds
- Works with real human photos

## Quick Start

### 1. One-Command Setup

Run the setup script to install dependencies and create the directory structure:

```bash
python scripts/setup_idm_vton.py
```

This will:
- Install `gradio_client` for API access
- Create the `models/real_humans/` directory structure
- Test connection to IDM-VTON on Hugging Face Spaces

### 2. Add Real Human Photos

The setup creates this structure:

```
models/real_humans/
├── male/
│   ├── S/  (add 3-5 photos here)
│   ├── M/  (add 3-5 photos here)
│   ├── L/  (add 3-5 photos here)
│   └── XL/ (add 3-5 photos here)
└── female/
    ├── S/  (add 3-5 photos here)
    ├── M/  (add 3-5 photos here)
    ├── L/  (add 3-5 photos here)
    └── XL/ (add 3-5 photos here)
```

Each directory has a `README.md` with detailed requirements.

**Photo Requirements:**
- Real human (not AI-generated or cartoon)
- Front-facing, standing pose
- Arms slightly away from body (for clothing fit)
- Clean background (white/gray preferred)
- High resolution (512x768 minimum)
- Format: JPG or PNG

### 3. Run Virtual Try-On

#### Using Auto-Selected Model

```bash
python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png
```

This will:
1. Automatically select a male/M model
2. Perform virtual try-on with the clothing
3. Save result to `output/idm_vton_result.png`

#### Specify Gender and Size

```bash
python scripts/demo_idm_vton.py --clothing your_shirt.jpg --gender female --size L
```

#### Use Your Own Person Image

```bash
python scripts/demo_idm_vton.py --person your_model.jpg --clothing your_shirt.jpg
```

#### Specify Clothing Type

```bash
# For upper body (shirts, t-shirts, jackets)
python scripts/demo_idm_vton.py --clothing shirt.jpg --clothing-type upper_body

# For lower body (pants, skirts)
python scripts/demo_idm_vton.py --clothing pants.jpg --clothing-type lower_body

# For dresses
python scripts/demo_idm_vton.py --clothing dress.jpg --clothing-type dresses
```

## Python API

### Basic Usage

```python
import sys
sys.path.insert(0, 'src')

from idm_vton import virtual_tryon

# Perform virtual try-on
result = virtual_tryon(
    person_image="models/realistic/male/M/front_001.jpg",
    clothing_image="samples/clothing/tshirt_blue.png",
    output_path="output/my_result.png",
    clothing_type="upper_body"
)

print(f"Result saved to: {result}")
```

### Advanced Usage with Client

```python
from idm_vton import IDMVTONClient

# Initialize client
client = IDMVTONClient()

# Perform try-on with custom parameters
result = client.try_on(
    person_image="path/to/person.jpg",
    clothing_image="path/to/clothing.jpg",
    clothing_type="upper_body",
    num_steps=30,          # More steps = better quality but slower
    guidance_scale=2.0,     # Controls how closely to follow the prompt
    seed=42                 # For reproducible results
)
```

### Using with Model Selector

```python
from idm_vton import virtual_tryon
from model_selector_real import RealModelSelector

# Select a model
selector = RealModelSelector(base_dir="models/real_humans")
person_image = selector.select_model("male", "M")

# Perform try-on
result = virtual_tryon(
    person_image=person_image,
    clothing_image="samples/clothing/tshirt_blue.png"
)
```

## Expected Results

After running the demo, you should see:

1. **Processing Time**: 30-60 seconds (first run may be slower)
2. **Output Quality**: Professional-grade, comparable to Myntra
3. **Natural Fit**: Clothing fits naturally on the model with proper shadows and folds
4. **Clean Integration**: Clothing blends seamlessly with the person

## Troubleshooting

### Connection Issues

If you see connection errors:

1. **Check Internet Connection**: IDM-VTON runs on Hugging Face Spaces and requires internet
2. **Try Again Later**: The Space may be busy or loading
3. **Check Status**: Visit https://huggingface.co/spaces/yisol/IDM-VTON

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Install gradio_client
pip install gradio_client>=0.10.0

# Install Pillow (if needed)
pip install Pillow
```

### No Models Found

If you get "No models available":

1. **Add Photos**: Add real human photos to `models/real_humans/`
2. **Use Existing Models**: Use `--base-dir models/realistic` to use existing models
3. **Provide Image**: Use `--person path/to/image.jpg` to specify an image directly

### Poor Results

If results are not good:

1. **Check Photo Quality**: Ensure person photo meets requirements (front-facing, clean background)
2. **Check Clothing Image**: Ensure clothing is clearly visible and isolated
3. **Adjust Parameters**: Try different `num_steps` or `guidance_scale` values
4. **Try Different Model**: Use a different person image

## File Structure

```
.
├── src/
│   └── idm_vton/
│       ├── __init__.py       # Module exports
│       ├── client.py         # IDMVTONClient class
│       └── tryon.py          # High-level virtual_tryon function
│
├── scripts/
│   ├── setup_idm_vton.py                 # One-command setup
│   ├── demo_idm_vton.py                  # Demo script
│   └── download_real_human_photos.py     # Photo management
│
├── models/
│   └── real_humans/           # Real human photos
│       ├── male/
│       │   ├── S/
│       │   ├── M/
│       │   ├── L/
│       │   └── XL/
│       ├── female/
│       │   ├── S/
│       │   ├── M/
│       │   ├── L/
│       │   └── XL/
│       └── metadata.json
│
└── tests/
    └── test_idm_vton.py       # Tests for IDM-VTON integration
```

## Comparison with Other Methods

| Feature | IDM-VTON | VITON-HD | VITON-Lite | Basic |
|---------|----------|----------|------------|-------|
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Speed | Moderate (30-60s) | Slow (GPU required) | Fast (1-2s) | Fast |
| GPU Required | No (Cloud) | Yes | No | No |
| Setup Complexity | Easy | Hard | Easy | Easy |
| Result Quality | Myntra-level | High | Good | Basic |

## Contributing

To improve the IDM-VTON integration:

1. Add more real human photos to `models/real_humans/`
2. Test with different clothing types
3. Report issues with specific combinations
4. Suggest improvements to the API

## License

This integration uses the IDM-VTON model via Hugging Face Spaces. Please respect the terms of use for the model and any photos you add.

## Support

For issues or questions:

1. Check this guide first
2. Review the error messages
3. Check the demo script examples
4. Open an issue on GitHub with:
   - Error message
   - Input images (if possible)
   - Steps to reproduce
