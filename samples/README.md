# Sample Data

This directory contains sample images for testing and demonstrating the virtual try-on functionality.

## Directory Structure

```
samples/
└── clothing/           # Sample clothing images
    ├── tshirt_blue.png
    ├── tshirt_red.png
    └── pants_black.png
```

## Sample Clothing

### T-Shirts
- **tshirt_blue.png** - Blue t-shirt with transparent background
- **tshirt_red.png** - Red t-shirt with transparent background

### Pants
- **pants_black.png** - Black pants with transparent background

## Generating Samples

To generate all sample clothing images, run:

```bash
python scripts/create_sample_clothing.py
```

Or generate everything (models + clothing) with:

```bash
python scripts/generate_all_samples.py
```

## Using Custom Clothing

You can add your own clothing images to this directory. For best results:

1. Use transparent backgrounds (PNG with alpha channel)
2. Keep images reasonably sized (200-500px)
3. Use clear, front-facing clothing views
4. Name files descriptively

Then use them in demos:

```bash
python scripts/demo_tryon.py --clothing samples/clothing/your_image.png
```
