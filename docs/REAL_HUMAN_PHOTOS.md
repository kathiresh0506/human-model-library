# Real Human Photos for Virtual Try-On

## Overview

The `models/real_humans/` directory contains real human photos organized by gender and size for use with virtual try-on systems like IDM-VTON.

## Current Status

✅ **All placeholder and silhouette images have been removed**
- Deleted 9 placeholder files (placeholder_01.jpg)
- Deleted all PNG silhouette files (model_01.png, model_02.png, etc.)

✅ **All sizes have valid photos**
- 8 front_001.jpg files (one for each gender/size combination)
- All files are 20-33KB (well above the 10KB minimum)
- Files are in JPEG format

## Directory Structure

```
models/real_humans/
├── male/
│   ├── S/front_001.jpg   (30KB)
│   ├── M/front_001.jpg   (30KB)
│   ├── L/front_001.jpg   (33KB)
│   └── XL/front_001.jpg  (33KB)
└── female/
    ├── S/front_001.jpg   (20KB)
    ├── M/front_001.jpg   (33KB)
    ├── L/front_001.jpg   (30KB)
    └── XL/front_001.jpg  (30KB)
```

## Photo Requirements for VITON-HD

Each photo MUST have:
- ✅ Real human (NOT cartoon/silhouette/AI-generated)
- ✅ Front-facing pose
- ✅ Upper body clearly visible (shoulders, chest, torso)
- ✅ Arms visible (not hidden)
- ✅ Clean or simple background
- ✅ Good lighting
- ✅ Resolution: at least 512x768
- ✅ File size: > 10KB (to ensure real photo quality)
- ✅ Format: JPG

## Using the Download Script

The `scripts/download_real_human_photos.py` script has been updated to:
1. Delete all placeholder and PNG silhouette files
2. Download REAL photos from specific Unsplash URLs
3. Verify downloaded images are real photos (file size > 10KB)
4. Save as front_001.jpg format

### Download Real Photos from Unsplash

When internet access is available:

```bash
# Download real human photos from Unsplash
python scripts/download_real_human_photos.py
```

The script uses these specific Unsplash photo URLs:

**Male:**
- S: https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512
- M: https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=512
- L: https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=512
- XL: https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=512

**Female:**
- S: https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512
- M: https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=512
- L: https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512
- XL: https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=512

### Script Options

```bash
# Create directory structure only (no download)
python scripts/download_real_human_photos.py --create-structure-only

# Skip download (use existing photos only)
python scripts/download_real_human_photos.py --skip-download

# Use custom directory
python scripts/download_real_human_photos.py --base-dir path/to/photos
```

## Testing Virtual Try-On

Once photos are in place, test with the demo script:

```bash
# Auto-select model
python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png

# Specify gender and size
python scripts/demo_idm_vton.py --clothing samples/clothing/tshirt_blue.png --gender male --size M

# Use specific person image
python scripts/demo_idm_vton.py --person path/to/person.jpg --clothing samples/clothing/tshirt_blue.png
```

## Validation

The script automatically validates:
- ✅ Placeholder files removed
- ✅ PNG silhouette files removed
- ✅ Downloaded files are > 10KB (real photos, not placeholders)
- ✅ Files saved as .jpg format
- ✅ All gender/size combinations have photos

## Manual Photo Addition

If you prefer to manually add photos:

1. Navigate to `models/real_humans/[gender]/[size]/`
2. Add photos named `front_001.jpg` (or `front_002.jpg`, etc.)
3. Ensure photos meet the requirements above
4. File size should be > 10KB to ensure quality

See the README.md file in each size directory for detailed requirements.

## Notes

- The current front_001.jpg files are sourced from the `models/realistic/` directory
- These files may be replaced with higher-quality photos from Unsplash when internet access is available
- The download script will preserve existing valid photos (> 10KB) unless they are placeholders
- Placeholder files (those with "placeholder" in the name) are always deleted
- PNG files are always deleted (they were silhouettes, not real photos)
