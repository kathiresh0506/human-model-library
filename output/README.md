# Output Directory

This directory contains the results of virtual try-on operations.

## Contents

When you run the demo scripts, the following files will be generated here:

- `demo_result.png` - Result of the try-on operation
- `comparison_demo_result.png` - Side-by-side comparison showing:
  - Original model
  - Clothing item
  - Final result

## Usage

Results are automatically saved here when you run:

```bash
python scripts/demo_tryon.py
```

You can specify custom output paths using the `--output` flag:

```bash
python scripts/demo_tryon.py --output output/my_custom_result.png
```

## Cleanup

Feel free to delete files from this directory as needed. The directory structure will be maintained.
