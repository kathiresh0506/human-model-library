# MakeHuman and Blender Setup Guide

This guide explains how to set up MakeHuman and Blender for generating photorealistic 3D human models for the virtual try-on system.

## Overview

The realistic model generation pipeline uses:
1. **MakeHuman** - Free 3D human model generator with customizable body parameters
2. **Blender** - Professional 3D rendering software for high-quality image output

**Note:** These tools are optional. If not installed, the system will use the enhanced 2D model generator as a fallback.

## MakeHuman Installation

### Windows

1. Download MakeHuman Community from: http://www.makehumancommunity.org/
2. Run the installer: `makehuman-community-1.2.0-windows.exe`
3. Follow the installation wizard
4. Default installation path: `C:\Program Files\MakeHuman Community`

### macOS

1. Download the macOS DMG from: http://www.makehumancommunity.org/
2. Open the DMG file
3. Drag MakeHuman to Applications folder
4. On first run, you may need to allow it in Security & Privacy settings

### Linux (Ubuntu/Debian)

```bash
# Add repository
sudo add-apt-repository ppa:makehuman-official/makehuman-community
sudo apt-get update

# Install
sudo apt-get install makehuman-community
```

## Blender Installation

### Windows

1. Download Blender from: https://www.blender.org/download/
2. Run the installer: `blender-3.6.5-windows-x64.msi`
3. Follow the installation wizard

### macOS

```bash
# Using Homebrew
brew install --cask blender
```

### Linux (Ubuntu/Debian)

```bash
# Official package
sudo snap install blender --classic
```

## Verify Installation

Check if both tools are accessible:

```bash
# Check if MakeHuman/Blender are installed
python scripts/create_realistic_models.py --check
```

## Pipeline Workflow

### Automated Generation

```bash
# Generate a model (auto-detects available tools)
python scripts/create_realistic_models.py \
  --gender male \
  --size M \
  --age young \
  --ethnicity asian \
  --output models/male/M/young/asian
```

### Fallback: Enhanced 2D Generator

If MakeHuman/Blender are not installed:

```bash
python scripts/create_sample_models.py
```

This creates realistic 2D models with:
- Facial features (eyes, nose, mouth)
- Hair styling
- 3D shading and highlights
- Natural body proportions
- Transparent RGBA background

## Rendering Settings

### Resolution

- **Portrait full-body**: 1024×2048 (recommended)
- **High quality**: 2048×4096

### Lighting Setup

The Blender renderer uses professional 3-point lighting:

1. **Key Light**: Main light source (front-side, bright)
2. **Fill Light**: Softens shadows (opposite side, dimmer)
3. **Rim Light**: Separates subject from background (back)

## Resources

- MakeHuman Wiki: http://www.makehumancommunity.org/wiki/
- Blender Manual: https://docs.blender.org/
- Project README: ../README.md
