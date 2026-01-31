"""
Blender renderer script.
Renders high-quality images from 3D human models using Blender.

Features:
- Professional 3-point lighting setup
- Multiple camera angles (front, side, back)
- High-quality rendering with anti-aliasing
- Transparent or solid background options
"""
import os
import subprocess
import logging
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class BlenderRenderer:
    """
    Renders 3D models using Blender with professional lighting and camera setup.
    """
    
    def __init__(self, blender_path: Optional[str] = None):
        """
        Initialize BlenderRenderer.
        
        Args:
            blender_path: Path to Blender executable
        """
        self.blender_path = blender_path or self._find_blender()
        self.has_blender = self.blender_path is not None
        
    def _find_blender(self) -> Optional[str]:
        """
        Attempt to find Blender installation.
        
        Returns:
            Path to Blender executable, or None if not found
        """
        # Try common paths and system PATH
        try:
            result = subprocess.run(['which', 'blender'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip()
                logger.info(f"Found Blender at: {path}")
                return path
        except:
            pass
        
        # Common installation paths
        possible_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found Blender at: {path}")
                return path
        
        logger.warning("Blender not found in common paths")
        return None
    
    def render_model(self,
                    model_path: str,
                    output_path: str,
                    view: str = 'front',
                    resolution: Tuple[int, int] = (1024, 2048),
                    transparent_bg: bool = True) -> bool:
        """
        Render a 3D model to an image with professional lighting.
        
        Args:
            model_path: Path to the 3D model file
            output_path: Path to save the rendered image
            view: View angle ('front', 'side', 'back')
            resolution: Output resolution (width, height)
            transparent_bg: Use transparent background
            
        Returns:
            True if successful, False otherwise
        """
        if not self.has_blender:
            logger.error("Blender not available - cannot render 3D models")
            logger.info("Install Blender from: https://www.blender.org/download/")
            return False
        
        try:
            logger.info(f"Rendering {view} view of {model_path}")
            
            # Create Blender Python script
            script = self._create_render_script(
                model_path, output_path, view, resolution, transparent_bg
            )
            
            # Save script to temporary file
            script_path = Path(output_path).parent / "render_script.py"
            with open(script_path, 'w') as f:
                f.write(script)
            
            # Run Blender in background mode
            cmd = [
                self.blender_path,
                '--background',
                '--python', str(script_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Render completed: {output_path}")
                # Clean up script
                script_path.unlink()
                return True
            else:
                logger.error(f"Render failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error rendering model: {e}")
            return False
    
    def _create_render_script(self,
                            model_path: str,
                            output_path: str,
                            view: str,
                            resolution: Tuple[int, int],
                            transparent_bg: bool = True) -> str:
        """
        Create Blender Python script for rendering with professional lighting.
        
        Args:
            model_path: Path to 3D model
            output_path: Output image path
            view: View angle
            resolution: Resolution tuple
            transparent_bg: Use transparent background
            
        Returns:
            Blender Python script as string
        """
        camera_positions = {
            'front': (0, -5, 1.5),
            'side': (5, 0, 1.5),
            'back': (0, 5, 1.5),
        }
        
        camera_rotations = {
            'front': (90, 0, 0),
            'side': (90, 0, 90),
            'back': (90, 0, 180),
        }
        
        cam_pos = camera_positions.get(view, (0, -5, 1.5))
        cam_rot = camera_rotations.get(view, (90, 0, 0))
        
        script = f"""
import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import model (try multiple formats)
model_imported = False
try:
    # Try MakeHuman MHX2 format
    bpy.ops.import_scene.makehuman_mhx2(filepath='{model_path}')
    model_imported = True
except:
    try:
        # Try FBX format
        bpy.ops.import_scene.fbx(filepath='{model_path}')
        model_imported = True
    except:
        try:
            # Try Collada/DAE format
            bpy.ops.wm.collada_import(filepath='{model_path}')
            model_imported = True
        except:
            print("ERROR: Could not import model - unsupported format")

if model_imported:
    # Setup camera
    bpy.ops.object.camera_add(location={cam_pos})
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians({cam_rot[0]}), 
                            math.radians({cam_rot[1]}), 
                            math.radians({cam_rot[2]}))
    bpy.context.scene.camera = camera
    
    # Professional 3-point lighting setup
    
    # Key light (main light) - front, slightly to the side
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 3))
    key_light = bpy.context.active_object
    key_light.data.energy = 200
    key_light.data.size = 2
    key_light.rotation_euler = (math.radians(60), 0, math.radians(-30))
    
    # Fill light (softer, opposite side) - reduces harsh shadows
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 80
    fill_light.data.size = 2
    fill_light.rotation_euler = (math.radians(45), 0, math.radians(30))
    
    # Rim/back light - creates depth and separation from background
    bpy.ops.object.light_add(type='AREA', location=(0, 2, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 100
    rim_light.data.size = 1.5
    rim_light.rotation_euler = (math.radians(120), 0, 0)
    
    # Ambient light for overall illumination
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    ambient = bpy.context.active_object
    ambient.data.energy = 0.3
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.resolution_x = {resolution[0]}
    scene.render.resolution_y = {resolution[1]}
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = '{output_path}'
    
    # Enable transparent background if requested
    scene.render.film_transparent = {str(transparent_bg)}
    
    if not {str(transparent_bg)}:
        # Use white/gray gradient background
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (0.95, 0.95, 0.95, 1)  # Light gray
    
    # Enable high quality rendering
    scene.render.engine = 'CYCLES'  # Use Cycles for better quality
    scene.cycles.samples = 128  # Good quality/speed balance
    scene.cycles.use_denoising = True
    
    # Anti-aliasing
    scene.render.filter_size = 1.5
    
    # Render
    bpy.ops.render.render(write_still=True)
    print(f"Render complete: {output_path}")
else:
    print("ERROR: Model import failed")
"""
        return script
    
    def render_multiple_views(self,
                            model_path: str,
                            output_dir: str,
                            views: Optional[List[str]] = None,
                            resolution: Tuple[int, int] = (1024, 2048),
                            transparent_bg: bool = True) -> bool:
        """
        Render multiple views of a model.
        
        Args:
            model_path: Path to 3D model
            output_dir: Directory to save rendered images
            views: List of views to render (default: front, side, back)
            resolution: Output resolution (portrait: 1024x2048)
            transparent_bg: Use transparent background
            
        Returns:
            True if all renders successful, False otherwise
        """
        if views is None:
            views = ['front', 'side', 'back']
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        success = True
        for view in views:
            output_path = os.path.join(output_dir, f"{view}.png")
            
            if not self.render_model(model_path, output_path, view, resolution, transparent_bg):
                logger.error(f"Failed to render {view} view")
                success = False
        
        return success
    
    def batch_render(self,
                    models: List[str],
                    output_base_dir: str,
                    views: Optional[List[str]] = None,
                    resolution: Tuple[int, int] = (512, 512)) -> dict:
        """
        Batch render multiple models.
        
        Args:
            models: List of model file paths
            output_base_dir: Base directory for outputs
            views: Views to render for each model
            resolution: Output resolution
            
        Returns:
            Dictionary mapping model paths to success status
        """
        results = {}
        
        for model_path in models:
            model_name = Path(model_path).stem
            output_dir = os.path.join(output_base_dir, model_name)
            
            try:
                success = self.render_multiple_views(
                    model_path, output_dir, views, resolution
                )
                results[model_path] = success
                
            except Exception as e:
                logger.error(f"Error rendering {model_path}: {e}")
                results[model_path] = False
        
        return results


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Render 3D models with Blender")
    parser.add_argument('--model', required=True, help='Path to 3D model')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--views', nargs='+', default=['front', 'side', 'back'],
                       help='Views to render')
    parser.add_argument('--width', type=int, default=512, help='Output width')
    parser.add_argument('--height', type=int, default=512, help='Output height')
    
    args = parser.parse_args()
    
    renderer = BlenderRenderer()
    success = renderer.render_multiple_views(
        args.model,
        args.output,
        args.views,
        (args.width, args.height)
    )
    
    if success:
        print(f"All renders completed successfully: {args.output}")
    else:
        print("Some renders failed")
