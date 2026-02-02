"""
Master script to generate all sample data and run a test try-on.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the generation scripts
from scripts.create_sample_models import create_sample_models
from scripts.create_sample_clothing import create_sample_clothing


def main():
    print("=" * 70)
    print("Generating All Sample Data for Virtual Try-On Demo")
    print("=" * 70)
    
    # Step 1: Create sample models
    print("\n📸 Step 1: Creating sample model images...")
    print("-" * 70)
    try:
        models = create_sample_models()
        print(f"✅ Created {len(models)} model(s)")
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        return 1
    
    # Step 2: Create sample clothing
    print("\n👕 Step 2: Creating sample clothing images...")
    print("-" * 70)
    try:
        clothing = create_sample_clothing()
        print(f"✅ Created {len(clothing)} clothing item(s)")
    except Exception as e:
        print(f"❌ Error creating clothing: {e}")
        return 1
    
    # Step 3: Run a test try-on
    print("\n🎨 Step 3: Running test try-on...")
    print("-" * 70)
    try:
        from src.model_selector import ModelSelector
        from src.clothing_fitter import ClothingFitter
        from src.utils import ImageLoader
        
        selector = ModelSelector()
        fitter = ClothingFitter()
        loader = ImageLoader()
        
        # Use the first model and first clothing item
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / "models" / "male" / "M" / "young" / "asian" / "front.png"
        clothing_path = base_dir / "samples" / "clothing" / "tshirt_blue.png"
        output_path = base_dir / "output" / "demo_result.png"
        
        # Load and process
        model_img = loader.load_image(str(model_path))
        
        # Load clothing with transparency support
        from PIL import Image
        import numpy as np
        clothing_pil = Image.open(clothing_path)
        if clothing_pil.mode == 'RGBA':
            clothing_img = np.array(clothing_pil)
            # Convert RGBA to RGB with white background
            rgb_clothing = np.zeros((clothing_img.shape[0], clothing_img.shape[1], 3), dtype=np.uint8)
            rgb_clothing[:, :] = 255
            alpha = clothing_img[:, :, 3:4] / 255.0
            clothing_img = (clothing_img[:, :, :3] * alpha + rgb_clothing * (1 - alpha)).astype(np.uint8)
        else:
            clothing_img = np.array(clothing_pil.convert('RGB'))
        
        # Fit clothing
        result = fitter.fit_clothing(model_img, clothing_img, 'shirt')
        
        if result is not None:
            loader.save_image(result, str(output_path))
            print(f"✅ Test try-on completed: {output_path}")
        else:
            print("⚠️  Test try-on completed but may need adjustment")
    except Exception as e:
        print(f"⚠️  Test try-on encountered issue: {e}")
        print("   (This is okay - the samples were still created)")
    
    # Summary
    print("\n" + "=" * 70)
    print("All samples generated successfully! 🎉")
    print("=" * 70)
    print("\nWhat was created:")
    print("  📸 Sample Models:")
    for model in models:
        print(f"     - {model}")
    print("\n  👕 Sample Clothing:")
    for item in clothing:
        print(f"     - {item}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("Run the demo with:")
    print("  python scripts/demo_tryon.py --gender male --size M --clothing samples/clothing/tshirt_blue.png")
    print("\nTry different options:")
    print("  python scripts/demo_tryon.py --gender female --size S --clothing samples/clothing/tshirt_red.png")
    print("  python scripts/demo_tryon.py --gender male --size L --age_group middle --ethnicity african --clothing samples/clothing/pants_black.png")
    print("\nCheck the output/ folder for results!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
