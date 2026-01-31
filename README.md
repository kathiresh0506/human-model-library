# Human Model Library

A complete virtual try-on system for fitting clothing on pre-built human models, similar to Myntra's virtual try-on feature. The library provides organized human models categorized by size, gender, age, and ethnicity, along with a FastAPI-based REST API for virtual try-on functionality.

## 🌟 Features

- **Pre-built Human Model Library**: Organized by gender, size, age, and ethnicity
- **Virtual Try-On API**: RESTful API for fitting clothing on models
- **Intelligent Model Selection**: Automatic selection of best-matching models
- **Size Recommendation**: Body measurement-based size recommendations
- **Pose Estimation**: Body keypoint detection using MediaPipe
- **Advanced Image Warping**: Clothing deformation and blending
- **Batch Model Generation**: Scripts for generating model variations
- **Docker Support**: Easy deployment with Docker

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Organization](#model-organization)
- [Usage Examples](#usage-examples)
- [Model Generation](#model-generation)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerized deployment
- (Optional) MakeHuman for 3D model generation
- (Optional) Blender for rendering

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/kathiresh0506/human-model-library.git
cd human-model-library

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build Docker image
docker build -t human-model-library .

# Run container
docker run -p 8000:8000 human-model-library
```

## 🎯 Quick Start

### Starting the API Server

```bash
# Start the FastAPI server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

### Basic Python Usage

```python
from src.model_selector import ModelSelector
from src.clothing_fitter import ClothingFitter

# Initialize components
selector = ModelSelector()
fitter = ClothingFitter()

# Select a model
model_path = selector.select_model(
    gender='male',
    size='M',
    age_group='young',
    ethnicity='asian'
)

# Perform virtual try-on
result = fitter.process_full_tryon(
    model_path=f"{model_path}/front.png",
    clothing_path="path/to/clothing.jpg",
    clothing_type="shirt",
    output_path="result.png"
)
```

## 🎨 Quick Demo

Want to test the virtual try-on without setting up the full API? Use our demo scripts!

### Generate Sample Data

First, generate sample model images and clothing items:

```bash
python scripts/generate_all_samples.py
```

This creates:
- 3 sample human models (male/female, different sizes and ethnicities)
- 3 sample clothing items (2 t-shirts, 1 pants)
- A test try-on result

### Run Demo Try-On

Try different combinations:

```bash
# Basic demo with male model and blue t-shirt
python scripts/demo_tryon.py --gender male --size M --clothing samples/clothing/tshirt_blue.png

# Female model with red t-shirt
python scripts/demo_tryon.py --gender female --size S --clothing samples/clothing/tshirt_red.png

# Male model with pants
python scripts/demo_tryon.py --gender male --size L --age_group middle --ethnicity african --clothing samples/clothing/pants_black.png
```

### View Output

Check the `output/` folder for:
- `demo_result.png` - The try-on result
- `comparison_demo_result.png` - Side-by-side comparison

## 📚 API Documentation

### Core Endpoints

#### Health Check
```
GET /health
```
Returns API health status.

#### List Models
```
GET /api/models?gender=male&size=M
```
List all available models with optional filters.

**Query Parameters:**
- `gender` (optional): male or female
- `size` (optional): S, M, L, or XL
- `age_group` (optional): young, middle, or senior
- `ethnicity` (optional): asian, african, caucasian, hispanic, middle_eastern

#### Select Model
```
POST /api/models/select
```
Select the best matching model based on criteria.

**Request Body:**
```json
{
  "gender": "male",
  "size": "M",
  "age_group": "young",
  "ethnicity": "asian"
}
```

#### Virtual Try-On
```
POST /api/tryon
```
Perform virtual try-on of clothing on a model.

**Request Body:**
```json
{
  "clothing_image": "data:image/png;base64,iVBORw0KGgo...",
  "clothing_type": "shirt",
  "gender": "male",
  "size": "M",
  "age_group": "young",
  "ethnicity": "asian"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Virtual try-on completed successfully",
  "result_image": "data:image/png;base64,iVBORw0KGgo...",
  "model_used": {
    "id": "male_m_young_asian_001",
    "gender": "male",
    "size": "M",
    "age_group": "young",
    "ethnicity": "asian"
  }
}
```

#### Size Recommendation
```
POST /api/models/size-recommendation
```
Get size recommendation based on body measurements.

**Request Body:**
```json
{
  "gender": "male",
  "chest": 98,
  "waist": 83,
  "hip": 98,
  "height": 177
}
```

## 📁 Project Structure

```
human-model-library/
├── config/                      # Configuration files
│   ├── sizes.yaml              # Size specifications
│   ├── ethnicities.yaml        # Ethnicity configurations
│   └── age_groups.yaml         # Age group definitions
├── src/                         # Core source code
│   ├── utils.py                # Utility functions
│   ├── model_selector.py       # Model selection logic
│   ├── pose_estimator.py       # Body keypoint detection
│   ├── warping.py              # Image warping utilities
│   └── clothing_fitter.py      # Virtual try-on implementation
├── api/                         # FastAPI application
│   ├── app.py                  # FastAPI setup
│   ├── schemas.py              # Pydantic models
│   └── routes/                 # API routes
│       ├── models.py           # Model endpoints
│       └── tryon.py            # Try-on endpoints
├── generator/                   # Model generation scripts
│   ├── makehuman_generator.py  # MakeHuman integration
│   ├── blender_renderer.py     # Blender rendering
│   └── batch_generate.py       # Batch generation
├── models/                      # Human model library
│   ├── male/
│   │   └── {size}/{age}/{ethnicity}/
│   └── female/
│       └── {size}/{age}/{ethnicity}/
├── tests/                       # Unit tests
│   ├── test_model_selector.py
│   └── test_clothing_fitter.py
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── Dockerfile                  # Docker configuration
└── README.md                   # This file
```

## 👤 Model Organization

Models are organized in a hierarchical directory structure:

```
models/
├── {gender}/           # male, female
    ├── {size}/         # S, M, L, XL
        ├── {age}/      # young, middle, senior
            ├── {ethnicity}/  # asian, african, caucasian, hispanic, middle_eastern
                ├── front.png
                ├── side.png
                ├── back.png
                └── metadata.json
```

### Size Chart

**Male:**
| Size | Chest (cm) | Waist (cm) | Hip (cm) | Height (cm) |
|------|------------|------------|----------|-------------|
| S    | 86-91      | 71-76      | 86-91    | 170-175     |
| M    | 96-101     | 81-86      | 96-101   | 175-180     |
| L    | 106-111    | 91-96      | 106-111  | 178-183     |
| XL   | 116-121    | 101-106    | 116-121  | 180-185     |

**Female:**
| Size | Bust (cm) | Waist (cm) | Hip (cm) | Height (cm) |
|------|-----------|------------|----------|-------------|
| S    | 82-86     | 62-66      | 88-92    | 160-165     |
| M    | 88-92     | 68-72      | 94-98    | 163-168     |
| L    | 96-100    | 76-80      | 102-106  | 165-170     |
| XL   | 104-108   | 84-88      | 110-114  | 167-172     |

## 💡 Usage Examples

### Example 1: Select a Model

```python
from src.model_selector import ModelSelector

selector = ModelSelector()

# Select by exact criteria
model = selector.select_model(
    gender='female',
    size='M',
    age_group='young',
    ethnicity='caucasian'
)

# Get model metadata
metadata = selector.get_model_metadata(model)
print(f"Selected model: {metadata['id']}")
```

### Example 2: Virtual Try-On

```python
from src.clothing_fitter import ClothingFitter
from src.utils import ImageLoader

fitter = ClothingFitter()
loader = ImageLoader()

# Load images
model_image = loader.load_image('models/male/M/young/asian/front.png')
clothing_image = loader.load_image('clothing/shirt.jpg')

# Perform try-on
result = fitter.fit_clothing(
    model_image=model_image,
    clothing_image=clothing_image,
    clothing_type='shirt'
)

# Save result
loader.save_image(result, 'output/result.png')
```

### Example 3: Size Recommendation

```python
from src.model_selector import ModelSelector

selector = ModelSelector()

# Get size recommendation based on measurements
size = selector.find_best_size_match(
    gender='male',
    chest=98,
    waist=83,
    hip=98,
    height=177
)

print(f"Recommended size: {size}")  # Output: M
```

### Example 4: API Call (cURL)

```bash
# Virtual try-on request
curl -X POST "http://localhost:8000/api/tryon" \
  -H "Content-Type: application/json" \
  -d '{
    "clothing_image": "data:image/png;base64,...",
    "clothing_type": "shirt",
    "gender": "male",
    "size": "M",
    "age_group": "young"
  }'
```

## 🔨 Model Generation

### Prerequisites for Model Generation

- **MakeHuman**: For generating 3D human models
- **Blender**: For rendering high-quality images

### Generate Single Model

```bash
cd generator

# Generate a specific model
python batch_generate.py \
  --gender male \
  --size M \
  --age-group young \
  --ethnicity asian
```

### Batch Generate All Models

```bash
cd generator

# Generate all model combinations
python batch_generate.py \
  --config-dir ../config \
  --models-dir ../models

# Generate with limit (for testing)
python batch_generate.py --limit 10
```

## ⚙️ Configuration

### Size Configuration (config/sizes.yaml)

Defines body measurements for each size category:

```yaml
male:
  M:
    chest:
      min: 96
      max: 101
      avg: 98.5
    waist:
      min: 81
      max: 86
      avg: 83.5
    # ... more measurements
```

### Ethnicity Configuration (config/ethnicities.yaml)

Defines characteristics for each ethnicity:

```yaml
ethnicities:
  asian:
    name: "Asian"
    description: "Asian ethnicity features"
    skin_tone_range: [0.6, 0.8]
    # ... more characteristics
```

### Age Group Configuration (config/age_groups.yaml)

Defines age ranges and characteristics:

```yaml
age_groups:
  young:
    name: "Young Adult"
    age_range: [18, 30]
    characteristics:
      physical:
        - "Smooth skin texture"
        # ... more characteristics
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model_selector.py

# Run with coverage
pytest --cov=src tests/
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8
```

### Code Style

This project follows PEP 8 style guidelines:

```bash
# Format code
black src/ api/ generator/

# Check style
flake8 src/ api/ generator/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe for pose estimation
- MakeHuman for 3D human model generation
- Blender for rendering capabilities
- FastAPI for the web framework

## 📞 Support

For support, please open an issue on GitHub or contact the maintainers.

## 🗺️ Roadmap

- [ ] Add more ethnicity variations
- [ ] Implement advanced pose variations
- [ ] Add support for multiple clothing items simultaneously
- [ ] Improve clothing segmentation
- [ ] Add AR/VR integration
- [ ] Mobile application support
- [ ] Real-time video try-on

---

**Note**: This is a production-ready implementation with proper error handling, logging, type hints, and documentation. For actual 3D model generation, ensure MakeHuman and Blender are properly installed and configured.
