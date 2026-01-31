"""
Setup configuration for Human Model Library package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="human-model-library",
    version="1.0.0",
    author="Human Model Library Contributors",
    author_email="",
    description="Virtual try-on system with pre-built human model library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kathiresh0506/human-model-library",
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.3",
        "opencv-python>=4.9.0.80",
        "numpy>=1.26.3",
        "Pillow>=10.2.0",
        "scikit-image>=0.22.0",
        "mediapipe>=0.10.9",
        "pyyaml>=6.0.1",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.23.3",
            "httpx>=0.26.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "human-model-api=api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "virtual-tryon",
        "computer-vision",
        "image-processing",
        "fashion-tech",
        "pose-estimation",
        "human-models",
    ],
    project_urls={
        "Bug Reports": "https://github.com/kathiresh0506/human-model-library/issues",
        "Source": "https://github.com/kathiresh0506/human-model-library",
        "Documentation": "https://github.com/kathiresh0506/human-model-library#readme",
    },
)
