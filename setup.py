#!/usr/bin/env python3
"""
Setup script for MLXVectorDB - High-Performance Vector Database for Apple Silicon
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    return []

# Core requirements
install_requires = [
    "mlx>=0.25.2",
    "numpy>=1.24.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "aiohttp>=3.8.0",
    "requests>=2.31.0",
    "psutil>=5.9.0",
    "structlog>=23.1.0",
    "rich>=13.0.0",
    "orjson>=3.9.0",
    "asyncio-throttle>=1.0.2",
]

# Development requirements
dev_requires = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.6.0",
    "pre-commit>=3.4.0",
]

# Performance/Monitoring requirements
monitoring_requires = [
    "prometheus-client>=0.17.0",
    "memory-profiler>=0.61.0",
    "py-spy>=0.3.14",
]

# Full requirements (dev + monitoring)
all_requires = dev_requires + monitoring_requires

setup(
    name="mlx-vector-db",
    version="0.1.0",
    author="Theseus-AT",
    author_email="your-email@domain.com",  # Optional: Ersetze mit deiner E-Mail
    description="High-Performance Vector Database optimized for Apple Silicon with MLX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Theseus-AT/mlx-vector-db",
    project_urls={
        "Bug Tracker": "https://github.com/Theseus-AT/mlx-vector-db/issues",
        "Documentation": "https://github.com/Theseus-AT/mlx-vector-db#readme",
        "Source Code": "https://github.com/Theseus-AT/mlx-vector-db",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml", "*.json"],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "monitoring": monitoring_requires,
        "all": all_requires,
    },
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "mlx-vector-db=main:main",  # Assuming main.py has a main() function
            "mlxvdb=main:main",  # Short alias
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "vector-database",
        "mlx",
        "apple-silicon",
        "machine-learning",
        "similarity-search",
        "embeddings",
        "rag",
        "retrieval-augmented-generation",
        "fastapi",
        "high-performance",
    ],
    
    # Platform restriction (Apple Silicon optimized)
    platforms=["darwin"],  # macOS only
    
    # License
    license="Apache License 2.0",
    
    # Additional metadata
    zip_safe=False,  # For better compatibility
)
