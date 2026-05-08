"""Setup configuration for the prostate cancer classification package."""
from setuptools import setup, find_packages

setup(
    name="prostate-cancer-ensemble",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Ensemble-based clinical classification model for prostate cancer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prostate-cancer-ensemble-platform",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "scikit-learn>=1.2.2",
        "xgboost>=1.7.6",
        "lightgbm>=4.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "joblib>=1.2.0",
        "openpyxl>=3.1.2",
    ],
    entry_points={
        "console_scripts": [
            "prostate-cancer-train=src.pipelines.training_pipeline:main",
        ],
    },
)
