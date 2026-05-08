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
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.1,<3",
        "pandas>=2.2,<3.1",
        "scikit-learn>=1.7,<1.9",
        "xgboost>=3.0,<4",
        "PyYAML>=6.0.2,<7",
        "matplotlib>=3.9,<3.11",
        "joblib>=1.4,<2",
        "openpyxl>=3.1.5,<4",
    ],
    extras_require={
        "lightgbm": ["lightgbm>=4.6,<5"],
        "viz": ["seaborn>=0.13.2,<0.14"],
    },
    entry_points={
        "console_scripts": [
            "prostate-cancer-train=src.pipelines.training_pipeline:main",
        ],
    },
)
