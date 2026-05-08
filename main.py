#!/usr/bin/env python
"""
Main entry point for the Prostate Cancer Classification Ensemble Model.
Run this script to execute the complete training pipeline.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set working directory
os.chdir(project_root)

from src.pipelines.training_pipeline import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)
