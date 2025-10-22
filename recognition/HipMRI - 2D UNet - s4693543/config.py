"""
File: config.py
Author: William Wright
Description: Configuration settings, hyperparameters, and paths for the project.
"""

# Paths
DATA_ROOT = "data"
IMG_DIR = f"{DATA_ROOT}/semantic_MRs_anon"
LBL_DIR = f"{DATA_ROOT}/semantic_labels_anon"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

# Training 
DEVICE = "cuda" 