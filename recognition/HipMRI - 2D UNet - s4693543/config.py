"""
File: config.py
Author: William Wright
Description: Configuration settings, hyperparameters, and paths for the project.
"""
import os

# Paths
DATA_ROOT = "data"
RANGPUR_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

IMG_DIR_TRAIN = os.path.join(DATA_ROOT, "keras_slices_train")
LBL_DIR_TRAIN = os.path.join(DATA_ROOT, "keras_slices_seg_train")

IMG_DIR_VAL   = os.path.join(DATA_ROOT, "keras_slices_validate")
LBL_DIR_VAL   = os.path.join(DATA_ROOT, "keras_slices_seg_validate")

IMG_DIR_TEST  = os.path.join(DATA_ROOT, "keras_slices_test")
LBL_DIR_TEST  = os.path.join(DATA_ROOT, "keras_slices_seg_test")

# Training 
DEVICE = "cuda" 