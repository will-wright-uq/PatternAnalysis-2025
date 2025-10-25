"""
File: dataset.py
Author: William Wright
Description: Data loader for loading and preprocessing the dataset.
"""
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from config import (
    IMG_DIR_TRAIN, LBL_DIR_TRAIN,
    IMG_DIR_VAL,   LBL_DIR_VAL,
    IMG_DIR_TEST,  LBL_DIR_TEST,
)

### CODE FROM ASSIGNMENT SPECIFICATION ###
import numpy as np
import nibabel as nib
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        
        affines.append(affine)
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images
    
### END OF CODE FROM ASSIGNMENT SPECIFICATION ###

def _sorted_niis(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.nii")))

def _pair_lists(img_dir: str, lbl_dir: str) -> Tuple[List[str], List[str]]:
    """Match image and label files by identical basename."""
    imgs = _sorted_niis(img_dir)
    lbls = _sorted_niis(lbl_dir)
    # map by basename (without extension)
    lbl_map = {os.path.splitext(os.path.basename(p))[0]: p for p in lbls}
    img_keep, lbl_keep = [], []
    for ip in imgs:
        key = os.path.splitext(os.path.basename(ip))[0]
        if key in lbl_map:
            img_keep.append(ip)
            lbl_keep.append(lbl_map[key])
    if len(img_keep) == 0:
        raise RuntimeError(f"No matched image/label pairs in {img_dir} and {lbl_dir}")
    return img_keep, lbl_keep

class HipMRIDataset(Dataset):
    """
    Wraps the spec-provided numpy loaders into a PyTorch Dataset.
    Returns:
      image: float32 tensor (1, H, W)  — per-slice z-scored if normImage=True
      mask:  long tensor   (H, W)      — integer labels [0..C-1]
    """
    def __init__(self, img_paths: List[str], lbl_paths: List[str], normalise: bool = True):
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths
        self.normalise = normalise

        # preload into memory
        self.X = load_data_2D(self.img_paths, normImage=self.normalise, categorical=False, dtype=np.float32)
        self.Y = load_data_2D(self.lbl_paths, normImage=False, categorical=False, dtype=np.float32)

        # Convert to tensors and fix shapes/dtypes
        # X: (N, H, W) -> (N, 1, H, W)
        X_t = torch.from_numpy(self.X).float().unsqueeze(1)
        # Y: (N, H, W) floats -> ints
        Y_t = torch.from_numpy(self.Y).long()

        self.X_t = X_t
        self.Y_t = Y_t

    def __len__(self):
        return self.X_t.shape[0]

    def __getitem__(self, idx):
        return self.X_t[idx], self.Y_t[idx]

def get_datasets():
    train_imgs, train_lbls = _pair_lists(IMG_DIR_TRAIN, LBL_DIR_TRAIN)
    val_imgs,   val_lbls   = _pair_lists(IMG_DIR_VAL,   LBL_DIR_VAL)
    test_imgs,  test_lbls  = _pair_lists(IMG_DIR_TEST,  LBL_DIR_TEST)

    ds_train = HipMRIDataset(train_imgs, train_lbls, normalize=True)
    ds_val   = HipMRIDataset(val_imgs,   val_lbls,   normalize=True)
    ds_test  = HipMRIDataset(test_imgs,  test_lbls,  normalize=True)
    return ds_train, ds_val, ds_test

def get_dataloaders(batch_size: int = 8, num_workers: int = 2, pin_memory: bool = True):
    ds_train, ds_val, ds_test = get_datasets()
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return dl_train, dl_val, dl_test