"""
File: dataset.py
Author: William Wright
Description: Data loader for loading and preprocessing the dataset.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


# Helper functions from the spec - slightly modified

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    To convert the single channel labels to a one-hot encoded tensor w/ shape (H, W, K+1)
    """
    channels = np.unique(arr)
    max_c = int(channels.max())
    res = np.zeros(arr.shape + (max_c + 1,), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_nifti_2d(path: str,
                  normalise: bool = False,
                  img_dtype=np.float32) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load 2D nifti file and convert to NumPy array
    """
    nifti = nib.load(path)
    arr = nifti.get_fdata(caching='unchanged')
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]

    # z-score normalisation (optional)
    if normalise:
        std = arr.std()
        arr = (arr - arr.mean()) / (std if std > 0 else 1.0)

    return arr.astype(img_dtype), None

### end of given functions from spec ###


class HipMRIDataset(Dataset):
    """
    A class to help load the HipMRI dataset
    """
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform=None,
                 normalise: bool = True,
                 categorical_masks: bool = False,
                 target_size: Optional[Tuple[int, int]] = (256, 128),
                 label_dtype=np.uint8):
        """
        Args:
            data_path: Base path to keras_slices_data folder (can be local or Rangpur path)
            split: 'train', 'val', or 'test' as per the data files
            transform: Optional callable to apply augmentation/pre-processing
            normalise: Z-score normalisation for the images
            categorical_masks: If True, masks are returned as one-hot; (C, H, W)
                               If False, masks are class indices (H, W)
            target_size: If not None, resize (H, W) to this size
            label_dtype: dtype used for masks before conversion to torch tensors
        """
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.normalise = normalise
        self.categorical_masks = categorical_masks
        self.target_size = target_size
        self.label_dtype = label_dtype

        # paths to each train/val/test image and label
        images_paths = {
            'train': 'keras_slices_train',
            'val'  : 'keras_slices_validate',
            'test' : 'keras_slices_test'
        }

        segs_paths = {
            'train': 'keras_slices_seg_train',
            'val'  : 'keras_slices_seg_validate',
            'test' : 'keras_slices_seg_test'
        }

        self.image_dir = os.path.join(data_path, images_paths[split])
        self.seg_dir = os.path.join(data_path, segs_paths[split])

        # find all the files ending wtih .nii.gz format (all of them)
        self.img_files = sorted(glob.glob(os.path.join(self.image_dir, '*.nii.gz')))

        # handle edge case...
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No .nii.gz images found in {self.image_dir}")

        # print number of samples for given split - for debugging
        print(f"[HipMRIDataset] - [Split: {split}] - [Samples: {len(self.img_files)}]")

    def __len__(self):
        return len(self.img_files)

    def _paired_paths(self, image_path: str) -> Tuple[str, str]:
        """
        Helper function to give an image file path and get the corresponding seg file path
        """
        img_filename = os.path.basename(image_path)

        # case_XXX.nii.gz  -> seg_XXX.nii.gz
        mask_filename = img_filename.replace('case_', 'seg_')

        return image_path, os.path.join(self.seg_dir, mask_filename)

    def __getitem__(self, idx: int):
        """
        This gets a single sample -> e.g HipMRIDataset[0] gets the first sample
        """
        image_path = self.img_files[idx]
        image_path, seg_path = self._paired_paths(image_path)

        img_np, _ = load_nifti_2d(image_path, normalise=self.normalise, img_dtype=np.float32)

        # ensure float32 and add channel dimension: (H, W) -> (1, H, W)
        img_np = np.expand_dims(img_np.astype(np.float32), axis=0)

        label_nifti = nib.load(seg_path)
        mask_np = label_nifti.get_fdata(caching='unchanged')

        # strip extra dim
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]  

        mask_np = np.rint(mask_np).astype(self.label_dtype)

        # One-hot encode
        if self.categorical_masks:
            mask_oh = to_channels(mask_np, dtype=self.label_dtype)  # (H, W, C) 
            # Convert to (C, H, W) for PyTorch
            mask_tensor = torch.from_numpy(mask_oh).permute(2, 0, 1).float()
        else:
            mask_tensor = torch.from_numpy(mask_np).long()

        img_tensor = torch.from_numpy(img_np).float()  # (1, H, W)

        # resizing helper block
        if self.target_size is not None:
            th, tw = self.target_size
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(th, tw),
                                       mode='bilinear', align_corners=False).squeeze(0)
            if self.categorical_masks:
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(th, tw),
                                            mode='nearest').squeeze(0)
            else:
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                            size=(th, tw), mode='nearest').squeeze(0).squeeze(0).long()

        # any transforms
        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        # return torch tensors
        return img_tensor, mask_tensor


def get_dataloaders(data_path: str,
                    batch_size: int = 8,
                    num_workers: int = 2,
                    categorical_masks: bool = False,
                    target_size: Optional[Tuple[int, int]] = (256, 128)):
    """
    Instantiate classes (data loaders) for each split; train/val/test
    """
    train_ds = HipMRIDataset(data_path, split='train',
                             normalise=True,
                             categorical_masks=categorical_masks,
                             target_size=target_size)
    val_ds = HipMRIDataset(data_path, split='val',
                           normalise=True,
                           categorical_masks=categorical_masks,
                           target_size=target_size)
    test_ds = HipMRIDataset(data_path, split='test',
                            normalise=True,
                            categorical_masks=categorical_masks,
                            target_size=target_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader