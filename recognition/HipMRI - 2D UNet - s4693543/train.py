"""
File: train.py
Author: William Wright
Description: Source code for training, validating, testing, and saving the model. Model is imported from modules.py, and data loader is from dataset.py. Losses and metrics are plotted during training. 
"""
from dataset import get_dataloaders
from modules import BasicUNet, MCDiceLoss, dice_score

import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def train_one_epoch(model, loader, loss_fn, opt, device):
    """
    Run one training epoch
    """
    model.train()
    total_loss = 0

    # dice scores for each class
    per_class_dice = {
        0: [],  # background
        1: [],  # body
        2: [],  # bones
        3: [],  # bladdr
        4: [],  # rectum
        5: [],  # prostate
    }

    progress_bar = tqdm(loader, desc="Training...", leave=False)

    for batch in progress_bar:
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # foreward pass
        opt.zero_grad(set_to_none=True)
        logits = model(images)
        batch_loss = loss_fn(logits, labels)

        # backprop
        batch_loss.backward()
        opt.step()
        total_loss += float(batch_loss)

        # store dice scores
        with torch.no_grad():
            dice_scores = dice_score(logits, labels, num_classes=6)
            for i, val in enumerate(dice_scores):
                per_class_dice[i].append(val)

        progress_bar.set_postfix({
            "dice loss": f"{batch_loss.item():.3f}",
            "prostate class dice": f"{dice_scores[5]:.3f}"
        })

    # average stuff across the epoch
    mean_loss = total_loss / max(1, len(loader))
    mean_dice = [float(np.mean(class_dice)) if class_dice else 0.0
                 for class_dice in per_class_dice]

    return mean_loss, mean_dice

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_dir = os.path.dirname(__file__)
    data_path = os.path.join(project_dir, "data")

    #testing if it runs
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        batch_size=2,
        num_workers=2,
        categorical_masks=False
    )

    # model + loss + optimiser 
    model = BasicUNet(in_channels=1, num_classes=6, base_features=32).to(device)
    loss_fn = MCDiceLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    # run a single epoch just to make sure it works
    print("\nRunning test on one training epoch...")
    train_loss, train_dice = train_one_epoch(model, train_loader, loss_fn, opt, device)

    print(f"\nFinished test epoch:")
    print(f"  Avg loss: {train_loss:.4f}")
    print(f"  Mean Dice per class: {train_dice}")