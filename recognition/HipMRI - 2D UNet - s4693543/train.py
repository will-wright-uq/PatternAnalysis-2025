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

import matplotlib.pyplot as plt

####### PLOTTING HELPERS ########
def plot_dice(train_dice, val_dice, output_path='dice_over_epochs.png'):
    """
    Plot per-class Dice coefficient over epochs (train vs val).
    Creates a 3x2 subplot layout for the 6 classes.
    """
    # epoch range based on training history
    epochs = range(1, len(train_dice) + 1)

    class_labels = [
        "Background",
        "Body",
        "Bones",
        "Bladder",
        "Rectum",
        "Prostate"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        train_vals = [epoch_dice[i] for epoch_dice in train_dice]
        val_vals = [epoch_dice[i] for epoch_dice in val_dice]

        ax.plot(epochs, train_vals, 'b-', label='Train')
        ax.plot(epochs, val_vals, 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.set_title(f"{class_labels[i]} Class: Dice Coefficient over Epochs")
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Per-class Dice plot saved to {output_path}")


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

    progress_bar = tqdm(loader, desc="Training...", leave=True)

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
        total_loss += batch_loss.item()

        # store dice scores FOR THIS BATCH ONLY
        with torch.no_grad():
            dice_scores = dice_score(logits, labels, num_classes=6)
            for i, val in enumerate(dice_scores):
                per_class_dice[i].append(val)

        progress_bar.set_postfix({"current batch prostate dice": f"{dice_scores[5]:.3f}"})

    # average stuff across the epoch
    mean_loss = total_loss / max(1, len(loader))
    mean_dice = [
        float(np.nanmean(v)) if v else 0.0
        for _, v in sorted(per_class_dice.items(), key=lambda kv: kv[0])
    ]

    return mean_loss, mean_dice

def evaluation(model, loader, loss_fn, device):
    """
    Evaluate the model
    """
    model.eval()
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
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation...", leave=True)

        for batch in progress_bar:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # foreward pass
            logits = model(images)
            batch_loss = loss_fn(logits, labels)
            total_loss += batch_loss.item()
            
            # store dice scores FOR THIS BATCH ONLY
            dice_scores = dice_score(logits, labels, num_classes=6)
            for i, score in enumerate(dice_scores):
                per_class_dice[i].append(score)
            
            progress_bar.set_postfix({"current batch prostate dice": f"{dice_scores[5]:.3f}"})
    
    mean_loss = total_loss / max(1, len(loader))
    mean_dice = [
        float(np.nanmean(v)) if v else 0.0
        for _, v in sorted(per_class_dice.items(), key=lambda kv: kv[0])
    ]

    return mean_loss, mean_dice

def main(data_path, num_epochs=50, batch_size=8, learning_rate=0.001, output_dir="outputs"):
    """
    Main training loop
    """
    #create folder to store models/outputs
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Getting data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=4,
        categorical_masks=False
    )

    # model
    print("Model created.")
    model = BasicUNet(in_channels=1, num_classes=6, base_features=32).to(device)

    # loss function is CE + Multi-Class Dice
    class_weights = torch.tensor([0.5, 0.5, 1.0, 1.5, 2.0, 2.0], device=device, dtype=torch.float32)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = MCDiceLoss()

    def loss_fn(logits, targets):
        return ce_loss(logits, targets) + dice_loss(logits, targets)

    # optimiser + scheduler
    optimiser = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 
        mode="max", 
        factor=0.25, 
        patience=5
    )

    # segmentation label mapping
    labels_map = {
        0: "Background",
        1: "Body",
        2: "Bones",
        3: "Bladder",
        4: "Rectum",
        5: "Prostate",
    }

    train_loss, val_loss = [], []
    train_dice_scores, val_dice_scores = [], []
    best_val_min_dice = 0

    print(f"----------{num_epochs} epochs----------")

    for epoch in range(1, num_epochs + 1):
        print("\n")
        print("*" * 40)
        print(f"Progress: Epoch {epoch}/{num_epochs}")
        print("*" * 40)

        tr_loss, tr_dice = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        va_loss, va_dice = evaluation(model, val_loader, loss_fn, device)

        # step scheduler on the hardest class (min over classes)
        min_val_dice = min(va_dice)
        scheduler.step(min_val_dice)

        train_loss.append(tr_loss)
        val_loss.append(va_loss)
        train_dice_scores.append(tr_dice)
        val_dice_scores.append(va_dice)

        # pretty print with labels
        def fmt_dice(dlist):
            return ", ".join(f"{labels_map[i]}: {dlist[i]:.4f}" for i in range(6))
        
        print(f"\nTRAIN RESULTS EPOCH {epoch}")
        print(f"    CE + Multi-Class Dice Loss (Train): {tr_loss:.4f}")
        print(f"    Class-Based Dice Coefficients (Train): {fmt_dice(tr_dice)}")
        print(f"VAL RESULTS EPOCH {epoch}")
        print(f"    CE + Multi-Class Dice Loss (Val): {va_loss:.4f}")
        print(f"    Class-Based Dice Coefficients (Val): {fmt_dice(va_dice)}")

        # checkpointing best model
        if min_val_dice > best_val_min_dice:
            best_val_min_dice = min_val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_dice": va_dice,
                    "val_loss": va_loss,
                },
                os.path.join(output_dir, "max_dice_model.pth"),
            )
            print(f"\nBest model saved with min dice: {best_val_min_dice:.4f}")

    print("\nCompleted model training; best model saved as max_dice_model.pth with dice:", round(best_val_min_dice, 4))

    plot_dice(train_dice_scores, val_dice_scores, output_path='outputs/dice_progress.png')


    return model, train_loss, val_loss, train_dice_scores, val_dice_scores

if __name__ == "__main__":
    # data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    project_dir = os.path.dirname(__file__)
    data_path = os.path.join(project_dir, "data")

    #hyperparams
    num_epochs = 10
    batch_size = 8
    learning_rate = 0.01
    
    # train the model with above params
    model, train_loss, val_loss, train_dice_scores, val_dice_scores = main(
        data_path=data_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir='outputs'
    )
