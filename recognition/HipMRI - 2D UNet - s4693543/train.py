"""
File: train.py
Author: William Wright
Description: Source code for training, validating, testing, and saving the model. Model is imported from modules.py, and data loader is from dataset.py. Losses and metrics are plotted during training. 
"""

from dataset import get_dataloaders
from modules import BasicUNet, MCDiceLoss

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

def plot_loss(train_losses, val_losses, output_path='loss_over_epochs.png'):
    """
    Plot training and validation loss over epochs.
    (Loss = Cross-Entropy + Dice Loss)
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (Avg. of Cross-Entropy and Dice Loss)')
    plt.title('Training and Validation Loss over Epochs')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Loss plot (Cross-Entropy + Dice) saved to {output_path}")


####### TRAINING AND EVALUATION ########
def train_one_epoch(model, loader, loss_fn, opt, device):
    """
    Run one training epoch
    """
    model.train()
    total_loss = 0

    # for dice coeff calculation
    eps = 1e-6
    inter = torch.zeros(6, device=device)
    p_sum = torch.zeros(6, device=device)
    t_sum = torch.zeros(6, device=device)

    progress_bar = tqdm(loader, desc="Training...", leave=True)

    for batch in progress_bar:
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward pass
        opt.zero_grad(set_to_none=True)
        logits = model(images)
        batch_loss = loss_fn(logits, labels)

        # backprop
        batch_loss.backward()
        opt.step()
        total_loss += batch_loss.item()

        # store dice scores FOR THIS BATCH ONLY
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            for c in range(6):
                pc = (pred == c).float()
                tc = (labels == c).float()
                inter[c] += (pc*tc).sum()
                p_sum[c] += pc.sum()
                t_sum[c] += tc.sum()

    # average stuff across the epoch
    mean_loss = total_loss / max(1, len(loader))
    epoch_dice = ((2*inter + eps) / (p_sum + t_sum + eps)).tolist()

    return mean_loss, epoch_dice

def evaluation(model, loader, loss_fn, device):
    """
    Evaluate the model
    """
    model.eval()
    total_loss = 0

    # for dice coeff calculation
    eps = 1e-6
    inter = torch.zeros(6, device=device)
    p_sum = torch.zeros(6, device=device)
    t_sum = torch.zeros(6, device=device)
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation...", leave=True)

        for batch in progress_bar:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # forward pass
            logits = model(images)
            batch_loss = loss_fn(logits, labels)
            total_loss += batch_loss.item()
            
            pred = torch.argmax(logits, dim=1)
            for c in range(6):
                pc = (pred == c).float()
                tc = (labels == c).float()
                inter[c] += (pc*tc).sum()
                p_sum[c] += pc.sum()
                t_sum[c] += tc.sum()
                
    mean_loss = total_loss / max(1, len(loader))
    epoch_dice = ((2*inter + eps) / (p_sum + t_sum + eps)).tolist()

    return mean_loss, epoch_dice

def main(data_path, num_epochs=50, batch_size=8, learning_rate=0.01, output_dir="outputs"):
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
        num_workers=2,
        categorical_masks=False
    )

    # model
    print("Model created.")
    model = BasicUNet(in_channels=1, num_classes=6, base_features=32).to(device)

    # loss function is CE + Multi-Class Dice
    # class_weights = torch.tensor([0.5, 0.5, 1.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float32)
    # ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = MCDiceLoss()

    def loss_fn(logits, targets):
        return dice_loss(logits, targets)
    
        # return 0.5*ce_loss(logits, targets) + 0.5*dice_loss(logits, targets)

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
        patience=10
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
        print("")
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
    plot_loss(train_loss, val_loss, output_path='outputs/loss_progress.png')

    return model, train_loss, val_loss, train_dice_scores, val_dice_scores

if __name__ == "__main__":
    # data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    project_dir = os.path.dirname(__file__)
    data_path = os.path.join(project_dir, "data")

    #hyperparams
    num_epochs = 50
    batch_size = 8
    learning_rate = 0.0001

    print("Starting training with params:"
          f"\n    Data path: {data_path}"
          f"\n    Num epochs: {num_epochs}"
          f"\n    Batch size: {batch_size}"
          f"\n    Learning rate: {learning_rate}\n")
    
    # train the model with above params
    model, train_loss, val_loss, train_dice_scores, val_dice_scores = main(
        data_path=data_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir='outputs'
    )
