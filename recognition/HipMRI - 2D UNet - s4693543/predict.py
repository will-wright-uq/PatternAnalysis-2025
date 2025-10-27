"""
File: predict.py
Author: William Wright
Description: Example usage of the trained model. Print any results or visualisations where possible. 
"""

import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

from tqdm import tqdm
from dataset import get_dataloaders
from modules import BasicUNet

def load_saved_model(saved_model_path, device):
    """
    Helper function to load a saved model from disk.
    """
    model = BasicUNet(in_channels=1, num_classes=6, base_features=32)

    saved_model = torch.load(saved_model_path, 
                             map_location=device, 
                             weights_only=False)
    
    model.load_state_dict(saved_model['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"\nLoading saved model from: {saved_model_path}")
    print(f"Best model loaded from epoch #{saved_model['epoch']} with:")
    print(f"    Train Loss scores: {saved_model['train_loss']}")
    print(f"    Train Dice scores: {saved_model['train_dice']}")
    print(f"    Validation Loss scores: {saved_model['val_loss']}")
    print(f"    Validation Dice scores: {saved_model['val_dice']}")
    
    return model, saved_model['train_dice'], saved_model['val_dice']

def evaluate(model, loader, device):
    """
    Evaluate the model on the test set and save predictions.
    """
    model.eval()

    # for dice coeff calculation
    eps = 1e-6
    inter = torch.zeros(6, device=device)
    p_sum = torch.zeros(6, device=device)
    t_sum = torch.zeros(6, device=device)

    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Testing...", leave=True)

        for batch in progress_bar:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # forward pass
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            test_preds.append(preds.cpu())
            test_targets.append(labels.cpu())

            for c in range(6):
                pc = (preds == c).float()
                tc = (labels == c).float()
                inter[c] += (pc*tc).sum()
                p_sum[c] += pc.sum()
                t_sum[c] += tc.sum()

    # dice calculation
    denom = p_sum + t_sum
    present = (t_sum > 0)
    dice = torch.zeros(6, device=device)
    dice[present] = (2 * inter[present] + eps) / (denom[present] + eps)
    epoch_dice = [float(dice[c]) if present[c].item() else None for c in range(6)]

    return epoch_dice, test_preds, test_targets

def plot_dice_by_split(train_dice, val_dice, test_dice, output_path):
    """
    Plot train/val/test dice score as a bar chart separated by class.
        x-axis: classes (with 3 bars per class for train/val/test)
        y-axis: dice score
    """
    classes = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, train_dice, width, label='Train', color='#4c72b0')
    ax.bar(x, val_dice, width, label='Validation', color='#55a868')
    ax.bar(x + width, test_dice, width, label='Test', color='#c44e52')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_title('Dice Score by Dataset Split and Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')

    for i, scores in enumerate([train_dice, val_dice, test_dice]):
        offset = (i - 1) * width
        for j, score in enumerate(scores):
            ax.text(j + offset, score + 0.015, f"{score:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved dice by split plot: {output_path}")

    return
    
def plot_seg_predictions(model=None, loader=None, device=None,
                         output_path="outputs/predictions.png",
                         num_samples=6,
                         require_all_classes_per_sample=False,
                         desired_classes=(0,1,2,3,4,5),
                         max_batches_to_scan=200):
    """
    Plot segmentation predictions against the ground truth.
    """
    #setup/config for plot
    class_labels = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']
    colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(class_labels)+0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    #helper function to compute mean dice for a single sample
    def mean_dice_ignore_absent(pred, target, n_classes=6, eps=1e-6):
        md = []
        for c in range(n_classes):
            p = (pred == c).astype(np.float32)
            t = (target == c).astype(np.float32)
            if t.sum() == 0:
                continue
            inter = (p * t).sum()
            md.append((2*inter + eps) / (p.sum() + t.sum() + eps))
        return float(np.mean(md)) if md else float('nan')

    samples = []
    model.eval()
    scanned_batches = 0
    with torch.no_grad():
        for images, labels in loader:
            scanned_batches += 1
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for b in range(images.size(0)):
                img  = images[b, 0].detach().cpu().numpy()
                targ = labels[b].detach().cpu().numpy().astype(np.int32)
                pred = preds[b].detach().cpu().numpy().astype(np.int32)

                #plot only samples containing all desired classes
                if require_all_classes_per_sample:
                    present = set(np.unique(targ).tolist())
                    if not set(desired_classes).issubset(present):
                        continue  # skip this sample
                
                #z-score normalisation for display
                disp = img
                if np.std(disp) > 0:
                    disp = (disp - disp.mean()) / disp.std()
                disp = (disp - disp.min()) / max(1e-6, disp.max() - disp.min())

                md = mean_dice_ignore_absent(pred, targ, n_classes=6)
                samples.append((disp, targ, pred, md))

                if len(samples) >= num_samples:
                    break

            if len(samples) >= num_samples:
                break
            if scanned_batches >= max_batches_to_scan:
                break
    
    # debugging
    if not samples:
        print("No samples to plot (none matched the class constraint).")
        return None
    if len(samples) < num_samples and require_all_classes_per_sample:
        print(f"Warning: only found {len(samples)} sample(s) containing all classes "
              f"after scanning {scanned_batches} batch(es).")

    n = len(samples)

    # setup plotting
    fig, axes = plt.subplots(
        nrows=n, ncols=3,
        figsize=(5, 3*n),
        gridspec_kw={'wspace': 0.005, 'hspace': 0.2}
    )

    plt.subplots_adjust(bottom=0.18, left=0.03, right=0.97, top=0.93, wspace=0.005, hspace=0.2)

    if n == 1:
        axes = np.array([axes])

    col_titles = ["Input", "Prediction", "Ground Truth"]
    for r, (disp, targ, pred, md) in enumerate(samples):
        # col 0: input image 
        ax0 = axes[r, 0]
        ax0.imshow(disp, cmap='gray')
        ax0.axis('off')
        if r == 0:
            ax0.set_title(col_titles[0], fontsize=13, pad=6)

        # col 1: prediction
        ax1 = axes[r, 1]
        ax1.imshow(pred, cmap=cmap, norm=norm, interpolation='nearest')
        ax1.axis('off')
        if r == 0:
            ax1.set_title(col_titles[1], fontsize=13, pad=6)
        
        #put dice score below predictions
        if np.isfinite(md):
            ax1.text(0.5, -0.05, f"Mean Dice: {md:.3f}",
                    transform=ax1.transAxes, ha='center', va='top',
                    fontsize=7, clip_on=False)

        # col 2: ground truth
        ax2 = axes[r, 2]
        ax2.imshow(targ, cmap=cmap, norm=norm, interpolation='nearest')
        ax2.axis('off')
        if r == 0:
            ax2.set_title(col_titles[2], fontsize=13, pad=6)

    #legend fixed on bottom
    patches = [Patch(facecolor=colors[i], edgecolor='none', label=f"{i}: {class_labels[i]}")
            for i in range(len(class_labels))]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.06))

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved segmentation visualisation: {output_path}")

    return output_path

def main():
    # data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    project_dir = os.path.dirname(__file__)
    data_path = os.path.join(project_dir, "data")

    saved_model_path = "outputs/max_dice_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    model, train_dice, val_dice = load_saved_model(saved_model_path, device)

    print("\nLoading the test dataset")
    _, _, test_loader = get_dataloaders(data_path, batch_size=1, num_workers=2)
    
    print("\nEvaluating performance on the test set...")
    mean_dice, test_preds, test_targets = evaluate(model, test_loader, device)
    
    #output test results
    print("\nTest Set Dice Scores by Class:")
    class_labels = [
        "Background",
        "Body",
        "Bones",
        "Bladder",
        "Rectum",
        "Prostate"
    ]
    for i, name in enumerate(class_labels):
        print(f"    {name}: Dice Score = {mean_dice[i]:.4f}")

    #call plotting fnuctions
    plot_dice_by_split(train_dice, val_dice, mean_dice, output_path="outputs/dice_by_split.png")
    plot_seg_predictions(
        model, test_loader, device,
        output_path="outputs/predictions.png",
        num_samples=2,
        require_all_classes_per_sample=True,
        max_batches_to_scan=500
    )

if __name__ == "__main__":
    main()