"""
File: predict.py
Author: William Wright
Description: Example usage of the trained model. Print any results or visualisations where possible. 
"""

import os
import torch

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
    print(f"Best model loaded from epoch: {saved_model['epoch']} with:")
    print(f"    Validation Loss scores: {saved_model['val_loss']}")
    print(f"    Validation Dice scores: {saved_model['val_dice']}")
    
    return model

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

def main():
    # data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    project_dir = os.path.dirname(__file__)
    data_path = os.path.join(project_dir, "data")

    saved_model_path = "outputs/max_dice_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    model = load_saved_model(saved_model_path, device)

    print("\nLoading the test dataset")
    _, _, test_loader = get_dataloaders(data_path, batch_size=8, num_workers=4)
    
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

if __name__ == "__main__":
    main()