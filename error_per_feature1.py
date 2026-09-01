# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:14:16 2026

@author: codett
"""

import os
import time
import numpy as np
import torch

from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MPN
from utils.custom_loss_functions import Masked_L2_loss

# Configuration

DATA_DIR = './data/'
CASE = '14'
MODEL_PATH = 'models/model_20260831-532.pt'

# These MUST match the arguements used during training
NFEATURE_DIM = 6
EFEATURE_DIM = 5
OUTPUT_DIM = 6
HIDDEN_DIM = 129
N_GNN_LAYERS = 4
K = 3
DROPOUT_RATE = 0.2
SAMPLE_NUMBER = 2000

# Main

@torch.no_grad()
def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    testset = PowerFlowData(
        root=DATA_DIR,
        case=CASE,
        split=[.5,.2,.3],
        task='test')
    sample_number = min(SAMPLE_NUMBER, len(testset))
    
    # Construct model
    model = MPN(
        nfeature_dim=NFEATURE_DIM,
        efeature_dim=EFEATURE_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_gnn_layers=N_GNN_LAYERS,
        K=K,
        dropout_rate=DROPOUT_RATE).to(device)
    
    # Load trained weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    eval_loss_fn = Masked_L2_loss(regularize=False)
    total_loss = 0.0
    total_time = 0.0 
    preds, targets, masks = [], [], []
    
    for i in range(sample_number):
        
        sample = testset[i].to(device)
        start = time.time()
        prediction = model(sample)
        total_time += time.time() - start
        total_loss += eval_loss_fn(
            prediction,
            sample.y,
            sample.x[:, 10:]).item()
        
        preds.append(prediction.cpu())
        targets.append(sample.y.cpu())
        masks.append(sample.x[:, 10:].cpu())
        
    # Average normalized Masked L2
    print()
    print("=" * 60)
    print(f"Masked L2 loss:       {total_loss / sample_number:.6f}")
    print(f"Average inference:    {total_time / sample_number:.6f} s")
    print("=" * 60)
    
    # Convert predictions back to physical units
    mean = testset.xymean[0].cpu()
    std = testset.xystd[0].cpu()
    preds = torch.stack(preds) * std + mean
    targets = torch.stack(targets) * std + mean
    masks = torch.stack(masks)
    
    # Onlt the first four outputs:
    # 0 : Voltage magnitude
    # 1 : Voltage angle
    # 2 : Active power
    # 3 : Reactive power
    
    preds = preds[:, :, :4]
    targets = targets[:, :, :4]
    masks = masks[:, :, :4]
    errors = preds - targets
    
    # Physical-unit error statistics
    feature_names = [
        "Voltage Magnitude (p.u.)",
        "Voltage Angle (deg)",
        "Active Power (MW)",
        "Reactive Power (MVAr)"]
    
    print()
    print("=" * 60)
    print("PHYSICAL-UNIT ERROR")
    print("=" * 60)
    
    for feature in range(4):
        
        # Only evaluate locations where this quantity was masked.
        valid = masks[:, :, feature] == 1
        feature_errors = errors[:, :, feature][valid]
        
        mae = torch.mean(torch.abs(feature_errors))
        rmse = torch.sqrt(torch.mean(feature_errors ** 2))
        max_error = torch.max(torch.abs(feature_errors))
        
        print()
        print(feature_names[feature])
        print(f"  MAE:       {mae.item():.6f}")
        print(f"  RMSE:      {rmse.item():.6f}")
        print(f"  Max error: {max_error.item():.6f}")
        
    print()
    print('=' * 60)
    
if __name__ == '__main__':
    main()
    
    