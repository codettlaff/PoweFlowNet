# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:25:37 2026
Author: Casey Dettlaff

Test PowerFlowNetMPN and report per-unit physical-unit errors.
PowerFlowNetMPN is the active architecture.
"""

import os
import time

import numpy as np
import torch

from datasets.PowerFlowData import PowerFlowData
from networks.PowerFlowNetMPN import PowerFlowNetMPN
from utils.custom_loss_functions import Masked_L2_loss

# Configuration
DATA_DIR = 'data'
CASE = '14'
MODEL_PATH = 'models/model_20260902-5811.pt'
SAMPLE_NUMBER = 2000

def get_checkpoint_arg(args, name, default):
    if args is None: return default
    if hasattr(args, name): return getattr(args, name)
    if isinstance(args, dict): return args.get(name, default)

# Main
@torch.no_grad()
def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Case:   {CASE}")
    
    # Load checkpoint first.
    # train.py stores the training arquements in checkpoint['args']
    # Use these to reconstruct the exact architecture where possible.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get('args', None)
    
    EFEATURE_DIM = 2 # Resistance and reactance only.
    
    NFEATURE_DIM = get_checkpoint_arg(
        checkpoint_args,
        'nfeature_dim',
        6)
    
    OUTPUT_DIM = get_checkpoint_arg(
        checkpoint_args,
        'output_dim',
        6)
    
    HIDDEN_DIM = get_checkpoint_arg(
        checkpoint_args,
        'hidden_dim',
        129)
    
    N_GNN_LAYERS = get_checkpoint_arg(
        checkpoint_args,
        'n_gnn_layers',
        4)
    
    K = get_checkpoint_arg(
        checkpoint_args,
        'K',
        3)
    
    DROPOUT_RATE = get_checkpoint_arg(
        checkpoint_args,
        'dropout_rate',
        0.2)
    
    print()
    print("Architecture")
    print("-" * 60)
    print(f"Node feature dimension: {NFEATURE_DIM}")
    print(f"Edge feature dimension: {EFEATURE_DIM}")
    print(f"Output dimension:       {OUTPUT_DIM}")
    print(f"Hidden dimension:       {HIDDEN_DIM}")
    print(f"GNN layers:             {N_GNN_LAYERS}")
    print(f"TAGConv K:              {K}")
    print(f"Dropout:                {DROPOUT_RATE}")
    print("-" * 60)
    
    # Load test dataset
    # normalize=True because the model was trained on normalized PowerFlowData representation.
    
    testset = PowerFlowData(
        root=DATA_DIR,
        case=CASE,
        split=[0.5, 0.2, 0.3],
        task='test',
        normalize=True)
    
    sample_number = min(SAMPLE_NUMBER, len(testset))
    print(f"Test samples available: {len(testset)}")
    print(f"Samples evaluated:      {sample_number}")
    
    node_in_dim, node_out_dim, edge_dim = (testset.get_data_dimensions())
    print()
    print("Dataset dimensions")
    print("-" * 60)
    print(f"data.x:       {node_in_dim}")
    print(f"data.y:       {node_out_dim}")
    print(f"data.edge_attr: {edge_dim}")
    print("-" * 60)
    
    # Construct PowerFlowNetMPN
    model = PowerFlowNetMPN(
        nfeature_dim=NFEATURE_DIM,
        efeature_dim=EFEATURE_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_gnn_layers=N_GNN_LAYERS,
        K=K,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Parameter count
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")
    
    # Evalutation
    eval_loss_fn = Masked_L2_loss(regularize=False)

    total_loss = 0.0
    total_time = 0.0

    preds = []
    targets = []
    masks = []
    
    for i in range(sample_number):
        
        sample = testset[i].to(device)
        
        # Inference timing
        start = time.perf_counter()
        prediction = model(sample)
        inference_time = time.perf_counter() - start
        total_time += inference_time
        
        # Prediction mask
        mask = sample.x[:, -NFEATURE_DIM:]
        
        # Masked L2
        loss = eval_loss_fn(
            prediction,
            sample.y,
            mask)
        
        # Save CPU copies
        preds.append(prediction.cpu())
        targets.append(sample.y.cpu())
        masks.append(mask.cpu())
        
    # Normalized evalutation statistics
    average_loss = total_loss / sample_number
    average_time = total_time / sample_number
    
    print()
    print("=" * 60)
    print("POWERFLOWNETMPN TEST RESULTS")
    print("=" * 60)
    print(f"Masked L2 loss:       {average_loss:.6f}")
    print(f"Average inference:    {average_time:.6f} s")
    print(f"Total inference:      {total_time:.3f} s")
    print("=" * 60)
    
    # Stack results
    preds = torch.stack(preds)
    targets = torch.stack(targets)
    masks = torch.stack(masks)
    
    # Convert preditions and targets back to physical units.
    # PowerFlowData stores the normalization statistics in xymean / xystd.
    mean = testset.xymean[0].cpu()
    std = testset.xystd[0].cpu()
    preds_physical = preds * (std + 1e-7) + mean
    targets_physical = targets * (std + 1e-7) + mean
    
    # Evaluate the four actual power-flow quantities
    # The model / dataset currently retains six outputs:
    # 0 = Voltage magnitude 
    # 1 = Voltage angle
    # 2 = Active power
    # 3 = Reactive power
    # 4 = Gs
    # 5 = Bs
    # PowerFlowNet paper predicts first four.
    
    preds_physical = preds_physical[:, :, :4]
    targets_physical = targets_physical[:, :, :4]
    masks_physical = masks[:, :, :4]
    
    errors = preds_physical - targets_physical
    feature_names = [
        'Voltage Magnitude (p.u.)',
        'Voltage Angle (deg)',
        'Active Power (MW)',
        'Reactive Power (MVAr)']
    
    # Physical-unit error statistics
    print()
    print("=" * 60)
    print("PHYSICAL-UNIT ERROR")
    print("=" * 60)
    
    for feature in range(4):
        
        # Only evaluate quantities for predicted variables.
        valid = masks_physical[:, :, feature] == 1
        feature_errors = errors[:, :, feature][valid]
        
        mae = torch.mean(torch.abs(feature_errors))
        rmse = torch.sqrt(torch.mean(feature_errors ** 2))
        max_error = torch.max(torch.abs(feature_errors))
        
        print()
        print(feature_names[feature])
        print(f"  MAE:       {mae.item():.6f}")
        print(f"  RMSE:      {rmse.item():.6f}")
        print(f"  Max error: {max_error.item():.6f}")
        
if __name__ == '__main__':
    main()
    
        