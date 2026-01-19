import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from jit_model import JiTFluidDiffusion
from jit_engine import train_jit_diffusion, evaluate_autoregressive, create_comparison_video
from jit_data import (split_data, 
                      compute_normalization_stats, 
                      normalize_data_list, 
                      create_jit_sequences,
                      FluidDynamicsDataset)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--prediction_type", required=True, type=str)
parser.add_argument("--loss_type", required=True, type=str)
parser.add_argument("--img_size", required=True, type=int)
parser.add_argument("--n_epochs", required=False, type=int)
parser.add_argument("--n_frames", required=False, type=int)
parser.add_argument("--data_path", required=True, type=str)
parser.add_argument("--patch_size", required=True, type=int)

args = parser.parse_args()

print(args)


# 1. Prepare Data

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

file_path = args.data_path

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            print(os.path.join(root, file))
    exit()

print(f"✓ File found: {file_path}")


# Load data
print("\n=== Loading data ===")
train_list, val_list, test_list = split_data(file_path, args.img_size)
print(f"Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")

# Normalize
print("\n=== Normalization ===")
frame_mean, frame_std, u_mean, u_std = compute_normalization_stats(train_list)
print(f"Frame: μ={frame_mean:.6f}, σ={frame_std:.6f}")
print(f"Control u: μ={u_mean:.6f}, σ={u_std:.6f}")

normalize_data_list(train_list, frame_mean, frame_std, u_mean, u_std)
normalize_data_list(val_list, frame_mean, frame_std, u_mean, u_std)
normalize_data_list(test_list, frame_mean, frame_std, u_mean, u_std)

# Create sequences
print("\n=== Creating sequences ===")
past_window = 10
X_train_frames, X_train_u_past, X_train_u_curr, Y_train = create_jit_sequences(train_list, past_window)
X_val_frames, X_val_u_past, X_val_u_curr, Y_val = create_jit_sequences(val_list, past_window)
print(f"Train sequences: {X_train_frames.shape[0]}, Val sequences: {X_val_frames.shape[0]}")

# DataLoaders
train_dataset = FluidDynamicsDataset(X_train_frames, X_train_u_past, X_train_u_curr, Y_train)
val_dataset = FluidDynamicsDataset(X_val_frames, X_val_u_past, X_val_u_curr, Y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)



















# 2 . Model Training
# Create model
print("\n=== Creating JiT model ===")
model = JiTFluidDiffusion(
    img_size=args.img_size,
    patch_size=args.patch_size,
    in_channels=1,
    past_window=past_window,
    hidden_size=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    bottleneck_dim=64,
    diffusion_steps=50,
    P_mean=-0.8,
    P_std=0.8,
    t_eps=0.05,
    noise_scale=1.0,
    prediction_type=args.prediction_type, # AJOUTÉ
    loss_type=args.loss_type             
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Train
print("\n=== Training JiT Fluid Diffusion ===")
output_path = f"output/{args.prediction_type}_{args.loss_type}_dim-{args.img_size}_patch-{args.patch_size}"
train_jit_diffusion(model, train_loader, val_loader, args.n_epochs, device, output_path, learning_rate=2e-4)


# 3. Evaluation
print("\n=== Evaluation on test set ===")
model.load_state_dict(torch.load(f'{output_path}/best_jit_fluid_model.pth'))

total_mse = 0
for i, test_case in enumerate(test_list):
    print(f"\n📊 Test case {i+1}/{len(test_list)} (amplitude: {test_case['amplitude']})")
    pred_frames, true_frames, mse = evaluate_autoregressive(
        model, test_case, past_window, device, num_frames=args.n_frames, 
        frame_mean=frame_mean, frame_std=frame_std, num_steps=50
    )
    total_mse += mse
    
    # Video
    gt_seq = true_frames.squeeze(1)
    pred_seq = pred_frames.squeeze(1)
    video_path = f"{output_path}/videos/"
    os.makedirs(video_path, exist_ok=True)
    create_comparison_video(gt_seq, pred_seq, test_case['amplitude'], save_path=f"{video_path}/jit_comparison_{test_case['amplitude']}.gif")

avg_test_mse = total_mse / len(test_list)
print(f"\n{'='*60}")
print(f"✅ Average test MSE: {avg_test_mse:.6f}")
print(f"{'='*60}")