import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


###############################
# 5) TRAINING LOOP
###############################
def train_jit_diffusion(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-4, best_model_path='models/best_jit_fluid_model.pth'):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for cond_frames, cond_u_past, cond_u_curr, target_frame in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            cond_frames = cond_frames.to(device)
            cond_u_past = cond_u_past.to(device)
            cond_u_curr = cond_u_curr.to(device)
            target_frame = target_frame.to(device)
            
            optimizer.zero_grad()
            loss = model(cond_frames, cond_u_past, cond_u_curr, target_frame)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond_frames, cond_u_past, cond_u_curr, target_frame in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
            ):
                cond_frames = cond_frames.to(device)
                cond_u_past = cond_u_past.to(device)
                cond_u_curr = cond_u_curr.to(device)
                target_frame = target_frame.to(device)
                
                loss = model(cond_frames, cond_u_past, cond_u_curr, target_frame)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.7f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! Val Loss: {best_val_loss:.6f}")
    
    print("\n✅ Training complete!")

###############################
# 6) AUTOREGRESSIVE EVALUATION
###############################
@torch.no_grad()
def evaluate_autoregressive(
    model,
    data_case,
    past_window,
    device,
    num_frames,
    frame_mean,
    frame_std,
    num_steps=50,
    delta_t=1.0
):
    model.eval()

    frames = data_case['frames']  # (T, 1, H, W)
    u = data_case['u']
    T = frames.shape[0]

    # initial history: x(t-past_window) ... x(t)
    history_frames = frames[:past_window].copy()

    pred_frames = []
    mse_list = []

    for i in range(num_frames):
        t_idx = past_window + i
        if t_idx >= T:
            break

        # conditioning
        cond_frames = history_frames[-past_window:]
        cond_u_past = u[t_idx - past_window:t_idx].reshape(-1, 1)
        cond_u_curr = np.array([u[t_idx]], dtype=np.float32)

        # to torch
        cond_frames_tensor = torch.tensor(
            cond_frames, dtype=torch.float32, device=device
        ).unsqueeze(0)
        past_u_tensor = torch.tensor(
            cond_u_past, dtype=torch.float32, device=device
        ).unsqueeze(0)
        curr_u_tensor = torch.tensor(
            cond_u_curr, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # ---- DIFFUSION: sample x_dot ----
        x_dot = model.sample(
            cond_frames_tensor,
            past_u_tensor,
            curr_u_tensor,
            num_steps=num_steps,
            method="heun",
        )

        # torch -> numpy
        x_dot_np = x_dot.squeeze(0).cpu().numpy()  # (1, H, W)

        # ---- SOLVER (Euler) ----
        x_curr = history_frames[-1]                # (1, H, W)
        x_next = x_curr + delta_t * x_dot_np       # (1, H, W)


        # store
        pred_frames.append(x_next)

        # MSE
        true_frame = frames[t_idx]
        mse = np.mean((x_next - true_frame) ** 2)
        mse_list.append(mse)

        # update history
        history_frames = np.concatenate(
            [history_frames, x_next[np.newaxis]],
            axis=0
        )
        history_frames = history_frames[-past_window:]

    pred_frames = np.array(pred_frames)
    true_frames = frames[past_window:past_window + len(pred_frames)]

    # denormalize
    pred_frames_denorm = pred_frames * frame_std + frame_mean
    true_frames_denorm = true_frames * frame_std + frame_mean

    avg_mse = float(np.mean(mse_list))
    print(f"Average MSE for amplitude {data_case['amplitude']}: {avg_mse:.6f}")

    return pred_frames_denorm, true_frames_denorm, avg_mse

###############################
# 7) VISUALIZATION
###############################
def create_comparison_video(gt_frames, pred_frames, amplitude, save_path):
    T = gt_frames.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Amplitude: {amplitude}")
    ax1, ax2 = axes
    ax1.set_title("Ground Truth")
    ax2.set_title("Predicted (JiT)")
    
    vmin = min(gt_frames.min(), pred_frames.min())
    vmax = max(gt_frames.max(), pred_frames.max())
    
    sns.heatmap(gt_frames[0], cmap="magma", vmin=vmin, vmax=vmax, center=0, ax=ax1, cbar=False, square=True)
    sns.heatmap(pred_frames[0], cmap="magma", vmin=vmin, vmax=vmax, center=0, ax=ax2, cbar=False, square=True)
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        sns.heatmap(gt_frames[frame], cmap="magma", vmin=vmin, vmax=vmax, center=0, ax=ax1, cbar=False, square=True)
        sns.heatmap(pred_frames[frame], cmap="magma", vmin=vmin, vmax=vmax, center=0, ax=ax2, cbar=False, square=True)
        ax1.set_title(f"Ground Truth (Frame {frame+1}/{T})")
        ax2.set_title(f"Predicted JiT (Frame {frame+1}/{T})")
    
    ani = animation.FuncAnimation(fig, update, frames=T, interval=200)
    ani.save(save_path, writer="ffmpeg", fps=5, dpi=200)
    print(f"Video saved: {save_path}")
    return save_path