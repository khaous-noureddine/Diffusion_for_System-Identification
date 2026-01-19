import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


###############################
# 3) JiT ARCHITECTURE COMPONENTS (from model_jit.py)
###############################
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2D sin-cos positional embedding"""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size=16, in_chans=1, pca_dim=128, embed_dim=384, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
    
    def forward(self, x):
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ControlEmbedder(nn.Module):
    """Embed control signals (u_past and u_curr) into hidden dimension"""
    def __init__(self, past_window, hidden_size):
        super().__init__()
        # Embed past_window + 1 (past + current) control values
        self.mlp = nn.Sequential(
            nn.Linear(past_window + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def forward(self, u_past, u_curr):
        # u_past: (B, past_window, 1), u_curr: (B, 1)
        u_combined = torch.cat([u_past.squeeze(-1), u_curr], dim=1)  # (B, past_window + 1)
        return self.mlp(u_combined)

def scaled_dot_product_attention(query, key, value, dropout_p=0.0):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)
    
    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)
    
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

###############################
# 4) JiT-BASED CONDITIONAL DIFFUSION MODEL
###############################
class JiTFluidDiffusion(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=16,
        in_channels=1,
        past_window=10,
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
        prediction_type="x-pred",
        loss_type="v-loss"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.past_window = past_window
        self.timesteps = diffusion_steps
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        
        # Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.control_embedder = ControlEmbedder(past_window, hidden_size)
        
        # Patch embedding pour la target frame (bruitée)
        self.x_embedder = BottleneckPatchEmbed(
            img_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )
        
        # Patch embedding pour les past frames de conditionnement
        # On va concaténer past_window frames en entrée
        self.cond_embedder = BottleneckPatchEmbed(
            img_size, patch_size, past_window, bottleneck_dim, hidden_size, bias=True
        )
        
        # Positional embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=0.0, proj_drop=0.0)
            for _ in range(depth)
        ])
        
        # Final prediction layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch embeddings
        for embedder in [self.x_embedder, self.cond_embedder]:
            w1 = embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(embedder.proj2.bias, 0)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x):
        """x: (N, num_patches, patch_size**2 * C) -> imgs: (N, C, H, W)"""
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def sample_t(self, n, device):
        """Sample timesteps from logit-normal distribution"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
    
    def forward(self, cond_frames, cond_u_past, cond_u_curr, target_frame):
        """
        Training forward pass with x-prediction and v-loss
        cond_frames: (B, past_window, 1, H, W)
        cond_u_past: (B, past_window, 1)
        cond_u_curr: (B, 1)
        target_frame: (B, 1, H, W)
        """
        B = target_frame.size(0)
        device = target_frame.device
        
        # Sample timestep
        t = self.sample_t(B, device).view(-1, *([1] * (target_frame.ndim - 1)))
        t_flat = t.flatten()
        
        # Add noise to target frame (forward diffusion)
        e = torch.randn_like(target_frame) * self.noise_scale
        z_t = t * target_frame + (1 - t) * e
        # Ground truth velocity
        # v = (target_frame - z_t) / (1 - t).clamp_min(self.t_eps)
        v = target_frame - e
        
        
        # Embed conditioning
        # Reshape cond_frames: (B, past_window, 1, H, W) -> (B, past_window, H, W)
        cond_frames_reshaped = cond_frames.squeeze(2)  # (B, past_window, H, W)
        cond_tokens = self.cond_embedder(cond_frames_reshaped)  # (B, num_patches, hidden_size)
        
        # Embed noisy target
        x_tokens = self.x_embedder(z_t)  # (B, num_patches, hidden_size)
        
        # Combine conditioning and noisy target tokens
        # Simple approach: add them
        tokens = x_tokens + cond_tokens + self.pos_embed
        
        # Time and control embedding
        t_emb = self.t_embedder(t_flat)
        control_emb = self.control_embedder(cond_u_past, cond_u_curr)
        c = t_emb + control_emb
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)
        
        
        # Final layer predicts x directly
        if self.prediction_type == "x-pred":
            x_pred_tokens = self.final_layer(tokens, c)
            x_pred = self.unpatchify(x_pred_tokens)
            
            if self.loss_type == "x-loss":
                loss = F.smooth_l1_loss(x_pred, target_frame)
                
            elif self.loss_type == "v-loss":
                v_pred = (x_pred - z_t) / (1 - t).clamp_min(self.t_eps)
                loss = F.smooth_l1_loss(v_pred, v)
            elif self.loss_type == "e-loss":
                e_pred = (z_t - t*x_pred) / (1 - t).clamp_min(self.t_eps)
                loss = F.smooth_l1_loss(e_pred, e)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        elif self.prediction_type == "v-pred":
            v_pred_tokens = self.final_layer(tokens, c)
            v_pred = self.unpatchify(v_pred_tokens)
            
            if self.loss_type == "x-loss":
                x_pred = (1 - t) * v_pred + z_t
                loss = F.smooth_l1_loss(x_pred, target_frame)
            elif self.loss_type == "v-loss":
                loss = F.smooth_l1_loss(v_pred, v)
            elif self.loss_type == "e-loss":
                e_pred = z_t - t * v_pred
                loss = F.smooth_l1_loss(e_pred, e)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            
        elif self.prediction_type == "e-pred":
            e_pred_tokens = self.final_layer(tokens, c)
            e_pred = self.unpatchify(e_pred_tokens)
            
            if self.loss_type == "x-loss":
                x_pred = (z_t - (1 - t) * e_pred) / t.clamp_min(self.t_eps)
                loss = F.smooth_l1_loss(x_pred, target_frame)
            elif self.loss_type == "v-loss":
                v_pred = (z_t - e_pred) / t.clamp_min(self.t_eps)
                loss = F.smooth_l1_loss(v_pred, v)
            elif self.loss_type == "e-loss":
                loss = F.smooth_l1_loss(e_pred, e)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
                
        
        return loss
        
    @torch.no_grad()
    def sample(self, cond_frames, cond_u_past, cond_u_curr, num_steps=50, method='heun'):
        """
        Generate a frame using reverse diffusion
        """
        B = cond_frames.size(0)
        device = cond_frames.device
        H, W = self.img_size, self.img_size
        
        # Start from noise
        z = self.noise_scale * torch.randn(B, 1, H, W, device=device)
        
        # Timesteps
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        
        # Embed conditioning (constant throughout sampling)
        cond_frames_reshaped = cond_frames.squeeze(2)
        cond_tokens = self.cond_embedder(cond_frames_reshaped)
        cond_tokens = cond_tokens + self.pos_embed
        
        # Control embedding (constant)
        control_emb = self.control_embedder(cond_u_past, cond_u_curr)
        
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            
            if method == 'euler':
                z = self._euler_step(z, t_curr, t_next, cond_tokens, control_emb)
            elif method == 'heun':
                z = self._heun_step(z, t_curr, t_next, cond_tokens, control_emb)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return z
    
    
    def _forward_sample(self, z, t, cond_tokens, control_emb):
        """Single forward pass during sampling - conversion to velocity"""
        B = z.size(0)
        t_scalar = t.expand(B)
        
        # 1. Obtenir les tokens du Transformer
        x_tokens = self.x_embedder(z)
        tokens = x_tokens + cond_tokens
        
        t_emb = self.t_embedder(t_scalar)
        c = t_emb + control_emb
        
        for block in self.blocks:
            tokens = block(tokens, c)
        
        # 2. Prédire et convertir selon le type
        raw_output_tokens = self.final_layer(tokens, c)
        output = self.unpatchify(raw_output_tokens)
        
        t_broadcast = t.view(-1, *([1] * (z.ndim - 1)))

        if self.prediction_type == "x-pred":
            # output is x_pred -> v = (x - z) / (1 - t)
            v_pred = (output - z) / (1.0 - t_broadcast).clamp_min(self.t_eps)
            
        elif self.prediction_type == "v-pred":
            # output is v_pred
            v_pred = output
            
        elif self.prediction_type == "e-pred":
            # output is e_pred -> v = (z - e) / t
            v_pred = (z - output) / t_broadcast.clamp_min(self.t_eps)
            
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
        return v_pred
    
    def _euler_step(self, z, t_curr, t_next, cond_tokens, control_emb):
        v_pred = self._forward_sample(z, t_curr, cond_tokens, control_emb)
        z_next = z + (t_next - t_curr) * v_pred
        return z_next
    
    def _heun_step(self, z, t_curr, t_next, cond_tokens, control_emb):
        # First Euler step
        v_pred_curr = self._forward_sample(z, t_curr, cond_tokens, control_emb)
        z_euler = z + (t_next - t_curr) * v_pred_curr
        
        # Second evaluation
        v_pred_next = self._forward_sample(z_euler, t_next, cond_tokens, control_emb)
        
        # Heun average
        v_pred = 0.5 * (v_pred_curr + v_pred_next)
        z_next = z + (t_next - t_curr) * v_pred
        return z_next