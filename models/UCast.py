import torch
import torch.nn as nn
import torch.nn.functional as F

# Add registry import
from core.registry import register_model

# ---------------------------
# Latent Query Attention
# ---------------------------
import os
import numpy as np

# ---------------------------
# Latent Query Attention
# ---------------------------
class LatentQueryAttention(nn.Module):
    def __init__(self, in_dim, latent_dim, head_dim, num_heads=1, dropout=0.1):
        super(LatentQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim

        self.latent_queries = nn.Parameter(torch.randn(latent_dim, head_dim * num_heads))
        self.q_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.k_proj = nn.Linear(in_dim, head_dim * num_heads)
        self.v_proj = nn.Linear(in_dim, head_dim * num_heads)
        self.out_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        B, L, _ = x.shape
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.q_proj(queries)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.view(B, self.latent_dim, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, values)
        out = out.transpose(1, 2).contiguous().view(B, self.latent_dim, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        if return_attn:
            return out, attn
        else:
            return out

# ---------------------------
# Downsampling Network
# ---------------------------
class HierarchicalLatentQueryNetwork(nn.Module):
    def __init__(self, orig_channels, time_dim, num_layers, head_dim, reduction_ratio=16, num_heads=1, dropout=0.1):
        super(HierarchicalLatentQueryNetwork, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        latent_dims = []
        current_channels = orig_channels
        for _ in range(num_layers):
            new_channels = max(1, current_channels // reduction_ratio)
            latent_dims.append(new_channels)
            current_channels = new_channels

        self.latent_dims = latent_dims
        current_in_dim = time_dim
        for latent_dim in latent_dims:
            self.layers.append(LatentQueryAttention(current_in_dim, latent_dim, head_dim, num_heads, dropout))
            current_in_dim = head_dim * num_heads
        self.norm_layers = nn.ModuleList([nn.LayerNorm(head_dim * num_heads) for _ in latent_dims])

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        x_base = x.transpose(1, 2)  # [B, C, T]
        skip_list = [x_base]
        x_down = x_base
        attn_maps = [] if return_attn else None
        for layer, norm in zip(self.layers, self.norm_layers):
            if return_attn:
                x_down, attn = layer(x_down, return_attn=True)
                attn_maps.append(attn.detach().cpu())
            else:
                x_down = layer(x_down)
            x_down = norm(x_down)
            skip_list.append(x_down)
        if return_attn:
            return skip_list[-1], skip_list, attn_maps
        else:
            return skip_list[-1], skip_list

# ---------------------------
# Upsampling Attention
# ---------------------------
class UpLatentQueryAttention(nn.Module):
    def __init__(self, q_in_dim, head_dim, num_heads=1, dropout=0.1):
        super(UpLatentQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(q_in_dim, head_dim * num_heads)
        self.k_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.v_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.out_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, skip, x):
        B, Lq, _ = skip.shape
        B, Lk, _ = x.shape
        queries = self.q_proj(skip)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, values)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out

# ---------------------------
# Upsampling Network
# ---------------------------
class HierarchicalUpsamplingNetwork(nn.Module):
    def __init__(self, num_layers, q_in_dim, head_dim, num_heads=1, dropout=0.1):
        super(HierarchicalUpsamplingNetwork, self).__init__()
        self.layers = nn.ModuleList([
            UpLatentQueryAttention(q_in_dim, head_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(q_in_dim) for _ in range(num_layers)
        ])

    def forward(self, x_bottom, skip_list):
        rev = list(reversed(skip_list))
        queries = rev[1:]
        x = x_bottom
        for layer, norm, query in zip(self.layers, self.norms, queries):
            x = norm(layer(query, x) + query)
            # x = layer(query, x)
        return x

# def covariance_loss(x, lambda_cov=1.0):
#     """
#     x: Tensor of shape [B, C, D]
#     Encourage features across time/channel to be decorrelated.
#     """
#     B, C, D = x.shape
#     x_reshaped = x.reshape(B * C, D)
#     x_centered = (x_reshaped - x_reshaped.mean(dim=0, keepdim=True)) / (x_reshaped.std(dim=0, keepdim=True) + 1e-5)
#     cov = (x_centered.T @ x_centered) / (B * C - 1)
#     loss = torch.sum(cov ** 2) - torch.sum(torch.diag(cov) ** 2)
#     loss = loss / (D * (D - 1))
#     return lambda_cov * loss

def covariance_loss(skip_list, lambda_cov=0.1, eps=1e-5):
    """
    skip_list: list of tensors, each tensor is of shape [B, C, D]
    Computes a normalized sum of negative log-determinants of covariance matrices.
    """
    total_loss = 0.0
    num_layers = len(skip_list) - 1  # exclude input
    for x in skip_list[1:]:
        B, C, D = x.shape
        x_reshaped = x.reshape(B * C, D)
        x_centered = (x_reshaped - x_reshaped.mean(dim=0, keepdim=True)) / (
            x_reshaped.std(dim=0, keepdim=True) + eps
        )
        cov = (x_centered.T @ x_centered) / (B * C - 1)
        cov = cov + eps * torch.eye(D, device=x.device, dtype=x.dtype)

        # normalize by dimension (D) to reduce scale variance
        loss = -torch.logdet(cov) / D
        total_loss += loss

    return lambda_cov * (total_loss / num_layers if num_layers > 0 else 0.0)

# ---------------------------
# Final Model
# ---------------------------
@register_model("UCast", paper="U-Cast: Learning Latent Hierarchical Channel Structure for High-Dimensional Time Series Forecasting", year=2024)
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.alpha = configs.alpha

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.pred_len)

        self.channel_reduction_net = HierarchicalLatentQueryNetwork(
            orig_channels=self.enc_in,
            time_dim=self.d_model,
            num_layers=configs.e_layers,
            head_dim=self.d_model,
            reduction_ratio=configs.channel_reduction_ratio,
            num_heads=1,
            dropout=configs.dropout
        )

        self.upsample_net = HierarchicalUpsamplingNetwork(
            num_layers=configs.e_layers,
            q_in_dim=self.d_model,
            head_dim=self.d_model,
            num_heads=1,
            dropout=configs.dropout
        )

        self.predict_layer = nn.Linear(self.d_model, self.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.transpose(1, 2)  # [B, C, T]
        x_enc = self.input_proj(x_enc)  # [B, C, d_model]
        x_enc = x_enc.transpose(1, 2)  # [B, d_model, C]
        x_bottom, skip_list = self.channel_reduction_net(x_enc)
        cov_loss = covariance_loss(skip_list, self.alpha)
        x_bottom = self.predict_layer(x_bottom)
        x_up = self.upsample_net(x_bottom, skip_list)
        dec_out = self.output_proj(x_up + x_enc.transpose(1, 2))  # [B, enc_in, pred_len]
        dec_out = dec_out.transpose(1, 2)  # [B, pred_len, enc_in]

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out, cov_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

# ---------------------------
# Test
# ---------------------------
def test_model():
    batch_size = 4
    seq_length = 336
    pred_length = 96
    input_channels = 1000
    d_model = 128
    x = torch.randn(batch_size, seq_length, input_channels)
    x_mark_enc = x_dec = x_mark_dec = None

    class Config:
        pass

    configs = Config()
    configs.task_name = 'forecast'
    configs.seq_len = seq_length
    configs.pred_len = pred_length
    configs.enc_in = input_channels
    configs.d_model = d_model
    configs.e_layers = 4
    configs.dropout = 0.1

    model = Model(configs)
    output = model(x, x_mark_enc, x, x_mark_dec)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_model()
