import math

import torch
import torch.nn as nn

def get_activation(activation_type):
    activations = {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "lrelu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
    }
    return activations[activation_type]()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)                # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                            # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # buffer: moves with .to(device), saved in state_dict, not trained
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        x = self.dropout(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_activation=True, activation_type="gelu"):
        super().__init__()

        linear_list = [nn.Linear(in_dim, out_dim)]

        if use_activation:
            linear_list.append(
                get_activation(
                    activation_type=activation_type))

        self.linear_layer = nn.Sequential(*linear_list)

    def forward(self, x):
        x = self.linear_layer(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1, activation_type="gelu"):
        super().__init__()
        self.mlp_block = nn.Sequential(
            LinearBlock(embedding_dim, hidden_dim, True, activation_type),
            nn.Dropout(dropout),
            LinearBlock(hidden_dim, embedding_dim, False),
            nn.Dropout(dropout))

    def forward(self, x):  # (N, Seq, dim)
        x = self.mlp_block(x)
        return x


# Multi-Head Attention Block for Encoder and Decoder.
class SelfAttentionBlock(nn.Module):
    def __init__(self, heads=8, embedding_dim=512, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(embedding_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True)

    def forward(self, x):  # (N, Seq, D)
        x_norm = self.norm_attn(x)
        x, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        return x         # (N, Seq, D)


# Multi-Head Attention Block for Decoder only.
class CrossAttentionBlock(nn.Module):
    def __init__(self, heads=8, embedding_dim=512, context_dim=None, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(context_dim or embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            kdim=context_dim,      # None -> embedding_dim
            vdim=context_dim,
            dropout=dropout,
            batch_first=True)

    def forward(self, x, context):
        # x:       (N, Seq_q, D)   -> queries
        # context: (N, Seq_kv, Dc) -> keys and values
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        x, _ = self.cross_attn(
            q, kv, kv,
            need_weights=False)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, heads=8, embedding_dim=512, hidden_dim=2048,
                 dropout=0.0, activation_type="gelu"):
        super().__init__()
        self.attn = SelfAttentionBlock(heads, embedding_dim, dropout)
        self.norm_mlp = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, hidden_dim, dropout, activation_type)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, heads=8, embedding_dim=512, hidden_dim=2048,
                 context_dim=None, dropout=0.0, activation_type="gelu"):
        super().__init__()
        self.self_attn = SelfAttentionBlock(heads, embedding_dim, dropout)
        self.cross_attn = CrossAttentionBlock(heads, embedding_dim, context_dim, dropout)
        self.norm_mlp = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, hidden_dim, dropout, activation_type)

    def forward(self, x, x_enc):
        x = x + self.self_attn(x)
        x = x + self.cross_attn(x, x_enc)
        x = x + self.mlp(self.norm_mlp(x))
        return x
