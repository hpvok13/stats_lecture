import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Transformer
from utils import get_activation_class


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        activation="gelu",
        bias=True,
        do_ln=True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if do_ln else nn.Identity()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias)
        self.activation = get_activation_class(activation)()
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(
        self, d_model, max_len=5000, dropout=0.1, bias=True, activation="gelu"
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

        self.ff = FeedForward(
            d_model,
            d_model * 2,
            d_model,
            bias=bias,
            activation=activation,
            do_ln=False,
        )

    def forward(self, x):
        x = x + self.ff(self.pe[: x.shape[0], ...])
        return self.dropout(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        max_len,
        activation,
        dropout,
        layer_norm_eps,
        bias,
    ):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.register_buffer(
            "mask", Transformer.generate_square_subsequent_mask(max_len)
        )
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=False,
        )
        self.feed_forward = FeedForward(
            d_model, ff_dim, d_model, activation, bias, True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        q, k, v = self.qkv(x).chunk(3, -1)
        mask = self.mask[: x.shape[0], : x.shape[0]]
        x, _ = self.attention(q, k, v, need_weights=False, attn_mask=mask)
        x += res
        res = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x += res
        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=512,
        max_len=5000,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
        batch_first=False,
        bias=True,
    ):
        super().__init__()
        self.batch_first = batch_first

        self.proj_in = nn.Linear(input_dim, d_model, bias=bias)

        self.pos_embed = PositionalEmbedding(
            d_model, max_len, dropout, bias, activation
        )

        self.attn = [
            AttentionBlock(
                d_model,
                num_heads,
                ff_dim,
                max_len,
                activation,
                dropout,
                layer_norm_eps,
                bias,
            )
            for _ in range(num_layers)
        ]
        self.attn = nn.Sequential(*self.attn)

        self.proj_out = nn.Linear(d_model, input_dim, bias=bias)

    def forward(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.proj_in(x)
        x = self.pos_embed(x)
        x = self.attn(x)
        x = self.proj_out(x)

        if self.batch_first:
            x = x.transpose(0, 1)

        return x
