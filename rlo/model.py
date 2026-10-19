"""Permutation-equivariant routing policy: six node sensors, two global sensors."""

import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.distance_weight = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, dist, use_distance=True):
        batch, nodes, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, nodes, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        if use_distance:
            scores = scores + self.distance_weight * dist[:, None]
        w = scores.softmax(-1)
        x = x + self.out((w @ v).transpose(1, 2).reshape(batch, nodes, width))
        return x + self.ff(self.norm2(x))


class Policy(nn.Module):
    def __init__(self, width=64, layers=2, distance_bias=True):
        super().__init__()
        self.width, self.distance_bias = width, distance_bias
        self.embed = nn.Linear(6, width)
        self.encoder = nn.ModuleList([Attention(width) for _ in range(layers)])
        self.global_embed = nn.Linear(2, width)
        self.query = nn.Linear(3 * width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, nodes, global_features, dist, mask):
        h = self.embed(nodes)
        for layer in self.encoder:
            h = layer(h, dist, self.distance_bias)
        current = nodes[:, :, 5].argmax(dim=1)
        here = h[torch.arange(len(h), device=h.device), current]
        context = torch.cat([here, h.mean(1), self.global_embed(global_features)], dim=-1)
        logits = (self.key(h) @ self.query(context).unsqueeze(-1)).squeeze(-1) / math.sqrt(self.width)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, self.value(context).squeeze(-1)
