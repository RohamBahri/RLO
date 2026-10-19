"""Permutation-equivariant CVRP policy: static attention encoder, dynamic decoder."""

import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by the number of attention heads")
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
    def __init__(self, width=64, layers=2, distance_bias=True, logit_clip=10.0):
        super().__init__()
        self.width, self.distance_bias, self.logit_clip = width, distance_bias, logit_clip
        self.embed = nn.Linear(4, width)
        self.encoder = nn.ModuleList([Attention(width) for _ in range(layers)])
        self.dynamic = nn.Linear(2, width)
        self.global_embed = nn.Linear(2, width)
        self.query = nn.Linear(3 * width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(), nn.Linear(width, 1))

    def encode(self, nodes, dist):
        """Run attention just once per instance; node features 0:4 never change."""
        h = self.embed(nodes[..., :4])
        for layer in self.encoder:
            h = layer(h, dist, self.distance_bias)
        return h

    def decode(self, static, nodes, global_features, mask):
        """Update only lightweight dynamic state at each routing decision."""
        h = static + self.dynamic(nodes[..., 4:6])
        current = nodes[..., 5].argmax(dim=1)
        here = h[torch.arange(len(h), device=h.device), current]
        context = torch.cat([here, h.mean(1), self.global_embed(global_features)], dim=-1)
        scores = (self.key(h) @ self.query(context).unsqueeze(-1)).squeeze(-1) / math.sqrt(self.width)
        if self.logit_clip is not None:
            scores = self.logit_clip * torch.tanh(scores / self.logit_clip)
        return scores.masked_fill(~mask, -1e9), self.value(context).squeeze(-1)

    def forward(self, nodes, global_features, dist, mask):
        return self.decode(self.encode(nodes, dist), nodes, global_features, mask)
