"""Read-only adapter for the archived, unmasked PyTorch MVP5 checkpoint."""

import math
import numpy as np
import torch
from torch import nn
from .problem import distances, features


class _Layer(nn.Module):
    def __init__(self, d=128, heads=4):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d, heads, d // heads
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.o_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.dist_bias = nn.Module()
        self.dist_bias.fc1 = nn.Linear(1, 8)
        self.dist_bias.fc2 = nn.Linear(8, 1)

    def forward(self, x, dist):
        b, n, d = x.shape
        h = self.ln1(x)
        q, k, v = [proj(h).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        bias = self.dist_bias.fc2(torch.relu(self.dist_bias.fc1(dist.unsqueeze(-1)))).squeeze(-1)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_head) + bias[:, None], dim=-1)
        x = x + self.o_proj((weights @ v).transpose(1, 2).contiguous().view(b, n, d))
        return x + self.ff(self.ln2(x))


class _MVP5(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(7, 128)
        self.layers = nn.ModuleList([_Layer() for _ in range(2)])
        self.ln_final = nn.LayerNorm(128)
        self.policy_q, self.policy_k = nn.Linear(128, 128), nn.Linear(128, 128)
        self.value_mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, feats, current, dist):
        h = self.embed(feats)
        for layer in self.layers:
            h = layer(h, dist)
        h = self.ln_final(h)
        q = self.policy_q(h[torch.arange(len(h), device=h.device), current])
        logits = (self.policy_k(h) @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(128)
        return logits


class MVP5Policy:
    """Callable instance -> greedy tour, with no implicit feasibility masking."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model = _MVP5().to(self.device).eval()
        # Only load archived, trusted checkpoints: older files contain pickled RNG state.
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"] if "model" in state else state, strict=True)

    @torch.no_grad()
    def __call__(self, instance):
        n = len(instance["demands"])
        visited = np.zeros(n, dtype=bool)
        current, load, tour = 0, 0.0, [0]
        dist = torch.tensor(distances(instance), dtype=torch.float32, device=self.device)[None]
        for _ in range(5 * n):  # same step cap as the archived unmasked environment
            f = torch.tensor(features(instance, visited, current, load), device=self.device)[None]
            logits = self.model(f, torch.tensor([current], device=self.device), dist)
            action = int(logits.argmax(dim=-1).item())
            tour.append(action)
            load = 0.0 if action == 0 else load + float(instance["demands"][action])
            visited[action] = action != 0 or visited[action]
            current = action
            if visited[1:].all() and current == 0:
                break
        return tour
