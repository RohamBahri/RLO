"""Research architectures sharing the verified masked-CVRP action interface.

The default attention configuration is parameter-compatible with rlo.model.Policy,
so completed Wave-1 checkpoints can initialize MBPI without conversion.
"""
import math
import torch
from torch import nn
from .model import Policy, Attention


class PointwiseBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                 nn.ReLU(), nn.Linear(2 * width, width))
    def forward(self, x):
        return x + self.net(x)


class ResearchPolicy(Policy):
    def __init__(self, width=128, layers=2, distance_bias=True, logit_clip=10.,
                 glimpse=False, encoder_type="attention", heads=4,
                 dynamic_reencode=False, hierarchy=False, codebook_size=32,
                 auxiliary=False):
        super().__init__(width=width, layers=layers, distance_bias=distance_bias,
                         logit_clip=logit_clip, glimpse=glimpse)
        if encoder_type not in ("attention", "raw", "mlp", "codebook"):
            raise ValueError("unknown encoder_type")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if heads != 4 and glimpse:
            raise ValueError("glimpse uses the base four-head decoder; disable for eight-head study")
        self.encoder_type = encoder_type
        self.heads = heads
        self.dynamic_reencode = bool(dynamic_reencode)
        self.hierarchy = bool(hierarchy)
        self.auxiliary_enabled = bool(auxiliary)
        self.codebook_size = int(codebook_size)
        if heads != 4 and encoder_type == "attention":
            self.encoder = nn.ModuleList([Attention(width, heads=heads) for _ in range(layers)])
        if encoder_type == "mlp":
            self.encoder = nn.ModuleList([PointwiseBlock(width) for _ in range(layers)])
        elif encoder_type in ("raw", "codebook"):
            self.encoder = nn.ModuleList()
        if encoder_type == "codebook":
            if codebook_size < 2:
                raise ValueError("codebook_size must be >= 2")
            self.codes = nn.Parameter(torch.randn(codebook_size, 4) * 0.2)
            self.code_values = nn.Parameter(torch.randn(codebook_size, width) * 0.1)
            self.code_temperature = 0.25
        if self.dynamic_reencode:
            self.dynamic_embed = nn.Linear(6, width)
        if self.hierarchy:
            self.route_gate = nn.Linear(3 * width, 2)
        if self.auxiliary_enabled:
            self.aux_heads = nn.Linear(3 * width, 3)

    def _encode(self, node_features, dist, *, dynamic=False):
        if self.encoder_type == "codebook":
            x = node_features[..., :4]
            assign = torch.softmax(-((x[..., None, :] - self.codes).square().sum(-1))
                                   / self.code_temperature, -1)
            h = self.embed(x) + (assign @ self.code_values)
        else:
            h = self.dynamic_embed(node_features) if dynamic else self.embed(node_features[..., :4])
        if self.encoder_type == "attention":
            for layer in self.encoder:
                h = layer(h, dist, self.distance_bias)
        elif self.encoder_type == "mlp":
            for layer in self.encoder:
                h = layer(h)
        return h

    def encode(self, nodes, dist):
        # The initial static encoder remains available in the dynamic ablation;
        # dynamic re-encoding below recomputes the representation each decision.
        return self._encode(nodes, dist)

    def _representation(self, static, nodes):
        if not self.dynamic_reencode:
            return static
        # All coordinates are in the original Euclidean units, matching encode().
        coords = nodes[..., :2]
        dist = torch.cdist(coords, coords, p=2)
        return self._encode(nodes, dist, dynamic=True)

    def _context(self, static, nodes, globals_):
        rep = self._representation(static, nodes)
        h = rep + self.dynamic(nodes[..., 4:6])
        current = nodes[..., 5].argmax(1)
        here = h[torch.arange(len(h), device=h.device), current]
        return torch.cat([here, h.mean(1), self.global_embed(globals_)], -1)

    def decode(self, static, nodes, global_features, mask):
        # For raw/MLP/codebook and dynamic attention, provide the same valid
        # query-pointer architecture as the common baseline.
        if not self.dynamic_reencode:
            base, val = super().decode(static, nodes, global_features, mask)
        else:
            dynamic_static = self._representation(static, nodes)
            base, val = super().decode(dynamic_static, nodes, global_features, mask)
        if not self.hierarchy:
            return base, val
        context = self._context(static, nodes, global_features)
        high = self.route_gate(context)
        gate_legal = torch.stack([mask[:, 0], mask[:, 1:].any(-1)], -1)
        high = high.masked_fill(~gate_legal, -1e9).log_softmax(-1)
        customer = base[:, 1:].log_softmax(-1)
        joint = torch.cat([high[:, :1], high[:, 1:] + customer], -1)
        return joint.masked_fill(~mask, -1e9), val

    def predict_aux(self, static, nodes, global_features):
        if not self.auxiliary_enabled:
            raise ValueError("auxiliary heads disabled")
        return self.aux_heads(self._context(static, nodes, global_features))

    def configuration(self):
        return dict(width=self.width, layers=len(self.encoder) if self.encoder_type != "raw" else 0,
                    distance_bias=self.distance_bias, logit_clip=self.logit_clip,
                    glimpse=self.glimpse, encoder_type=self.encoder_type, heads=self.heads,
                    dynamic_reencode=self.dynamic_reencode, hierarchy=self.hierarchy,
                    codebook_size=self.codebook_size, auxiliary=self.auxiliary_enabled)


def from_saved(path, device="cpu"):
    """Load trusted research or v2/v3 anchor weights, not optimizer/replay pickle."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("wave3_version") == 1:
        model = ResearchPolicy(**state["architecture"])
    elif state.get("architecture_version") in (2, 3):
        model = ResearchPolicy(**state["architecture"])
    else:
        raise ValueError("unrecognized model checkpoint")
    model.load_state_dict(state["model"])
    return model.to(device).eval()
