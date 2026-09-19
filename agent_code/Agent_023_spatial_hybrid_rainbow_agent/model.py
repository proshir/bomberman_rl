"""Spatial dueling quantile network and portable checkpoint helpers."""

from pathlib import Path
import torch
from torch import nn
from .config import ACTION_FEATURES, GLOBAL_FEATURES, N_ACTIONS, QUANTILES, RESIDUAL_BLOCKS, TRUNK_WIDTH

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width))
    def forward(self, x): return torch.nn.functional.silu(x + self.layers(x))

class SpatialHybridRainbow(nn.Module):
    """Dueling quantiles, with a shared state embedding plus action features."""
    def __init__(self, width=TRUNK_WIDTH, blocks=RESIDUAL_BLOCKS, quantiles=QUANTILES):
        super().__init__(); self.width, self.blocks, self.quantiles = width, blocks, quantiles
        self.stem = nn.Sequential(nn.Conv2d(20, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU())
        self.residual = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.state = nn.Sequential(nn.Linear(width + GLOBAL_FEATURES, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.value = nn.Linear(256, quantiles)
        self.advantage = nn.Sequential(nn.Linear(256 + ACTION_FEATURES, 128), nn.SiLU(), nn.Linear(128, quantiles))
    def forward(self, spatial, global_values, actions):
        trunk = self.residual(self.stem(spatial)).mean(dim=(-1, -2))
        state = self.state(torch.cat((trunk, global_values), dim=1)); value = self.value(state).unsqueeze(1)
        action_state = state[:, None, :].expand(-1, N_ACTIONS, -1)
        advantage = self.advantage(torch.cat((action_state, actions), dim=-1))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    def q_values(self, spatial, global_values, actions): return self(spatial, global_values, actions).mean(dim=-1)

def load_checkpoint(path, device="cpu"):
    data = torch.load(Path(path), map_location=device, weights_only=False)
    if data.get("format") != "spatial_hybrid_rainbow_v1": raise ValueError("not an Agent_023 checkpoint")
    model = SpatialHybridRainbow(**data["architecture"]); model.load_state_dict(data["model"]); return model, data

def save_checkpoint(path, model, **metadata):
    payload = {"format": "spatial_hybrid_rainbow_v1", "architecture": {"width": model.width, "blocks": model.blocks, "quantiles": model.quantiles}, "model": model.state_dict(), **metadata}
    Path(path).parent.mkdir(parents=True, exist_ok=True); torch.save(payload, Path(path))
