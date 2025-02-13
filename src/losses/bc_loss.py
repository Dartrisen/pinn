# src/losses/bc_loss.py
import torch
from torch import nn

from src.losses.base_loss import ConditionBase


class DirichletCondition(ConditionBase):
    """Dirichlet boundary condition: φ(x, ...) = target."""
    def __init__(self, target=0.0):
        self.target = target

    def compute(self, model: nn.Module, points: torch.Tensor) -> torch.Tensor:
        phi = model(points)
        target_tensor = torch.full_like(phi, self.target, dtype=torch.float32, device=points.device)
        return torch.mean((phi - target_tensor) ** 2)


# Function-based interfaces for quick use
def dirichlet_bc_loss(model, bc_points, target=0.0):
    return DirichletCondition(target).compute(model, bc_points)
