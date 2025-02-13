# src/losses/ic_loss.py
import numpy as np
import torch
from torch import nn

from src.losses.base_loss import ConditionBase


class NeumannCondition(ConditionBase):
    """Neumann boundary condition (first derivative constraint): φ'(x) = target."""
    def __init__(self, target: float = 0.0):
        self.target = target

    def compute(self, model: nn.Module, points: torch.Tensor) -> torch.Tensor:
        points = points.clone().detach().requires_grad_(True)
        phi = model(points)
        d_phi = torch.autograd.grad(phi, points, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        target_tensor = torch.full_like(d_phi, self.target, dtype=torch.float32, device=points.device)
        return torch.mean((d_phi - target_tensor) ** 2)


def derivative_ic_loss(model, ic_points, target=1 / np.pi):
    return NeumannCondition(target).compute(model, ic_points)
