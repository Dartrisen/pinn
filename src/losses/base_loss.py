# src/losses/base_loss.py
import torch
from torch import nn


class ConditionBase:
    """Base class for boundary and initial conditions."""
    def compute(self, model: nn.Module, points: torch.Tensor):
        raise NotImplementedError("Subclasses must implement compute()")
