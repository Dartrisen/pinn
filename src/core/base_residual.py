# src/core/base_residual.py
from abc import ABC, abstractmethod

import torch

from src.core.generic_pinn import GenericPINNSolver


class ResidualBase(ABC):
    @abstractmethod
    def compute(self, model: GenericPINNSolver, collocation_points: torch.tensor):
        """
        Computes the PDE residual loss.
        Should return a torch.Tensor.
        """
        pass
