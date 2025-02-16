# src/residuals/wave.py
import torch

from src.core.base_residual import ResidualBase
from src.models.wave1d import WavePINN


class Wave1DResidual(ResidualBase):
    def __init__(self, c: float = 1.0):
        self.c = c  # c: Wave speed (default is 1.0).

    def compute(self, model: WavePINN, collocation_points: torch.tensor) -> torch.Tensor:
        """
        Computes the mean squared PDE residual for the 1D wave equation:
            u_tt(x,t) - c^2 * u_xx(x,t) = 0,
        where collocation_points is a tensor of shape (N,2) with columns (x, t).

        Args:
            model: A torch.nn.Module that accepts (x,t) and outputs u(x,t).
            collocation_points: torch.Tensor of shape (N,2).

        Returns:
            torch.Tensor: Mean squared residual.
        """
        c = self.c
        collocation_points = collocation_points.clone().detach().requires_grad_(True)
        u = model(collocation_points)  # shape: (N,1)

        # Compute first derivatives with respect to x and t.
        grad_u = torch.autograd.grad(u, collocation_points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        # Compute second derivatives.
        u_xx = torch.autograd.grad(u_x, collocation_points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_tt = torch.autograd.grad(u_t, collocation_points, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 1:2]

        residual = u_tt - c ** 2 * u_xx
        return torch.mean(residual ** 2)


def wave_pde_residual(model, collocation_points, c: float = 1.0) -> torch.Tensor:
    residual_obj = Wave1DResidual(c)
    return residual_obj.compute(model, collocation_points)
