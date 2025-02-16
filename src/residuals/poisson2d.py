# src/residuals/poisson2d.py
import numpy as np
import torch

from src.core.base_residual import ResidualBase
from src.models.poisson2d import Poisson2DPINN


def f_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Source term for the 2D Poisson equation.
    Here: f(x,y) = sin(pi*x)*sin(pi*y).
    """
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


class Poisson2DResidual(ResidualBase):
    def compute(self, model: Poisson2DPINN, collocation_points: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared residual for the 2D Poisson equation:
            u_xx + u_yy + f(x,y) = 0,
        where f(x,y) = sin(pi*x)*sin(pi*y).
        """
        # Make sure the input has gradients.
        collocation_points = collocation_points.clone().detach().requires_grad_(True)
        u = model(collocation_points)  # Expect shape (N,1)

        # Separate the coordinates: x and y.
        x = collocation_points[:, 0:1]
        y = collocation_points[:, 1:2]

        # First derivatives.
        grad_u = torch.autograd.grad(u, collocation_points,
                                     grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]

        # Second derivatives.
        u_xx = torch.autograd.grad(u_x, collocation_points,
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, collocation_points,
                                   grad_outputs=torch.ones_like(u_y),
                                   create_graph=True)[0][:, 1:2]

        residual = u_xx + u_yy + f_2d(x, y)
        return torch.mean(residual ** 2)


def poisson2d_residual(model: Poisson2DPINN, collocation_points: torch.Tensor):
    """
    Convenience function to compute the 2D Poisson residual.
    """
    residual_obj = Poisson2DResidual()
    return residual_obj.compute(model, collocation_points)
