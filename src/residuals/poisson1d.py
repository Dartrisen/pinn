# src/residuals/poisson1d.py
import numpy as np
import torch

from src.core.base_residual import ResidualBase
from src.models.poisson1d import PoissonPINN


def f_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Source term for the 2D Poisson equation.
    Here: f(x) = sin(pi*x).
    """
    return torch.sin(np.pi * x)


class Poisson1DResidual(ResidualBase):
    def compute(self, model: PoissonPINN, x_coll: torch.Tensor) -> torch.Tensor:
        """
            Computes the mean squared PDE residual for the Poisson equation:
                φ''(x) - f(x) = 0.

            Uses automatic differentiation.

            Args:
                model (PoissonPINN): The PINN model.
                x_coll (torch.Tensor): Collocation points.

            Returns:
                torch.Tensor: Mean squared PDE residual.
            """
        x_coll = x_coll.clone().detach().requires_grad_(True)
        phi = model(x_coll)
        d_phi = torch.autograd.grad(phi, x_coll, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        d2_phi = torch.autograd.grad(d_phi, x_coll, grad_outputs=torch.ones_like(d_phi), create_graph=True)[0]
        residual = d2_phi - f_1d(x_coll)
        return torch.mean(residual ** 2)


# A convenient function interface (to pass into GenericPINNSolver)
def poisson1d_residual(model, collocation_points):
    residual_obj = Poisson1DResidual()
    return residual_obj.compute(model, collocation_points)
