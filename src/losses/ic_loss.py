# src/losses/ic_loss.py
import numpy as np
import torch

from src.losses.conditions import NeumannCondition


def derivative_ic_loss(model, ic_points, target=-1 / np.pi):
    return NeumannCondition(target).compute(model, ic_points)


def wave_ic_loss(model, ic_points):
    """
    Enforces the initial conditions for the 1D wave equation:
        u(x,0) = sin(pi*x)   and   u_t(x,0) = 0.
    Args:
        model: Model
        ic_points: Tensor of shape (N_ic,2) with t=0.
    """
    # Enforce u(x,0) = sin(pi*x)
    u_ic = model(ic_points)
    x = ic_points[:, 0:1]
    u_target = torch.sin(np.pi * x)
    loss_u = torch.mean((u_ic - u_target) ** 2)

    # Enforce u_t(x,0) = 0
    ic_points_grad = ic_points.clone().detach().requires_grad_(True)
    u_ic_grad = model(ic_points_grad)
    grad_u = torch.autograd.grad(u_ic_grad, ic_points_grad,
                                 grad_outputs=torch.ones_like(u_ic_grad),
                                 create_graph=True)[0]
    u_t = grad_u[:, 1:2]
    loss_ut = torch.mean(u_t ** 2)

    return loss_u + loss_ut
