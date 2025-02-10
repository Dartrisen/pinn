#!/usr/bin/env python3
"""
Example driver for solving the Poisson equation using a generic PINN framework.
Test Problem:
    Find φ(x) such that
        φ''(x) + sin(πx) = 0,  for x ∈ [0,1],
    with Dirichlet boundary conditions:
        φ(0) = φ(1) = 0.
Analytic solution:
    φ(x) = (1/π²)*sin(πx)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.generic_pinn import GenericPINNSolver
from src.poisson_pinn import PoissonPINN

# =============================================================================
# Constants & Hyperparameters
# =============================================================================
L = 1.0  # Domain: x in [0, 1]
N_COLL = 2000  # Number of collocation points (for PDE residual)
N_BC = 2  # Number of boundary points (x = 0 and x = 1)
EPOCHS = 1000  # Number of training epochs
LEARNING_RATE = 1e-3  # Optimizer learning rate
LOSS_WEIGHTS = {'pde': 1.0, 'bc': 10.0, 'ic': 10.0}  # Loss weighting

# =============================================================================
# Data Generation
# =============================================================================
# Collocation points for PDE residual (converted to a torch tensor)
x_coll = np.linspace(0, L, N_COLL).reshape(-1, 1)
x_coll_tensor = torch.tensor(x_coll, dtype=torch.float32)

# Boundary points (x = 0 and x = L) as a torch tensor
x_bc = np.array([[0.0], [L]])
x_bc_tensor = torch.tensor(x_bc, dtype=torch.float32)


# =============================================================================
# PDE-Specific Functions
# =============================================================================
def rho(x: torch.Tensor) -> torch.Tensor:
    """
    Analytic charge density function: ρ(x) = sin(πx)

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: sin(πx)
    """
    return torch.sin(np.pi * x)


def poisson_pde_residual(model: PoissonPINN, x_coll: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared PDE residual for the Poisson equation:
        φ''(x) + ρ(x) = 0.

    Uses automatic differentiation.

    Args:
        model (PoissonPINN): The PINN model.
        x_coll (torch.Tensor): Collocation points.

    Returns:
        torch.Tensor: Mean squared PDE residual.
    """
    # Ensure x_coll requires gradients
    x_coll = x_coll.clone().detach().requires_grad_(True)
    phi = model(x_coll)
    # First derivative: dφ/dx
    d_phi = torch.autograd.grad(phi, x_coll,
                                grad_outputs=torch.ones_like(phi),
                                create_graph=True)[0]
    # Second derivative: d²φ/dx²
    d2_phi = torch.autograd.grad(d_phi, x_coll,
                                 grad_outputs=torch.ones_like(d_phi),
                                 create_graph=True)[0]
    residual = d2_phi + rho(x_coll)
    return torch.mean(residual ** 2)


def poisson_bc_loss(model: PoissonPINN, x_bc: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error for the Dirichlet boundary conditions:
        φ(0) = φ(L) = 0.

    Args:
        model (PoissonPINN): The PINN model.
        x_bc (torch.Tensor): Boundary points.

    Returns:
        torch.Tensor: Mean squared BC loss.
    """
    phi_bc = model(x_bc)
    return torch.mean(phi_bc ** 2)


def poisson_ic_loss(model: PoissonPINN, ic_points: torch.Tensor) -> torch.Tensor:
    """
    Enforces a derivative constraint at x = 0 (as an initial condition):
        φ'(0) should equal 1/π.

    Args:
        model (PoissonPINN): The PINN model.
        ic_points (torch.Tensor): Points where the derivative is enforced.

    Returns:
        torch.Tensor: Mean squared derivative error.
    """
    ic_points = ic_points.clone().detach().requires_grad_(True)
    phi_ic = model(ic_points)
    grad_phi_ic = torch.autograd.grad(phi_ic, ic_points,
                                      grad_outputs=torch.ones_like(phi_ic),
                                      create_graph=True)[0]
    target = torch.tensor([[1 / np.pi]], dtype=torch.float32, device=ic_points.device)
    return torch.mean((grad_phi_ic - target) ** 2)


# =============================================================================
# Main Routine
# =============================================================================
def main() -> None:
    """
    Main routine to configure, train, and evaluate the PINN for the Poisson equation.
    """
    # Optionally use the derivative (initial) constraint at x = 0
    use_deriv_constraint = True
    ic_loss_func = poisson_ic_loss if use_deriv_constraint else None
    ic_points_tensor = torch.tensor([[0.0]], dtype=torch.float32) if use_deriv_constraint else None

    # Initialize the model and create the PINN solver instance
    model = PoissonPINN()
    solver = GenericPINNSolver(
        model=model,
        pde_residual_func=poisson_pde_residual,
        bc_loss_func=poisson_bc_loss,
        ic_loss_func=ic_loss_func,
        collocation_points=x_coll_tensor,
        bc_points=x_bc_tensor,
        ic_points=ic_points_tensor,
        device=torch.device('cpu')
    )

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the PINN model
    loss_history = solver.train(EPOCHS, optimizer, LOSS_WEIGHTS, log_interval=50)

    # Evaluate the trained model on a plotting grid
    x_plot = np.linspace(0, L, 100).reshape(-1, 1)
    phi_pred = solver.predict(x_plot)
    phi_analytic = (1 / np.pi ** 2) * np.sin(np.pi * x_plot)

    # Plot the solution comparison and training loss history
    plt.figure(figsize=(12, 5))

    # Plot analytic vs. PINN solution
    plt.subplot(1, 2, 1)
    plt.plot(x_plot, phi_analytic, label='Analytic', lw=2)
    plt.plot(x_plot, phi_pred, '--', label='PINN', lw=2)
    plt.xlabel('x')
    plt.ylabel('φ(x)')
    plt.title('Solution Comparison')
    plt.legend()

    # Plot training loss history
    plt.subplot(1, 2, 2)
    epochs_logged = np.arange(0, EPOCHS, 50)
    plt.plot(epochs_logged, loss_history, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
