import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.core.generic_pinn import GenericPINNSolver
from src.losses.bc_loss import dirichlet_bc_loss
from src.models.poisson1d import PoissonPINN
from src.residuals.poisson1d import poisson1d_residual
from src.utils.data_generation import (
    generate_1d_collocation_points,
    generate_1d_boundary_points,
)

# =============================================================================
# Hyperparameters & Domain Settings
# =============================================================================
DEVICE = torch.device("cpu")
X_MIN, X_MAX = 0.0, 1.0
NUM_COLL = 1000

EPOCHS = 1000
LEARNING_RATE = 1e-3
LOSS_WEIGHT_PDE = 1.0
LOSS_WEIGHT_BC = 1.0
LOSS_WEIGHT_IC = 1.0
LOSS_WEIGHTS = {"pde": LOSS_WEIGHT_PDE, "bc": LOSS_WEIGHT_BC, "ic": LOSS_WEIGHT_IC}


def main() -> None:
    # =============================================================================
    # Generate Training Data (using your existing utils)
    # =============================================================================
    x_coll_tensor = generate_1d_collocation_points(X_MIN, X_MAX, NUM_COLL).to(DEVICE)
    x_bc_tensor = generate_1d_boundary_points(X_MIN, X_MAX).to(DEVICE)

    # Optionally, we define an initial condition loss.
    # For 1D Poisson we might enforce a derivative condition at 0.
    use_deriv_constraint = False
    ic_loss_func = None
    ic_points_tensor = None
    if use_deriv_constraint:
        from src.losses.ic_loss import derivative_ic_loss
        ic_loss_func = derivative_ic_loss
        ic_points_tensor = torch.tensor([[0.0]], dtype=torch.float32)

    # =============================================================================
    # Instantiate Model & Solver
    # =============================================================================
    model = PoissonPINN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Instantiate your generic PINN solver (from your core module) with wave-specific functions.
    solver = GenericPINNSolver(
        model=model,
        residual_func=poisson1d_residual,
        bc_loss_func=dirichlet_bc_loss,
        ic_loss_func=ic_loss_func,
        collocation_points=x_coll_tensor,
        bc_points=x_bc_tensor,
        ic_points=ic_points_tensor,
        device=DEVICE,
    )

    loss_history = solver.train(EPOCHS, optimizer, LOSS_WEIGHTS, log_interval=50)

    # =============================================================================
    # Evaluation and Plotting
    # =============================================================================
    x_plot = np.linspace(X_MIN, X_MAX, 100).reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        phi_pred = solver.predict(x_plot)

    phi_analytic = -(1 / np.pi ** 2) * np.sin(np.pi * x_plot)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_plot, phi_analytic, label="Analytic", lw=2)
    plt.plot(x_plot, phi_pred, "--", label="PINN", lw=2)
    plt.xlabel("x")
    plt.ylabel("φ(x)")
    plt.title("Solution Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    epochs_logged = np.arange(0, EPOCHS, 50)
    plt.plot(epochs_logged, loss_history, "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
