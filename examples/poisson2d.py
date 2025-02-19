import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.core.generic_pinn import GenericPINNSolver
from src.models.poisson2d import Poisson2DPINN
from src.residuals.poisson2d import poisson2d_residual
from src.losses.bc_loss import dirichlet_bc_loss
from src.utils.data_generation import (
    generate_2d_collocation_points,
    generate_2d_boundary_points,
)

# =============================================================================
# Hyperparameters & Domain Settings
# =============================================================================
DEVICE = torch.device("cpu")
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.0, 1.0, 0.0, 1.0  # Domain: [0,1] x [0,1]
NUM_COLL = 2000

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
    x_coll_tensor = generate_2d_collocation_points(X_MIN, X_MAX, Y_MIN, Y_MAX, NUM_COLL).to(DEVICE)
    x_bc_tensor = generate_2d_boundary_points(X_MIN, X_MAX, Y_MIN, Y_MAX, num_points_per_edge=100).to(DEVICE)

    use_deriv_constraint = False
    ic_loss_func = None
    ic_points_tensor = None
    if use_deriv_constraint:
        from src.losses.ic_loss import derivative_ic_loss
        ic_loss_func = derivative_ic_loss
        ic_points_tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    # =============================================================================
    # Instantiate Model & Solver
    # =============================================================================
    model = Poisson2DPINN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    solver = GenericPINNSolver(
        model=model,
        residual_func=poisson2d_residual,
        bc_loss_func=dirichlet_bc_loss,
        ic_loss_func=ic_loss_func,
        collocation_points=x_coll_tensor,
        bc_points=x_bc_tensor,
        ic_points=ic_points_tensor,
        device=DEVICE
    )

    loss_history = solver.train(EPOCHS, optimizer, LOSS_WEIGHTS, log_interval=100)

    # =============================================================================
    # Evaluation and Plotting
    # =============================================================================
    n_points = 50
    x_vals = np.linspace(X_MIN, X_MAX, n_points)
    y_vals = np.linspace(Y_MIN, Y_MAX, n_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.vstack([X.flatten(), Y.flatten()]).T

    model.eval()
    with torch.no_grad():
        u_pred = solver.predict(grid_points).reshape(n_points, n_points)

    U_analytic = 1 / (2 * np.pi**2) * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Plot analytic and predicted solutions
    plt.figure(figsize=(10, 8))

    # Plot 2D contours
    plt.subplot(2, 2, 1)
    cp1 = plt.contourf(X, Y, U_analytic, levels=50, cmap='viridis')
    plt.colorbar(cp1)
    plt.title("Analytic Solution")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2, 2, 2)
    cp2 = plt.contourf(X, Y, u_pred, levels=50, cmap='viridis')
    plt.colorbar(cp2)
    plt.title("PINN Predicted Solution")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot 1D slices
    # Central x-slice (fixed y)
    mid_y_idx = n_points // 2
    plt.subplot(2, 2, 3)
    plt.plot(x_vals, U_analytic[mid_y_idx, :], 'b-', label='Analytic')
    plt.plot(x_vals, u_pred[mid_y_idx, :], 'r--', label='PINN')
    plt.title(f"Solution along x-axis (y = {y_vals[mid_y_idx]:.1f})")
    plt.xlabel("x")
    plt.ylabel("u(x,y)")
    plt.legend()
    plt.grid(True)

    # Central y-slice (fixed x)
    mid_x_idx = n_points // 2
    plt.subplot(2, 2, 4)
    plt.plot(y_vals, U_analytic[:, mid_x_idx], 'b-', label='Analytic')
    plt.plot(y_vals, u_pred[:, mid_x_idx], 'r--', label='PINN')
    plt.title(f"Solution along y-axis (x = {x_vals[mid_x_idx]:.1f})")
    plt.xlabel("y")
    plt.ylabel("u(x,y)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
