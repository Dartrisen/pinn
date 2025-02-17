"""
PINN Example for the 1D Wave Equation

We solve:
    u_tt(x,t) - c**2 * u_xx(x,t) = 0,  for (x,t) in [0,1] x [0,1],
    c = 1,
with boundary conditions:
    u(0,t) = u(1,t) = 0,  for t in [0,1],
and initial conditions:
    u(x,0) = sin(pi*x),
    u_t(x,0) = 0,  for x in [0,1].

The analytic solution is:
    u(x,t) = cos(pi*t)*sin(pi*x).
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.core.generic_pinn import GenericPINNSolver
from src.models.wave1d import WavePINN
from src.residuals.wave1d import wave_pde_residual
from src.losses.bc_loss import dirichlet_bc_loss
from src.losses.ic_loss import wave_ic_loss
from src.utils.data_generation import (
    generate_wave_collocation_points,
    generate_wave_boundary_points,
    generate_wave_initial_points
)

# =============================================================================
# Hyperparameters & Domain Settings
# =============================================================================
DEVICE = torch.device("cpu")
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
NUM_COLL = 2000  # Number of collocation points in the domain
NUM_BC = 100  # Number of boundary points along each boundary edge
NUM_IC = 100  # Number of points for initial conditions

EPOCHS = 1000
LEARNING_RATE = 1e-3
LOSS_WEIGHT_PDE = 1.0
LOSS_WEIGHT_BC = 10.0
LOSS_WEIGHT_IC = 10.0
LOSS_WEIGHTS = {'pde': LOSS_WEIGHT_PDE, 'bc': LOSS_WEIGHT_BC, 'ic': LOSS_WEIGHT_IC}

# =============================================================================
# Generate Training Data (using your existing utils)
# =============================================================================
collocation_points = generate_wave_collocation_points(NUM_COLL, X_MIN, X_MAX, T_MIN, T_MAX).to(DEVICE)
bc_points = generate_wave_boundary_points(NUM_BC, T_MIN, T_MAX).to(DEVICE)
ic_points = generate_wave_initial_points(NUM_IC, X_MIN, X_MAX, t_val=0.0).to(DEVICE)

# =============================================================================
# Instantiate Model & Solver
# =============================================================================
model = WavePINN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Instantiate your generic PINN solver (from your core module) with wave-specific functions.
solver = GenericPINNSolver(
    model=model,
    residual_func=wave_pde_residual,
    bc_loss_func=lambda model, points: dirichlet_bc_loss(model, points, target=0.0),
    ic_loss_func=wave_ic_loss,
    collocation_points=collocation_points,
    bc_points=bc_points,
    ic_points=ic_points,
    device=DEVICE
)

loss_history = solver.train(EPOCHS, optimizer, loss_weights=LOSS_WEIGHTS, log_interval=50)


# =============================================================================
# Evaluation and Plotting
# =============================================================================
def analytic_solution(x, t):
    """Analytic solution: u(x,t)=cos(pi*t)*sin(pi*x)."""
    return np.cos(np.pi * t) * np.sin(np.pi * x)


# Evaluate at a fixed time, e.g., t = 0.75.
t_eval = 0.75
x_plot = np.linspace(X_MIN, X_MAX, 100)[:, None]
t_plot = t_eval * np.ones_like(x_plot)
plot_points = np.hstack((x_plot, t_plot))

model.eval()
with torch.no_grad():
    u_pred = solver.predict(plot_points)

u_exact = analytic_solution(x_plot, t_eval).flatten()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_plot, u_exact, label='Analytic', lw=2)
plt.plot(x_plot, u_pred, '--', label='PINN', lw=2)
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title(f'Solution at t = {t_eval}')
plt.legend()

plt.subplot(1, 2, 2)
epochs_logged = np.arange(0, EPOCHS, 50)
plt.plot(epochs_logged, loss_history, 'o-')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss History')

plt.tight_layout()
plt.show()
