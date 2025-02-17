#!/usr/bin/env python3
"""
PINN Example for the 1D Wave Equation

We solve:
    u_tt(x,t) - u_xx(x,t) = 0,  for (x,t) in [0,1] x [0,1],
with boundary conditions:
    u(0,t) = u(1,t) = 0,  t in [0,1],
and initial conditions:
    u(x,0) = sin(pi*x),
    u_t(x,0) = 0,  x in [0,1].

The analytic solution is:
    u(x,t) = cos(pi*t)*sin(pi*x).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =============================================================================
# Hyperparameters and Domain Settings
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


# =============================================================================
# Neural Network Model for the Wave Equation
# =============================================================================
class WavePINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_hidden_layers=2):
        """
        A simple fully connected neural network.
        Input: (x,t), Output: u(x,t)
        """
        super(WavePINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Helper Functions to Generate Training Data
# =============================================================================
def generate_collocation_points(num_points):
    """Generate collocation points in the domain [X_MIN, X_MAX] x [T_MIN, T_MAX]."""
    x = np.random.uniform(X_MIN, X_MAX, num_points)[:, None]
    t = np.random.uniform(T_MIN, T_MAX, num_points)[:, None]
    return torch.tensor(np.hstack((x, t)), dtype=torch.float32)


def generate_boundary_points(num_points):
    """
    Generate boundary points along x = X_MIN and x = X_MAX for t in [T_MIN, T_MAX].
    Returns a tensor of shape (2*num_points, 2).
    """
    t_vals = np.linspace(T_MIN, T_MAX, num_points)[:, None]
    x_left = X_MIN * np.ones_like(t_vals)
    x_right = X_MAX * np.ones_like(t_vals)
    bc_left = np.hstack((x_left, t_vals))
    bc_right = np.hstack((x_right, t_vals))
    bc = np.vstack((bc_left, bc_right))
    return torch.tensor(bc, dtype=torch.float32)


def generate_initial_points(num_points):
    """
    Generate initial condition points at t = T_MIN for x in [X_MIN, X_MAX].
    Returns a tensor of shape (num_points, 2).
    """
    x_vals = np.linspace(X_MIN, X_MAX, num_points)[:, None]
    t_vals = T_MIN * np.ones_like(x_vals)
    ic = np.hstack((x_vals, t_vals))
    return torch.tensor(ic, dtype=torch.float32)


# =============================================================================
# Analytical Solution
# =============================================================================
def analytic_solution(x, t):
    """
    Analytic solution: u(x,t) = cos(pi*t)*sin(pi*x)
    """
    return np.cos(np.pi * t) * np.sin(np.pi * x)


# =============================================================================
# Loss Functions
# =============================================================================
def wave_pde_residual(model, collocation_points):
    """
    Computes the mean squared PDE residual for the wave equation:
      u_tt - u_xx = 0.

    collocation_points: Tensor of shape (N,2) with columns (x,t).
    """
    # Enable gradients for collocation points
    collocation_points = collocation_points.clone().detach().requires_grad_(True)
    u = model(collocation_points)  # shape: (N,1)

    # Compute gradients with respect to x and t
    grad_u = torch.autograd.grad(u, collocation_points,
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    # Second derivatives: u_xx and u_tt
    u_xx = torch.autograd.grad(u_x, collocation_points,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, collocation_points,
                               grad_outputs=torch.ones_like(u_t),
                               create_graph=True)[0][:, 1:2]

    residual = u_tt - u_xx
    return torch.mean(residual ** 2)


def wave_bc_loss(model, bc_points):
    """
    Enforce Dirichlet boundary conditions: u(0,t)=u(1,t)=0.
    bc_points: Tensor of shape (N_bc,2).
    """
    u_bc = model(bc_points)
    return torch.mean(u_bc ** 2)


def wave_ic_loss(model, ic_points):
    """
    Enforce initial conditions:
      u(x,0) = sin(pi*x) and u_t(x,0) = 0.

    ic_points: Tensor of shape (N_ic,2) with t=0.
    """
    # u(x,0)
    u_ic = model(ic_points)
    x = ic_points[:, 0:1]
    # Target for u(x,0)
    u_target = torch.sin(np.pi * x)
    loss_u = torch.mean((u_ic - u_target) ** 2)

    # Now enforce initial velocity u_t(x,0)=0.
    ic_points = ic_points.clone().detach().requires_grad_(True)
    u_ic = model(ic_points)
    grad_u = torch.autograd.grad(u_ic, ic_points,
                                 grad_outputs=torch.ones_like(u_ic),
                                 create_graph=True)[0]
    u_t = grad_u[:, 1:2]
    loss_ut = torch.mean(u_t ** 2)

    return loss_u + loss_ut


# =============================================================================
# Training Loop
# =============================================================================
def train_wave_pinn(model, collocation_points, bc_points, ic_points, epochs, optimizer):
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_pde = wave_pde_residual(model, collocation_points)
        loss_bc = wave_bc_loss(model, bc_points)
        loss_ic = wave_ic_loss(model, ic_points)
        loss = (LOSS_WEIGHT_PDE * loss_pde +
                LOSS_WEIGHT_BC * loss_bc +
                LOSS_WEIGHT_IC * loss_ic)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            loss_history.append(loss.item())
            print(f"Epoch {epoch:04d}: Total Loss = {loss.item():.6f}, "
                  f"PDE Loss = {loss_pde.item():.6f}, BC Loss = {loss_bc.item():.6f}, "
                  f"IC Loss = {loss_ic.item():.6f}")
    return loss_history


# =============================================================================
# Main Routine
# =============================================================================
def main():
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Generate training data
    collocation_pts = generate_collocation_points(NUM_COLL).to(DEVICE)
    bc_pts = generate_boundary_points(NUM_BC).to(DEVICE)
    ic_pts = generate_initial_points(NUM_IC).to(DEVICE)

    # Instantiate model, optimizer
    model = WavePINN(input_dim=2, hidden_dim=50, output_dim=1, num_hidden_layers=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    loss_history = train_wave_pinn(model, collocation_pts, bc_pts, ic_pts, EPOCHS, optimizer)

    # Evaluate and compare with the analytic solution at a fixed time, say t = 0.5
    t_eval = 0.75
    x_plot = np.linspace(X_MIN, X_MAX, 100)[:, None]
    t_plot = t_eval * np.ones_like(x_plot)
    X_T = np.hstack((x_plot, t_plot))

    model.eval()
    with torch.no_grad():
        X_T_tensor = torch.tensor(X_T, dtype=torch.float32).to(DEVICE)
        u_pred = model(X_T_tensor).cpu().numpy().flatten()

    u_exact = analytic_solution(x_plot, t_eval).flatten()

    # Plot the solution comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_plot, u_exact, label='Analytic', lw=2)
    plt.plot(x_plot, u_pred, '--', label='PINN', lw=2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Solution at t = {t_eval}')
    plt.legend()

    # Plot the training loss history
    plt.subplot(1, 2, 2)
    epochs_logged = np.arange(0, EPOCHS, 50)
    plt.plot(epochs_logged, loss_history, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss History')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
