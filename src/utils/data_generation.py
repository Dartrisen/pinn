# src/utils/data_generation.py
import numpy as np
import torch


def generate_1d_collocation_points(x_min, x_max, num_points):
    x = np.linspace(x_min, x_max, num_points).reshape(-1, 1)
    return torch.tensor(x, dtype=torch.float32)


def generate_1d_boundary_points(x_min, x_max):
    return torch.tensor([[x_min], [x_max]], dtype=torch.float32)


def generate_2d_collocation_points(x_min, x_max, y_min, y_max, num_points):
    # Uniformly sample interior points
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    points = np.stack([x, y], axis=-1)
    return torch.tensor(points, dtype=torch.float32)


def generate_2d_boundary_points(x_min, x_max, y_min, y_max, num_points_per_edge=100):
    # Create points along the boundaries of a rectangle.
    x_edge = np.linspace(x_min, x_max, num_points_per_edge)
    y_edge = np.linspace(y_min, y_max, num_points_per_edge)

    bottom = np.stack([x_edge, np.full_like(x_edge, y_min)], axis=-1)
    top = np.stack([x_edge, np.full_like(x_edge, y_max)], axis=-1)
    left = np.stack([np.full_like(y_edge, x_min), y_edge], axis=-1)
    right = np.stack([np.full_like(y_edge, x_max), y_edge], axis=-1)

    points = np.concatenate([bottom, top, left, right], axis=0)
    return torch.tensor(points, dtype=torch.float32)


def generate_wave_collocation_points(num_points, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
    """
    Generates collocation points in the space-time domain [x_min,x_max] x [t_min,t_max].

    Returns:
        torch.Tensor of shape (num_points, 2).
    """
    x = np.random.uniform(x_min, x_max, num_points)[:, None]
    t = np.random.uniform(t_min, t_max, num_points)[:, None]
    return torch.tensor(np.hstack((x, t)), dtype=torch.float32)


def generate_wave_boundary_points(num_points, t_min=0.0, t_max=1.0):
    """
    Generates boundary points for x=0 and x=1 with t in [t_min,t_max].

    Returns:
        torch.Tensor of shape (2*num_points, 2).
    """
    t_vals = np.linspace(t_min, t_max, num_points)[:, None]
    bc_left = np.hstack((np.zeros_like(t_vals), t_vals))
    bc_right = np.hstack((np.ones_like(t_vals), t_vals))
    bc = np.vstack((bc_left, bc_right))
    return torch.tensor(bc, dtype=torch.float32)


def generate_wave_initial_points(num_points, x_min=0.0, x_max=1.0, t_val=0.0):
    """
    Generates initial condition points at t=t_val for x in [x_min, x_max].

    Returns:
        torch.Tensor of shape (num_points, 2).
    """
    x_vals = np.linspace(x_min, x_max, num_points)[:, None]
    t_vals = t_val * np.ones_like(x_vals)
    return torch.tensor(np.hstack((x_vals, t_vals)), dtype=torch.float32)
