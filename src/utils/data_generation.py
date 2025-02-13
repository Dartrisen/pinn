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
