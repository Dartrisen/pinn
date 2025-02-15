# src/losses/bc_loss.py
from src.losses.conditions import DirichletCondition


# Function-based interfaces for quick use
def dirichlet_bc_loss(model, bc_points, target=0.0):
    return DirichletCondition(target).compute(model, bc_points)
