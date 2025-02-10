# Generic PINN Framework in PyTorch

This repository implements a modular, generic Physics-Informed Neural Network (PINN) framework in PyTorch designed for solving partial differential equations (PDEs) and systems of PDEs. The framework is structured to allow users to easily plug in custom PDE residuals, boundary conditions, and initial conditions, enabling rapid experimentation with various physical models.

## Repository Structure

generic-pinn-framework/
├── src/
│   ├── generic_pinn.py     # Generic PINN solver class.
│   └── poisson_pinn.py     # Example PINN model for the Poisson equation.
├── main.py                 # Example driver script that sets up and runs the Poisson example.
├── requirements.txt        # Python dependencies.
└── README.md               # This file.

## Problem statement

Find $\phi(x)$ such that
$$\phi''(x) + \sin(\pi x) = 0, \quad x \in [0,1],$$
with Dirichlet boundary conditions
$$\phi(0) = \phi(1) = 0.$$

Analytic Solution:
$$\phi(x) = \frac{1}{\pi^2} \sin(\pi x)$$

## Result
<img src="https://github.com/Dartrisen/pinn/blob/main/result.png" width="50%" height="50%">
