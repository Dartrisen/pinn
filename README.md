# Generic PINN Framework in PyTorch

This repository implements a modular, generic Physics-Informed Neural Network (PINN) framework in PyTorch designed for solving partial differential equations (PDEs) and systems of PDEs. The framework is organized into several modules (core, models, residuals, losses, utils) that allow users to easily plug in custom PDE residuals, boundary conditions, and initial conditions. Currently, the framework includes implementations for:
- **Poisson 1D**
- **Poisson 2D**
- **Wave 1D**

This modular design facilitates rapid experimentation with various physical models and makes it straightforward to extend the framework to new PDEs.

## Problem Statements and Analytic Solutions

### Poisson 1D
We solve:
$$ \phi''(x) - \sin(\pi x) = 0, \quad x \in [0,1], $$
with Dirichlet boundary conditions:
$$ \phi(0) = \phi(1) = 0. $$
Analytic Solution:
$$ \phi(x) = -\frac{1}{\pi^2}\sin(\pi x). $$

### Poisson 2D
We solve a 2D Poisson problem:
$$
\Delta \phi(x,y) + \sin(\pi x)\sin(\pi y)= 0, \quad (x,y) \in [0,1]\times[0,1],
$$
with Dirichlet boundary conditions:
$$
\phi(x,y) = 0 \quad \text{on } \partial([0,1]\times[0,1]).
$$
*(An analytic solution can be derived in simple cases; see the code for details.)*

### Wave 1D
We solve the 1D wave equation:
$$
u_{tt}(x,t) - u_{xx}(x,t) = 0, \quad (x,t) \in [0,1]\times[0,1],
$$
with boundary conditions:
$$
u(0,t) = u(1,t) = 0, \quad t \in [0,1],
$$
and initial conditions:
$$
u(x,0) = \sin(\pi x), \quad u_t(x,0) = 0, \quad x \in [0,1].
$$
Analytic Solution:
$$
u(x,t)=\cos(\pi t)\sin(\pi x).
$$

## Features

- **Modular Design:** Organized into core, models, residuals, losses, and utils, making it easy to swap components.
- **Multiple PDE Examples:** Supports 1D Poisson, 2D Poisson, and 1D Wave equations.
- **Comprehensive Plotting:** Each example produces multiple plots:
  - Comparison of the PINN-predicted solution vs. the analytic solution.
  - Training loss history.
  - Additional diagnostic plots as needed.
- **Extensible:** Easily add new PDEs by implementing additional residual and loss functions and updating the data generation utilities.


## Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/dartrisen/pinn.git
cd pinn
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python examples/poisson_1d.py
python examples/poisson_2d.py
python examples/wave1d.py
```

## Results
**Poisson 1D**
<img src="https://github.com/Dartrisen/pinn/blob/main/poisson1d.png" width="100%" height="100%">

**Poisson 2D**
<img src="https://github.com/Dartrisen/pinn/blob/main/poisson2d.png" width="100%" height="100%">

## Extending the Framework

To add new PDEs:

- Implement a New Residual: Create a file in src/residuals/ (e.g., for a nonlinear PDE).
- Define New Conditions: Extend or create new loss functions in src/losses/ for boundary/initial conditions.
- Update Data Generation: Modify src/utils/data_generation.py to generate collocation/BC/IC points for new domains.
- Reuse Existing Models: Use or extend the models in src/models/.
