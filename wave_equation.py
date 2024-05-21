import torch
from torch import nn


class Network(nn.Module):
    """This neural network is designed to take two input features and produce a single output.

    In the context of Physics Informed Neural Networks (PINNs), this network serves as a universal
    function approximator to estimate the solution of a given differential equation.
    """
    def __init__(self, num_hidden: int, dim_hidden: int):
        super(Network, self).__init__()

        self.feature_block = nn.Sequential(
            nn.Linear(2, dim_hidden),
            *[nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.Tanh()) for _ in range(num_hidden - 1)]
        )
        self.classifier = nn.Linear(dim_hidden, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = torch.cat([x, t], dim=1)
        x = self.feature_block(x)
        x = self.classifier(x)
        return x


def f(model: Network, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return model(x, t)


def grad(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Computes the partial derivative of an output with respect to an input."""
    f_grad = output
    for _ in range(order):
        f_grad = torch.autograd.grad(
            f_grad,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]
    return f_grad
