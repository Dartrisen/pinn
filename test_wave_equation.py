import torch
import unittest
from wave_equation import f, grad, Network


class TestGradients(unittest.TestCase):
    def setUp(self):
        x_raw = torch.linspace(0, 1, 100, requires_grad=True)
        t_raw = torch.linspace(0, 1, 100, requires_grad=True)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        self.x = grids[0].flatten().reshape(-1, 1)
        self.t = grids[1].flatten().reshape(-1, 1)

        self.model = Network(num_hidden=2, dim_hidden=3)
