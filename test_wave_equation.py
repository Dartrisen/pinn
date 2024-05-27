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

    def test_gradients(self):
        eps = 1e-4
        f_x_derivative = (f(self.model, self.x + eps, self.t) - f(self.model, self.x - eps, self.t)) / (2 * eps)
        f_value = f(self.model, self.x, self.t)
        f_x_autoderivative = grad(f_value, self.x, order=1)
        is_matching_x = torch.allclose(f_x_derivative.T, f_x_autoderivative.T, atol=1e-2, rtol=1e-2)
        self.assertTrue(is_matching_x)


if __name__ == '__main__':
    unittest.main()
