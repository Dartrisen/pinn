# src/models/poisson_net.py
from src.models.base_model import FullyConnectedNet


class PoissonPINN(FullyConnectedNet):
    def __init__(self):
        super(PoissonPINN, self).__init__(input_dim=1, hidden_dim=10, output_dim=1, num_hidden_layers=2)
