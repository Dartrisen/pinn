# src/models/poisson1d.py
from src.models.base_model import NN


class PoissonPINN(NN):
    def __init__(self):
        super(PoissonPINN, self).__init__(n_input=1, n_hidden=10, n_layers=2, n_output=1)
