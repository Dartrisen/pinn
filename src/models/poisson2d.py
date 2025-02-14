# src/models/poisson2d.py
from src.models.base_model import NN


class Poisson2DPINN(NN):
    def __init__(self):
        super(Poisson2DPINN, self).__init__(n_input=2, n_hidden=50, n_layers=4, n_output=1)
