# src/models/wave1d.py
from src.models.base_model import NN


class WavePINN(NN):
    def __init__(self):
        super(WavePINN, self).__init__(n_input=2, n_hidden=50, n_layers=2, n_output=1)
