import torch
import torch.nn as nn
from src.models.base_model import FullyConnectedNet


class Poisson2DPINN(FullyConnectedNet):
    def __init__(self):
        super(Poisson2DPINN, self).__init__(input_dim=2, hidden_dim=50, num_hidden_layers=3, output_dim=1)
        # layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        # for _ in range(num_hidden_layers):
        #     layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     layers.append(nn.Tanh())
        # layers.append(nn.Linear(hidden_dim, output_dim))
        # self.net = nn.Sequential(*layers)

    # def forward(self, x):
    #     return self.net(x)
