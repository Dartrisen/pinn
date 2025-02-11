# src/models/basic_net.py
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, num_hidden_layers=2):
        super(FullyConnectedNet, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
