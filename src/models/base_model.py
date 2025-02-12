# src/models/base_model.py
import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, n_output):
        super().__init__()
        self.fcs = nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
            ]) for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = (x-x.mean())/x.std()
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class NN_SC1(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.alpha = nn.parameter.Parameter(data=torch.ones(1))
        self.fcs = nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
            ]) for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = (x-mean)/std
        x = self.alpha*x
        x = torch.exp(x)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class NN_SC2(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.alpha = nn.parameter.Parameter(data=-1*torch.ones(1))
        self.fcs = nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
            ]) for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = (x-mean)/std
        x = self.alpha*x
        x = torch.exp(x)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
