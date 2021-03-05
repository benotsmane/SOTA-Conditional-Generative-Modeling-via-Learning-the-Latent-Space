import torch
from torch import nn


class Net_H(nn.Module):
    def __init__(self):
        super(Net_H, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(1, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 3), )

    def forward(self, x):
        return self.mlp(x)


class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(3 + 1, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 3))

    def forward(self, h, z):
        # Concat h and z
        h_z = torch.cat((h, z), dim=1)
        return self.mlp(h_z)


class Net_Z(nn.Module):
    def __init__(self):
        super(Net_Z, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(3 + 1, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.layer_z = nn.Linear(256, 1)
        self.layer_rho = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, h, z):
        # Concat h and z
        h_z = torch.cat((h, z), dim=1)
        out = self.mlp(h_z)
        return self.layer_z(out), self.layer_rho(out)
