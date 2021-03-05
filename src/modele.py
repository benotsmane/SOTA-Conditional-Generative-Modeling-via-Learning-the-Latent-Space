import torch
import torch.nn as nn
from utils import weights_init, sample_z
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net_H(nn.Module):
    def __init__(self):
        super(Net_H, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, (4, 4), stride=1, padding=3, dilation=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 4000, (4, 4), stride=2, padding=0),
            nn.BatchNorm2d(4000),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, X):
        skip_connection_layers = []
        h = self.conv_1(X)
        skip_connection_layers.append(h.clone())
        h = self.conv_2(h)
        skip_connection_layers.append(h.clone())
        h = self.conv_3(h)
        skip_connection_layers.append(h.clone())
        h = self.conv_4(h)
        skip_connection_layers.append(h.clone())
        h = self.conv_5(h)
        skip_connection_layers.append(h.clone())

        return h, skip_connection_layers


class Net_Z(nn.Module):
    def __init__(self):
        super(Net_Z, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=3, padding=0, dilation=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=6, stride=3, padding=0, dilation=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=4, stride=3, padding=0, dilation=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True)

        )

        self.dense_layers = nn.Sequential(
            nn.Linear(4704, 20),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(20, 10),
            # nn.LeakyReLU(inplace=True)
        )

    def forward(self, h, z):
        # Concat h and z
        h_z = torch.cat((h, z), dim=2)

        z_t = self.conv_layers(h_z)
        # Reshaping
        z_t = z_t.view(z_t.shape[0], 1, -1)

        z_t = self.dense_layers(z_t)
        return z_t


class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(4010, 512, 4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=1, padding=3, dilation=2),
            nn.Tanh()
        )

    def forward(self, h, z, skip_connect_layers):
        # Concat h and z
        h_z = torch.cat((h, z), dim=1)
        output = self.deconv_1(h_z)
        output = self.deconv_2(output) + skip_connect_layers[2]  # adding skip connection
        output = self.deconv_3(output) + skip_connect_layers[1]  # adding skip connection
        output = self.deconv_4(output)
        output = self.deconv_5(output)

        return output


if __name__ == "__main__":
    # Create the models
    net_H = Net_H().to(device)
    net_Z = Net_Z().to(device)
    net_G = Net_G().to(device)

    # Initialize models weights
    net_H.apply(weights_init)
    net_Z.apply(weights_init)
    net_G.apply(weights_init)

    # Print the models
    print(net_H)
    print(net_Z)
    print(net_G)

    """Test outputs size"""
    z = torch.zeros(10, 1, 10).to(device)
    X = torch.zeros(10, 1, 32, 32).to(device)
    with torch.no_grad():
        h, skip_connect_layers = net_H(X)
        z = net_Z(h.squeeze(3).permute(0, 2, 1), z)
        output = net_G(h, z[:, :, :10].permute(0, 2, 1).unsqueeze(3), skip_connect_layers)

    print(h.shape)  # expected: [10, 4000, 1, 1]
    print(z.shape)  # expected: [10, 1, 10]
    print(output.shape)  # expected: [10, 1, 32, 32]

    """Test vector z shape"""
    print(sample_z(10, z_size=10).size())  # expected size [10,1,10]
