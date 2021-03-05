import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from utils import weights_init, freeze, unfreeze
from networks_toy import Net_G, Net_H, Net_Z
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_batch(batch_size):
    x = torch.rand(batch_size, device=device).view(-1, 1)
    k = random.randint(0, 1)
    sign = torch.tensor([[1] if i % 2 == k else [-1] for i in range(batch_size)], device=device)
    y1 = sign * 4 * x
    y2 = sign * 4 * x ** 2
    y3 = sign * 4 * x ** 3
    y = torch.cat([y1, y2, y3], dim=1)
    return x, y


if __name__ == "__main__":
    N_EPOCHS = 400
    N_BATCHS = 150
    STEPS = 20
    Z_DIM = 1
    BATCH_SIZE = 3  # Important to keep this small ! (<10)
    alpha = 1e-4  # for lipshitz regularization
    beta_z = 0.001  # lr for z updating
    lr = 1e-4  # lr for networks Z, H, and G
    debug = False
    SAVE = True

    net_H = Net_H().to(device)
    net_Z = Net_Z().to(device)
    net_G = Net_G().to(device)
    # # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2
    net_H.apply(weights_init)
    net_Z.apply(weights_init)
    net_G.apply(weights_init)

    # Loss
    criterion = nn.MSELoss()

    # Optimizers
    optimizer_Z = optim.Adam(net_Z.parameters(), lr=lr)
    params = list(net_H.parameters()) + list(net_G.parameters())
    optimizer_H_G = optim.Adam(params, lr=lr)

    net_H.train()
    net_Z.train()
    net_G.train()
    for epoch in range(N_EPOCHS):
        cumul_loss_Z = 0
        cumul_loss_z = 0
        cumul_loss_rho = 0
        cumul_loss_H_G = 0
        nb_batch = 0
        for _ in tqdm(range(N_BATCHS)):
            x, y = get_batch(BATCH_SIZE)
            bsize = x.shape[0]
            # Freeze H and G networks
            freeze(net_H)
            freeze(net_G)

            h = net_H(x)
            # # # generate init z
            z_t = 0.2 * torch.rand((bsize, 1), device=device) - 0.1
            z_t.requires_grad = True
            loss_Z = 0
            loss_rho = 0
            loss_z = 0
            v_prev = 0
            v = 0
            for p in range(STEPS):
                z = z_t.clone().detach()
                y_hat = net_G(h, z_t)
                loss = criterion(y_hat, y)
                loss.backward()

                with torch.no_grad():
                    # update with momentum
                    v = 0.002 * v_prev - beta_z * z_t.grad.data
                    z_t.data += v.data
                    # z_t.data -= beta_z * z_t.grad.data
                    v_prev = v.clone()
                    z_t.data = torch.clip(z_t.data, -.1, .1)

                z_t.grad = None
                optimizer_Z.zero_grad()

                rho = torch.norm(z_t - z, p=1, dim=1, keepdim=True)
                # adding noise
                h_noise = h + 0.001 * torch.randn_like(h)
                z_noise = z + 0.001 * torch.randn_like(z)

                z_t_hat, rho_hat = net_Z(h, z)
                z_t_hat_noise, rho_hat_noise = net_Z(h_noise, z_noise)

                loss_z = criterion(z_t_hat, z_t.detach()) + alpha * criterion(z_t_hat_noise, z_t_hat.detach())
                loss_rho = criterion(rho_hat, rho.detach()) + alpha * criterion(rho_hat_noise, rho_hat.detach())
                loss_Z = loss_z + loss_rho
                loss_Z.backward()
                optimizer_Z.step()

            cumul_loss_Z += loss_Z
            cumul_loss_z += loss_z
            cumul_loss_rho += loss_rho
            z_t = z_t.detach()
            optimizer_H_G.zero_grad()

            # Unfreeze network H and G
            unfreeze(net_H)
            unfreeze(net_G)
            h = net_H(x)
            y_hat = net_G(h, z_t)

            # adding noise for lipshitz regularization
            h_noise = h + 0.001 * torch.randn_like(h)
            z_t_noise = z_t + 0.001 * torch.randn_like(z_t)
            y_hat_noise = net_G(h_noise, z_t_noise)

            loss_H_G = criterion(y_hat, y) + alpha * criterion(y_hat_noise, y_hat.detach())
            loss_H_G.backward()

            optimizer_H_G.step()

            cumul_loss_H_G += loss_H_G
            nb_batch += 1

        print('Epoch: %d\tLoss_Z: %.6f\tLoss_z: %.6f\tLoss_rho: %.6f\tLoss_H_G: %.6f'
              % (epoch, cumul_loss_Z / nb_batch, cumul_loss_z / nb_batch, cumul_loss_rho / nb_batch,
                 cumul_loss_H_G / nb_batch))

        # checkpoint
        if SAVE and (epoch % 10 == 0 or epoch == N_EPOCHS - 1) and (epoch > 0):
            torch.save({'epoch': epoch,
                        'Z_state_dict': net_Z.state_dict(),
                        'G_state_dict': net_G.state_dict(),
                        'H_state_dict': net_H.state_dict(),
                        'optimizer_H_G_state_dict': optimizer_H_G.state_dict(),
                        'optimizer_Z_state_dict': optimizer_Z.state_dict(),
                        },
                       f"models/toy_model.pth")

    toy_checkpoint = torch.load("models/toy_model.pth")

    net_G.load_state_dict(toy_checkpoint['G_state_dict'])
    net_H.load_state_dict(toy_checkpoint['H_state_dict'])
    net_Z.load_state_dict(toy_checkpoint['Z_state_dict'])

    freeze(net_H)
    freeze(net_G)
    freeze(net_Z)

    net_H.eval()
    net_Z.eval()
    net_G.eval()
    x_array = []
    z_array = []
    y_array = []
    y_hat_array = []
    for _ in tqdm(range(1000)):
        x, y = get_batch(BATCH_SIZE)
        z_t = .2 * torch.rand((BATCH_SIZE, 1), device=device) - .1
        # z_t = sample_z(BATCH_SIZE, z_size=1).reshape(-1, 1)
        z_t.requires_grad = True
        v_prev = 0
        for p in range(4):
            # print(" update :", p)
            z = z_t.clone().detach()
            y_hat = net_G(net_H(x), z_t)
            loss = criterion(y_hat, y)
            loss.backward()
            x_array.append(x.cpu().numpy())
            z_array.append(z_t.cpu().detach().numpy())
            y_array.append(y.cpu().numpy())
            y_hat_array.append(y_hat.cpu().detach().numpy())
            with torch.no_grad():
                # Update with momentum
                v = 0.002 * v_prev - beta_z * z_t.grad.data
                z_t.data += v.data
                # z_t.data -= beta_z * z_t.grad.data
                v_prev = v.clone()
                z_t.data = torch.clip(z_t.data, -.1, .1)

            z_pred, rho = net_Z(net_H(x), z)
            z_t_2 = z + rho * (z_pred - z) / torch.clip(torch.abs((z_pred - z)), 1e-8, float("inf"))
            z_t_2.data = torch.clip(z_t_2.data, -0.1, .1)
            rho_true = torch.norm(z_t - z, p=1, dim=1, keepdim=True)
            z_t.grad.zero_()
            y_hat = net_G(net_H(x), z_t)

            x_array.append(x.cpu().numpy())
            z_array.append(z_t.cpu().detach().numpy())
            y_array.append(y.cpu().numpy())
            y_hat_array.append(y_hat.cpu().detach().numpy())

    x_array = np.vstack(x_array)
    z_array = np.vstack(z_array)
    y_array = np.vstack(y_array)
    y_hat_array = np.vstack(y_hat_array)

    E = np.sqrt(np.sum((y_hat_array - y_array) ** 2, axis=1, keepdims=True))
    plt.scatter(x_array, z_array, c=-E, cmap=plt.cm.YlOrBr, s=0.5)
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.show()

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 0], s=0.5)
    plt.scatter(x_array, y_array[:, 0], s=0.5)
    plt.show()

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 1], s=0.5)
    plt.scatter(x_array, y_array[:, 1], s=0.5)
    plt.show()

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 2], s=0.5)
    plt.scatter(x_array, y_array[:, 2], s=0.5)
    plt.show()

    # Inference with momentum correction
    x_array = []
    z_array = []
    y_array = []
    y_hat_array = []
    for _ in tqdm(range(500)):
        x, y = get_batch(BATCH_SIZE)
        z_t = .2 * torch.rand((BATCH_SIZE, 1), device=device) - .1
        for _ in range(STEPS):
            z = z_t.clone().detach()
            with torch.no_grad():
                z_pred, rho = net_Z(net_H(x), z)
                z_t = z_pred + rho * (z_pred-z) / torch.clip(torch.abs((z_pred-z)), 1e-8, float("inf"))
                z_t.data = torch.clip(z_t.data, -0.1, 0.1)
        y_hat = net_G(net_H(x), z_t)

        x_array.append(x.cpu().numpy())
        z_array.append(z_t.cpu().detach().numpy())
        y_array.append(y.cpu().numpy())
        y_hat_array.append(y_hat.cpu().detach().numpy())

    x_array = np.vstack(x_array)
    z_array = np.vstack(z_array)
    y_array = np.vstack(y_array)
    y_hat_array = np.vstack(y_hat_array)

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 0], s=0.5)
    plt.scatter(x_array, y_array[:, 0], s=0.5)
    plt.show()

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 1], s=0.5)
    plt.scatter(x_array, y_array[:, 1], s=0.5)
    plt.show()

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 2], s=0.5)
    plt.scatter(x_array, y_array[:, 2], s=0.5)
    plt.show()

    # # Inference without momentum correction
    x_array = []
    z_array = []
    y_array = []
    y_hat_array = []
    for _ in tqdm(range(500)):
        x, y = get_batch(BATCH_SIZE)
        z_t = .2 * torch.rand((BATCH_SIZE, 1), device=device) - .1
        for _ in range(STEPS):
            z = z_t.clone().detach()
            with torch.no_grad():
                z_pred, rho = net_Z(net_H(x), z)
                # rho = torch.clip(rho ** 2, 0., 10.)
                z_t = z_pred
                z_t.data = torch.clip(z_t.data, -0.1, 0.1)

        y_hat = net_G(net_H(x), z_t)

        x_array.append(x.cpu().numpy())
        z_array.append(z_t.cpu().detach().numpy())
        y_array.append(y.cpu().numpy())
        y_hat_array.append(y_hat.cpu().detach().numpy())

    x_array = np.vstack(x_array)
    z_array = np.vstack(z_array)
    y_array = np.vstack(y_array)
    y_hat_array = np.vstack(y_hat_array)

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 0], s=0.5)
    plt.scatter(x_array, y_array[:, 0], s=0.5)
    plt.show()

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 1], s=0.5)
    plt.scatter(x_array, y_array[:, 1], s=0.5)
    plt.show()

    fig = plt.figure()
    plt.scatter(x_array, y_hat_array[:, 2], s=0.5)
    plt.scatter(x_array, y_array[:, 2], s=0.5)
    plt.show()