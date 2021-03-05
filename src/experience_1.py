import torch
import torch.nn as nn
import torch.optim as optim
from modele import Net_Z, Net_H, Net_G
from utils import get_mnist_data, plot_images, add_noise_to_image, freeze, unfreeze, momentum_correction, \
                  weights_init, sample_z

workers = 2

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16

# Get MNIST data
train_dataloader, test_dataloader = get_mnist_data(batch_size, workers=workers)

# Plot some training images
real_batch = next(iter(train_dataloader))
plot_images(real_batch[0].to(device), title="Training Images")

# Create mask
mask = torch.ones((1, 32, 32), device=device, dtype=torch.bool)
mask[0, 4:14, 4:20] = 0

masked_batch = real_batch[0] * mask.unsqueeze(0).cpu()
plot_images(masked_batch.to(device), title="Masked Images")

# Hyper-params + networks initialize

EPOCH = 5
STEPS = 20
alpha = 1e-5
beta_z = 1
lr = 1e-5
noise_variance = 1e-3
experiment = 'data with noise'  # possible values : "data with noise" , "multi mode" or ""

# Create the models
net_H = Net_H().to(device)
net_Z = Net_Z().to(device)
net_G = Net_G().to(device)

# Initialize weights
net_H.apply(weights_init)
net_Z.apply(weights_init)
net_G.apply(weights_init)

# Loss
criterion = nn.L1Loss()

# Optimizers
optimizer_Z = optim.Adam(net_Z.parameters(), lr=lr)
params = list(net_H.parameters()) + list(net_G.parameters())
optimizer_H_G = optim.Adam(params, lr=lr)

"""Training"""

net_H.train()
net_Z.train()
net_G.train()

for i in range(EPOCH):
    cumul_loss_Z = 0
    cumul_loss_H_G = 0
    nb_batch = 0
    loss_Z = 0
    for X, _ in train_dataloader:
        X = add_noise_to_image(X, ratio_noise_per_batch=0.2)

        # put a mask on images
        input_masked = X.to(device) * mask

        # Freeze H network
        freeze(net_H)
        H, skip_connect_layers = net_H(input_masked)

        # freeze G network
        freeze(net_G)

        # generate init z
        z_t = sample_z(X.shape[0], z_size=10)
        z_t.requires_grad = True

        for _ in range(STEPS):
            ###########################
            # ###### updating z  #### #
            ###########################
            z = z_t.clone().detach()

            X_hat = net_G(H, z_t.permute(0, 2, 1).unsqueeze(3), skip_connect_layers)
            loss = criterion(X_hat * ~mask, X.to(device) * ~mask)
            loss.backward()

            with torch.no_grad():
                z_t -= beta_z * z_t.grad

            z_t.grad.zero_()

            ###########################
            # #  updating network Z # #
            ###########################
            optimizer_Z.zero_grad()

            z_t_hat = net_Z(H.squeeze(3).permute(0, 2, 1), z)
            v_t_hat = z_t_hat[:, :, 10:]
            z_t_hat = z_t_hat[:, :, :10]

            # adding noise
            H_with_noise = H + torch.randn(H.shape, device=device) * noise_variance
            z_with_noise = z + torch.randn(z.shape, device=device) * noise_variance
            z_t_hat_with_noise = net_Z(H_with_noise.squeeze(3).permute(0, 2, 1), z_with_noise)
            z_t_hat_with_noise = z_t_hat_with_noise[:, :, :10]

            # loss L_Z
            # formula: L1[z_t+1, Z(z_t, h)] + alpha * L1[Z(z_t + e, h + e), Z(z_t, h)]      with e as noise
            # All information in paper page 18 section 2.4

            loss_z = criterion(z_t_hat, z_t.detach()) + alpha * criterion(z_t_hat_with_noise, z_t_hat.detach())
            loss_v = criterion(v_t_hat, (z_t - z))
            loss_Z = loss_z + loss_v

            loss_Z.backward()

            optimizer_Z.step()

        cumul_loss_Z += loss_Z
        z_t = z_t.detach()

        #################################
        # #  updating network H and G # #
        #################################
        optimizer_H_G.zero_grad()

        # Unfreeze network H and G
        unfreeze(net_H)
        unfreeze(net_G)

        H, skip_connect_layers = net_H(input_masked)

        X_hat = net_G(H, z_t.permute(0, 2, 1).unsqueeze(3), skip_connect_layers)

        # adding noise
        H_with_noise = H + torch.randn_like(H, device=device) * noise_variance
        z_t_with_noise = z_t + torch.randn_like(z_t, device=device) * noise_variance
        X_hat_with_noise = net_G(H_with_noise, z_t_with_noise.permute(0, 2, 1).unsqueeze(3), skip_connect_layers)

        # loss E_hat
        # formula: L1[y_GT, G(z, h)] + alpha * L1[G(z + e, h + e), G(z, h)]      with e as noise
        # All information in paper page 18 section 2.4
        loss_H_G = criterion(X_hat * ~mask, X.to(device) * ~mask) + alpha * criterion(X_hat_with_noise * ~mask,
                                                                                      X_hat.detach() * ~mask)
        loss_H_G.backward()

        optimizer_H_G.step()

        cumul_loss_H_G += loss_H_G
        nb_batch += 1

        if nb_batch % 25 == 0:
            print('Batch: %d\tLoss_Z: %.6f\tLoss_H_G: %.6f' % (nb_batch, loss_Z, loss_H_G))

    print('Epoch: %d\tLoss_Z: %.5f\tLoss_H_G: %.5f' % (i, cumul_loss_Z / nb_batch, cumul_loss_H_G / nb_batch))

    # Saving model

    torch.save(net_H.state_dict(), 'network_H.cpkt')
    torch.save(net_Z.state_dict(), 'network_Z.cpkt')
    torch.save(net_G.state_dict(), 'network_G.cpkt')

"""Test"""

net_H.eval()
net_Z.eval()
net_G.eval()

with torch.no_grad():
    for X, _ in test_dataloader:
        input_masked = X.to(device) * mask

        H, skip_connect_layers = net_H(input_masked)

        # init vector z
        z = sample_z(X.shape[0], z_size=10)
        z2 = sample_z(X.shape[0], z_size=10)
        print(z[0])
        print(z2[0])

        z = momentum_correction(net_Z, H, z, STEPS)
        z2 = momentum_correction(net_Z, H, z2, STEPS)

        print(z[0])
        print(z2[0])
        output = net_G(H, z.permute(0, 2, 1).unsqueeze(3), skip_connect_layers)
        output2 = net_G(H, z2.permute(0, 2, 1).unsqueeze(3), skip_connect_layers)
        output *= ~mask
        output += input_masked
        output2 *= ~mask
        output2 += input_masked

        plot_images(X, title="Real Images")

        plot_images(input_masked, title="Real Images with mask")

        plot_images(output, title="Inpainted Images")

        plot_images(output2, title="Inpainted Images")

        break
