import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_random_mask_mnist(input_batch):
    bsize, c, h, w = input_batch.size()
    mask = torch.ones((c, h, w), device=device, dtype=torch.bool)
    if torch.rand(1).item() > 0.5:
        if torch.rand(1).item() > 0.5:
            mask[0, 4:16, 4:28] = 0
        else:
            mask[0, 16:28, 4:28] = 0
    else:
        if torch.rand(1).item() > 0.5:
            mask[0, 4:28, 4:16] = 0
        else:
            mask[0, 4:28, 16:28] = 0
    return mask


def select_white_line_images(data, proba):
    result = data.clone()
    if torch.rand(1).item() <= proba:
        result[:, 0, 8:11, 6:16] = 1
    return result


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def freeze(network):
    for parameter in network.parameters():
        parameter.requires_grad = False


def unfreeze(network):
    for parameter in network.parameters():
        parameter.requires_grad = True


def sample_z(batch_size, z_size):
    # sampling z from the surface of the n-dimensional sphere
    x = torch.randn(batch_size, z_size, device=device)
    z = x.clone()
    x_norm = z.pow(2).sum(dim=1, keepdim=True).sqrt().expand(-1, z_size)
    z = x / x_norm
    return z.unsqueeze(1)


def momentum_correction(net_Z, H, z, STEPS):
    # optimize z
    for _ in range(STEPS):
        z_t = net_Z(H.squeeze(3).permute(0, 2, 1), z)
        v_t = z_t[:, :, 10:]
        z_t = z_t[:, :, :10]
        rho = torch.sum(torch.abs(v_t), dim=2, keepdim=True)
        z_t = z + rho * (z_t - z) / torch.norm((z_t - z), dim=2, keepdim=True)
        z = z_t
    return z


def plot_images(images, title=""):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def get_mnist_data(batch_size, workers=2):
    """Dataset"""
    dataroot = '/tmp/mnist'
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    train_dataset = dset.MNIST(dataroot, train=True, download=True, transform=transform)
    test_dataset = dset.MNIST(dataroot, train=False, download=True, transform=transform)
    # Create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return train_dataloader, test_dataloader


def add_noise_to_image(batch_images, ratio_noise_per_batch=0.2):
    output = batch_images.clone()
    # Get a random top half of an image
    top_half = batch_images[torch.randint(batch_images.size(0), (1,)), :, :int(batch_images.size(2)/2), :]

    # Indice of images that we want to modify by adding replacing their top half by the random top half
    indice_images = torch.randperm(output.size(0))[:int(output.size(0)*ratio_noise_per_batch)]

    output[indice_images, :, :int(output.size(2)/2), :] = top_half

    return output