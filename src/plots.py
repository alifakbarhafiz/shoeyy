"""Small grids for the sanity check at the end of a training notebook."""

import math

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_grid(images, nrow=8, title=None):
    images = images.detach().cpu().clamp(0, 1)
    grid = make_grid(images, nrow=nrow)
    rows = math.ceil(images.size(0) / nrow)
    plt.figure(figsize=(nrow * 1.15, max(1.5, rows * 1.3)))
    array = grid.permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        plt.imshow(array[:, :, 0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(array, vmin=0, vmax=1)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
