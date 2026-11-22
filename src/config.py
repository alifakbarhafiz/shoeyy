"""Fixed experiment settings. Every notebook imports these so the comparison stays fair."""

import random

import numpy as np
import torch

DRIVE_ROOT = "/content/drive/MyDrive/dl-sesi07"
SEED = 69
BATCH_SIZE = 128
LATENT_DIM = 64
NOISE_DIM = 100
FID_SAMPLES = 5000
GRID_SAMPLES = 16
RECON_SAMPLES = 8
DIFFUSION_STEPS = 1000
SPARSITY_LAMBDA = 1e-3
NEAR_ZERO = 1e-3
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CLASS_A = 0
CLASS_B = 9
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_signed(images):
    """Map pixels from [0, 1] to [-1, 1]."""
    return images * 2 - 1


def to_unit(images):
    """Map pixels from [-1, 1] to [0, 1]."""
    return ((images.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)


def standard_normal(rows, cols, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator)


def standard_image_noise(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, 1, 28, 28, generator=generator)
