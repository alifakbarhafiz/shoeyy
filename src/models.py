"""The six generators and the MLP used in the limitation experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DIFFUSION_STEPS, LATENT_DIM, NOISE_DIM


class MLPClassifier(nn.Module):
    """784 → 256 → 64 → 10. The 64-d vector is the latent used in section 2.1."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, 10)

    def features(self, images):
        hidden = F.relu(self.fc1(images.view(images.size(0), -1)))
        return F.relu(self.fc2(hidden))

    def forward(self, images):
        return self.fc3(self.features(images))


class MLPDecoder(nn.Module):
    """Decoder trained on frozen classifier features. 64 → 256 → 784."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT_DIM, 256)
        self.fc2 = nn.Linear(256, 784)

    def forward(self, latent):
        hidden = F.relu(self.fc1(latent))
        return torch.sigmoid(self.fc2(hidden)).view(-1, 1, 28, 28)


class VanillaAE(nn.Module):
    """784 → 256 → 64 → 256 → 784. Sparse AE uses this same net with an L1 term."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, LATENT_DIM),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, images):
        return self.encoder(images.view(images.size(0), -1))

    def decode(self, latent):
        return self.decoder(latent).view(-1, 1, 28, 28)

    def forward(self, images):
        latent = self.encode(images)
        return self.decode(latent), latent


class VAE(nn.Module):
    """Same widths as the vanilla AE. The 64-d code is sampled from N(mu, sigma)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc_mu = nn.Linear(256, LATENT_DIM)
        self.fc_logvar = nn.Linear(256, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, images):
        hidden = F.relu(self.fc1(images.view(images.size(0), -1)))
        return self.fc_mu(hidden), self.fc_logvar(hidden).clamp(-20, 20)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent):
        return self.decoder(latent).view(-1, 1, 28, 28)

    def forward(self, images):
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar


class MLPGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.net(noise).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, images):
        return self.net(images.view(images.size(0), -1)).view(-1)


class DCGANGenerator(nn.Module):
    """Project z to 7×7, then stride-2 transposed convolutions to 14 and 28."""

    def __init__(self):
        super().__init__()
        self.project = nn.Linear(NOISE_DIM, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        hidden = self.project(noise).view(-1, 128, 7, 7)
        return self.net(hidden)


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, images):
        return self.net(images).view(-1)


def dcgan_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device) / half
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, images, time_embedding):
        hidden = self.conv1(F.silu(self.norm1(images)))
        hidden = hidden + self.time_proj(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(images)


class EpsilonUNet(nn.Module):
    """Small U-Net on 28×28. Input is padded to 32 so both pools land on even maps."""

    def __init__(self, base=32, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTime(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = ResBlock(base, base, time_dim)
        self.down2 = ResBlock(base, base * 2, time_dim)
        self.down3 = ResBlock(base * 2, base * 4, time_dim)
        self.mid = ResBlock(base * 4, base * 4, time_dim)
        self.up2 = ResBlock(base * 4 + base * 2, base * 2, time_dim)
        self.up1 = ResBlock(base * 2 + base, base, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(base), base),
            nn.SiLU(),
            nn.Conv2d(base, 1, 3, padding=1),
        )

    def forward(self, images, timesteps):
        time_embedding = self.time_mlp(timesteps)
        padded = F.pad(images, (2, 2, 2, 2))
        hidden = self.in_conv(padded)
        skip_32 = self.down1(hidden, time_embedding)
        skip_16 = self.down2(F.avg_pool2d(skip_32, 2), time_embedding)
        hidden = self.down3(F.avg_pool2d(skip_16, 2), time_embedding)
        hidden = self.mid(hidden, time_embedding)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.up2(torch.cat([hidden, skip_16], dim=1), time_embedding)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.up1(torch.cat([hidden, skip_32], dim=1), time_embedding)
        return self.out(hidden)[:, :, 2:-2, 2:-2]


class DDPM(nn.Module):
    """Noise-prediction DDPM with a linear beta schedule."""

    def __init__(self, steps=DIFFUSION_STEPS):
        super().__init__()
        self.steps = steps
        self.unet = EpsilonUNet()
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

    def loss(self, images):
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, self.steps, (images.size(0),), device=images.device)
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
        noisy = alpha_bar.sqrt() * images + (1 - alpha_bar).sqrt() * noise
        prediction = self.unet(noisy, timesteps)
        return F.mse_loss(prediction, noise)

    def _denoise_step(self, images, step):
        batch = images.size(0)
        timesteps = torch.full((batch,), step, device=images.device, dtype=torch.long)
        beta = self.betas[timesteps].view(batch, 1, 1, 1)
        alpha = self.alphas[timesteps].view(batch, 1, 1, 1)
        alpha_bar = self.alpha_bars[timesteps].view(batch, 1, 1, 1)
        prediction = self.unet(images, timesteps)
        mean = (1 / alpha.sqrt()) * (images - beta / (1 - alpha_bar).sqrt() * prediction)
        if step == 0:
            return mean
        alpha_bar_prev = self.alpha_bars_prev[timesteps].view(batch, 1, 1, 1)
        variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        return mean + variance.clamp_min(1e-20).sqrt() * torch.randn_like(images)

    @torch.no_grad()
    def sample(self, noise):
        """Reverse chain. noise is x_T in [-1, 1] scale (standard normal)."""
        images = noise
        for step in reversed(range(self.steps)):
            images = self._denoise_step(images, step)
        return images

    @torch.no_grad()
    def trajectory(self, noise, keep):
        """keep is a set of remaining step counts, plus steps for the pure-noise frame."""
        frames = {}
        images = noise
        if self.steps in keep:
            frames[self.steps] = images.detach().cpu()
        for step in reversed(range(self.steps)):
            images = self._denoise_step(images, step)
            if step in keep:
                frames[step] = images.detach().cpu()
        return frames
