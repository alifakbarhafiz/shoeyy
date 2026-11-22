"""Test-set scores shared by every model: MSE, PSNR, SSIM, latency, and FID."""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity
from torchvision.models import inception_v3, Inception_V3_Weights

from src.config import FID_SAMPLES, NEAR_ZERO

_INCEPTION = None


def psnr_from_mse(mse):
    if mse <= 0:
        return 99.0
    return float(10 * math.log10(1.0 / mse))


def activation_summary(latent):
    latent = latent.detach()
    return {
        "mean_abs_latent": float(latent.abs().mean()),
        "frac_near_zero": float((latent.abs() < NEAR_ZERO).float().mean()),
    }


def stability_from_curve(values):
    curve = np.asarray(values, dtype=float)
    if curve.size == 0 or not np.isfinite(curve).all():
        return "diverged"
    if curve[-1] <= curve[0]:
        return "stable"
    return "not converged"


def gan_stability(generator_loss, discriminator_loss):
    generator = np.asarray(generator_loss, dtype=float)
    discriminator = np.asarray(discriminator_loss, dtype=float)
    if generator.size == 0 or not (
        np.isfinite(generator).all() and np.isfinite(discriminator).all()
    ):
        return "diverged"
    generator_tail = generator[-10:]
    discriminator_tail = discriminator[-10:]
    if discriminator_tail.mean() < 0.05:
        return "discriminator loss near zero"
    if generator_tail.std() > max(float(generator_tail.mean()), 1e-8):
        return "generator loss oscillating"
    return "stable"


@torch.no_grad()
def reconstruction_scores(reconstruct, loader, device):
    """reconstruct(x) -> image in [0, 1] with the same shape as x."""
    mse_sum = 0.0
    pixel_count = 0
    psnr_values = []
    ssim_values = []
    for images, _ in loader:
        images = images.to(device)
        restored = reconstruct(images).clamp(0, 1)
        mse_sum += F.mse_loss(restored, images, reduction="sum").item()
        pixel_count += images.numel()
        originals = images[:, 0].cpu().numpy()
        outputs = restored[:, 0].cpu().numpy()
        for original, output in zip(originals, outputs):
            pixel_mse = float(np.mean((original - output) ** 2))
            psnr_values.append(psnr_from_mse(pixel_mse))
            ssim_values.append(
                float(structural_similarity(original, output, data_range=1.0))
            )
    return {
        "test_mse": mse_sum / pixel_count,
        "test_psnr": float(np.mean(psnr_values)),
        "test_ssim": float(np.mean(ssim_values)),
    }


def measure_latency_ms(fn, device, warmup=10, repeats=30):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def _frechet(mu_real, sigma_real, mu_fake, sigma_fake):
    offset = np.eye(sigma_real.shape[0]) * 1e-6
    covmean = sqrtm((sigma_real + offset) @ (sigma_fake + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    difference = mu_real - mu_fake
    distance = difference.dot(difference) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(distance)


def _inception(device):
    global _INCEPTION
    if _INCEPTION is None:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        _INCEPTION = model.to(device)
    return _INCEPTION


@torch.no_grad()
def _embed(images, device, batch_size=16):
    """Inception pool features. images are (N, 1, 28, 28) in [0, 1]."""
    model = _inception(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    features = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start : start + batch_size].to(device)
        batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        embedded = model(batch)
        if not torch.is_tensor(embedded):
            embedded = embedded.logits
        features.append(embedded.cpu())
    return torch.cat(features).numpy()


def _gaussian(features):
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@torch.no_grad()
def fid_score(fake_images, real_images, device):
    """Same Inception pipeline for every generative model. Images are in [0, 1]."""
    fake_images = fake_images[:FID_SAMPLES].cpu()
    real_images = real_images[:FID_SAMPLES].cpu()
    cache = Path("artifacts") / "fid_real.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        real_features = np.load(cache)
        if len(real_features) != len(real_images):
            real_features = _embed(real_images, device)
            np.save(cache, real_features)
    else:
        real_features = _embed(real_images, device)
        np.save(cache, real_features)
    fake_features = _embed(fake_images, device)
    mu_real, sigma_real = _gaussian(real_features)
    mu_fake, sigma_fake = _gaussian(fake_features)
    return _frechet(mu_real, sigma_real, mu_fake, sigma_fake)


def real_images_for_fid(loader):
    images = []
    seen = 0
    for batch, _ in loader:
        images.append(batch)
        seen += batch.size(0)
        if seen >= FID_SAMPLES:
            break
    return torch.cat(images)[:FID_SAMPLES]
