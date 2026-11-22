"""Fashion-MNIST, one seeded 80/10/10 split for every notebook."""

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

from src.config import BATCH_SIZE, CLASS_A, CLASS_B, RECON_SAMPLES, SEED

_SPLIT_CACHE = None


def get_splits(root="data"):
    """Return the same train, val, and test subsets on every call."""
    global _SPLIT_CACHE
    if _SPLIT_CACHE is None:
        transform = transforms.ToTensor()
        official_train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        official_test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
        full = ConcatDataset([official_train, official_test])
        generator = torch.Generator().manual_seed(SEED)
        train_set, val_set, test_set = random_split(
            full, [56000, 7000, 7000], generator=generator
        )
        _SPLIT_CACHE = (train_set, val_set, test_set)
    return _SPLIT_CACHE


def make_loader(dataset, shuffle, seed=None):
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(SEED if seed is None else seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def fixed_test_images(n=RECON_SAMPLES):
    """The same n test images for every reconstruction figure."""
    _, _, test_set = get_splits()
    images, labels = [], []
    for index in range(n):
        image, label = test_set[index]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images), labels


def class_pair():
    """One test image from class 0 and one from class 9."""
    _, _, test_set = get_splits()
    found = {}
    for index in range(len(test_set)):
        image, label = test_set[index]
        label = int(label)
        if label in (CLASS_A, CLASS_B) and label not in found:
            found[label] = image.unsqueeze(0)
        if len(found) == 2:
            break
    return found[CLASS_A], found[CLASS_B]
