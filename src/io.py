"""Save and load everything the figure notebook needs from Google Drive."""

import json
from pathlib import Path

import torch


def artifact_dir(name=None):
    path = Path("artifacts") if name is None else Path("artifacts") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(name, filename, payload):
    path = artifact_dir(name) / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_json(name, filename):
    path = artifact_dir(name) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the {name} notebook first.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_torch(name, filename, payload):
    path = artifact_dir(name) / filename
    torch.save(payload, path)
    return path


def load_torch(name, filename, map_location="cpu"):
    path = artifact_dir(name) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the {name} notebook first.")
    return torch.load(path, map_location=map_location, weights_only=False)


def save_shared(filename, payload):
    path = artifact_dir() / filename
    torch.save(payload, path)
    return path


def load_shared(filename, map_location="cpu"):
    path = artifact_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run a training notebook first.")
    return torch.load(path, map_location=map_location, weights_only=False)


def checkpoint_path(name):
    return artifact_dir(name) / "checkpoint.pt"


def load_checkpoint(name, map_location="cpu"):
    path = checkpoint_path(name)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)
