"""
Training Utilities
Helper functions for training, seeding, and device management
"""

import torch
import numpy as np
import random
import os
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # PennyLane
    try:
        from pennylane import numpy as pnp

        pnp.random.seed(seed)
    except ImportError:
        pass


def get_device(prefer_cpu: bool = False) -> torch.device:
    """
    Get best available device

    Args:
        prefer_cpu: Force CPU even if GPU available (for quantum simulations)

    Returns:
        torch.device
    """
    if prefer_cpu:
        return torch.device("cpu")

    # Check for Kaggle environment
    if os.path.exists("/kaggle"):
        # On Kaggle, prefer CPU for quantum simulations
        return torch.device("cpu")

    # Otherwise use GPU if available
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def save_model_state(model, optimizer, epoch, save_path):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        save_path: Path to save checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )


def load_model_state(model, optimizer, load_path, device="cpu"):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        load_path: Path to checkpoint
        device: Device to load to

    Returns:
        Loaded epoch number
    """
    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0)


def create_output_dirs(base_dir: str = "./results"):
    """
    Create standard output directory structure

    Args:
        base_dir: Base directory for results

    Returns:
        Dictionary with directory paths
    """
    base_path = Path(base_dir)

    dirs = {
        "checkpoints": base_path / "checkpoints",
        "metrics": base_path / "metrics",
        "plots": base_path / "plots",
        "logs": base_path / "logs",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return {k: str(v) for k, v in dirs.items()}


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_lr(optimizer):
    """
    Get current learning rate from optimizer

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
