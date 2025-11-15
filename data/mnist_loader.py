"""
Unified MNIST Data Loader
Supports binary and multi-class classification with configurable image sizes.
Handles filtering, preprocessing, and dataloaders for all models.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List, Optional
import numpy as np


def load_mnist_data(
    image_size: int = 4,
    batch_size: int = 32,
    classes: Optional[List[int]] = None,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    seed: int = 42,
    normalize: bool = True,
    data_dir: str = "./data",
    num_workers: int = 0,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Load and preprocess MNIST dataset

    Args:
        image_size: Size to resize images to (image_size × image_size)
        batch_size: Batch size for dataloaders
        classes: List of classes to include (None = all 10 classes)
                 Example: [0, 1] for binary classification
        train_samples: Maximum training samples (None = all)
        test_samples: Maximum test samples (None = all)
        seed: Random seed for reproducibility
        normalize: Whether to normalize images to [-1, 1]
        data_dir: Directory to store/load MNIST data
        num_workers: Number of dataloader workers
        shuffle_train: Whether to shuffle training data
        shuffle_test: Whether to shuffle test data

    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader
        info: Dictionary with dataset information

    Examples:
        >>> # Binary classification (0 vs 1)
        >>> train_loader, test_loader, info = load_mnist_data(
        ...     image_size=4, batch_size=32, classes=[0, 1]
        ... )

        >>> # Full 10-class with larger images
        >>> train_loader, test_loader, info = load_mnist_data(
        ...     image_size=8, batch_size=64, classes=None
        ... )
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define transforms
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    transform = transforms.Compose(transform_list)

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Filter classes if specified
    if classes is not None:
        train_dataset = filter_classes(train_dataset, classes)
        test_dataset = filter_classes(test_dataset, classes)

    # Limit samples if specified
    if train_samples is not None and train_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_samples]
        train_dataset = Subset(train_dataset, indices)

    if test_samples is not None and test_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:test_samples]
        test_dataset = Subset(test_dataset, indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Collect info
    info = {
        "image_size": image_size,
        "batch_size": batch_size,
        "classes": classes if classes is not None else list(range(10)),
        "num_classes": len(classes) if classes is not None else 10,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "normalized": normalize,
        "seed": seed,
    }

    return train_loader, test_loader, info


def filter_classes(dataset, classes: List[int]):
    """
    Filter dataset to only include specified classes

    Args:
        dataset: MNIST dataset
        classes: List of class indices to keep

    Returns:
        Subset of dataset containing only specified classes
    """
    if hasattr(dataset, "targets"):
        # Original dataset
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for cls in classes:
                mask |= targets == cls
            indices = torch.where(mask)[0]
        else:
            # List or numpy array
            indices = [i for i, t in enumerate(targets) if t in classes]
    else:
        # Already a Subset - need to access underlying dataset
        base_dataset = dataset.dataset
        subset_indices = dataset.indices
        targets = base_dataset.targets

        if isinstance(targets, torch.Tensor):
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for cls in classes:
                mask |= targets == cls
            all_indices = torch.where(mask)[0]
            # Filter to only include indices in our subset
            indices = [idx for idx in all_indices if idx in subset_indices]
        else:
            indices = [i for i in subset_indices if targets[i] in classes]

    return Subset(dataset if hasattr(dataset, "targets") else dataset.dataset, indices)


def get_class_distribution(dataloader: DataLoader) -> dict:
    """
    Get class distribution in a dataloader

    Args:
        dataloader: PyTorch DataLoader

    Returns:
        Dictionary mapping class indices to counts
    """
    class_counts = {}

    for _, labels in dataloader:
        for label in labels:
            label = label.item()
            class_counts[label] = class_counts.get(label, 0) + 1

    return dict(sorted(class_counts.items()))


def visualize_samples(
    dataloader: DataLoader,
    num_samples: int = 25,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
):
    """
    Visualize random samples from dataloader

    Args:
        dataloader: PyTorch DataLoader
        num_samples: Number of samples to show
        figsize: Figure size (width, height)
        save_path: Path to save figure (None = display only)
    """
    import matplotlib.pyplot as plt

    # Get one batch
    images, labels = next(iter(dataloader))

    # Limit to num_samples
    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Create grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num_samples):
        img = images[idx].squeeze().cpu().numpy()

        # Denormalize if needed
        if img.min() < 0:
            img = img * 0.5 + 0.5

        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(f"Label: {labels[idx].item()}")
        axes[idx].axis("off")

    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def get_sample_batch(dataloader: DataLoader, batch_idx: int = 0):
    """
    Get a specific batch from dataloader

    Args:
        dataloader: PyTorch DataLoader
        batch_idx: Index of batch to retrieve

    Returns:
        images, labels tensors
    """
    for i, (images, labels) in enumerate(dataloader):
        if i == batch_idx:
            return images, labels

    raise IndexError(f"Batch index {batch_idx} out of range")


def print_dataset_info(info: dict):
    """
    Print formatted dataset information

    Args:
        info: Info dictionary from load_mnist_data
    """
    print("=" * 60)
    print("MNIST Dataset Information")
    print("=" * 60)
    print(f"Image Size:       {info['image_size']}×{info['image_size']}")
    print(f"Batch Size:       {info['batch_size']}")
    print(f"Classes:          {info['classes']}")
    print(f"Num Classes:      {info['num_classes']}")
    print(f"Train Samples:    {info['train_samples']:,}")
    print(f"Test Samples:     {info['test_samples']:,}")
    print(f"Normalized:       {info['normalized']}")
    print(f"Random Seed:      {info['seed']}")
    print("=" * 60)


# Quick test
if __name__ == "__main__":
    print("Testing MNIST data loader...")

    # Test 1: Binary classification (0 vs 1) with 4x4 images
    print("\n=== Test 1: Binary Classification (4×4) ===")
    train_loader, test_loader, info = load_mnist_data(
        image_size=4, batch_size=32, classes=[0, 1], seed=42
    )

    print_dataset_info(info)

    # Check class distribution
    print("\nClass Distribution:")
    train_dist = get_class_distribution(train_loader)
    test_dist = get_class_distribution(test_loader)
    print(f"  Train: {train_dist}")
    print(f"  Test:  {test_dist}")

    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch Shape:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")

    # Test 2: Full 10-class with 8x8 images
    print("\n=== Test 2: 10-Class Classification (8×8) ===")
    train_loader_full, test_loader_full, info_full = load_mnist_data(
        image_size=8,
        batch_size=64,
        classes=None,  # All classes
        train_samples=1000,  # Limit for testing
        test_samples=200,
        seed=42,
    )

    print_dataset_info(info_full)

    # Test 3: Limited samples
    print("\n=== Test 3: Limited Samples ===")
    train_loader_limited, test_loader_limited, info_limited = load_mnist_data(
        image_size=4,
        batch_size=16,
        classes=[0, 1, 2, 3, 4],
        train_samples=500,
        test_samples=100,
        seed=42,
    )

    print_dataset_info(info_limited)

    # Test 4: Visualization
    print("\n=== Test 4: Visualization ===")
    visualize_samples(
        train_loader, num_samples=16, figsize=(8, 8), save_path="test_mnist_samples.png"
    )

    print("\n✓ MNIST data loader test passed!")
