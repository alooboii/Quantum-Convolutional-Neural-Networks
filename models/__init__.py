"""
Models Package
Centralized model registry and factory for all implementations
"""

from .base_model import BaseModel
from .qfcn import QFCN
from .classical_cnn import ClassicalCNN, ClassicalCNN_Large, get_classical_cnn
from .cong_qcnn import CongQCNN

# Model Registry
MODEL_REGISTRY = {
    "qfcn": QFCN,
    "cong_qcnn": CongQCNN,
    "classical": ClassicalCNN,
    "classical_large": ClassicalCNN_Large,
}

# Aliases for convenience
MODEL_ALIASES = {
    "quantum_fourier": "qfcn",
    "spectral": "qfcn",
    "fourier": "qfcn",
    "cong": "cong_qcnn",
    "original_qcnn": "cong_qcnn",
    "mera": "cong_qcnn",
    "cnn": "classical",
    "baseline": "classical",
}


def get_model(model_name: str, **kwargs):
    """
    Factory function to create models by name

    Args:
        model_name: Name of the model ('qfcn', 'cong_qcnn', 'classical', etc.)
        **kwargs: Model-specific arguments
            - num_classes: Number of output classes (default: 2)
            - image_size: Size of input images (default: 4)
            - n_qubits_start: For QFCN (default: 4)
            - n_qubits: For Cong QCNN (default: 8)

    Returns:
        Model instance

    Examples:
        >>> model = get_model('qfcn', num_classes=2)
        >>> model = get_model('classical', image_size=8, num_classes=10)
        >>> model = get_model('cong_qcnn', n_qubits=8)
    """
    # Resolve aliases
    model_name = model_name.lower()
    if model_name in MODEL_ALIASES:
        model_name = MODEL_ALIASES[model_name]

    # Check if model exists
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. " f"Available models: {available}"
        )

    # Get model class
    model_class = MODEL_REGISTRY[model_name]

    # Extract common arguments
    num_classes = kwargs.get("num_classes", 2)
    image_size = kwargs.get("image_size", 4)

    # Create model with appropriate arguments
    if model_name == "qfcn":
        n_qubits_start = kwargs.get("n_qubits_start", 4)
        n_qubits_l2 = kwargs.get("n_qubits_l2", 2)
        return model_class(
            n_qubits_start=n_qubits_start,
            n_qubits_l2=n_qubits_l2,
            num_classes=num_classes,
        )

    elif model_name == "cong_qcnn":
        n_qubits = kwargs.get("n_qubits", 8)
        return model_class(n_qubits=n_qubits, num_classes=num_classes)

    elif model_name in ["classical", "classical_large"]:
        return model_class(
            input_channels=1, num_classes=num_classes, image_size=image_size
        )

    else:
        # Generic fallback
        return model_class(**kwargs)


def get_model_auto(model_name: str, image_size: int = 4, num_classes: int = 2):
    """
    Automatic model creation with smart defaults based on image size

    Args:
        model_name: Name of the model
        image_size: Size of input images
        num_classes: Number of output classes

    Returns:
        Model instance with appropriate configuration

    Examples:
        >>> model = get_model_auto('classical', image_size=4)  # Returns ClassicalCNN
        >>> model = get_model_auto('classical', image_size=8)  # Returns ClassicalCNN_Large
    """
    if model_name.lower() in ["classical", "cnn", "baseline"]:
        return get_classical_cnn(image_size=image_size, num_classes=num_classes)

    elif model_name.lower() in ["qfcn", "quantum_fourier", "spectral"]:
        # QFCN expects 4x4 = 16 pixels for 4 qubits
        if image_size == 4:
            return get_model(
                "qfcn", n_qubits_start=4, n_qubits_l2=2, num_classes=num_classes
            )
        elif image_size == 8:
            # For 8x8 = 64 pixels, we'd need 6 qubits (2^6 = 64)
            # But let's stick with 4 qubits and downsample/pool
            return get_model(
                "qfcn", n_qubits_start=4, n_qubits_l2=2, num_classes=num_classes
            )
        else:
            return get_model("qfcn", num_classes=num_classes)

    elif model_name.lower() in ["cong_qcnn", "cong", "mera"]:
        # Cong QCNN expects 8 pixels (2x4 or 8x1)
        return get_model("cong_qcnn", n_qubits=8, num_classes=num_classes)

    else:
        return get_model(model_name, image_size=image_size, num_classes=num_classes)


def list_models():
    """
    List all available models

    Returns:
        Dictionary with model names and descriptions
    """
    models_info = {
        "qfcn": {
            "name": "QFCN (Quantum Fourier CNN)",
            "paper": "Shen & Liu, 2021",
            "parameters": 14,
            "type": "quantum",
            "description": "Spectral convolution using QFT",
        },
        "cong_qcnn": {
            "name": "Cong QCNN (Original QCNN)",
            "paper": "Cong et al., Nature Physics 2019",
            "parameters": 63,
            "type": "quantum",
            "description": "MERA + QEC architecture",
        },
        "classical": {
            "name": "Classical CNN (Small)",
            "paper": "N/A (baseline)",
            "parameters": "~170",
            "type": "classical",
            "description": "Standard PyTorch CNN for 4x4 images",
        },
        "classical_large": {
            "name": "Classical CNN (Large)",
            "paper": "N/A (baseline)",
            "parameters": "~15K",
            "type": "classical",
            "description": "Deeper CNN for 8x8+ images",
        },
    }

    return models_info


def compare_models(model_names: list = None):
    """
    Compare specifications of multiple models

    Args:
        model_names: List of model names to compare (default: all)

    Returns:
        Comparison table as string
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    models_info = list_models()

    # Build comparison table
    lines = []
    lines.append("=" * 100)
    lines.append(
        f"{'Model':<20} {'Type':<12} {'Parameters':<12} {'Paper':<30} {'Description':<30}"
    )
    lines.append("=" * 100)

    for name in model_names:
        if name in models_info:
            info = models_info[name]
            lines.append(
                f"{info['name']:<20} "
                f"{info['type']:<12} "
                f"{str(info['parameters']):<12} "
                f"{info['paper']:<30} "
                f"{info['description']:<30}"
            )

    lines.append("=" * 100)

    return "\n".join(lines)


# Export all
__all__ = [
    "BaseModel",
    "QFCN",
    "CongQCNN",
    "ClassicalCNN",
    "ClassicalCNN_Large",
    "get_model",
    "get_model_auto",
    "list_models",
    "compare_models",
    "MODEL_REGISTRY",
]


# Quick test
if __name__ == "__main__":
    print("Testing models package...")

    print("\n=== Available Models ===")
    print(compare_models())

    print("\n=== Testing Model Creation ===")

    # Test QFCN
    print("\n1. Creating QFCN...")
    qfcn = get_model("qfcn", num_classes=2)
    print(f"   ✓ {qfcn.__class__.__name__} created")
    print(f"   Parameters: {qfcn.get_num_parameters()}")

    # Test Cong QCNN
    print("\n2. Creating Cong QCNN...")
    cong = get_model("cong_qcnn", num_classes=2)
    print(f"   ✓ {cong.__class__.__name__} created")
    print(f"   Parameters: {cong.get_num_parameters()}")

    # Test Classical CNN
    print("\n3. Creating Classical CNN...")
    classical = get_model("classical", image_size=4, num_classes=2)
    print(f"   ✓ {classical.__class__.__name__} created")
    print(f"   Parameters: {classical.get_num_parameters()}")

    # Test aliases
    print("\n4. Testing aliases...")
    fourier = get_model("fourier", num_classes=2)
    print(f"   ✓ 'fourier' alias -> {fourier.__class__.__name__}")

    baseline = get_model("baseline", image_size=4, num_classes=2)
    print(f"   ✓ 'baseline' alias -> {baseline.__class__.__name__}")

    # Test auto selection
    print("\n5. Testing auto selection...")
    auto_small = get_model_auto("classical", image_size=4, num_classes=2)
    print(f"   ✓ Auto (4x4) -> {auto_small.__class__.__name__}")

    auto_large = get_model_auto("classical", image_size=8, num_classes=2)
    print(f"   ✓ Auto (8x8) -> {auto_large.__class__.__name__}")

    # Test error handling
    print("\n6. Testing error handling...")
    try:
        bad_model = get_model("nonexistent_model")
    except ValueError as e:
        print(f"   ✓ Caught error: {str(e)[:50]}...")

    print("\n✓ Models package test passed!")
