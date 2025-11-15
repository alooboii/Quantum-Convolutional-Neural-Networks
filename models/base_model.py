"""
Abstract Base Model
Defines the interface that all models (quantum and classical) must implement
for unified training, evaluation, and comparison.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models in the benchmark

    All models (QFCN, Cong QCNN, Classical CNN, etc.) must inherit from this
    and implement the required methods.

    This ensures:
    - Unified interface for training
    - Consistent evaluation metrics
    - Fair comparison across models
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            x: Input tensor [batch_size, channels, height, width]

        Returns:
            logits: Output tensor [batch_size, num_classes]
        """
        pass

    @abstractmethod
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters

        Returns:
            Total count of trainable parameters
        """
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """
        Get model type identifier

        Returns:
            One of: 'quantum', 'classical', 'hybrid'
        """
        pass

    @abstractmethod
    def get_circuit_depth(self) -> str:
        """
        Get circuit depth (for quantum models) or 'N/A' for classical

        Returns:
            Circuit depth description (e.g., 'O(log n)', 'N/A')
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information

        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.__class__.__name__,
            "type": self.get_model_type(),
            "num_parameters": self.get_num_parameters(),
            "circuit_depth": self.get_circuit_depth(),
            "framework": self._get_framework(),
        }

    def _get_framework(self) -> str:
        """
        Detect which framework the model uses

        Returns:
            Framework name ('pennylane', 'qiskit', 'pytorch', etc.)
        """
        if hasattr(self, "dev") or hasattr(self, "q_layer"):
            # Check for PennyLane
            try:
                import pennylane as qml

                if hasattr(self, "dev") and isinstance(self.dev, qml.Device):
                    return "pennylane"
                if hasattr(self, "q_layer") and isinstance(
                    self.q_layer, qml.qnn.TorchLayer
                ):
                    return "pennylane"
            except:
                pass

            # Check for Qiskit
            try:
                from qiskit import QuantumCircuit

                if hasattr(self, "circuit") and isinstance(
                    self.circuit, QuantumCircuit
                ):
                    return "qiskit"
            except:
                pass

            return "quantum (unknown)"
        else:
            return "pytorch"

    def count_parameters_by_layer(self) -> Dict[str, int]:
        """
        Count parameters for each named module

        Returns:
            Dictionary mapping layer names to parameter counts
        """
        param_dict = {}
        for name, module in self.named_modules():
            if name == "":  # Skip the model itself
                continue
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if num_params > 0:
                param_dict[name] = num_params
        return param_dict

    def get_parameter_efficiency(self) -> float:
        """
        Calculate parameter efficiency metric

        Returns:
            Parameters per output dimension (lower is more efficient)
        """
        # Get output dimension from a dummy forward pass
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 4, 4)  # Assume 4x4 input
                output = self.forward(dummy_input)
                output_dim = output.shape[-1]

            return self.get_num_parameters() / output_dim
        except:
            return float("inf")

    def summary(self, input_size: tuple = (1, 1, 4, 4), device: str = "cpu") -> str:
        """
        Print model summary similar to Keras model.summary()

        Args:
            input_size: Size of input tensor (batch, channels, height, width)
            device: Device to run summary on

        Returns:
            Summary string
        """
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append(f"Model: {self.__class__.__name__}")
        summary_lines.append("=" * 80)

        # Model info
        info = self.get_model_info()
        summary_lines.append(f"Type: {info['type']}")
        summary_lines.append(f"Framework: {info['framework']}")
        summary_lines.append(f"Total Parameters: {info['num_parameters']:,}")
        summary_lines.append(f"Circuit Depth: {info['circuit_depth']}")
        summary_lines.append("-" * 80)

        # Layer-wise parameters
        summary_lines.append("Layer-wise Parameters:")
        param_dict = self.count_parameters_by_layer()
        for name, count in param_dict.items():
            summary_lines.append(f"  {name:40s} {count:>10,}")

        summary_lines.append("=" * 80)

        # Test forward pass
        try:
            with torch.no_grad():
                test_input = torch.randn(*input_size).to(device)
                test_output = self.forward(test_input)
            summary_lines.append(f"Input shape:  {tuple(test_input.shape)}")
            summary_lines.append(f"Output shape: {tuple(test_output.shape)}")
        except Exception as e:
            summary_lines.append(f"Forward pass test failed: {e}")

        summary_lines.append("=" * 80)

        return "\n".join(summary_lines)

    def save(self, path: str):
        """
        Save model state dict

        Args:
            path: Path to save model
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_info": self.get_model_info(),
            },
            path,
        )

    def load(self, path: str, device: str = "cpu"):
        """
        Load model state dict

        Args:
            path: Path to load model from
            device: Device to load model to
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("model_info", {})


class DummyModel(BaseModel):
    """
    Dummy model for testing the base class
    """

    def __init__(self, num_classes=2):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_type(self):
        return "classical"

    def get_circuit_depth(self):
        return "N/A"


# Quick test
if __name__ == "__main__":
    print("Testing BaseModel abstract class...")

    # Test with dummy model
    model = DummyModel(num_classes=2)

    print("\n=== Model Info ===")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    print("\n=== Layer-wise Parameters ===")
    param_dict = model.count_parameters_by_layer()
    for name, count in param_dict.items():
        print(f"{name}: {count}")

    print("\n=== Parameter Efficiency ===")
    efficiency = model.get_parameter_efficiency()
    print(f"Parameters per output dim: {efficiency:.2f}")

    print("\n=== Model Summary ===")
    print(model.summary())

    print("\n=== Save/Load Test ===")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pth")

        # Save
        model.save(save_path)
        print(f"✓ Model saved to {save_path}")

        # Load
        model_loaded = DummyModel(num_classes=2)
        loaded_info = model_loaded.load(save_path)
        print(f"✓ Model loaded from {save_path}")
        print(f"  Loaded info: {loaded_info}")

    print("\n✓ BaseModel abstract class test passed!")
