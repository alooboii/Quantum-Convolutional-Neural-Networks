"""
Cong QCNN - Original Quantum Convolutional Neural Network
Based on: Cong, Choi & Lukin, Nature Physics 2019
Implementation adapted from Qiskit tutorial

Key features:
- MERA (Multi-scale Entanglement Renormalization Ansatz) reversed
- Quantum Error Correction inspired pooling
- Parametric 2-qubit unitaries N(α,β,γ)
- 63 parameters for 8 qubits

Architecture:
    8 pixels → ZFeatureMap → 8 qubits
           → Conv → Pool → 4 qubits (discard 0,1,2,3)
           → Conv → Pool → 2 qubits (discard 4,5)
           → Conv → Pool → 1 qubit (discard 6)
           → Measure Z → 2 classes
"""

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


class CongQCNN(nn.Module):
    """
    Cong QCNN implementation using PennyLane

    Original paper uses Qiskit, but we adapt to PennyLane for consistency.
    Uses parametric 2-qubit unitaries based on N(α,β,γ) = exp(i[α σ_x⊗σ_x + β σ_y⊗σ_y + γ σ_z⊗σ_z])

    Parameters: 63 total
        - Conv1 (8 qubits): 24 params (8 gates × 3 params)
        - Pool1 (8→4): 12 params (4 gates × 3 params)
        - Conv2 (4 qubits): 12 params
        - Pool2 (4→2): 6 params
        - Conv3 (2 qubits): 6 params
        - Pool3 (2→1): 3 params
    """

    def __init__(self, n_qubits=8, num_classes=2):
        super(CongQCNN, self).__init__()

        self.n_qubits = n_qubits
        self.num_classes = num_classes

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build the QCNN circuit
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def qcnn_circuit(inputs, params):
            """
            Full QCNN circuit

            Args:
                inputs: 8 pixel values
                params: All trainable parameters (63 total)

            Returns:
                Expectation value of Z on final qubit
            """
            param_idx = 0

            # === ENCODING: ZFeatureMap ===
            # Simple angle encoding (similar to Qiskit's ZFeatureMap)
            for i in range(n_qubits):
                qml.RZ(inputs[i], wires=i)

            # === LAYER 1: Conv + Pool (8 → 4 qubits) ===
            # Convolutional Layer 1
            param_idx = self._conv_layer(n_qubits, params, param_idx)

            # Pooling Layer 1 (keep qubits 4,5,6,7; discard 0,1,2,3)
            sources = [0, 1, 2, 3]
            sinks = [4, 5, 6, 7]
            param_idx = self._pool_layer(sources, sinks, params, param_idx)

            # === LAYER 2: Conv + Pool (4 → 2 qubits) ===
            # Convolutional Layer 2 (only on qubits 4,5,6,7)
            param_idx = self._conv_layer(4, params, param_idx, offset=4)

            # Pooling Layer 2 (keep qubits 6,7; discard 4,5)
            sources = [4, 5]
            sinks = [6, 7]
            param_idx = self._pool_layer(sources, sinks, params, param_idx)

            # === LAYER 3: Conv + Pool (2 → 1 qubit) ===
            # Convolutional Layer 3 (only on qubits 6,7)
            param_idx = self._conv_layer(2, params, param_idx, offset=6)

            # Pooling Layer 3 (keep qubit 7; discard 6)
            sources = [6]
            sinks = [7]
            param_idx = self._pool_layer(sources, sinks, params, param_idx)

            # === MEASUREMENT ===
            # Measure Z expectation on final qubit (qubit 7)
            return qml.expval(qml.PauliZ(7))

        # Store circuit
        self._qcnn_circuit = qcnn_circuit

        # Calculate total parameters
        # Conv1: 8 qubits → (4 even pairs + 4 odd pairs) × 3 = 24
        # Pool1: 4 pairs × 3 = 12
        # Conv2: 4 qubits → (2 even + 2 odd) × 3 = 12
        # Pool2: 2 pairs × 3 = 6
        # Conv3: 2 qubits → (1 even + 1 odd) × 3 = 6
        # Pool3: 1 pair × 3 = 3
        # Total: 24 + 12 + 12 + 6 + 6 + 3 = 63

        total_params = 63

        # Create TorchLayer
        weight_shapes = {"params": (total_params,)}
        self.q_layer = qml.qnn.TorchLayer(qcnn_circuit, weight_shapes)

        # Classical head: Map single measurement to 2 classes
        # Use a simple linear layer
        self.classical_head = nn.Linear(1, num_classes)

    def _two_qubit_unitary(self, params, wires):
        """
        Parametric 2-qubit unitary N(α,β,γ)

        Circuit from Vatan & Williams (2004):
        Based on exp(i[α σ_x⊗σ_x + β σ_y⊗σ_y + γ σ_z⊗σ_z])

        Decomposition:
            RZ(-π/2) on qubit 1
            CNOT(1,0)
            RZ(θ₀) on qubit 0
            RY(θ₁) on qubit 1
            CNOT(0,1)
            RY(θ₂) on qubit 1
            CNOT(1,0)
            RZ(π/2) on qubit 0
        """
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[2], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(np.pi / 2, wires=wires[0])

    def _conv_layer(self, num_qubits, params, param_idx, offset=0):
        """
        Convolutional layer: Apply 2-qubit unitaries to neighboring pairs

        Args:
            num_qubits: Number of qubits in this layer
            params: All parameters
            param_idx: Current parameter index
            offset: Qubit offset (for layers operating on subset)

        Returns:
            Updated param_idx
        """
        qubits = list(range(offset, offset + num_qubits))

        # Even pairs: (0,1), (2,3), (4,5), (6,7)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            self._two_qubit_unitary(params[param_idx : param_idx + 3], [q1, q2])
            param_idx += 3

        # Odd pairs with circular: (1,2), (3,4), (5,6), (7,0)
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [qubits[0]]):
            self._two_qubit_unitary(params[param_idx : param_idx + 3], [q1, q2])
            param_idx += 3

        return param_idx

    def _pool_layer(self, sources, sinks, params, param_idx):
        """
        Pooling layer: Apply 2-qubit unitaries between source and sink pairs
        Then "discard" source qubits (stop operating on them)

        Args:
            sources: Qubits to be "discarded"
            sinks: Qubits to keep
            params: All parameters
            param_idx: Current parameter index

        Returns:
            Updated param_idx
        """
        for source, sink in zip(sources, sinks):
            # Similar to conv but without final CNOT (asymmetric pooling)
            qml.RZ(-np.pi / 2, wires=sink)
            qml.CNOT(wires=[sink, source])
            qml.RZ(params[param_idx], wires=source)
            qml.RY(params[param_idx + 1], wires=sink)
            qml.CNOT(wires=[source, sink])
            qml.RY(params[param_idx + 2], wires=sink)
            # Note: No final CNOT and RZ - this creates asymmetry for pooling

            param_idx += 3

        return param_idx

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 1, H, W]
               For 8 qubits, expects 8 pixels (could be 2x4 or 8x1)

        Returns:
            logits: Output tensor [batch_size, num_classes]
        """
        batch_size = x.shape[0]

        # Flatten to 8 pixels
        x = torch.flatten(x, start_dim=1)

        # Ensure we have 8 pixels (pad or truncate if needed)
        if x.shape[1] < self.n_qubits:
            # Pad with zeros
            padding = torch.zeros(
                batch_size, self.n_qubits - x.shape[1], device=x.device
            )
            x = torch.cat([x, padding], dim=1)
        elif x.shape[1] > self.n_qubits:
            # Truncate or resize
            x = x[:, : self.n_qubits]

        # Quantum circuit: [batch, 8] -> [batch, 1]
        x = self.q_layer(x)

        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [batch] -> [batch, 1]

        # Classical head: [batch, 1] -> [batch, num_classes]
        logits = self.classical_head(x)

        return logits

    def get_num_parameters(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_type(self):
        """Return model type identifier"""
        return "quantum"

    def get_circuit_depth(self):
        """Return theoretical circuit depth"""
        # Each conv layer: 2 passes × (num_qubits) gates
        # Each pool layer: num_pairs gates
        # Approximate depth is sum of all gates in sequence
        return "O(log n) via MERA structure"


# Quick test
if __name__ == "__main__":
    print("Testing Cong QCNN implementation...")

    # Create model
    model = CongQCNN(n_qubits=8, num_classes=2)

    # Test with 2x4 images (8 pixels)
    batch_size = 4
    x = torch.randn(batch_size, 1, 2, 4)

    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_parameters()}")
    print(f"Model type: {model.get_model_type()}")
    print(f"Circuit depth: {model.get_circuit_depth()}")

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")

    # Test with different input sizes
    print("\n=== Testing input flexibility ===")
    x_4x4 = torch.randn(2, 1, 4, 4)  # 16 pixels, will truncate to 8
    output_4x4 = model(x_4x4)
    print(f"4×4 input -> output shape: {output_4x4.shape}")

    x_8x1 = torch.randn(2, 1, 8, 1)  # 8 pixels, perfect fit
    output_8x1 = model(x_8x1)
    print(f"8×1 input -> output shape: {output_8x1.shape}")

    print("\n✓ Cong QCNN implementation test passed!")
