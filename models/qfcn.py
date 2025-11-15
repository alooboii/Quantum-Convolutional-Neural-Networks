"""
Quantum Fourier Convolutional Neural Network (QFCN)
Based on: Shen & Liu, 2021 (arXiv:2106.10421)

Key features:
- Spectral convolution using QFT
- AmplitudeEmbedding for efficient encoding
- 14 trainable parameters (most efficient!)
"""

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp


class QFCN(nn.Module):
    """
    Quantum Fourier Convolutional Network
    
    Architecture:
        16 pixels → AmplitudeEmbedding → 4 qubits
                 → QFT → RZ filters → iQFT
                 → Pooling → 2 qubits
                 → QFT → RZ filters → iQFT
                 → Pooling → 1 qubit
                 → Classical head → 2 classes
    
    Parameters: 14 total
        - Layer 1: 4 (filter) + 2 (pool) = 6
        - Layer 2: 2 (filter) + 2 (pool) = 4
        - Classical: 1*2 + 2 (bias) = 4
    """
    
    def __init__(self, n_qubits_start=4, n_qubits_l2=2, num_classes=2):
        super(QFCN, self).__init__()
        
        self.n_qubits_start = n_qubits_start
        self.n_qubits_l2 = n_qubits_l2
        self.num_classes = num_classes
        
        # Quantum devices
        self.dev_l1 = qml.device("default.qubit", wires=n_qubits_start)
        self.dev_l2 = qml.device("default.qubit", wires=n_qubits_l2)
        
        # Define QNodes
        @qml.qnode(self.dev_l1, interface="torch", diff_method="parameter-shift")
        def qcnn_layer_1(inputs, filter_params, pool_params):
            """Layer 1: 4 qubits -> 2 qubits"""
            # Encoding: 16 pixels -> 4 qubits via AmplitudeEmbedding
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits_start), normalize=True)
            
            # Spectral Convolution: QFT -> RZ filters -> iQFT
            qml.QFT(wires=range(n_qubits_start))
            for i in range(n_qubits_start):
                qml.RZ(filter_params[i], wires=i)
            qml.adjoint(qml.QFT)(wires=range(n_qubits_start))
            
            # Spatial Pooling: 4 qubits -> 2 qubits (keep qubits 1 and 3)
            # Pool (q0, q1) -> keep q1
            qml.CRZ(pool_params[0], wires=[0, 1])
            qml.PauliX(wires=0)
            qml.CRX(pool_params[1], wires=[0, 1])
            qml.PauliX(wires=0)
            
            # Pool (q2, q3) -> keep q3
            qml.CRZ(pool_params[0], wires=[2, 3])
            qml.PauliX(wires=2)
            qml.CRX(pool_params[1], wires=[2, 3])
            qml.PauliX(wires=2)
            
            # Measure qubits 1 and 3
            return qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(3))
        
        @qml.qnode(self.dev_l2, interface="torch", diff_method="parameter-shift")
        def qcnn_layer_2(inputs, filter_params, pool_params):
            """Layer 2: 2 qubits -> 1 qubit"""
            # Encoding: 2 features -> 2 qubits via AngleEmbedding
            qml.AngleEmbedding(features=inputs, wires=range(n_qubits_l2), rotation='Y')
            
            # Spectral Convolution
            qml.QFT(wires=range(n_qubits_l2))
            for i in range(n_qubits_l2):
                qml.RZ(filter_params[i], wires=i)
            qml.adjoint(qml.QFT)(wires=range(n_qubits_l2))
            
            # Spatial Pooling: 2 qubits -> 1 qubit
            qml.CRZ(pool_params[0], wires=[0, 1])
            qml.PauliX(wires=0)
            qml.CRX(pool_params[1], wires=[0, 1])
            qml.PauliX(wires=0)
            
            # Measure qubit 1
            return qml.expval(qml.PauliZ(1))
        
        # Store QNode references
        self._qcnn_layer_1 = qcnn_layer_1
        self._qcnn_layer_2 = qcnn_layer_2
        
        # Create TorchLayers (handles parameters automatically)
        weight_shapes_l1 = {
            "filter_params": (n_qubits_start,),
            "pool_params": (2,)
        }
        weight_shapes_l2 = {
            "filter_params": (n_qubits_l2,),
            "pool_params": (2,)
        }
        
        self.q_layer_1 = qml.qnn.TorchLayer(qcnn_layer_1, weight_shapes_l1)
        self.q_layer_2 = qml.qnn.TorchLayer(qcnn_layer_2, weight_shapes_l2)
        
        # Classical head
        self.classical_head = nn.Linear(1, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 1, 4, 4]
        
        Returns:
            logits: Output tensor [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Flatten: [batch, 1, 4, 4] -> [batch, 16]
        x = torch.flatten(x, start_dim=1)
        
        # Layer 1: [batch, 16] -> [batch, 2]
        x = self.q_layer_1(x)
        
        # Layer 2: [batch, 2] -> [batch, 1]
        x = self.q_layer_2(x)
        
        # Ensure correct shape for Linear layer
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
        # QFT depth is O(n^2) for n qubits, but we use 4 and 2 qubits
        # Layer 1: QFT(4) + RZ(4) + iQFT(4) + Pooling ≈ 2*O(16) + 4 + const
        # Layer 2: QFT(2) + RZ(2) + iQFT(2) + Pooling ≈ 2*O(4) + 2 + const
        return "O(n²) where n=max(4,2)=4"


# Quick test
if __name__ == "__main__":
    print("Testing QFCN implementation...")
    
    # Create model
    model = QFCN(n_qubits_start=4, n_qubits_l2=2, num_classes=2)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 4, 4)
    
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
    
    print("\n✓ QFCN implementation test passed!")