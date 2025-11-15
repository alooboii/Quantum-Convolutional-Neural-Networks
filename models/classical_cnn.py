"""
Classical Convolutional Neural Network Baseline
Standard PyTorch CNN for performance comparison

Architecture:
    Input (4x4) → Conv2D(8) → ReLU → Conv2D(4) → ReLU → Flatten → FC(2)

Parameters: ~170 (much more than quantum models)
Purpose: Establish performance ceiling and efficiency baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalCNN(nn.Module):
    """
    Classical CNN baseline for MNIST classification

    Architecture designed to match quantum model complexity while
    providing a fair classical comparison.
    """

    def __init__(self, input_channels=1, num_classes=2, image_size=4):
        super(ClassicalCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=8,
            kernel_size=2,
            stride=1,
            padding=0,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=2, stride=1, padding=0
        )

        # Calculate size after convolutions
        # After conv1: (4-2+1) = 3x3
        # After conv2: (3-2+1) = 2x2
        conv_output_size = 2 * 2 * 4  # 16

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 1, 4, 4]

        Returns:
            logits: Output tensor [batch_size, num_classes]
        """
        # Conv layer 1: [batch, 1, 4, 4] -> [batch, 8, 3, 3]
        x = self.conv1(x)
        x = F.relu(x)

        # Conv layer 2: [batch, 8, 3, 3] -> [batch, 4, 2, 2]
        x = self.conv2(x)
        x = F.relu(x)

        # Flatten: [batch, 4, 2, 2] -> [batch, 16]
        x = torch.flatten(x, start_dim=1)

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x

    def get_num_parameters(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_type(self):
        """Return model type identifier"""
        return "classical"

    def get_circuit_depth(self):
        """Return circuit depth (N/A for classical)"""
        return "N/A (classical)"


class ClassicalCNN_Large(nn.Module):
    """
    Larger classical CNN for 8x8 or larger images
    """

    def __init__(self, input_channels=1, num_classes=2, image_size=8):
        super(ClassicalCNN_Large, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size

        # Convolutional layers with batch norm
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size
        # After 3 pooling layers: image_size / (2^3)
        final_size = image_size // 8
        if final_size < 1:
            final_size = 1
        conv_output_size = 64 * final_size * final_size

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten and FC layers
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_type(self):
        return "classical"

    def get_circuit_depth(self):
        return "N/A (classical)"


# Factory function
def get_classical_cnn(image_size=4, num_classes=2):
    """
    Factory function to get appropriate classical CNN based on image size

    Args:
        image_size: Size of input images
        num_classes: Number of output classes

    Returns:
        ClassicalCNN or ClassicalCNN_Large instance
    """
    if image_size <= 4:
        return ClassicalCNN(
            input_channels=1, num_classes=num_classes, image_size=image_size
        )
    else:
        return ClassicalCNN_Large(
            input_channels=1, num_classes=num_classes, image_size=image_size
        )


# Quick test
if __name__ == "__main__":
    print("Testing Classical CNN implementation...")

    # Test small CNN (4x4 images)
    print("\n=== Testing ClassicalCNN (4x4) ===")
    model_small = ClassicalCNN(num_classes=2, image_size=4)
    x_small = torch.randn(4, 1, 4, 4)

    print(f"Input shape: {x_small.shape}")
    output_small = model_small(x_small)
    print(f"Output shape: {output_small.shape}")
    print(f"Total parameters: {model_small.get_num_parameters()}")

    # Test backward pass
    loss = output_small.sum()
    loss.backward()
    print("✓ Backward pass successful")

    # Test large CNN (8x8 images)
    print("\n=== Testing ClassicalCNN_Large (8x8) ===")
    model_large = ClassicalCNN_Large(num_classes=2, image_size=8)
    x_large = torch.randn(4, 1, 8, 8)

    print(f"Input shape: {x_large.shape}")
    output_large = model_large(x_large)
    print(f"Output shape: {output_large.shape}")
    print(f"Total parameters: {model_large.get_num_parameters()}")

    # Test factory function
    print("\n=== Testing Factory Function ===")
    model_auto_small = get_classical_cnn(image_size=4, num_classes=2)
    model_auto_large = get_classical_cnn(image_size=8, num_classes=2)
    print(f"4x4 model type: {type(model_auto_small).__name__}")
    print(f"8x8 model type: {type(model_auto_large).__name__}")

    print("\n✓ Classical CNN implementation test passed!")
