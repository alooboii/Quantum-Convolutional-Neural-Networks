"""
Unified Trainer
Provides a consistent training loop for all models (quantum and classical)
with comprehensive logging, checkpointing, and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable
import time
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm


class Trainer:
    """
    Unified trainer for quantum and classical models

    Features:
    - Automatic device selection (CPU/GPU)
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Training history tracking
    - Progress bars
    - Kaggle environment detection
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        device: str = "auto",
        lr: float = 0.001,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_dir: str = "./results/checkpoints",
        model_name: str = "model",
        verbose: bool = True,
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Test/validation data loader
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            device: Device to train on ('auto', 'cpu', 'cuda')
            lr: Learning rate (used if optimizer is None)
            scheduler: Learning rate scheduler (optional)
            save_dir: Directory to save checkpoints
            model_name: Name for saving checkpoints
            verbose: Whether to print progress
        """
        # Device setup
        if device == "auto":
            # Check if we're on Kaggle
            self.device = self._detect_device()
        else:
            self.device = torch.device(device)

        # Model setup
        self.model = model.to(self.device)
        self.model_name = model_name

        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Loss function
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        # Scheduler
        self.scheduler = scheduler

        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_loss = float("inf")
        self.verbose = verbose

        # History tracking
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "learning_rate": [],
            "epoch_time": [],
        }

        # Metrics
        self.total_train_time = 0.0
        self.convergence_epoch = None  # Epoch when 90% accuracy reached

        if self.verbose:
            print(f"Trainer initialized:")
            print(f"  Device: {self.device}")
            print(f"  Model: {model.__class__.__name__}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Test batches: {len(test_loader)}")

    def _detect_device(self) -> torch.device:
        """Detect best available device"""
        # Check for Kaggle environment
        if os.path.exists("/kaggle"):
            # On Kaggle, prefer CPU for quantum simulations
            return torch.device("cpu")

        # Otherwise use GPU if available
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if self.verbose:
            pbar = tqdm(
                self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]"
            )
        else:
            pbar = self.train_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            if self.verbose and isinstance(pbar, tqdm):
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
                )

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on test set

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        if self.verbose:
            pbar = tqdm(self.test_loader, desc=f"Epoch {self.current_epoch + 1} [Test]")
        else:
            pbar = self.test_loader

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if self.verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{100.*correct/total:.2f}%",
                        }
                    )

        avg_loss = running_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def train(
        self,
        epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best: bool = True,
        save_every: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Train the model for multiple epochs

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs (None = disabled)
            save_best: Save checkpoint when validation improves
            save_every: Save checkpoint every N epochs (None = disabled)

        Returns:
            Training history dictionary
        """
        if self.verbose:
            print(f"\nStarting training for {epochs} epochs...")
            print("=" * 80)

        start_time = time.time()
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Evaluate
            test_metrics = self.evaluate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Record metrics
            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["test_loss"].append(test_metrics["loss"])
            self.history["test_accuracy"].append(test_metrics["accuracy"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            self.history["epoch_time"].append(epoch_time)

            # Check for convergence (first time reaching 90% test accuracy)
            if self.convergence_epoch is None and test_metrics["accuracy"] >= 90.0:
                self.convergence_epoch = epoch + 1

            # Print summary
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                    f"Test Loss: {test_metrics['loss']:.4f} | "
                    f"Test Acc: {test_metrics['accuracy']:.2f}% | "
                    f"Time: {epoch_time:.1f}s"
                )

            # Save best model
            if save_best and test_metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = test_metrics["accuracy"]
                self.best_loss = test_metrics["loss"]
                self.save_checkpoint(f"{self.model_name}_best.pth")
                epochs_without_improvement = 0

                if self.verbose:
                    print(f"  → New best accuracy: {self.best_accuracy:.2f}%")
            else:
                epochs_without_improvement += 1

            # Save periodic checkpoint
            if save_every is not None and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"{self.model_name}_epoch_{epoch + 1}.pth")

            # Early stopping
            if early_stopping_patience is not None:
                if epochs_without_improvement >= early_stopping_patience:
                    if self.verbose:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        print(f"Best accuracy: {self.best_accuracy:.2f}%")
                    break

        self.total_train_time = time.time() - start_time

        if self.verbose:
            print("=" * 80)
            print(f"Training completed in {self.total_train_time:.1f}s")
            print(f"Best test accuracy: {self.best_accuracy:.2f}%")
            if self.convergence_epoch:
                print(f"Converged (90% acc) at epoch: {self.convergence_epoch}")

        return self.history

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.save_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "history": self.history,
            "model_name": self.model_name,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        if self.verbose:
            print(f"  → Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.save_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.verbose:
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"  Epoch: {self.current_epoch}")
            print(f"  Best accuracy: {self.best_accuracy:.2f}%")

    def save_history(self, filename: str = None):
        """
        Save training history to JSON

        Args:
            filename: Name of JSON file (default: {model_name}_history.json)
        """
        if filename is None:
            filename = f"{self.model_name}_history.json"

        history_path = self.save_dir / filename

        # Add metadata
        history_with_metadata = {
            "history": self.history,
            "metadata": {
                "model_name": self.model_name,
                "total_epochs": self.current_epoch + 1,
                "best_accuracy": self.best_accuracy,
                "best_loss": self.best_loss,
                "total_train_time": self.total_train_time,
                "convergence_epoch": self.convergence_epoch,
                "device": str(self.device),
            },
        }

        with open(history_path, "w") as f:
            json.dump(history_with_metadata, f, indent=2)

        if self.verbose:
            print(f"History saved: {history_path}")

    def get_summary(self) -> Dict:
        """
        Get training summary statistics

        Returns:
            Dictionary with summary stats
        """
        return {
            "model_name": self.model_name,
            "total_epochs": self.current_epoch + 1,
            "best_test_accuracy": self.best_accuracy,
            "best_test_loss": self.best_loss,
            "final_train_accuracy": (
                self.history["train_accuracy"][-1]
                if self.history["train_accuracy"]
                else 0.0
            ),
            "final_test_accuracy": (
                self.history["test_accuracy"][-1]
                if self.history["test_accuracy"]
                else 0.0
            ),
            "total_train_time": self.total_train_time,
            "avg_epoch_time": (
                np.mean(self.history["epoch_time"])
                if self.history["epoch_time"]
                else 0.0
            ),
            "convergence_epoch": self.convergence_epoch,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }


# Quick test
if __name__ == "__main__":
    print("Testing Trainer...")

    # Create dummy model and data
    from torch.utils.data import TensorDataset

    # Dummy data
    X_train = torch.randn(100, 1, 4, 4)
    y_train = torch.randint(0, 2, (100,))
    X_test = torch.randn(20, 1, 4, 4)
    y_test = torch.randint(0, 2, (20,))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 2)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    model = DummyModel()

    # Test trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=0.01,
        save_dir="./test_results",
        model_name="test_model",
        verbose=True,
    )

    # Train
    history = trainer.train(epochs=3, early_stopping_patience=5, save_best=True)

    # Get summary
    summary = trainer.get_summary()
    print("\n=== Training Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Save history
    trainer.save_history()

    print("\n✓ Trainer test passed!")
