"""
Model Evaluator
Comprehensive evaluation, metrics computation, and visualization
for comparing quantum and classical models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison

    Features:
    - Accuracy, precision, recall, F1
    - Confusion matrices
    - Training curve visualization
    - Model comparison tables
    - Inference time measurement
    - Parameter efficiency analysis
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator

        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            device: Device to evaluate on
            class_names: Names of classes for visualization
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.class_names = class_names

        # Results cache
        self.predictions = None
        self.targets = None
        self.probabilities = None

    def evaluate(self, verbose: bool = True) -> Dict[str, float]:
        """
        Comprehensive evaluation

        Args:
            verbose: Print results

        Returns:
            Dictionary with all metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Accumulate metrics
                total_loss += loss.item()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Cache results
        self.predictions = np.array(all_predictions)
        self.targets = np.array(all_targets)
        self.probabilities = np.array(all_probabilities)

        # Calculate metrics
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.test_loader)

        # Per-class metrics
        num_classes = len(np.unique(self.targets))
        per_class_acc = []

        for cls in range(num_classes):
            cls_mask = self.targets == cls
            cls_correct = (self.predictions[cls_mask] == cls).sum()
            cls_total = cls_mask.sum()
            if cls_total > 0:
                per_class_acc.append(100.0 * cls_correct / cls_total)
            else:
                per_class_acc.append(0.0)

        results = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "per_class_accuracy": per_class_acc,
            "num_samples": total,
            "num_correct": correct,
        }

        if verbose:
            print("=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print(f"Overall Accuracy: {accuracy:.2f}%")
            print(f"Average Loss:     {avg_loss:.4f}")
            print(f"Total Samples:    {total}")
            print(f"Correct:          {correct}")

            if self.class_names:
                print("\nPer-Class Accuracy:")
                for i, (name, acc) in enumerate(zip(self.class_names, per_class_acc)):
                    print(f"  {name}: {acc:.2f}%")

            print("=" * 60)

        return results

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix

        Returns:
            Confusion matrix as numpy array
        """
        if self.predictions is None:
            self.evaluate(verbose=False)

        return confusion_matrix(self.targets, self.predictions)

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        normalize: bool = False,
    ):
        """
        Plot confusion matrix

        Args:
            save_path: Path to save figure
            figsize: Figure size
            normalize: Normalize by true labels
        """
        cm = self.get_confusion_matrix()

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=self.class_names if self.class_names else "auto",
            yticklabels=self.class_names if self.class_names else "auto",
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_classification_report(self) -> str:
        """
        Get scikit-learn classification report

        Returns:
            Classification report string
        """
        if self.predictions is None:
            self.evaluate(verbose=False)

        return classification_report(
            self.targets, self.predictions, target_names=self.class_names
        )

    def measure_inference_time(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Measure inference time

        Args:
            num_samples: Number of samples to test

        Returns:
            Dictionary with timing metrics
        """
        import time

        self.model.eval()

        # Get sample batch
        sample_input, _ = next(iter(self.test_loader))
        sample_input = sample_input[:num_samples].to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(sample_input)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                _ = self.model(sample_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        times = np.array(times)

        return {
            "mean_time": times.mean(),
            "std_time": times.std(),
            "min_time": times.min(),
            "max_time": times.max(),
            "samples_per_second": num_samples / times.mean(),
        }

    def plot_prediction_samples(
        self,
        num_samples: int = 25,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 12),
    ):
        """
        Plot sample predictions

        Args:
            num_samples: Number of samples to show
            save_path: Path to save figure
            figsize: Figure size
        """
        if self.predictions is None:
            self.evaluate(verbose=False)

        # Get sample images
        sample_images, sample_targets = next(iter(self.test_loader))
        num_samples = min(num_samples, len(sample_images))

        # Get predictions for these samples
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sample_images[:num_samples].to(self.device))
            _, sample_predictions = outputs.max(1)
            sample_predictions = sample_predictions.cpu().numpy()

        # Plot
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()

        for idx in range(num_samples):
            img = sample_images[idx].squeeze().cpu().numpy()

            # Denormalize
            if img.min() < 0:
                img = img * 0.5 + 0.5

            true_label = sample_targets[idx].item()
            pred_label = sample_predictions[idx]

            axes[idx].imshow(img, cmap="gray")

            # Color: green if correct, red if wrong
            color = "green" if true_label == pred_label else "red"

            if self.class_names:
                title = f"T: {self.class_names[true_label]}\nP: {self.class_names[pred_label]}"
            else:
                title = f"True: {true_label}\nPred: {pred_label}"

            axes[idx].set_title(title, color=color, fontsize=8)
            axes[idx].axis("off")

        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Prediction samples saved to {save_path}")
        else:
            plt.show()

        plt.close()


class MultiModelComparator:
    """
    Compare multiple models on the same test set
    """

    def __init__(self, test_loader: DataLoader, device: str = "cpu"):
        """
        Initialize comparator

        Args:
            test_loader: Test data loader
            device: Device to evaluate on
        """
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.results = {}

    def add_model(self, name: str, model: nn.Module, history: Optional[Dict] = None):
        """
        Add a model to compare

        Args:
            name: Model identifier
            model: Trained model
            history: Training history (optional)
        """
        evaluator = ModelEvaluator(model, self.test_loader, str(self.device))
        metrics = evaluator.evaluate(verbose=False)

        # Get model info
        model_info = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "model_type": (
                model.get_model_type()
                if hasattr(model, "get_model_type")
                else "unknown"
            ),
            "evaluator": evaluator,
        }

        # Add history if provided
        if history:
            model_info["history"] = history
            model_info["best_train_acc"] = max(history.get("train_accuracy", [0]))
            model_info["best_test_acc"] = max(history.get("test_accuracy", [0]))
            model_info["total_train_time"] = sum(history.get("epoch_time", [0]))

        self.results[name] = model_info

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table as pandas DataFrame

        Returns:
            DataFrame with model comparison
        """
        data = []

        for name, info in self.results.items():
            row = {
                "Model": name,
                "Type": info["model_type"],
                "Parameters": info["num_parameters"],
                "Test Accuracy (%)": f"{info['accuracy']:.2f}",
                "Test Loss": f"{info['loss']:.4f}",
            }

            if "total_train_time" in info:
                row["Train Time (s)"] = f"{info['total_train_time']:.1f}"

            if "best_test_acc" in info:
                row["Best Test Acc (%)"] = f"{info['best_test_acc']:.2f}"

            data.append(row)

        df = pd.DataFrame(data)
        return df

    def print_comparison(self):
        """Print comparison table"""
        df = self.get_comparison_table()
        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

    def plot_training_curves(
        self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Plot training curves for all models

        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for name, info in self.results.items():
            if "history" in info:
                history = info["history"]
                epochs = range(1, len(history["train_loss"]) + 1)

                # Loss
                ax1.plot(
                    epochs,
                    history["train_loss"],
                    label=f"{name} (train)",
                    linestyle="--",
                )
                ax1.plot(epochs, history["test_loss"], label=f"{name} (test)")

                # Accuracy
                ax2.plot(
                    epochs,
                    history["train_accuracy"],
                    label=f"{name} (train)",
                    linestyle="--",
                )
                ax2.plot(epochs, history["test_accuracy"], label=f"{name} (test)")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training Accuracy Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrices(
        self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Plot confusion matrices for all models side by side

        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        num_models = len(self.results)
        fig, axes = plt.subplots(1, num_models, figsize=figsize)

        if num_models == 1:
            axes = [axes]

        for ax, (name, info) in zip(axes, self.results.items()):
            evaluator = info["evaluator"]
            cm = evaluator.get_confusion_matrix()

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(f'{name}\nAcc: {info["accuracy"]:.2f}%')
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrices saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def save_comparison_report(self, save_dir: str):
        """
        Save comprehensive comparison report

        Args:
            save_dir: Directory to save report
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        df = self.get_comparison_table()
        df.to_csv(save_dir / "comparison_table.csv", index=False)

        # Save detailed results
        detailed_results = {}
        for name, info in self.results.items():
            detailed_results[name] = {
                "accuracy": float(info["accuracy"]),
                "loss": float(info["loss"]),
                "num_parameters": int(info["num_parameters"]),
                "model_type": info["model_type"],
            }

            if "history" in info:
                detailed_results[name]["history"] = info["history"]

        with open(save_dir / "detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Generate plots
        self.plot_training_curves(save_path=str(save_dir / "training_curves.png"))
        self.plot_confusion_matrices(save_path=str(save_dir / "confusion_matrices.png"))

        print(f"\nComparison report saved to {save_dir}")


# Quick test
if __name__ == "__main__":
    print("Testing Evaluator...")

    # Create dummy model and data
    from torch.utils.data import TensorDataset, DataLoader

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 2)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

        def get_model_type(self):
            return "classical"

    # Dummy data
    X_test = torch.randn(50, 1, 4, 4)
    y_test = torch.randint(0, 2, (50,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Test single model evaluation
    print("\n=== Testing Single Model Evaluation ===")
    model = DummyModel()
    evaluator = ModelEvaluator(model, test_loader, class_names=["0", "1"])

    results = evaluator.evaluate()
    print("\nConfusion Matrix:")
    print(evaluator.get_confusion_matrix())

    print("\nClassification Report:")
    print(evaluator.get_classification_report())

    # Test multi-model comparison
    print("\n=== Testing Multi-Model Comparison ===")
    comparator = MultiModelComparator(test_loader)

    model1 = DummyModel()
    model2 = DummyModel()

    dummy_history = {
        "train_loss": [0.5, 0.3, 0.2],
        "test_loss": [0.6, 0.4, 0.3],
        "train_accuracy": [70, 80, 85],
        "test_accuracy": [65, 75, 80],
        "epoch_time": [1.0, 1.0, 1.0],
    }

    comparator.add_model("Model 1", model1, dummy_history)
    comparator.add_model("Model 2", model2, dummy_history)

    comparator.print_comparison()

    print("\n✓ Evaluator test passed!")
