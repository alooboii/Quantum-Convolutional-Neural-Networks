"""
Benchmark All Models
Main script to train and compare all models on MNIST
"""

import sys
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models import get_model, compare_models
from data.mnist_loader import load_mnist_data, print_dataset_info
from training import Trainer, MultiModelComparator
from training.utils import set_seed, get_device, create_output_dirs, format_time
import time


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark all QCNN models on MNIST")

    # Dataset
    parser.add_argument(
        "--image-size", type=int, default=4, help="Image size (4, 8, 10, etc.)"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Classes to include (default: 0 1)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Limit training samples (None = all)",
    )
    parser.add_argument(
        "--test-samples", type=int, default=None, help="Limit test samples (None = all)"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Early stopping patience (None = disabled)",
    )

    # Models
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["qfcn", "classical", "cong_qcnn"],
        help="Models to benchmark",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu/cuda/auto)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("QUANTUM CNN BENCHMARK - MNIST Comparison")
    print("=" * 80)

    # Set seed
    set_seed(args.seed)
    print(f"\nRandom seed set to: {args.seed}")

    # Create output directories
    output_dirs = create_output_dirs(args.output_dir)
    print(f"Output directory: {args.output_dir}")

    # Device
    device = get_device(prefer_cpu=(args.device == "cpu"))
    print(f"Device: {device}")

    # Load data
    print("\n" + "-" * 80)
    print("Loading MNIST dataset...")
    print("-" * 80)

    train_loader, test_loader, data_info = load_mnist_data(
        image_size=args.image_size,
        batch_size=args.batch_size,
        classes=args.classes,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )

    print_dataset_info(data_info)

    # Available models info
    print("\n" + "-" * 80)
    print("Available Models:")
    print("-" * 80)
    print(compare_models(args.models))

    # Training configuration
    print("\n" + "-" * 80)
    print("Training Configuration:")
    print("-" * 80)
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Learning Rate:   {args.lr}")
    print(
        f"Early Stopping:  {args.early_stopping if args.early_stopping else 'Disabled'}"
    )
    print("-" * 80)

    # Train each model
    all_results = {}
    comparator = MultiModelComparator(test_loader, device=str(device))

    for model_name in args.models:
        print("\n" + "=" * 80)
        print(f"Training: {model_name.upper()}")
        print("=" * 80)

        try:
            # Create model
            model = get_model(
                model_name,
                num_classes=data_info["num_classes"],
                image_size=args.image_size,
            )

            print(f"\nModel: {model.__class__.__name__}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                lr=args.lr,
                device=str(device),
                save_dir=output_dirs["checkpoints"],
                model_name=model_name,
                verbose=args.verbose,
            )

            # Train
            start_time = time.time()
            history = trainer.train(
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping,
                save_best=True,
            )
            train_time = time.time() - start_time

            # Get summary
            summary = trainer.get_summary()
            summary["total_train_time"] = train_time

            print(f"\nTraining completed in {format_time(train_time)}")
            print(f"Best test accuracy: {summary['best_test_accuracy']:.2f}%")

            # Save history
            trainer.save_history(f"{model_name}_history.json")

            # Add to comparator
            comparator.add_model(model_name, model, history)

            # Store results
            all_results[model_name] = summary

        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Generate comparison report
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    comparator.print_comparison()

    # Save comparison report
    comparator.save_comparison_report(output_dirs["plots"])

    # Save summary results
    summary_path = Path(output_dirs["metrics"]) / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"config": vars(args), "data_info": data_info, "results": all_results},
            f,
            indent=2,
        )

    print(f"\nBenchmark summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
