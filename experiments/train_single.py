"""
Train Single Model
Script to train a single model with custom configuration
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models import get_model
from data.mnist_loader import load_mnist_data, print_dataset_info
from training import Trainer, ModelEvaluator
from training.utils import set_seed, get_device, create_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single QCNN model")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (qfcn, classical, cong_qcnn)",
    )

    # Dataset
    parser.add_argument("--image-size", type=int, default=4)
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--early-stopping", type=int, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training {args.model.upper()} on MNIST")
    print("=" * 60)

    # Setup
    set_seed(args.seed)
    device = get_device(prefer_cpu=(args.device == "cpu"))
    output_dirs = create_output_dirs(args.output_dir)

    # Load data
    train_loader, test_loader, data_info = load_mnist_data(
        image_size=args.image_size,
        batch_size=args.batch_size,
        classes=args.classes,
        seed=args.seed,
    )

    print_dataset_info(data_info)

    # Create model
    model = get_model(
        args.model, num_classes=data_info["num_classes"], image_size=args.image_size
    )

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=args.lr,
        device=str(device),
        save_dir=output_dirs["checkpoints"],
        model_name=args.model,
        verbose=True,
    )

    history = trainer.train(
        epochs=args.epochs, early_stopping_patience=args.early_stopping, save_best=True
    )

    # Evaluate
    evaluator = ModelEvaluator(model, test_loader, str(device))
    results = evaluator.evaluate()

    # Save results
    trainer.save_history()
    evaluator.plot_confusion_matrix(
        save_path=str(Path(output_dirs["plots"]) / f"{args.model}_confusion_matrix.png")
    )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
