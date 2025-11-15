# Quick Start Guide

## Installation
```bash
pip install -r requirements.txt
```

## Run Full Benchmark
```bash
# All models, 4x4 images, binary classification
python experiments/benchmark_all.py \
    --image-size 4 \
    --classes 0 1 \
    --epochs 20 \
    --models qfcn classical cong_qcnn

# Or use the shell script
bash run_benchmark.sh
```

## Train Single Model
```bash
# QFCN only
python experiments/train_single.py \
    --model qfcn \
    --epochs 20 \
    --batch-size 32

# Classical CNN
python experiments/train_single.py \
    --model classical \
    --epochs 20 \
    --image-size 8
```

## Python API
```python
from models import get_model
from data.mnist_loader import load_mnist_data
from training import Trainer

# Load data
train_loader, test_loader, info = load_mnist_data(
    image_size=4,
    classes=[0, 1],
    batch_size=32
)

# Create model
model = get_model('qfcn', num_classes=2)

# Train
trainer = Trainer(model, train_loader, test_loader)
history = trainer.train(epochs=20)

# Evaluate
summary = trainer.get_summary()
print(f"Accuracy: {summary['best_test_accuracy']:.2f}%")
```

## Results

Results are saved to `./results/`:
- `checkpoints/` - Model weights
- `metrics/` - JSON/CSV metrics
- `plots/` - Training curves, confusion matrices

## Kaggle

Upload the repository and run:
```python
!python experiments/benchmark_all.py --epochs 10 --device cpu
```

Results saved to `/kaggle/working/results/`