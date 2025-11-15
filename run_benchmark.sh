#!/bin/bash
# Quick benchmark script

echo "=========================================="
echo "Quantum CNN MNIST Benchmark"
echo "=========================================="

# Binary classification (0 vs 1) on 4x4 images
python experiments/benchmark_all.py \
    --image-size 4 \
    --classes 0 1 \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --models qfcn classical cong_qcnn \
    --seed 42 \
    --output-dir ./results \
    --device cpu \
    --verbose

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "Results saved to: ./results"
echo "=========================================="
