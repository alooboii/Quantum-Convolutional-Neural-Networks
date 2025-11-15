"""
Training Package
Unified training and evaluation utilities
"""

from .trainer import Trainer
from .evaluator import ModelEvaluator, MultiModelComparator

__all__ = ["Trainer", "ModelEvaluator", "MultiModelComparator"]
