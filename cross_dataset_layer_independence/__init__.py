"""
Cross-Dataset Layer Independence Analysis

This package tests whether layer performance is independent of the dataset
by conducting paired t-tests between best and worst performing layers
across different datasets.
"""

from .cross_dataset_analyzer import CrossDatasetLayerAnalyzer
from .dataset_model_registry import (
    DATASET_MODEL_REGISTRY,
    get_datasets,
    get_models,
    get_common_models,
    get_config
)

__version__ = '1.0.0'
__all__ = [
    'CrossDatasetLayerAnalyzer',
    'DATASET_MODEL_REGISTRY',
    'get_datasets',
    'get_models',
    'get_common_models',
    'get_config'
]
