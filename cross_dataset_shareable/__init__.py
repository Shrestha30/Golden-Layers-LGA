"""
Cross-Dataset Layer Independence Analysis Package

Modules:
    analyzers: Analyzer classes for different datasets
    config: Configuration and dataset registry
    heatmap_utils: Utilities for heatmap generation
    plot_heatmap: Main plotting script
"""

from .analyzers import (
    GoldenLayerAnalyzer,
    GoldenLayerAnalyzerWikiBio,
    GoldenLayerAnalyzerWikiRecent
)
from .config import get_config, get_datasets, get_models, DATASET_MODEL_REGISTRY
from .heatmap_utils import load_analyzer, highlight_mark_layers, compute_oracle_vs_golden

__all__ = [
    'GoldenLayerAnalyzer',
    'GoldenLayerAnalyzerWikiBio',
    'GoldenLayerAnalyzerWikiRecent',
    'get_config',
    'get_datasets',
    'get_models',
    'DATASET_MODEL_REGISTRY',
    'load_analyzer',
    'highlight_mark_layers',
    'compute_oracle_vs_golden',
]
