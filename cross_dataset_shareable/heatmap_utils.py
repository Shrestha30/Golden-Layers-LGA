#!/usr/bin/env python3
"""
Heatmap Utilities for Layer Performance Visualization

This module provides utilities for:
1. Loading and initializing analyzers
2. Computing metrics and correlations
3. Drawing heatmaps with golden layer highlighting
"""

import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ks_2samp, ttest_rel

from analyzers import GoldenLayerAnalyzer, GoldenLayerAnalyzerWikiBio, GoldenLayerAnalyzerWikiRecent
from config import get_config, DATA_ROOT


# =============================================================================
# Metric Mappings
# =============================================================================

METRIC_MAPPINGS = {
    'zsre': {
        'Rewrite': 'post_rewrite',
        'Rephrase': 'post_rephrase',
        'Locality': 'post_neighborhood',
        'Portability': 'post_one_hop',
        'Fluency': None,
        'Overall': 'post_overall',
    },
    'counterfact': {
        'Rewrite': 'post_rewrite',
        'Rephrase': 'post_rephrase',
        'Locality': 'post_neighborhood',
        'Portability': 'post_one_hop',
        'Fluency': None,
        'Overall': 'post_overall',
    },
    'wikibio': {
        'Rewrite': 'post_rewrite_acc',
        'Rephrase': None,
        'Locality': 'post_locality_acc',
        'Portability': None,
        'Fluency': 'post_fluency',
        'Overall': 'post_overall',
    },
    'wikirecent': {
        'Rewrite': 'post_rewrite_acc',
        'Rephrase': None,
        'Locality': 'post_locality_acc',
        'Portability': 'post_portability_acc',
        'Fluency': 'post_fluency',
        'Overall': 'overall',
    },
    'wikicounterfact': {
        'Rewrite': 'post_rewrite_acc',
        'Rephrase': None,
        'Locality': 'post_locality_acc',
        'Portability': 'post_portability_acc',
        'Fluency': 'post_fluency',
        'Overall': 'overall',
    },
}


def get_metric_key(dataset, metric_name):
    """Get the actual metric key for a dataset and metric display name."""
    if dataset not in METRIC_MAPPINGS:
        return None
    return METRIC_MAPPINGS[dataset].get(metric_name)


# =============================================================================
# Analyzer Loading
# =============================================================================

def load_analyzer(dataset, model):
    """
    Load the appropriate analyzer for a dataset-model pair.

    Args:
        dataset: Dataset name (e.g., 'zsre', 'wikibio')
        model: Model name (e.g., 'gpt2-xl', 'llama2-7b')

    Returns:
        (analyzer, config) tuple
    """
    config = get_config(dataset, model)

    # Select analyzer class based on configuration
    analyzer_type = config.get('analyzer_class', 'default')

    if analyzer_type == 'wikibio':
        analyzer_class = GoldenLayerAnalyzerWikiBio
    elif analyzer_type == 'wikirecent':
        analyzer_class = GoldenLayerAnalyzerWikiRecent
    else:
        analyzer_class = GoldenLayerAnalyzer

    # Create analyzer
    data_dir = config['data_dir']
    n_layers = config.get('n_layers', 48)

    try:
        analyzer = analyzer_class(data_dir=str(data_dir), n_layers=n_layers)
    except TypeError:
        analyzer = analyzer_class(data_dir=str(data_dir))

    # Load data
    analyzer.load_data(verbose=False)

    # Compute overall metric if available
    if hasattr(analyzer, 'compute_overall_metric'):
        analyzer.compute_overall_metric(verbose=False)

    # Split dataset
    proxy_size = config.get('proxy_size', 100)
    analyzer.split_dataset(proxy_size=proxy_size, random_seed=42, verbose=False)
    analyzer.compute_proxy_test_means(verbose=False)

    return analyzer, config


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_correlation(proxy_means, test_means):
    """Compute Spearman correlation between proxy and test layer means."""
    if hasattr(proxy_means, 'numpy'):
        proxy_means = proxy_means.numpy()
    if hasattr(test_means, 'numpy'):
        test_means = test_means.numpy()

    valid_mask = ~(np.isnan(proxy_means) | np.isnan(test_means))

    if valid_mask.sum() < 3:
        return np.nan

    proxy_valid = proxy_means[valid_mask]
    test_valid = test_means[valid_mask]

    corr, _ = spearmanr(proxy_valid, test_valid)
    return corr


def compute_oracle_vs_golden(analyzer, metric_key, higher_is_better=True, golden_layer_proxy=True):
    """
    Compare oracle (sample-wise best) vs golden layer (fixed from proxy) performance.

    Args:
        analyzer: Loaded analyzer with results and test_indices
        metric_key: The metric key in analyzer.results
        higher_is_better: If True, oracle = max; if False (e.g., fluency), oracle = min
        golden_layer_proxy: If True, use proxy set to find golden layer

    Returns:
        dict with analysis results
    """
    data = analyzer.results[metric_key]
    if hasattr(data, 'numpy'):
        data = data.numpy()

    test_indices = analyzer.test_indices
    if hasattr(test_indices, 'numpy'):
        test_indices = test_indices.numpy()
    test_indices = np.array(test_indices)

    test_data = data[test_indices]

    if golden_layer_proxy:
        proxy_means = analyzer.proxy_means[metric_key]
    else:
        proxy_means = analyzer.test_means[metric_key]

    if hasattr(proxy_means, 'numpy'):
        proxy_means = proxy_means.numpy()

    if higher_is_better:
        best_perf = np.nanmax(proxy_means)
    else:
        best_perf = np.nanmin(proxy_means)

    golden_layers = np.where(proxy_means == best_perf)[0].tolist()
    golden_layer = golden_layers[0]

    if higher_is_better:
        oracle_perf = np.nanmax(test_data, axis=1)
    else:
        oracle_perf = np.nanmin(test_data, axis=1)

    golden_perf = test_data[:, golden_layer]

    if higher_is_better:
        gaps = oracle_perf - golden_perf
    else:
        gaps = golden_perf - oracle_perf

    valid_mask = ~(np.isnan(oracle_perf) | np.isnan(golden_perf))
    valid_gaps = gaps[valid_mask]

    if valid_mask.sum() >= 3:
        corr, _ = pearsonr(oracle_perf[valid_mask], golden_perf[valid_mask])
        ks_stat, ks_pvalue = ks_2samp(oracle_perf[valid_mask], golden_perf[valid_mask])
    else:
        corr = np.nan
        ks_stat, ks_pvalue = np.nan, np.nan

    t_stat, p_two_tailed = ttest_rel(oracle_perf[valid_mask], golden_perf[valid_mask])

    return {
        'golden_layer': golden_layer,
        'golden_layers': golden_layers,
        'oracle_perf': oracle_perf,
        'golden_perf': golden_perf,
        'gaps': gaps,
        'mean_gap': float(np.nanmean(valid_gaps)) if len(valid_gaps) > 0 else np.nan,
        'median_gap': float(np.nanmedian(valid_gaps)) if len(valid_gaps) > 0 else np.nan,
        'n_samples': int(valid_mask.sum()),
        'higher_is_better': higher_is_better,
        'correlation': float(corr),
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        't_stat': float(t_stat),
        'p_two_tailed': float(p_two_tailed),
    }


# =============================================================================
# Heatmap Drawing Utilities
# =============================================================================

def get_segments(indices):
    """Groups consecutive indices into segments. E.g., [14, 15, 16, 20, 21] -> [(14, 16), (20, 21)]"""
    if not indices:
        return []
    indices = sorted(list(set(indices)))
    segments = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            segments.append((start, indices[i-1]))
            start = indices[i]
    segments.append((start, indices[-1]))
    return segments


def highlight_mark_layers(golden_layers, dataset_golden_layers, dataset_names, best_layers, ax):
    """
    Add visual highlights to heatmap for golden layers and best layers.

    Args:
        golden_layers: Set of layer indices in the golden region (union)
        dataset_golden_layers: Dict mapping dataset -> list of proxy-selected layers
        dataset_names: List of dataset names (row labels)
        best_layers: Dict mapping dataset -> list of best test layers
        ax: Matplotlib axes object
    """
    # 1. Highlight GOLDEN_LAYERS (Vertical blocks - gold border)
    golden_segments = get_segments(golden_layers)
    lw_gold = 2
    offset_gold = 0.02

    for start, end in golden_segments:
        width = (end - start) + 1
        rect = mpatches.Rectangle(
            (start - 0.5 + offset_gold, -0.5 + offset_gold),
            width - 2*offset_gold, len(dataset_names) - 2*offset_gold,
            fill=False, edgecolor='gold',
            linewidth=lw_gold, linestyle='-', zorder=4
        )
        ax.add_patch(rect)

    # 2. Highlight DATASET_GOLDEN_LAYERS (Horizontal blocks - green border)
    lw_green = 1
    offset_green = 0.02

    for row, dataset in enumerate(dataset_names):
        if dataset in dataset_golden_layers:
            dataset_segments = get_segments(dataset_golden_layers[dataset])
            for start, end in dataset_segments:
                width = (end - start) + 1
                rect = mpatches.Rectangle(
                    (start - 0.5 + offset_green, row - 0.5 + offset_green),
                    width - 2*offset_green, 1 - 2*offset_green,
                    fill=False, edgecolor='limegreen',
                    linewidth=lw_green, linestyle='-', zorder=5
                )
                ax.add_patch(rect)

    # 3. Add best layer markers (star)
    for row, dataset in enumerate(dataset_names):
        if dataset in best_layers:
            layers = best_layers[dataset]
            ax.plot(layers, [row]*len(layers), marker='*', color='white',
                    markersize=10, markeredgecolor='black', markeredgewidth=0.8,
                    linestyle='None', zorder=6)
