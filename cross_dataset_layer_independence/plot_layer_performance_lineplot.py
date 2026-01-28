#!/usr/bin/env python3
"""
Layer-wise Performance Line Plot

Generates a line plot showing performance across all layers for each dataset.

X-axis: Layer (0-47)
Y-axis: Performance (Rewrite Accuracy on test set)
Each line: One dataset

Golden layers {15, 16, 17, 18, 19} are highlighted with vertical bands/markers.
If golden layers cluster together with high performance, it visually proves
that layer selection is dataset-independent.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

from dataset_model_registry import get_config, DATASET_MODEL_REGISTRY

# Golden layers to highlight (union of all proxy-selected layers)
GOLDEN_LAYERS = {15, 16, 17, 18, 19}

# Dataset-specific golden layers
DATASET_GOLDEN_LAYERS = {
    'wikirecent': [17],
    'wikibio': [19],
    'zsre': [15, 17, 18, 19],
    'wikicounterfact': [15],
    'counterfact': [16]
}

# Dataset order for plotting
DATASETS = ['wikirecent', 'wikibio', 'zsre', 'wikicounterfact', 'counterfact']

# Pretty names for legend
DATASET_NAMES = {
    'wikirecent': 'WikiRecent',
    'wikibio': 'WikiBio',
    'zsre': 'ZSRE',
    'wikicounterfact': 'WikiCounterfact',
    'counterfact': 'Counterfact'
}

# Colors for each dataset
DATASET_COLORS = {
    'wikirecent': '#1f77b4',     # blue
    'wikibio': '#ff7f0e',        # orange
    'zsre': '#2ca02c',           # green
    'wikicounterfact': '#d62728', # red
    'counterfact': '#9467bd'      # purple
}

# Metric mapping
METRIC_MAP = {
    'zsre': 'post_rewrite',
    'counterfact': 'post_rewrite',
    'wikibio': 'post_rewrite_acc',
    'wikicounterfact': 'post_rewrite_acc',
    'wikirecent': 'post_rewrite_acc'
}


def load_analyzer(dataset, model):
    """Load the golden layer analyzer for a dataset."""
    config = get_config(dataset, model)

    # Load the API module
    api_path = config['api_path']
    spec = importlib.util.spec_from_file_location("analyzer_module", api_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get analyzer class
    api_class_name = config.get('api_class', 'GoldenLayerAnalyzer')
    analyzer_class = getattr(module, api_class_name)

    # Create analyzer
    data_dir = config['base_dir'] / config.get('data_dir', 'allLayerPerformance')
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


def collect_performance_data(model='gpt2-xl'):
    """
    Collect test set performance for all layers across all datasets.

    Returns:
        performance_data: dict mapping dataset -> np.array of shape (n_layers,)
        best_layers: dict mapping dataset -> best layer index
    """
    print("=" * 70)
    print("Collecting Performance Data for Line Plot")
    print("=" * 70)

    n_layers = 48
    performance_data = {}
    best_layers = {}

    for dataset in DATASETS:
        print(f"\nLoading {dataset}...")

        try:
            analyzer, config = load_analyzer(dataset, model)
            metric = METRIC_MAP[dataset]

            # Get test means for all layers
            test_means = analyzer.test_means[metric].numpy()
            performance_data[dataset] = test_means

            # Find best layer (max performance)
            best_layer = np.argmax(test_means)
            best_perf = test_means[best_layer]
            best_layers[dataset] = best_layer

            print(f"  Test samples: {len(analyzer.test_indices)}")
            print(f"  Best layer: {best_layer} (perf: {best_perf:.4f})")
            print(f"  Golden layer {DATASET_GOLDEN_LAYERS[dataset][0]} perf: {test_means[DATASET_GOLDEN_LAYERS[dataset][0]]:.4f}")

        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
            performance_data[dataset] = np.full(n_layers, np.nan)

    return performance_data, best_layers


def plot_lineplot(performance_data, best_layers, output_path='results/layer_performance_lineplot.png'):
    """
    Create the line plot visualization.
    """
    print("\n" + "=" * 70)
    print("Creating Line Plot")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(14, 7))

    n_layers = 48
    x = np.arange(n_layers)

    # Add shaded region for golden layers
    golden_min = min(GOLDEN_LAYERS) - 0.5
    golden_max = max(GOLDEN_LAYERS) + 0.5
    ax.axvspan(golden_min, golden_max, alpha=0.2, color='gold',
               label='Golden Layer Region')

    # Plot vertical lines for each golden layer
    for gl in GOLDEN_LAYERS:
        ax.axvline(x=gl, color='gold', linestyle='--', alpha=0.5, linewidth=1)

    # Plot each dataset as a line
    for dataset in DATASETS:
        perf = performance_data[dataset]
        color = DATASET_COLORS[dataset]
        label = DATASET_NAMES[dataset]

        ax.plot(x, perf, color=color, linewidth=2, label=label, marker='o',
                markersize=3, alpha=0.8)

        # Mark the dataset-specific golden layer with a star
        for gl in DATASET_GOLDEN_LAYERS[dataset]:
            ax.plot(gl, perf[gl], marker='*', color=color, markersize=15,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)

    # Customize plot
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rewrite Accuracy (Test Set)', fontsize=12)
    ax.set_title('Layer-wise Performance Across Datasets\n'
                 'Golden layers {15,16,17,18,19} highlighted; '
                 'Dataset-specific golden layer marked with ★', fontsize=12)

    # Set x-axis ticks
    ax.set_xticks(range(0, 48, 5))
    ax.set_xlim(-0.5, 47.5)

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-')

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    # Move golden layer region to end of legend
    ax.legend(handles, labels, loc='lower right', fontsize=9, ncol=2)

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved line plot to: {output_path}")

    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved PDF to: {pdf_path}")

    plt.close()

    return fig


def plot_zoomed_golden_region(performance_data, best_layers, output_path='results/layer_performance_golden_zoom.png'):
    """
    Create a zoomed-in plot focusing on the golden layer region.
    """
    print("\n" + "=" * 70)
    print("Creating Zoomed Golden Region Plot")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Focus on layers 10-25 (around golden region)
    zoom_start = 10
    zoom_end = 25
    x = np.arange(zoom_start, zoom_end + 1)

    # Add shaded region for golden layers
    golden_min = min(GOLDEN_LAYERS) - 0.5
    golden_max = max(GOLDEN_LAYERS) + 0.5
    ax.axvspan(golden_min, golden_max, alpha=0.2, color='gold',
               label='Golden Layer Region')

    # Plot vertical lines for each golden layer
    for gl in GOLDEN_LAYERS:
        ax.axvline(x=gl, color='gold', linestyle='--', alpha=0.5, linewidth=1)

    # Plot each dataset as a line (zoomed)
    for dataset in DATASETS:
        perf = performance_data[dataset][zoom_start:zoom_end+1]
        color = DATASET_COLORS[dataset]
        label = DATASET_NAMES[dataset]

        ax.plot(x, perf, color=color, linewidth=2.5, label=label, marker='o',
                markersize=6, alpha=0.9)

        # Mark the dataset-specific golden layer with a star
        for gl in DATASET_GOLDEN_LAYERS[dataset]:
            if zoom_start <= gl <= zoom_end:
                ax.plot(gl, performance_data[dataset][gl], marker='*', color=color,
                        markersize=20, markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # Customize plot
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rewrite Accuracy (Test Set)', fontsize=12)
    ax.set_title('Golden Layer Region Performance (Zoomed)\n'
                 'All datasets show high performance in layers 15-19', fontsize=12)

    # Set x-axis ticks
    ax.set_xticks(range(zoom_start, zoom_end + 1))
    ax.set_xlim(zoom_start - 0.5, zoom_end + 0.5)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-')

    # Create legend
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved zoomed plot to: {output_path}")

    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved PDF to: {pdf_path}")

    plt.close()

    return fig


def print_golden_layer_stats(performance_data):
    """Print statistics about golden layer performance."""
    print("\n" + "=" * 70)
    print("GOLDEN LAYER PERFORMANCE STATISTICS")
    print("=" * 70)

    # Header
    print(f"\n{'Dataset':<18} " + " ".join([f"{'L'+str(gl):<8}" for gl in sorted(GOLDEN_LAYERS)]) + f" {'Mean':<8} {'Best Layer':<12}")
    print("-" * 80)

    for dataset in DATASETS:
        perf = performance_data[dataset]
        golden_perfs = [perf[gl] for gl in sorted(GOLDEN_LAYERS)]
        mean_golden = np.mean(golden_perfs)
        best_layer = np.argmax(perf)
        best_perf = perf[best_layer]

        perf_str = " ".join([f"{p:.4f}  " for p in golden_perfs])
        print(f"{DATASET_NAMES[dataset]:<18} {perf_str} {mean_golden:.4f}   L{best_layer} ({best_perf:.4f})")

    # Average across datasets
    print("-" * 80)
    avg_perfs = []
    for gl in sorted(GOLDEN_LAYERS):
        avg = np.mean([performance_data[d][gl] for d in DATASETS])
        avg_perfs.append(avg)
    overall_mean = np.mean(avg_perfs)

    perf_str = " ".join([f"{p:.4f}  " for p in avg_perfs])
    print(f"{'AVERAGE':<18} {perf_str} {overall_mean:.4f}")

    # Variance analysis
    print("\n" + "=" * 70)
    print("VARIANCE ACROSS DATASETS (Lower = More Consistent)")
    print("=" * 70)
    print(f"\n{'Layer':<10} {'Mean Perf':<12} {'Std Dev':<12} {'CV (%)':<12}")
    print("-" * 50)

    for gl in sorted(GOLDEN_LAYERS):
        perfs = [performance_data[d][gl] for d in DATASETS]
        mean_p = np.mean(perfs)
        std_p = np.std(perfs)
        cv = (std_p / mean_p * 100) if mean_p > 0 else 0
        print(f"Layer {gl:<4} {mean_p:<12.4f} {std_p:<12.4f} {cv:<12.2f}")

    # Compare golden vs non-golden
    print("\n" + "=" * 70)
    print("GOLDEN vs NON-GOLDEN LAYERS")
    print("=" * 70)

    golden_all = []
    non_golden_all = []

    for dataset in DATASETS:
        perf = performance_data[dataset]
        for layer in range(48):
            if layer in GOLDEN_LAYERS:
                golden_all.append(perf[layer])
            else:
                non_golden_all.append(perf[layer])

    print(f"\nGolden layers (n={len(golden_all)}):")
    print(f"  Mean: {np.mean(golden_all):.4f}")
    print(f"  Std:  {np.std(golden_all):.4f}")

    print(f"\nNon-golden layers (n={len(non_golden_all)}):")
    print(f"  Mean: {np.mean(non_golden_all):.4f}")
    print(f"  Std:  {np.std(non_golden_all):.4f}")

    print(f"\nDifference: {np.mean(golden_all) - np.mean(non_golden_all):.4f}")


def main():
    print("=" * 70)
    print("Layer-wise Performance Line Plot")
    print("Model: GPT2-XL | Metric: Rewrite Accuracy")
    print("=" * 70)

    # Collect performance data
    performance_data, best_layers = collect_performance_data('gpt2-xl')

    # Print statistics
    print_golden_layer_stats(performance_data)

    # Create main line plot
    output_path = 'results/layer_performance_lineplot_rewrite.png'
    plot_lineplot(performance_data, best_layers, output_path)

    # Create zoomed plot
    zoom_path = 'results/layer_performance_golden_zoom_rewrite.png'
    plot_zoomed_golden_region(performance_data, best_layers, zoom_path)

    # Save raw data
    np.save('results/performance_data_rewrite.npy', performance_data)
    print(f"\n✓ Saved raw data to: results/performance_data_rewrite.npy")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
