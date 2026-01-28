#!/usr/bin/env python3
"""
Layer-wise Performance Deviation Heatmap - GPT2-XL (Overall Metric)

Generates a heatmap showing absolute deviation from the best-performing layer
per dataset for the Overall metric on GPT2-XL.

Rows: Datasets (WikiRecent, WikiBio, ZSRE, WikiCounterfact, Counterfact)
Columns: All 48 layers (0-47)
Cell value: |Perf(dataset, layer) - max_l' Perf(dataset, l')|

Note: All datasets use 'post_overall' as the overall metric name.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

from dataset_model_registry import get_config, DATASET_MODEL_REGISTRY
import heatmap_api  # noqa: F401

# Number of layers for GPT2-XL
N_LAYERS = 48

# Golden layers to highlight (will be computed from proxy analysis)
GOLDEN_LAYERS = set()

# Dataset-specific golden layers (will be computed)
DATASET_GOLDEN_LAYERS = {}

# Dataset order for plotting
DATASETS = ['wikirecent', 'wikibio', 'zsre', 'wikicounterfact', 'counterfact']

# Metric mapping for Overall
# NOTE: WikiRecent and WikiCounterfact use 'overall' not 'post_overall' in their API
METRIC_MAP = {
    'wikirecent': 'overall',
    'wikibio': 'post_overall',
    'zsre': 'post_overall',
    'wikicounterfact': 'overall',
    'counterfact': 'post_overall',
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

    # Create analyzer with correct n_layers
    data_dir = config['base_dir'] / config.get('data_dir', 'allLayerPerformance')
    n_layers = config.get('n_layers', N_LAYERS)
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


def compute_deviation_matrix(model='gpt2-xl'):
    """
    Compute the deviation matrix for all datasets.

    Returns:
        deviation_matrix: np.array of shape (n_datasets, n_layers)
        dataset_names: list of dataset names
        best_layers: dict mapping dataset to best layer index
    """
    print("=" * 70)
    print("Computing Deviation Matrix for Overall Metric (GPT2-XL)")
    print("=" * 70)

    deviation_matrix = np.zeros((len(DATASETS), N_LAYERS))
    best_layers = {}
    test_means_dict = {}
    golden_layers_set = set()
    dataset_golden = {}

    for i, dataset in enumerate(DATASETS):
        print(f"\nLoading {dataset}...")

        try:
            analyzer, config = load_analyzer(dataset, model)
            metric = METRIC_MAP[dataset]

            # Check if metric exists
            if metric not in analyzer.test_means:
                print(f"  Warning: {metric} not found in test_means")
                print(f"  Available: {list(analyzer.test_means.keys())}")
                deviation_matrix[i, :] = np.nan
                continue

            # Get test means for all layers
            test_means = analyzer.test_means[metric].numpy()
            test_means_dict[dataset] = test_means

            # 1. Find the maximum performance value
            best_perf = np.max(test_means)

            # 2. Find ALL indices where performance equals that maximum
            # np.where returns a tuple, so we take the first element [0]
            best_layers_list = np.where(test_means == best_perf)[0].tolist()

            # Now, best_layers_list is an array/list like [14] or [14, 15, 16]
            best_layers[dataset] = best_layers_list

            # Find proxy best layer(s) for golden layer identification
            # Include ALL layers that achieve the best performance (handle ties)
            proxy_means = analyzer.proxy_means[metric].numpy()
            proxy_best_perf = np.max(proxy_means)
            proxy_best_layers = np.where(proxy_means == proxy_best_perf)[0].tolist()
            golden_layers_set.update(proxy_best_layers)
            dataset_golden[dataset] = proxy_best_layers

            # Compute deviation for each layer
            deviation = np.abs(test_means - best_perf)
            deviation_matrix[i, :] = deviation

            print(f"  Test samples: {len(analyzer.test_indices)}")
            print(f"  Test best layer: {best_layers_list} (perf: {best_perf:.4f})")
            print(f"  Proxy best layer(s): {proxy_best_layers} (perf: {proxy_best_perf:.4f})")

        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
            import traceback
            traceback.print_exc()
            deviation_matrix[i, :] = np.nan

    # Update global golden layers
    global GOLDEN_LAYERS, DATASET_GOLDEN_LAYERS
    GOLDEN_LAYERS = golden_layers_set
    DATASET_GOLDEN_LAYERS = dataset_golden

    return deviation_matrix, DATASETS, best_layers, test_means_dict


def plot_heatmap(deviation_matrix, dataset_names, best_layers, output_path='results/layer_deviation_heatmap_overall.png'):
    """
    Create the heatmap visualization.
    """
    print("\n" + "=" * 70)
    print("Creating Heatmap (GPT2-XL Overall)")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Create custom colormap (cool to warm: blue -> white -> red)
    cmap = plt.cm.RdYlBu_r  # Reversed: blue=low, red=high

    # Plot heatmap
    im = ax.imshow(deviation_matrix, cmap=cmap, aspect='auto',
                   vmin=0, vmax=np.nanmax(deviation_matrix), interpolation='none')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Absolute Deviation from Best Layer', fontsize=12)

    # Set axis labels
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)

    # Set y-axis ticks (datasets)
    ax.set_yticks(range(len(dataset_names)))
    pretty_names = {
        'wikirecent': 'WikiRecent',
        'wikibio': 'WikiBio',
        'zsre': 'ZSRE',
        'wikicounterfact': 'WikiCounterfact',
        'counterfact': 'Counterfact'
    }
    ax.set_yticklabels([pretty_names.get(d, d) for d in dataset_names], fontsize=10)

    # Set x-axis ticks (layers) - show every 5th layer
    ax.set_xticks(range(0, N_LAYERS, 5))
    ax.set_xticklabels(range(0, N_LAYERS, 5), fontsize=9)

    heatmap_api.highlight_mark_layers(GOLDEN_LAYERS, DATASET_GOLDEN_LAYERS, dataset_names, best_layers, ax)

    # Title
    golden_str = '{' + ', '.join(map(str, sorted(GOLDEN_LAYERS))) + '}'
    ax.set_title(f'Layer-wise Performance Deviation from Best Layer (Overall - GPT2-XL)\n'
                 f'Gold = Golden region {golden_str}; Green = Dataset proxy-selected; '
                 '★ = Best Performing Layer', fontsize=11)

    # Create legend (placed below the heatmap to avoid overlap)
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=2,
                       label=f'Golden Region {golden_str} (GPT2-XL)'),
        mpatches.Patch(facecolor='none', edgecolor='limegreen', linewidth=3,
                       label='Dataset Proxy-Selected (GPT2-XL)'),
        plt.Line2D([0], [0], marker='*', color='white', markeredgecolor='black',
                   markersize=12, linestyle='None', label='Best Performing Layer (Test Set)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=9, frameon=True)
    ax.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend below

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved heatmap to: {output_path}")

    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved PDF to: {pdf_path}")

    plt.close()

    return fig


def print_summary_table(deviation_matrix, dataset_names, best_layers, test_means_dict):
    """Print a summary table of golden layer performance."""
    print("\n" + "=" * 70)
    print("SUMMARY: Golden Layer Performance Across Datasets (GPT2-XL Overall)")
    print("=" * 70)

    # Header
    print(f"\n{'Dataset':<18} {'Best Layer':<12} {'Golden Layers Deviation':<40}")
    print("-" * 70)

    for i, dataset in enumerate(dataset_names):
        if dataset not in best_layers:
            print(f"{dataset:<18} {'N/A':<12} {'N/A':<40}")
            continue

        best_layer = best_layers[dataset]

        # Get deviations for all golden layers
        golden_devs = []
        for gl in sorted(GOLDEN_LAYERS):
            dev = deviation_matrix[i, gl]
            golden_devs.append(f"L{gl}:{dev:.3f}")

        golden_str = ", ".join(golden_devs)
        print(f"{dataset:<18} {', '.join(map(str, best_layer)):<12} {golden_str}")

    # Compute average deviation for golden vs non-golden layers
    print("\n" + "=" * 70)
    print("COMPARISON: Golden vs Non-Golden Layer Average Deviation")
    print("=" * 70)

    golden_indices = list(GOLDEN_LAYERS)
    non_golden_indices = [i for i in range(N_LAYERS) if i not in GOLDEN_LAYERS]

    print(f"\n{'Dataset':<18} {'Avg Golden Dev':<18} {'Avg Non-Golden Dev':<18} {'Ratio':<10}")
    print("-" * 70)

    valid_rows = []
    for i, dataset in enumerate(dataset_names):
        if np.all(np.isnan(deviation_matrix[i, :])):
            print(f"{dataset:<18} {'N/A':<18} {'N/A':<18} {'N/A':<10}")
            continue

        valid_rows.append(i)
        golden_avg = np.nanmean(deviation_matrix[i, golden_indices])
        non_golden_avg = np.nanmean(deviation_matrix[i, non_golden_indices])
        ratio = golden_avg / non_golden_avg if non_golden_avg > 0 else 0

        print(f"{dataset:<18} {golden_avg:<18.4f} {non_golden_avg:<18.4f} {ratio:<10.3f}")

    if valid_rows:
        # Overall average
        overall_golden_avg = np.nanmean(deviation_matrix[valid_rows][:, golden_indices])
        overall_non_golden_avg = np.nanmean(deviation_matrix[valid_rows][:, non_golden_indices])
        overall_ratio = overall_golden_avg / overall_non_golden_avg if overall_non_golden_avg > 0 else 0

        print("-" * 70)
        print(f"{'OVERALL':<18} {overall_golden_avg:<18.4f} {overall_non_golden_avg:<18.4f} {overall_ratio:<10.3f}")

        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        print(f"\nGolden layers have {overall_ratio:.1%} of the average deviation of non-golden layers.")
        if overall_ratio < 0.5:
            print("→ Golden layers show CONSISTENTLY LOW deviation across datasets!")
            print("→ This supports the claim of dataset-independent layer selection for Overall metric.")
        else:
            print("→ Golden layers do not show significantly lower deviation.")


def main():
    print("=" * 70)
    print("Layer-wise Performance Deviation Heatmap - GPT2-XL Overall")
    print("Model: GPT2-XL | Metric: Overall")
    print("Datasets: WikiRecent, WikiBio, ZSRE, WikiCounterfact, Counterfact")
    print("=" * 70)

    # Compute deviation matrix
    deviation_matrix, dataset_names, best_layers, test_means_dict = compute_deviation_matrix('gpt2-xl')

    # Print summary
    print_summary_table(deviation_matrix, dataset_names, best_layers, test_means_dict)

    # Create heatmap
    output_path = 'results/layer_deviation_heatmap_overall.png'
    plot_heatmap(deviation_matrix, dataset_names, best_layers, output_path)

    # Save raw data
    np.save('results/deviation_matrix_overall.npy', deviation_matrix)
    print(f"\n✓ Saved raw data to: results/deviation_matrix_overall.npy")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
