#!/usr/bin/env python3
"""
Valley of Consistency Plot - Rephrase Accuracy

Generates a line plot showing absolute deviation from the best layer for each dataset.
This version uses Rephrase Accuracy metric, supported by ZSRE and Counterfact.

X-axis: Layer (0-47)
Y-axis: Absolute Deviation from Best Layer
Each line: One dataset (ZSRE, Counterfact)

Golden layers for Rephrase:
- ZSRE: Layer 16
- Counterfact: Layer 18
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

from dataset_model_registry import get_config, DATASET_MODEL_REGISTRY

# Golden layers for Rephrase metric
GOLDEN_LAYERS = {16, 18}

# Dataset-specific golden layers (proxy-selected) for Rephrase
DATASET_GOLDEN_LAYERS = {
    'zsre': [16],
    'counterfact': [18]
}

# Dataset order for plotting (only datasets that support rephrase)
DATASETS = ['zsre', 'counterfact']

# Pretty names for legend
DATASET_NAMES = {
    'zsre': 'ZSRE',
    'counterfact': 'Counterfact'
}

# Colors for each dataset
DATASET_COLORS = {
    'zsre': '#2ca02c',           # green
    'counterfact': '#9467bd'     # purple
}

# Metric mapping for Rephrase
METRIC_MAP = {
    'zsre': 'post_rephrase',
    'counterfact': 'post_rephrase'
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


def collect_deviation_data(model='gpt2-xl'):
    """
    Collect deviation from best layer for all layers across all datasets.

    For each dataset:
    - Best layer = proxy-selected golden layer for rephrase
    - Deviation[layer] = |Performance(Best Layer) - Performance(layer)|

    Returns:
        deviation_data: dict mapping dataset -> np.array of shape (n_layers,)
        performance_data: dict mapping dataset -> np.array of shape (n_layers,)
        best_layers: dict mapping dataset -> best layer index
    """
    print("=" * 70)
    print("Collecting Deviation Data for Valley of Consistency Plot (Rephrase)")
    print("=" * 70)

    n_layers = 48
    deviation_data = {}
    performance_data = {}
    best_layers = {}

    for dataset in DATASETS:
        print(f"\nLoading {dataset}...")

        try:
            analyzer, config = load_analyzer(dataset, model)
            metric = METRIC_MAP[dataset]

            # Check if metric exists
            if metric not in analyzer.test_means:
                print(f"  Warning: {metric} not found in test_means")
                print(f"  Available: {list(analyzer.test_means.keys())}")
                deviation_data[dataset] = np.full(n_layers, np.nan)
                performance_data[dataset] = np.full(n_layers, np.nan)
                continue

            # Get test means for all layers
            test_means = analyzer.test_means[metric].numpy()
            performance_data[dataset] = test_means

            # Get the proxy-selected golden layer for rephrase
            golden_layer = DATASET_GOLDEN_LAYERS[dataset][0]
            best_layers[dataset] = golden_layer
            best_perf = test_means[golden_layer]

            # Compute deviation for each layer
            deviation = np.abs(test_means - best_perf)
            deviation_data[dataset] = deviation

            print(f"  Test samples: {len(analyzer.test_indices)}")
            print(f"  Golden layer (Rephrase): {golden_layer} (perf: {best_perf:.4f})")
            print(f"  Max deviation: {np.max(deviation):.4f} at layer {np.argmax(deviation)}")

        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
            import traceback
            traceback.print_exc()
            deviation_data[dataset] = np.full(n_layers, np.nan)
            performance_data[dataset] = np.full(n_layers, np.nan)

    return deviation_data, performance_data, best_layers


def plot_valley(deviation_data, best_layers, output_path='results/valley_of_consistency_rephrase.png'):
    """
    Create the Valley of Consistency visualization for Rephrase.
    """
    print("\n" + "=" * 70)
    print("Creating Valley of Consistency Plot (Rephrase)")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(14, 7))

    n_layers = 48
    x = np.arange(n_layers)

    # Add shaded region for golden layers (16-18)
    golden_min = min(GOLDEN_LAYERS) - 0.5
    golden_max = max(GOLDEN_LAYERS) + 0.5
    ax.axvspan(golden_min, golden_max, alpha=0.2, color='gold',
               label='Golden Layer Region')

    # Plot vertical lines for each golden layer
    for gl in GOLDEN_LAYERS:
        ax.axvline(x=gl, color='gold', linestyle='--', alpha=0.5, linewidth=1)

    # Plot each dataset as a line
    for dataset in DATASETS:
        dev = deviation_data[dataset]
        if np.all(np.isnan(dev)):
            print(f"  Skipping {dataset} - no valid data")
            continue

        color = DATASET_COLORS[dataset]
        label = DATASET_NAMES[dataset]

        ax.plot(x, dev, color=color, linewidth=2, label=label, marker='o',
                markersize=3, alpha=0.8)

        # Mark the dataset-specific golden layer(s) with a star (deviation = 0)
        for gl in DATASET_GOLDEN_LAYERS[dataset]:
            ax.plot(gl, dev[gl], marker='*', color=color, markersize=15,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)

    # Customize plot
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Absolute Deviation from Golden Layer', fontsize=12)
    ax.set_title('Valley of Consistency: Deviation from Best Layer (Rephrase Accuracy)\n'
                 'Golden layers {16, 18} highlighted; '
                 'Dataset-specific golden layer marked with ★', fontsize=12)

    # Set x-axis ticks
    ax.set_xticks(range(0, 48, 5))
    ax.set_xlim(-0.5, 47.5)

    # Set y-axis limits (deviation is always >= 0)
    ax.set_ylim(-0.02, None)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-')

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved valley plot to: {output_path}")

    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved PDF to: {pdf_path}")

    plt.close()

    return fig


def plot_valley_zoomed(deviation_data, best_layers, output_path='results/valley_of_consistency_rephrase_zoomed.png'):
    """
    Create a zoomed-in plot focusing on the golden layer region.
    """
    print("\n" + "=" * 70)
    print("Creating Zoomed Valley Plot (Rephrase)")
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
        dev = deviation_data[dataset]
        if np.all(np.isnan(dev)):
            continue

        dev_zoomed = dev[zoom_start:zoom_end+1]
        color = DATASET_COLORS[dataset]
        label = DATASET_NAMES[dataset]

        ax.plot(x, dev_zoomed, color=color, linewidth=2.5, label=label, marker='o',
                markersize=6, alpha=0.9)

        # Mark the dataset-specific golden layer with a star
        for gl in DATASET_GOLDEN_LAYERS[dataset]:
            if zoom_start <= gl <= zoom_end:
                ax.plot(gl, deviation_data[dataset][gl], marker='*', color=color,
                        markersize=20, markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # Customize plot
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Absolute Deviation from Golden Layer', fontsize=12)
    ax.set_title('Valley of Consistency - Zoomed (Rephrase Accuracy)\n'
                 'Both datasets converge near zero around layers 16-18', fontsize=12)

    # Set x-axis ticks
    ax.set_xticks(range(zoom_start, zoom_end + 1))
    ax.set_xlim(zoom_start - 0.5, zoom_end + 0.5)

    # Set y-axis limits
    ax.set_ylim(-0.01, None)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-')

    # Create legend
    ax.legend(loc='upper right', fontsize=9)

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


def print_valley_stats(deviation_data, performance_data):
    """Print statistics about the valley of consistency."""
    print("\n" + "=" * 70)
    print("VALLEY OF CONSISTENCY STATISTICS (Rephrase)")
    print("=" * 70)

    # Header
    print(f"\n{'Dataset':<18} {'Golden Layer':<12} {'Dev at Golden':<15} {'Mean Dev (16-18)':<18} {'Mean Dev (Other)':<18}")
    print("-" * 85)

    golden_range = list(range(16, 19))  # 16, 17, 18
    other_layers = [i for i in range(48) if i not in golden_range]

    for dataset in DATASETS:
        dev = deviation_data[dataset]
        if np.all(np.isnan(dev)):
            print(f"{DATASET_NAMES[dataset]:<18} {'N/A':<12} {'N/A':<15} {'N/A':<18} {'N/A':<18}")
            continue

        golden_layer = DATASET_GOLDEN_LAYERS[dataset][0]
        dev_at_golden = dev[golden_layer]
        mean_dev_golden = np.mean([dev[l] for l in golden_range])
        mean_dev_other = np.mean([dev[l] for l in other_layers])

        print(f"{DATASET_NAMES[dataset]:<18} {golden_layer:<12} {dev_at_golden:<15.4f} {mean_dev_golden:<18.4f} {mean_dev_other:<18.4f}")

    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL VALLEY DEPTH ANALYSIS")
    print("=" * 70)

    # Average deviation at each layer across all datasets
    valid_datasets = [d for d in DATASETS if not np.all(np.isnan(deviation_data[d]))]

    if not valid_datasets:
        print("No valid data available")
        return

    avg_dev_per_layer = np.zeros(48)
    for layer in range(48):
        avg_dev_per_layer[layer] = np.mean([deviation_data[d][layer] for d in valid_datasets])

    # Find the "valley" - layers with lowest average deviation
    sorted_layers = np.argsort(avg_dev_per_layer)
    print(f"\nLayers with lowest average deviation (top 10):")
    for i, layer in enumerate(sorted_layers[:10]):
        print(f"  {i+1}. Layer {layer}: avg dev = {avg_dev_per_layer[layer]:.4f}")

    # Compare golden vs non-golden
    golden_avg = np.mean([avg_dev_per_layer[l] for l in golden_range])
    non_golden_avg = np.mean([avg_dev_per_layer[l] for l in other_layers])

    print(f"\nGolden layers (16-18) average deviation: {golden_avg:.4f}")
    print(f"Other layers average deviation: {non_golden_avg:.4f}")
    print(f"Ratio (golden/other): {golden_avg/non_golden_avg:.2%}")

    if golden_avg < non_golden_avg * 0.5:
        print("\n→ Strong valley effect: Golden layers have <50% of other layers' deviation")
        print("→ This supports dataset-independent layer selection for Rephrase!")
    elif golden_avg < non_golden_avg * 0.75:
        print("\n→ Moderate valley effect: Golden layers have lower deviation than others")
    else:
        print("\n→ Weak valley effect: Golden layers not significantly better")


def main():
    print("=" * 70)
    print("Valley of Consistency Plot - Rephrase Accuracy")
    print("Model: GPT2-XL | Metric: Rephrase Accuracy")
    print("Datasets: ZSRE (Layer 16), Counterfact (Layer 18)")
    print("=" * 70)

    # Collect deviation data
    deviation_data, performance_data, best_layers = collect_deviation_data('gpt2-xl')

    # Print statistics
    print_valley_stats(deviation_data, performance_data)

    # Create main valley plot
    output_path = 'results/valley_of_consistency_rephrase.png'
    plot_valley(deviation_data, best_layers, output_path)

    # Create zoomed plot
    zoom_path = 'results/valley_of_consistency_rephrase_zoomed.png'
    plot_valley_zoomed(deviation_data, best_layers, zoom_path)

    # Save raw data
    np.save('results/deviation_data_rephrase.npy', deviation_data)
    print(f"\n✓ Saved raw data to: results/deviation_data_rephrase.npy")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
