#!/usr/bin/env python3
"""
Combined Layer-wise Performance Deviation Heatmap - Rewrite Accuracy

Creates a publication-quality combined figure showing layer deviation heatmaps
for four models: GPT-2XL, LLaMA2-7B, Qwen3-32B, and Gemma3-12B.

Layout: 4 rows (one per model), horizontally aligned
- Row 1: GPT-2XL (48 layers, 5 datasets)
- Row 2: LLaMA2-7B (32 layers, 5 datasets)
- Row 3: Qwen3-32B (64 layers, 3 datasets: zsre, counterfact, wikibio)
- Row 4: Gemma3-12B (48 layers, 3 datasets: zsre, counterfact, wikibio)

Each subplot shows:
- Rows: Datasets
- Columns: Layers
- Cell color: Absolute deviation from best-performing layer
- Gold border: Golden region (union of proxy-selected layers)
- Green border: Dataset-specific proxy-selected layers
- Star marker: Best performing layer on test set
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import importlib.util
import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataset_model_registry import get_config, DATASET_MODEL_REGISTRY
import heatmap_api

# Model configurations
MODEL_CONFIGS = {
    'gpt2-xl': {
        'display_name': 'GPT-2 XL',
        'n_layers': 48,
        'datasets': ['wikirecent', 'wikibio', 'zsre', 'wikicounterfact', 'counterfact'],
        'tick_step': 8,
    },
    'llama2-7b': {
        'display_name': 'LLaMA2-7B',
        'n_layers': 32,
        'datasets': ['wikirecent', 'wikibio', 'zsre', 'wikicounterfact', 'counterfact'],
        'tick_step': 4,
    },
    'qwen3-32b': {
        'display_name': 'Qwen3-32B',
        'n_layers': 64,
        'datasets': ['zsre', 'counterfact', 'wikibio'],
        'tick_step': 8,
    },
    'gemma3-12b': {
        'display_name': 'Gemma3-12B',
        'n_layers': 48,
        'datasets': ['wikirecent', 'wikibio', 'zsre', 'wikicounterfact', 'counterfact'],
        'tick_step': 8,
    },
}

# Metric mapping
METRIC_MAP = {
    'zsre': 'post_rewrite',
    'counterfact': 'post_rewrite',
    'wikibio': 'post_rewrite_acc',
    'wikicounterfact': 'post_rewrite_acc',
    'wikirecent': 'post_rewrite_acc'
}

# Pretty names for datasets
DATASET_PRETTY_NAMES = {
    'wikirecent': 'WikiRecent',
    'wikibio': 'WikiBio',
    'zsre': 'ZSRE',
    'wikicounterfact': 'WikiCounterfact',
    'counterfact': 'Counterfact'
}


def load_analyzer(dataset, model):
    """Load the golden layer analyzer for a dataset."""
    config = get_config(dataset, model)

    api_path = config['api_path']
    spec = importlib.util.spec_from_file_location("analyzer_module", api_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    api_class_name = config.get('api_class', 'GoldenLayerAnalyzer')
    analyzer_class = getattr(module, api_class_name)

    data_dir = config['base_dir'] / config.get('data_dir', 'allLayerPerformance')
    n_layers = config.get('n_layers', 48)

    try:
        analyzer = analyzer_class(data_dir=str(data_dir), n_layers=n_layers)
    except TypeError:
        analyzer = analyzer_class(data_dir=str(data_dir))

    analyzer.load_data(verbose=False)

    if hasattr(analyzer, 'compute_overall_metric'):
        analyzer.compute_overall_metric(verbose=False)

    proxy_size = config.get('proxy_size', 100)
    analyzer.split_dataset(proxy_size=proxy_size, random_seed=42, verbose=False)
    analyzer.compute_proxy_test_means(verbose=False)

    return analyzer, config


def compute_model_data(model_name):
    """
    Compute deviation matrix and related data for a single model.

    Returns:
        dict with deviation_matrix, datasets, best_layers, golden_layers, dataset_golden_layers
    """
    config = MODEL_CONFIGS[model_name]
    datasets = config['datasets']
    n_layers = config['n_layers']

    deviation_matrix = np.zeros((len(datasets), n_layers))
    best_layers = {}
    test_means_dict = {}
    golden_layers_set = set()
    dataset_golden = {}

    print(f"\n  Loading {config['display_name']}...")

    for i, dataset in enumerate(datasets):
        try:
            analyzer, _ = load_analyzer(dataset, model_name)
            metric = METRIC_MAP[dataset]

            if metric not in analyzer.test_means:
                print(f"    Warning: {metric} not found for {dataset}")
                deviation_matrix[i, :] = np.nan
                continue

            test_means = analyzer.test_means[metric].numpy()
            test_means_dict[dataset] = test_means

            best_perf = np.nanmax(test_means)
            best_layer_list = np.where(np.isclose(test_means, best_perf, atol=1e-3))[0].tolist()
            best_layers[dataset] = best_layer_list

            proxy_means = analyzer.proxy_means[metric].numpy()
            proxy_best_perf = np.nanmax(proxy_means)
            proxy_best_layers = np.where(proxy_means == proxy_best_perf)[0].tolist()
            golden_layers_set.update(proxy_best_layers)
            dataset_golden[dataset] = proxy_best_layers

            deviation = np.abs(test_means - best_perf)
            deviation_matrix[i, :] = deviation

            print(f"    {dataset}: best={best_layer_list}, proxy={proxy_best_layers}")

        except Exception as e:
            print(f"    Error loading {dataset}: {e}")
            deviation_matrix[i, :] = np.nan

    return {
        'deviation_matrix': deviation_matrix,
        'proxy_scores': proxy_means,
        'test_scores': test_means,
        "all_proxy_scores": analyzer.proxy_means,
        "all_test_scores": analyzer.test_means,
        'datasets': datasets,
        'best_layers': best_layers,
        'golden_layers': golden_layers_set,
        'dataset_golden_layers': dataset_golden,
        'n_layers': n_layers,
    }


def plot_combined_heatmap(model_data_dict, output_path='results/combined_rewrite_heatmap.pdf', model_order = ['gpt2-xl', 'llama2-7b', 'qwen3-32b', 'gemma3-12b']):
    """
    Create a combined publication-quality heatmap figure.
    """
    print("\n" + "=" * 70)
    print("Creating Combined Heatmap Figure")
    print("=" * 70)

    # Figure setup with proper sizing for 4 models
    # Use gridspec for precise control
    fig = plt.figure(figsize=(14, 12))

    # Create gridspec: 4 rows with different heights proportional to number of datasets
    n_datasets = [len(MODEL_CONFIGS[m]['datasets']) for m in model_order]

    # Create gridspec with height ratios based on number of datasets
    gs = GridSpec(len(model_order), 2, figure=fig,
                  height_ratios=n_datasets,
                  width_ratios=[50, 1],  # Main plot and colorbar
                  hspace=0.15, wspace=0.05)

    # Shared colormap and normalization
    cmap = plt.cm.RdYlBu_r

    # Find global vmax for consistent color scale
    all_deviations = []
    for model_name in model_order:
        data = model_data_dict[model_name]
        all_deviations.append(np.nanmax(data['deviation_matrix']))
    global_vmax = max(all_deviations)
    norm = Normalize(vmin=0, vmax=global_vmax)

    axes = []
    images = []

    for row, model_name in enumerate(model_order):
        data = model_data_dict[model_name]
        config = MODEL_CONFIGS[model_name]

        # Create subplot
        ax = fig.add_subplot(gs[row, 0])
        axes.append(ax)

        # Plot heatmap
        im = ax.imshow(data['deviation_matrix'], cmap=cmap, aspect='auto',
                       norm=norm, interpolation='none')
        images.append(im)

        # Y-axis: datasets
        ax.set_yticks(range(len(data['datasets'])))
        ax.set_yticklabels([DATASET_PRETTY_NAMES.get(d, d) for d in data['datasets']],
                          fontsize=12)

        # X-axis: layers
        tick_step = config['tick_step']
        n_layers = config['n_layers']
        ax.set_xticks(range(0, n_layers, tick_step))
        ax.set_xticklabels(range(0, n_layers, tick_step), fontsize=13)

        # Only show x-label on bottom plot
        if row == len(model_order) - 1:
            ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')

        # Model label on left
        ax.set_ylabel(config['display_name'], fontsize=12, fontweight='bold')

        # Add golden region and layer markers
        heatmap_api.highlight_mark_layers(
            data['golden_layers'],
            data['dataset_golden_layers'],
            data['datasets'],
            data['best_layers'],
            ax
        )

        ax.grid(False)

    # Add shared colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.set_label('Absolute Deviation from Best Layer', fontsize=12, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    # Main title
    # fig.suptitle('Layer-wise Performance Deviation Heatmap: Rewrite Accuracy\n'
    #              'Gold = Golden Region (Proxy-Selected Union); '
    #              'Green = Dataset Proxy-Selected; '
    #              '★ = Best Test Layer',
    #              fontsize=12, fontweight='bold', y=0.98)

    # Create legend at the bottom
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=2,
                       label='Golden Layers (Union)'),
        mpatches.Patch(facecolor='none', edgecolor='limegreen', linewidth=2,
                       label='Proxy Dataset Selected Layer(s)'),
        plt.Line2D([0], [0], marker='*', color='white', markeredgecolor='black',
                   markersize=10, linestyle='None', label='Optimal Layer(s)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=12, frameon=True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figures
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as PDF (primary for papers)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\n✓ Saved PDF: {output_path}")

    # Save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved PNG: {png_path}")

    plt.close()

    return fig


def print_summary(model_data_dict, model_order=['gpt2-xl', 'llama2-7b', 'qwen3-32b', 'gemma3-12b']):
    """Print summary statistics for all models."""
    print("\n" + "=" * 80)
    print("SUMMARY: Golden Layer Statistics Across Models")
    print("=" * 80)

    for model_name in model_order:
        data = model_data_dict[model_name]
        config = MODEL_CONFIGS[model_name]

        golden_layers = sorted(data['golden_layers'])
        golden_indices = list(data['golden_layers'])
        n_layers = config['n_layers']
        non_golden_indices = [i for i in range(n_layers) if i not in golden_indices]

        print(f"\n{config['display_name']} ({n_layers} layers):")
        print(f"  Golden layers: {golden_layers}")

        # Compute average deviation
        valid_mask = ~np.all(np.isnan(data['deviation_matrix']), axis=1)
        if valid_mask.any() and golden_indices:
            valid_matrix = data['deviation_matrix'][valid_mask]
            golden_avg = np.nanmean(valid_matrix[:, golden_indices])
            non_golden_avg = np.nanmean(valid_matrix[:, non_golden_indices]) if non_golden_indices else 0
            ratio = golden_avg / non_golden_avg if non_golden_avg > 0 else 0

            print(f"  Avg Golden Dev:     {golden_avg:.4f}")
            print(f"  Avg Non-Golden Dev: {non_golden_avg:.4f}")
            print(f"  Ratio:              {ratio:.3f} ({ratio*100:.1f}%)")


DATA_CACHE_PATH = Path(__file__).parent / 'results' / 'model_data_dict.pt'


def save_model_data(model_data_dict, model_order, filepath=DATA_CACHE_PATH):
    """Save model_data_dict using torch.save (handles tensors, numpy, dicts, sets natively)."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'model_data_dict': model_data_dict,
        'model_order': model_order,
    }

    torch.save(save_data, filepath)
    print(f"\n✓ Saved model_data_dict to: {filepath}")


def load_model_data(filepath=DATA_CACHE_PATH):
    """Load model_data_dict using torch.load."""
    save_data = torch.load(filepath, weights_only=False)
    return save_data['model_data_dict'], save_data['model_order']


def main():
    print("=" * 80)
    print("Combined Layer-wise Performance Deviation Heatmap")
    print("Metric: Rewrite Accuracy")
    print("Models: GPT-2 XL, LLaMA2-7B, Gemma3-12B")
    print("=" * 80)

    # Check if cached data exists
    if DATA_CACHE_PATH.exists():
        print(f"\nLoading cached data from: {DATA_CACHE_PATH}")
        model_data_dict, model_order = load_model_data()
        print(f"Loaded models: {model_order}")
    else:
        # Compute data for all models
        print("\nNo cached data found. Computing from source...")
        model_data_dict = {}
        model_order = ['gpt2-xl', 'llama2-7b', 'gemma3-12b']
        for model_name in model_order:
            model_data_dict[model_name] = compute_model_data(model_name)

        # Save for future use / sharing
        save_model_data(model_data_dict, model_order)

    # Print summary
    print_summary(model_data_dict, model_order=model_order)

    # Create combined heatmap
    output_path = 'results/combined_rewrite_heatmap.pdf'
    plot_combined_heatmap(model_data_dict, output_path, model_order=model_order)

    # Save raw data
    for model_name in model_order:
        data = model_data_dict[model_name]
        np.save(f'results/deviation_matrix_{model_name}_rewrite.npy',
                data['deviation_matrix'])
    print(f"\n✓ Saved raw deviation matrices")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
