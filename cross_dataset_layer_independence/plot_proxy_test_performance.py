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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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

METRIC_LIST = ['Rewrite', "Rephrase", "Locality", "Portability", "Fluency", "Overall"]

METRIC_MAP = {
    'zsre': {
        "Rewrite": 'post_rewrite',
        "Rephrase": 'post_rephrase',
        "Locality": 'post_neighborhood',
        "Portability": 'post_one_hop',
        "Overall": 'post_overall',
    },
    'counterfact': {
        "Rewrite": 'post_rewrite',
        "Rephrase": 'post_rephrase',
        "Locality": 'post_neighborhood',
        "Portability": 'post_one_hop',
        "Overall": 'post_overall',
    },
    'wikibio': {
        "Rewrite": 'post_rewrite_acc',
        "Locality": 'post_locality_acc',
        "Fluency": 'post_fluency',
        "Overall": 'post_overall',
    },
    'wikicounterfact': {
        "Rewrite": 'post_rewrite_acc',
        "Locality": 'post_locality_acc',
        "Portability": 'post_portability_acc',
        "Fluency": 'post_fluency',
        "Overall": 'overall',
    },
    'wikirecent': {
        "Rewrite": 'post_rewrite_acc',
        "Locality": 'post_locality_acc',
        "Portability": 'post_portability_acc',
        "Fluency": 'post_fluency',
        "Overall": 'overall',
    }
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
    test_std_dict = {}
    golden_layers_set = set()
    dataset_golden = {}
    proxy_std_dict = {}

    print(f"\n  Loading {config['display_name']}...")

    for i, dataset in enumerate(datasets):
        best_layers[dataset] = {}
        test_means_dict[dataset] = {}
        dataset_golden[dataset] = {}
        test_std_dict[dataset] = {}
        proxy_std_dict[dataset] = {}
        for metric_high in METRIC_LIST:
            try:
                analyzer, _ = load_analyzer(dataset, model_name)
                metric = METRIC_MAP[dataset][metric_high]

                print(f"  Dataset: {dataset} (analyzer.test_means keys: {list(analyzer.test_means.keys())})")

                if metric not in analyzer.test_means:
                    print(f"    Warning: {metric} not found for {dataset}")
                    deviation_matrix[i, :] = np.nan
                    continue

                test_means = analyzer.test_means[metric].numpy()
                # test_means_dict[dataset] = {}
                test_means_dict[dataset][metric_high] = test_means
                test_std_dict[dataset][metric_high] = analyzer.test_stds[metric].numpy()
                proxy_std_dict[dataset][metric_high] = analyzer.proxy_stds[metric].numpy()


                best_perf = np.nanmax(test_means)
                best_layer_list = np.where(test_means == best_perf)[0].tolist()
                # best_layers[dataset] = {}
                best_layers[dataset][metric_high] = best_layer_list

                proxy_means = analyzer.proxy_means[metric].numpy()
                proxy_best_perf = np.nanmax(proxy_means)
                proxy_best_layers = np.where(proxy_means == proxy_best_perf)[0].tolist()
                golden_layers_set.update(proxy_best_layers)
                # dataset_golden[dataset] = {}
                dataset_golden[dataset][metric_high] = proxy_best_layers

                deviation = np.abs(test_means - best_perf)
                deviation_matrix[i, :] = deviation

                print(f"    {dataset}: best={best_layer_list}, proxy={proxy_best_layers}")

            except Exception as e:
                print(f"    Error loading {dataset}: {e}")
                deviation_matrix[i, :] = np.nan

    return {
        'deviation_matrix': deviation_matrix,
        'proxy_scores': proxy_means,
        'test_scores': test_means_dict,
        'test_stds': test_std_dict,
        'proxy_stds': proxy_std_dict,
        "all_proxy_scores": analyzer.proxy_means,
        "all_test_scores": analyzer.test_means,
        'datasets': datasets,
        'best_layers': best_layers,
        'golden_layers': golden_layers_set,
        'dataset_golden_layers': dataset_golden,
        'n_layers': n_layers,
    }

import matplotlib.pyplot as plt
import os

def plot_layer_selection_scatter(model_data):
    """
    Academic-quality scatter plot:
      - x: layer index
      - y: performance
      - color: metric
      - marker: dataset
      - solid: test-selected
      - hatched box: proxy-selected
    """

    test_means = model_data["test_scores"]
    best_layers = model_data["best_layers"]
    dataset_golden = model_data["dataset_golden_layers"]
    datasets = model_data["datasets"]

    # ---- Visual mappings ----
    metric_colors = {
        "Rewrite": "tab:blue",
        "Rephrase": "tab:orange",
        "Locality": "tab:green",
        "Portability": "tab:red",
        "Fluency": "tab:purple",
        "Overall": "tab:brown",
    }

    dataset_markers = {
        "zsre": "o",
        "counterfact": "^",
        "wikibio": "D",
        "wikicounterfact": "P",
        "wikirecent": "X",
    }

    alpha_test = 0.7
    alpha_proxy = 0.7

    fig, ax = plt.subplots(figsize=(10, 6))

    def minmax_scale(arr):
        arr = np.asarray(arr, dtype=float)
        min_v = np.nanmin(arr) -0.2
        max_v = np.nanmax(arr) +0.2
        if max_v - min_v == 0:
            return np.zeros_like(arr)
        return (arr - min_v) / (max_v - min_v)



    # ---- Plot ----
    for dataset in datasets:

        for metric, scores in test_means[dataset].items():
            if metric == "Fluency":
                scores = minmax_scale(scores)

            color = metric_colors.get(metric, "gray")
            marker = dataset_markers.get(dataset, "o")

            # Test-selected (solid)
            for layer in best_layers[dataset].get(metric, []):
                ax.scatter(
                    layer,
                    scores[layer],
                    color=color,
                    marker=marker,
                    alpha=alpha_test,
                    s=80,
                    edgecolor="black",
                    linewidth=0.5,
                )

            # Proxy-selected (hatched square)
            for layer in dataset_golden[dataset].get(metric, []):
                ax.scatter(
                    layer,
                    scores[layer],
                    facecolor="none",
                    edgecolor=color,
                    marker=marker,      # ← dataset-specific marker
                    alpha=alpha_proxy,
                    s=100,
                    linewidth=2.5,
                    hatch="",
                )


    # ---- Labels ----
    ax.set_xlabel("Layer selected")
    ax.set_ylabel("Performance")
    ax.set_title("Layer Selection vs Performance")

    ax.grid(True, linestyle="--", alpha=0.3)

    # ---- Legends ----
    metric_legend = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, label=m, markersize=8)
        for m, c in metric_colors.items()
    ]

    dataset_legend = [
        plt.Line2D([0], [0], marker=mk, color='k',
                   linestyle='None', label=ds, markersize=8)
        for ds, mk in dataset_markers.items()
    ]

    style_legend = [
        Line2D(
            [0], [0],
            marker='o',
            color='k',
            linestyle='None',
            label='Test-selected',
            markerfacecolor='k',
            alpha=alpha_test,
        ),
        Patch(
            facecolor='none',
            edgecolor='k',
            hatch='',
            label='Proxy-selected',
            alpha=alpha_proxy,
        ),
    ]


    ax.legend(
        handles=metric_legend + dataset_legend + style_legend,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False,
    )

    plt.tight_layout()

    # ---- Save outputs ----
    output_pdf = 'results/plot_proxy_test_performance.pdf'
    output_png = output_pdf.replace('.pdf', '.png')

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    # Vector PDF (publication-ready)
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')

    # High-resolution raster
    plt.savefig(output_png, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved scatter plot to: {output_pdf} and {output_png}")

    plt.close(fig)



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


DATA_CACHE_PATH = Path(__file__).parent / 'results' / 'model_data_dict_scatter.pt'


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

import matplotlib.pyplot as plt
import numpy as np

def plot_layer_selection_bar(model_data, dataset_name):
    """
    Academic-quality bar plot for a single dataset:
      - One group per metric
      - Two bars per group: 
          1) Proxy-selected (dataset golden layer) - light/transparent
          2) Test-selected (best layer) - solid
      - If multiple best layers, pick the first one
    """
    test_means = model_data["test_scores"]
    best_layers = model_data["best_layers"]
    dataset_golden = model_data["dataset_golden_layers"]
    test_std = model_data["test_stds"]


    metrics = list(test_means[dataset_name].keys())
    n_metrics = len(metrics)

    # Performance values
    proxy_values = []
    best_values = []
    proxy_layers_list = []
    best_layers_list = []
    test_std_list = []
    proxy_std_list = []

    print(dataset_name)
    for metric in metrics:
        # Proxy-selected layer(s) performance: mean if multiple
        print(metric)
        proxy_layers = dataset_golden[dataset_name].get(metric, [])
        if len(proxy_layers) == 0:
            proxy_values.append(np.nan)
        else:
            proxy_values.append(float(test_means[dataset_name][metric][proxy_layers[0]]))
            proxy_std_list.append(float(model_data["proxy_stds"][dataset_name][metric][proxy_layers[0]]))
            proxy_layers_list.append(proxy_layers[0])
        # print("Proxy Perf.: ",test_means[dataset_name][metric][proxy_layers[0]])
        # print("proxy Layer: ", proxy_layers[0])

        # Best layer performance: pick first best layer
        best_layer_list = best_layers[dataset_name].get(metric, [])
        if len(best_layer_list) == 0:
            best_values.append(np.nan)
        else:
            l = best_layer_list[0]
            best_values.append(float(test_means[dataset_name][metric][l]))
            test_std_list.append(float(test_std[dataset_name][metric][l]))
            best_layers_list.append(l)
        # print("Test Perf.: ",test_means[dataset_name][metric][l])
        # print("Test Layer: ", l)
        # print("-------------")

    print('Dataset name: ',dataset_name)
    print('test_std',test_std_list)
    print('proxy_std',proxy_std_list)
    print(metrics)
    res_ = {
        dataset_name:{
            "proxy": proxy_values,
            "test": best_values,
            "proxy_std": proxy_std_list,
            "test_std": test_std_list,
            "layers_p": proxy_layers_list,
            "layers_t": best_layers_list,
            'metrics': metrics
        }
    }
    def pretty_print_results(res, float_fmt="{:.7g}"):
        print("{")
        for dataset, values in res.items():
            print(f'    "{dataset.title()}": {{')
            
            for key, val in values.items():
                print(f'        "{key}": ', end="")
                
                # ---- Layers as strings ----
                if key in ("layers_p", "layers_t") and isinstance(val, list):
                    layer_str = ", ".join(f'"{x}"' for x in val)
                    print(f"[{layer_str}],")
                
                # ---- Float lists ----
                elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
                    nums = ", ".join(float_fmt.format(x) for x in val)
                    print(f"[{nums}],")
                
                # ---- String lists ----
                elif isinstance(val, list) and all(isinstance(x, str) for x in val):
                    strs = ", ".join(f'"{x}"' for x in val)
                    print(f"[{strs}],")
                
                else:
                    print(f"{val},")
            
            print("    }")
        print("}")


    pretty_print_results(res_)

        

    # ---- Plot ----
    x = np.arange(len(proxy_values))  # metric positions
    width = 0.20  # bar width

    colors = ["blue", "green", "orange","purple", "red", "brown"]

    print(len(proxy_values))
    print(len(best_values))
    print(width)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_proxy = ax.bar(x - width/2, proxy_values, width, alpha=0.3, color = colors[: len(proxy_values)])
    bars_best = ax.bar(x + width/2, best_values, width, alpha=1.0, color = colors[: len(proxy_values)])

    # Labels & ticks
    # Tick positions: one per bar
    tick_positions = []
    tick_labels = []
    for i in range(len(proxy_values)):
        tick_positions.extend([x[i] - width/2, x[i] + width/2])
        tick_labels.extend(["Proxy", "Test"])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)

    # ax.set_ylabel("Performance")
    # ax.set_title(f"Layer Performance for Dataset: {dataset_name}")
    print('Processed for dataset:', dataset_name)

    sp_metric = []
    for m in metrics:
        sp_metric.append(m)

    # --- Separate legend ---
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[i], label=sp_metric[i]) for i in range(len(proxy_values))]
    ax.legend(handles=legend_handles, title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')


    # ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.grid(False)

    plt.tight_layout()
    # plt.show()
    # ---- Save outputs ----
    output_pdf = 'results/plot_proxy_test_performance.pdf'
    output_png = output_pdf.replace('.pdf', '.png')

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    # Vector PDF (publication-ready)
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')

    # High-resolution raster
    plt.savefig(output_png, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved scatter plot to: {output_pdf} and {output_png}")

    plt.close(fig)


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
        model_order = ['gemma3-12b']#['gpt2-xl', 'llama2-7b', 'gemma3-12b']
        for model_name in model_order:
            model_data_dict[model_name] = compute_model_data(model_name)

        # Save for future use / sharing
        save_model_data(model_data_dict, model_order)

    # Print summary
    # print_summary(model_data_dict, model_order=model_order)

    # Create combined heatmap
    output_path = 'results/plot_proxy_test_performance.pdf'
    # plot_combined_heatmap(model_data_dict, output_path, model_order=model_order)
    # plot_layer_selection_scatter(model_data_dict['gemma3-12b'])
    for i in range(5):
        plot_layer_selection_bar(model_data_dict['gemma3-12b'], model_data_dict['gemma3-12b']['datasets'][i])  # Example for ZSRE

    # Save raw data
    # for model_name in model_order:
    #     data = model_data_dict[model_name]
    #     np.save(f'results/deviation_matrix_{model_name}_rewrite.npy',
    #             data['deviation_matrix'])
    # print(f"\n✓ Saved raw deviation matrices")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
