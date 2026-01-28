#!/usr/bin/env python3
"""
Generate LaTeX table for Oracle vs Golden Layer Analysis across datasets and models.

Usage:
    python generate_latex_table.py --metric Rewrite
    python generate_latex_table.py --metric Overall --models gpt2-xl llama2-7b
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).parent))

import oracle_vs_golden_analysis
import heatmap_api
import dataset_model_registry


# Dataset display names and order
DATASETS = ['zsre', 'wikibio', 'wikicounterfact', 'wikirecent', 'counterfact']
DATASET_DISPLAY_NAMES = {
    'zsre': 'ZSRE',
    'wikibio': 'WikiBio',
    'wikicounterfact': 'WikiCounterfact',
    'wikirecent': 'WikiRecent',
    'counterfact': 'Counterfact'
}

# Model display names
MODEL_DISPLAY_NAMES = {
    'gpt2-xl': 'GPT2',
    'llama2-7b': 'LLaMA2',
    'gemma3-12b': 'Gemma3',
    'qwen3-32b': 'Qwen3'
}

MODEL_DISPLAY_NAMESL = {
    'gpt2-xl': 'GPT-2 XL',
    'llama2-7b': 'LLaMA2-7B',
    'gemma3-12b': 'Gemma3-12B',
    'qwen3-32b': 'Qwen3'
}


def get_oracle_golden_stats(dataset, model, metric_name, higher_is_better=True, golden_layer_proxy=False):
    """
    Get oracle mean/std, golden mean/std, and statistical significance for a dataset/model/metric.

    Args:
        dataset: Dataset name
        model: Model name
        metric_name: Display metric name (e.g., 'Rewrite', 'Overall')
        higher_is_better: True for most metrics, False for Fluency
        golden_layer_proxy: If True, select golden layer from proxy set; if False, from test set

    Returns:
        dict with 'oracle_mean', 'oracle_std', 'golden_mean', 'golden_std', 'stat_diff', 'available'
        If metric is not supported for dataset, returns {'available': False}
    """
    # Check if metric is supported for this dataset
    metric_key = heatmap_api.get_metric_key(dataset, metric_name)
    if metric_key is None:
        return {'available': False}

    try:
        # Load analyzer
        analyzer, config = heatmap_api.load_analyzer(dataset, model, dataset_model_registry)

        # Get test data
        data = analyzer.results[metric_key]
        if hasattr(data, 'numpy'):
            data = data.numpy()

        test_indices = analyzer.test_indices
        if hasattr(test_indices, 'numpy'):
            test_indices = test_indices.numpy()
        test_indices = np.array(test_indices)

        test_data = data[test_indices]

        # Get golden layer from test set (or proxy set if golden_layer_proxy=True)
        if golden_layer_proxy:
            layer_means = analyzer.proxy_means[metric_key]
        else:
            layer_means = analyzer.test_means[metric_key]

        if hasattr(layer_means, 'numpy'):
            layer_means = layer_means.numpy()
    
        if higher_is_better:
            best_perf = np.nanmax(layer_means)
            best_perf_proxy = np.nanmax(analyzer.proxy_means[metric_key].numpy())
            best_perf_test = np.nanmax(analyzer.test_means[metric_key].numpy())
        else:
            best_perf = np.nanmin(layer_means)
            best_perf_proxy = np.nanmin(analyzer.proxy_means[metric_key].numpy())
            best_perf_test = np.nanmin(analyzer.test_means[metric_key].numpy())

        golden_layers_proxy = np.where(analyzer.proxy_means[metric_key].numpy() == best_perf_proxy)[0].tolist()
        golden_layers = np.where(layer_means == best_perf)[0].tolist()
        golden_layers_test = np.where(analyzer.test_means[metric_key].numpy() == best_perf_test)[0].tolist()
        golden_layer = golden_layers[0]

        # Oracle performance (sample-wise best)
        if higher_is_better:
            oracle_perf = np.nanmax(test_data, axis=1)
        else:
            oracle_perf = np.nanmin(test_data, axis=1)

        # Golden layer performance
        golden_perf = test_data[:, golden_layer]

        # Compute statistics
        oracle_mean = np.nanmean(oracle_perf)
        oracle_std = np.nanstd(oracle_perf)
        golden_mean = np.nanmean(golden_perf)
        golden_std = np.nanstd(golden_perf)

        # Statistical significance (t-test)
        _, p_two_tailed = ttest_ind(oracle_perf, golden_perf, nan_policy='omit')
        stat_diff = 'Yes' if p_two_tailed <= 0.05 else 'No'

        return {
            'available': True,
            'oracle_mean': oracle_mean,
            'oracle_std': oracle_std,
            'golden_mean': golden_mean,
            'golden_std': golden_std,
            'stat_diff': stat_diff,
            'p_value': p_two_tailed,
            'golden_layer': golden_layer,
            'proxy_perf': analyzer.test_means[metric_key].numpy()[golden_layers_proxy[0]],
            'test_perf': best_perf_test,
            'proxy_layer': golden_layers_proxy[0],#",".join(map(str, golden_layers_proxy)),
            'test_layer': golden_layers_test[0],#",".join(map(str, golden_layers_test)),
        }

    except Exception as e:
        print(f"Error loading {dataset}/{model}: {e}")
        return {'available': False}


def format_value(mean, std):
    """Format mean ± std in LaTeX notation."""
    return f"{mean:.4f}$_{{\\pm{std:.4f}}}$"

def format_value_layer_perf(layer, perf):
    """Format mean ± std in LaTeX notation."""
    return f"{layer} & {perf:.4f}"


def generate_latex_table(metric_name, models, higher_is_better=True, golden_layer_proxy=False, verbose=False):
    """
    Generate LaTeX table for a given metric across all datasets and specified models.

    Args:
        metric_name: Display metric name (e.g., 'Rewrite', 'Overall')
        models: List of model names
        higher_is_better: True for most metrics, False for Fluency
        golden_layer_proxy: If True, select golden layer from proxy set; if False, from test set
        verbose: Print progress

    Returns:
        str: LaTeX table code
    """
    # Collect data for all combinations
    results = {}
    for model in models:
        results[model] = {}
        for dataset in DATASETS:
            if verbose:
                print(f"Processing {dataset}/{model}...")
            results[model][dataset] = get_oracle_golden_stats(
                dataset, model, metric_name, higher_is_better, golden_layer_proxy
            )

    # Build LaTeX table
    n_datasets = len(DATASETS)
    col_spec = "C{1cm}| c |" + " ".join([f"C{{2cm}}" for _ in DATASETS])

    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header = "Model & Method & " + " & ".join([DATASET_DISPLAY_NAMES[d] for d in DATASETS]) + "\\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows for each model
    for model in models:
        model_display = MODEL_DISPLAY_NAMES.get(model, model.upper())

        # Oracle row
        oracle_values = []
        for dataset in DATASETS:
            r = results[model][dataset]
            if r['available']:
                oracle_values.append(format_value(r['oracle_mean'], r['oracle_std']))
            else:
                oracle_values.append(" -- ")

        oracle_row = f"\\multirow{{2}}{{*}}{{\\rotatebox{{90}}{{{model_display}}}}}&Sample-Wise Optimal Layer & " + " & ".join(oracle_values) + "\\\\"
        lines.append(oracle_row)

        # Golden row
        golden_values = []
        for dataset in DATASETS:
            r = results[model][dataset]
            if r['available']:
                golden_values.append(format_value(r['golden_mean'], r['golden_std']))
            else:
                golden_values.append("--")

        golden_row = "        &Golden Layer & " + " & ".join(golden_values) + "\\\\"
        lines.append(golden_row)

        # Cmidrule
        lines.append(f"        \\cmidrule{{2-{n_datasets + 2}}}")

        # Statistical difference row
        stat_values = []
        for dataset in DATASETS:
            r = results[model][dataset]
            if r['available']:
                stat_values.append(r['stat_diff'])
            else:
                stat_values.append("--")

        stat_row = "&Statistical Difference & " + " & ".join(stat_values) + " \\\\"
        lines.append(stat_row)
        lines.append("\\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")

    return "\n".join(lines)

def generate_latex_table_layers(metric_name, models, higher_is_better=True, golden_layer_proxy=False, verbose=False):
    """
    Generate LaTeX table for a given metric across all datasets and specified models.

    Args:
        metric_name: Display metric name (e.g., 'Rewrite', 'Overall')
        models: List of model names
        higher_is_better: True for most metrics, False for Fluency
        golden_layer_proxy: If True, select golden layer from proxy set; if False, from test set
        verbose: Print progress

    Returns:
        str: LaTeX table code
    """
    # Collect data for all combinations
    results = {}
    for model in models:
        results[model] = {}
        for dataset in DATASETS:
            if verbose:
                print(f"Processing {dataset}/{model}...")
            results[model][dataset] = get_oracle_golden_stats(
                dataset, model, metric_name, higher_is_better, golden_layer_proxy
            )

    # Build LaTeX table
    n_datasets = len(DATASETS)
    col_spec = "C{1cm}| c |" + " ".join([f"C{{2cm}}" for _ in DATASETS])

    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header = "Model & Method & " + " & ".join([DATASET_DISPLAY_NAMES[d] for d in DATASETS]) + "\\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows for each model
    for model in models:
        model_display = MODEL_DISPLAY_NAMESL.get(model, model.upper())

        # Oracle row
        oracle_values = []
        for dataset in DATASETS:
            r = results[model][dataset]
            if r['available']:
                oracle_values.append(format_value_layer_perf(r['proxy_layer'], r['proxy_perf']))
            else:
                # oracle_values.append("-- & --")
                c=1+1

        oracle_row = f"\\multirow{{2}}{{*}}{{\\rotatebox{{0}}{{{model_display}}}}}&Proxy & " + " & ".join(oracle_values) + "\\\\"
        lines.append(oracle_row)

        # Golden row
        golden_values = []
        for dataset in DATASETS:
            r = results[model][dataset]
            if r['available']:
                golden_values.append(format_value_layer_perf(r['test_layer'], r['test_perf']))
            else:
                # golden_values.append("-- & --")
                c=2+2

        golden_row = "        &Test & " + " & ".join(golden_values) + "\\\\"
        lines.append(golden_row)

        lines.append("\\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")

    return "\n".join(lines)

# def generate_latex_table(metric_name, models, higher_is_better=True, golden_layer_proxy=False, verbose=False):
#     """
#     Generate LaTeX table for a given metric across all datasets and specified models.

#     Args:
#         metric_name: Display metric name (e.g., 'Rewrite', 'Overall')
#         models: List of model names
#         higher_is_better: True for most metrics, False for Fluency
#         golden_layer_proxy: If True, select golden layer from proxy set; if False, from test set
#         verbose: Print progress

#     Returns:
#         str: LaTeX table code
#     """
#     # Collect data for all combinations
#     results = {}
#     for model in models:
#         results[model] = {}
#         for dataset in DATASETS:
#             if verbose:
#                 print(f"Processing {dataset}/{model}...")
#             results[model][dataset] = get_oracle_golden_stats(
#                 dataset, model, metric_name, higher_is_better, golden_layer_proxy
#             )

#     # Build LaTeX table
#     n_datasets = len(DATASETS)
#     col_spec = "C{1cm}| c |" + " ".join([f"C{{2cm}}" for _ in DATASETS])

#     lines = []
#     lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
#     lines.append("\\toprule")

#     # Header row
#     header = "Model & Method & " + " & ".join([DATASET_DISPLAY_NAMES[d] for d in DATASETS]) + "\\\\"
#     lines.append(header)
#     lines.append("\\midrule")

#     # Data rows for each model
#     for model in models:
#         model_display = MODEL_DISPLAY_NAMES.get(model, model.upper())

#         # Oracle row
#         oracle_values = []
#         for dataset in DATASETS:
#             r = results[model][dataset]
#             if r['available']:
#                 oracle_values.append(format_value(r['oracle_mean'], r['oracle_std']))
#             else:
#                 oracle_values.append("--")

#         oracle_row = f"\\multirow{{3}}{{*}}{{\\rotatebox{{90}}{{{model_display}}}}}&Sample-Wise Optimal Layer & " + " & ".join(oracle_values) + "\\\\"
#         lines.append(oracle_row)

#         # Golden row
#         golden_values = []
#         for dataset in DATASETS:
#             r = results[model][dataset]
#             if r['available']:
#                 golden_values.append(format_value(r['golden_mean'], r['golden_std']))
#             else:
#                 golden_values.append("--")

#         golden_row = "        &Golden Layer & " + " & ".join(golden_values) + "\\\\"
#         lines.append(golden_row)

#         # Cmidrule
#         lines.append(f"        \\cmidrule{{2-{n_datasets + 2}}}")

#         # Statistical difference row
#         stat_values = []
#         for dataset in DATASETS:
#             r = results[model][dataset]
#             if r['available']:
#                 stat_values.append(r['stat_diff'])
#             else:
#                 stat_values.append("--")

#         stat_row = "&Statistical Difference & " + " & ".join(stat_values) + " \\\\"
#         lines.append(stat_row)
#         lines.append("\\midrule")

#     # Remove last midrule and add bottomrule
#     lines[-1] = "\\bottomrule"
#     lines.append("\\end{tabular}")

#     return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate LaTeX table for Oracle vs Golden analysis')
    parser.add_argument('--metric', type=str, default='Rewrite',
                        help='Metric name (Rewrite, Rephrase, Locality, Portability, Fluency, Overall)')
    parser.add_argument('--models', nargs='+', default=['gpt2-xl', 'llama2-7b', 'gemma3-12b'],
                        help='Model names to include')
    parser.add_argument('--golden-layer-proxy', action='store_true',
                        help='Select golden layer from proxy set (default: use test set)')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()

    # Fluency is lower-is-better
    higher_is_better = args.metric.lower() != 'fluency'

    print(f"\n{'='*70}")
    print(f"Generating LaTeX table for metric: {args.metric}")
    print(f"Models: {args.models}")
    print(f"Golden layer from: {'proxy set' if args.golden_layer_proxy else 'test set'}")
    print(f"{'='*70}\n")

    latex_table = generate_latex_table(
        args.metric,
        args.models,
        higher_is_better,
        golden_layer_proxy=args.golden_layer_proxy,
        verbose=args.verbose
    )

    print(latex_table)

    # Also save to file
    output_file = f"results/oracle_golden_table_{args.metric.lower()}.tex"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    main()
