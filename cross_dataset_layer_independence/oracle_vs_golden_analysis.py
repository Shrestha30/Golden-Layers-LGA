#!/usr/bin/env python3
"""
Oracle vs Golden Layer Analysis

Compares fixed golden layer (selected from proxy set) performance against
sample-wise oracle (best possible layer for each sample) on test set.

For each test sample:
- Oracle: performance at the best layer for that specific sample
- Golden: performance at the fixed golden layer selected from proxy

Outputs:
- Scatter plot: Oracle (x) vs Golden (y) with diagonal reference
- Statistics: Mean and median performance gap (oracle - golden)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

import dataset_model_registry
import heatmap_api


def plot_oracle_vs_golden(result, dataset, model, metric_name, output_path):
    """
    Create scatter plot of oracle vs golden layer performance.

    Args:
        result: dict from compute_oracle_vs_golden()
        dataset: Dataset name
        model: Model name
        metric_name: Display name of metric (e.g., 'Overall')
        output_path: Path to save the figure
    """
    oracle = result['oracle_perf']
    golden = result['golden_perf']

    # Filter out NaN values
    valid_mask = ~(np.isnan(oracle) | np.isnan(golden))
    oracle_valid = oracle[valid_mask]
    golden_valid = golden[valid_mask]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with small markers and transparency
    ax.scatter(oracle_valid, golden_valid,
               s=15, alpha=0.3, c='steelblue', edgecolors='none')

    # Add diagonal reference line (y=x, perfect agreement)
    min_val = min(oracle_valid.min(), golden_valid.min())
    max_val = max(oracle_valid.max(), golden_valid.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'r--', linewidth=1.5, label='Perfect Agreement (y=x)')

    # Labels and title
    ax.set_xlabel(f'Oracle Performance (sample-wise best)', fontsize=12)
    ax.set_ylabel(f'Golden Layer {result["golden_layer"]} Performance', fontsize=12)
    ax.set_title(f'Oracle vs Golden Layer Performance\n'
                 f'{dataset.upper()} - {model.upper()} - {metric_name}\n'
                 f'Mean Gap: {result["mean_gap"]:.4f}, Median Gap: {result["median_gap"]:.4f}',
                 fontsize=11)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    # Add statistics text box
    golden_layers_str = str(result.get('golden_layers', [result['golden_layer']]))
    stats_text = (f'N = {result["n_samples"]} samples\n'
                  f'Golden Layer(s) = {golden_layers_str}\n'
                  f'Using Layer = {result["golden_layer"]}\n'
                  f'Mean Gap = {result["mean_gap"]:.4f}\n'
                  f'Median Gap = {result["median_gap"]:.4f}\n'
                  f'Std Dev of Gap = {result["std_gap"]:.4f}\n'
                  f'Correlation = {result["correlation"]:.4f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    plt.close()

def plot_ecdf_gaps(result, dataset, model, metric_name, output_path):
    """
    Create an ECDF plot of sample-wise performance gaps (oracle - golden).

    Args:
        result: dict from compute_oracle_vs_golden()
        dataset: Dataset name
        model: Model name
        metric_name: Display name of metric (e.g., 'Overall')
        output_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    oracle = result['oracle_perf']
    golden = result['golden_perf']

    # Filter out NaN values
    valid_mask = ~(np.isnan(oracle) | np.isnan(golden))
    oracle_valid = oracle[valid_mask]
    golden_valid = golden[valid_mask]

    # Compute gaps
    gaps = oracle_valid - golden_valid
    gaps_sorted = np.sort(gaps)
    cdf = np.arange(1, len(gaps_sorted)+1) / len(gaps_sorted)

    # Plot ECDF
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gaps_sorted, cdf, marker='.', linestyle='none',
            alpha=0.6, color='steelblue')

    # Labels and title
    ax.set_xlabel('Performance Gap (Oracle - Golden)', fontsize=12)
    ax.set_ylabel('Cumulative Fraction of Test Samples', fontsize=12)
    ax.set_title(f'ECDF of Sample-wise Performance Gaps\n'
                 f'{dataset.upper()} - {model.upper()} - {metric_name}',
                 fontsize=11)

    # Grid
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    golden_layers_str = str(result.get('golden_layers', [result['golden_layer']]))
    stats_text = (f'N = {result["n_samples"]} samples\n'
                  f'Golden Layer(s) = {golden_layers_str}\n'
                  f'Using Layer = {result["golden_layer"]}\n'
                  f'Mean Gap = {result["mean_gap"]:.4f}\n'
                  f'Median Gap = {result["median_gap"]:.4f}\n'
                  f'Std Dev of Gap = {result["std_gap"]:.4f}\n'
                  f'Correlation = {result["correlation"]:.4f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved ECDF: {output_path}")

    # Also save PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved ECDF: {pdf_path}")

    plt.close()


def run_analysis(dataset, model, metric_name, higher_is_better=True, golden_layer_proxy=True):
    """
    Run oracle vs golden analysis for a specific dataset/model/metric.

    Args:
        dataset: Dataset name (e.g., 'zsre')
        model: Model name (e.g., 'llama2-7b')
        metric_name: Display metric name (e.g., 'Overall', 'Rewrite')
        higher_is_better: True for most metrics, False for Fluency

    Returns:
        result dict from compute_oracle_vs_golden
    """
    print("=" * 70)
    print(f"Oracle vs Golden Layer Analysis")
    print(f"Dataset: {dataset.upper()}, Model: {model.upper()}, Metric: {metric_name}")
    print("=" * 70)

    # Load analyzer
    print("\nLoading data...")
    analyzer, config = heatmap_api.load_analyzer(dataset, model, dataset_model_registry)
    # print(analyzer.proxy_means)

    # Get metric key
    metric_key = heatmap_api.get_metric_key(dataset, metric_name)
    if metric_key is None:
        raise ValueError(f"Metric '{metric_name}' not supported for dataset '{dataset}'")

    print(f"Metric key: {metric_key}")
    print(f"Test samples: {len(analyzer.test_indices)}")

    # Compute oracle vs golden
    print("\nComputing oracle vs golden comparison...")
    result = heatmap_api.compute_oracle_vs_golden(analyzer, metric_key, higher_is_better, golden_layer_proxy=golden_layer_proxy)

    # Print results
    # print("\n" + "=" * 70)
    # print("RESULTS")
    # print("=" * 70)
    # golden_layers = result.get('golden_layers', [result['golden_layer']])
    # print(f"Golden Layer(s) (from proxy): {golden_layers}")
    # print(f"Using Layer: {result['golden_layer']} (first of ties)")
    # print(f"Valid test samples: {result['n_samples']}")
    # print(f"Mean Gap (oracle - golden): {result['mean_gap']:.4f}")
    # print(f"Median Gap (oracle - golden): {result['median_gap']:.4f}")
    # print(f"Std Dev of Gap: {result['std_gap']:.4f}")
    # print(f"Correlation (oracle vs golden): {result['correlation']:.4f}")

    # # Generate plot
    # output_dir = Path('results')
    # output_path = output_dir / f'oracle_vs_golden_{dataset}_{model.replace("-", "_")}_{metric_name.lower()}.png'
    # output_path_ecdf = output_dir / f'oracle_vs_golden_{dataset}_{model.replace("-", "_")}_{metric_name.lower()}_ecdf.png'
    # plot_oracle_vs_golden(result, dataset, model, metric_name, output_path)
    # plot_ecdf_gaps(result, dataset, model, metric_name, output_path_ecdf)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Oracle vs Golden Layer Analysis')
    parser.add_argument('--dataset', type=str, default='zsre',
                        help='Dataset name (zsre, counterfact, wikibio, wikirecent, wikicounterfact)')
    parser.add_argument('--model', type=str, default='llama2-7b',
                        help='Model name (gpt2-xl, llama2-7b, qwen3-32b)')
    parser.add_argument('--metric', type=str, default='Overall',
                        help='Metric name (Rewrite, Rephrase, Locality, Portability, Fluency, Overall)')
    args = parser.parse_args()

    # Fluency is lower-is-better
    higher_is_better = args.metric.lower() != 'fluency'

    result = run_analysis(args.dataset, args.model, args.metric, higher_is_better)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
