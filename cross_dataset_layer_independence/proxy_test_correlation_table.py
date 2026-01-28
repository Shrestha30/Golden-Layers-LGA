#!/usr/bin/env python3
"""
Proxy-Test Layer Performance Correlation Table

Calculates the Pearson correlation between proxy set and test set
layer-wise performance for each metric across all datasets.

Output: Table with datasets as rows and metrics as columns.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import dataset_model_registry
import heatmap_api


def get_datasets_for_model(model):
    """Get all datasets available for a model."""
    datasets = []
    for key in dataset_model_registry.DATASET_MODEL_REGISTRY.keys():
        if key[1] == model:
            datasets.append(key[0])
    return datasets


def compute_all_correlations(model):
    """
    Compute correlations for all datasets and metrics for a given model.

    Args:
        model: Model name (e.g., 'gpt2-xl', 'llama2-7b')

    Returns:
        DataFrame with datasets as rows and metrics as columns
    """
    print("=" * 80)
    print(f"Computing Proxy-Test Correlations for {model.upper()}")
    print("=" * 80)

    datasets = get_datasets_for_model(model)
    results = {}

    for dataset in datasets:
        print(f"\nLoading {dataset}...")

        try:
            analyzer, config = heatmap_api.load_analyzer(dataset, model, dataset_model_registry)

            dataset_results = {}

            for metric_name in heatmap_api.METRIC_COLUMNS:
                metric_key = heatmap_api.get_metric_key(dataset, metric_name)

                if metric_key is None:
                    dataset_results[metric_name] = 'N/A'
                    print(f"  {metric_name}: N/A (not supported)")
                elif metric_key not in analyzer.proxy_means or metric_key not in analyzer.test_means:
                    dataset_results[metric_name] = 'N/A'
                    print(f"  {metric_name}: N/A (not found: {metric_key})")
                else:
                    proxy = analyzer.proxy_means[metric_key]
                    test = analyzer.test_means[metric_key]
                    corr = heatmap_api.compute_correlation(proxy, test)

                    if np.isnan(corr):
                        dataset_results[metric_name] = 'N/A'
                        print(f"  {metric_name}: N/A (insufficient data)")
                    else:
                        dataset_results[metric_name] = corr
                        print(f"  {metric_name}: {corr:.4f}")

            results[dataset] = dataset_results

        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset] = {m: 'N/A' for m in heatmap_api.METRIC_COLUMNS}

    # Create DataFrame
    df = pd.DataFrame(results).T
    df = df[heatmap_api.METRIC_COLUMNS]
    df.index.name = 'Dataset'

    return df


def format_table(df):
    """Format DataFrame for display with proper alignment."""
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )
    return formatted


def print_summary_statistics(df, model):
    """Print summary statistics for the correlation table."""
    print("\n" + "=" * 80)
    print(f"SUMMARY STATISTICS ({model.upper()})")
    print("=" * 80)

    numeric_df = df.replace('N/A', np.nan).astype(float)

    for col in heatmap_api.METRIC_COLUMNS:
        values = numeric_df[col].dropna()
        if len(values) > 0:
            print(f"\n{col}:")
            print(f"  Mean correlation: {values.mean():.4f}")
            print(f"  Min correlation:  {values.min():.4f}")
            print(f"  Max correlation:  {values.max():.4f}")
            print(f"  Datasets with data: {len(values)}/{len(df)}")

    all_values = numeric_df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    if len(all_values) > 0:
        print(f"\nOverall average correlation: {np.mean(all_values):.4f}")


def main(model='gpt2-xl'):
    print("=" * 80)
    print("Proxy-Test Layer Performance Correlation Table")
    print(f"Model: {model.upper()}")
    print("=" * 80)

    df = compute_all_correlations(model)

    print("\n" + "=" * 80)
    print(f"CORRELATION TABLE ({model.upper()}) - Pearson r")
    print("=" * 80)

    formatted_df = format_table(df)
    print("\n" + formatted_df.to_string())

    # Save to CSV
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)

    model_safe = model.replace("-", "_")
    csv_path = output_dir / f'proxy_test_correlation_{model_safe}.csv'
    df.to_csv(csv_path)
    print(f"\n✓ Saved to: {csv_path}")

    print_summary_statistics(df, model)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute proxy-test correlation table')
    parser.add_argument('--model', type=str, default='gpt2-xl',
                        help='Model name (gpt2-xl, llama2-7b, qwen3-32b)')
    args = parser.parse_args()
    main(args.model)
