#!/usr/bin/env python3
"""
Influence Method Comparison Script

Compares multiple influence-based layer selection methods:
1. Load influence scores for all available methods
2. Use proxy set to select best layers for each method
3. Evaluate test set performance at those layers
4. Generate comparison table sorted by Overall performance

Usage:
    python run_method_comparison.py --model llama2-7b --dataset wikicounterfact
    python run_method_comparison.py --model gpt2-xl --dataset zsre
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import influence_api
import influence_registry
import heatmap_api
import dataset_model_registry


def load_all_influence_methods(model, dataset, expected_n_samples=None, expected_n_layers=None):
    """
    Load all available influence score files for a model/dataset pair.

    Args:
        model: Model name
        dataset: Dataset name
        expected_n_samples: Expected number of samples (optional, for validation)
        expected_n_layers: Expected number of layers (optional, for validation)

    Returns:
        List of dicts with 'name', 'scores', 'mode'
    """
    files = influence_registry.find_influence_files(model, dataset)

    if not files:
        print(f"No influence files found for {model}/{dataset}")
        print("Check that the paths in influence_registry.py are correct.")
        return []

    # Build expected shape if provided
    expected_shape = None
    if expected_n_samples is not None and expected_n_layers is not None:
        expected_shape = (expected_n_samples, expected_n_layers)

    methods = []
    for filepath, inf_type in files:
        # Load with error handling
        scores, error = influence_api.load_influence_scores(filepath, expected_shape=expected_shape)

        if error:
            print(f"  ✗ {inf_type}: {error}")
            continue

        if scores is None:
            print(f"  ✗ {inf_type}: Could not load scores")
            continue

        # Determine mode based on influence type
        if 'causaltrace' in inf_type.lower():
            mode = 'max'
        else:
            mode = 'min'

        if "OverallNewOverallHat" in inf_type:
            print("")

        methods.append({
            'name': inf_type,
            'scores': scores,
            'mode': mode,
            'filepath': filepath
        })
        print(f"  ✓ Loaded: {inf_type} (shape: {scores.shape})")

    return methods


def run_comparison(model, dataset, top_n=19, baseline_method='CausalTraceProba.last', mode='min', outlier=False):
    """
    Run full comparison of influence methods for a model/dataset.

    Args:
        model: Model name
        dataset: Dataset name
        top_n: Number of top positive and negative methods to show
        baseline_method: Name of baseline method for comparison
    """
    print("=" * 100)
    print(f"Influence Method Comparison: {model.upper()} - {dataset.upper()}")
    print("=" * 100)

    # 1. Load performance analyzer
    print("\n[1] Loading performance data...")
    try:
        analyzer, config = heatmap_api.load_analyzer(dataset, model, dataset_model_registry)
        n_samples = analyzer.n_samples
        n_layers = analyzer.n_layers
        print(f"    Loaded {len(analyzer.proxy_indices)} proxy, {len(analyzer.test_indices)} test samples")
        print(f"    Expected shape: ({n_samples}, {n_layers})")
    except Exception as e:
        print(f"    Error loading performance data: {e}")
        return None

    # 2. Load influence methods
    print("\n[2] Loading influence score files...")
    methods = load_all_influence_methods(
        model, dataset,
        expected_n_samples=n_samples,
        expected_n_layers=n_layers
    )

    if not methods:
        return None

    print(f"    Loaded {len(methods)} influence methods")
    top_n = len(methods)//2+1

    # 3. Generate comparison table
    print("\n[3] Computing layer selections and performance...")
    df, baseline_perf = influence_api.generate_method_comparison_table(
        analyzer=analyzer,
        influence_methods=methods,
        baseline_method_name=baseline_method,
        include_layer_17=True,
        mode=mode,
        outlier=outlier
    )

    # 4. Just show all methods sorted by Overall (top_n logic commented out)
    # if 'Overall' in df.columns:
    #     # Convert Overall to float for filtering
    #     def safe_float(x):
    #         try:
    #             return float(x)
    #         except:
    #             return np.nan
    #
    #     df['_overall_float'] = df['Overall'].apply(safe_float)
    #
    #     # Top positive
    #     df_positive = df[df['_overall_float'].notna()].nlargest(top_n, '_overall_float')
    #
    #     # Top negative (if any with very low performance)
    #     # For now, just show all sorted
    #
    #     # Drop helper column
    #     df = df.drop(columns=['_overall_float'], errors='ignore')
    #     df_positive = df_positive.drop(columns=['_overall_float'], errors='ignore')

    # 5. Print results
    print("\n")
    influence_api.print_comparison_table(df, baseline_perf, model, dataset)

    # 6. Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'method_comparison_{model}_{dataset}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nSaved: {output_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Compare influence-based layer selection methods')
    parser.add_argument('--model', type=str, default='llama2-7b',
                        help='Model name (gpt2-xl, llama2-7b, qwen3-32b, gemma3-12b, emmet)')
    parser.add_argument('--dataset', type=str, default='wikicounterfact',
                        help='Dataset name (zsre, counterfact, wikibio, wikirecent, wikicounterfact)')
    parser.add_argument('--top-n', type=int, default=19,
                        help='Number of top methods to show')
    parser.add_argument('--baseline', type=str, default='CausalTraceProba.last',
                        help='Baseline method name for comparison')
    args = parser.parse_args()

    df = run_comparison(
        model=args.model,
        dataset=args.dataset,
        top_n=args.top_n,
        baseline_method=args.baseline
    )

    if df is not None:
        print("\n" + "=" * 100)
        print("COMPLETE")
        print("=" * 100)


if __name__ == '__main__':
    main()
