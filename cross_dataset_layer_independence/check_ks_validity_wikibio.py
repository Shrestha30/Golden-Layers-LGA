#!/usr/bin/env python3
"""
K-S Test Validity Check for WikiBio

Analyzes duplicate/tie rates in performance distributions to determine
if Kolmogorov-Smirnov test is appropriate.

Checks three layers:
1. Layer 19: WikiBio proxy-selected (best on proxy)
2. Layer 15: ZSRE proxy-selected (cross-dataset comparison)
3. Worst layer: Lowest test mean on WikiBio
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

from dataset_model_registry import get_config

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

def analyze_distribution(data, name):
    """Analyze a single distribution for duplicates and ties."""
    # Remove NaN values
    data_clean = data[~np.isnan(data)]

    n_total = len(data_clean)
    unique_values = np.unique(data_clean)
    n_unique = len(unique_values)

    # Count occurrences of each value
    value_counts = Counter(data_clean)

    # Sort by count (descending)
    sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

    # Calculate duplicate rate
    duplicate_rate = 100 * (1 - n_unique / n_total)

    print(f"\n{'='*70}")
    print(f"Distribution: {name}")
    print(f"{'='*70}")
    print(f"Total samples:     {n_total}")
    print(f"Unique values:     {n_unique}")
    print(f"Duplicate rate:    {duplicate_rate:.2f}%")
    print(f"\nValue distribution (top 10):")
    print(f"{'Value':>10} {'Count':>8} {'Percentage':>12}")
    print(f"{'-'*32}")

    for value, count in sorted_counts[:10]:
        pct = 100 * count / n_total
        print(f"{value:>10.4f} {count:>8} {pct:>11.2f}%")

    if len(sorted_counts) > 10:
        print(f"... and {len(sorted_counts) - 10} more unique values")

    return {
        'data': data_clean,
        'n_total': n_total,
        'n_unique': n_unique,
        'duplicate_rate': duplicate_rate,
        'value_counts': value_counts,
        'sorted_counts': sorted_counts
    }

def compare_distributions(dist1_info, dist2_info, name1, name2):
    """Compare two distributions to assess K-S test validity."""
    print(f"\n{'='*70}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*70}")

    # Combined unique values
    all_values = set(dist1_info['value_counts'].keys()) | set(dist2_info['value_counts'].keys())
    n_combined_unique = len(all_values)

    # Calculate tie rate (values that appear in both distributions)
    common_values = set(dist1_info['value_counts'].keys()) & set(dist2_info['value_counts'].keys())

    # Estimate tie percentage (rough approximation)
    tie_count = 0
    for val in common_values:
        tie_count += min(dist1_info['value_counts'][val], dist2_info['value_counts'][val])

    total_samples = dist1_info['n_total'] + dist2_info['n_total']
    tie_pct = 100 * (2 * tie_count) / total_samples

    print(f"\nCombined statistics:")
    print(f"  Unique values (dist 1):  {dist1_info['n_unique']}")
    print(f"  Unique values (dist 2):  {dist2_info['n_unique']}")
    print(f"  Combined unique values:  {n_combined_unique}")
    print(f"  Common values:           {len(common_values)}")
    print(f"  Estimated tie rate:      {tie_pct:.2f}%")

    # K-S test validity assessment
    print(f"\n{'='*70}")
    print("K-S Test Validity Assessment")
    print(f"{'='*70}")

    issues = []

    # Check for binary/highly discrete data
    if n_combined_unique <= 5:
        issues.append(f"Highly discrete data (only {n_combined_unique} unique values)")

    # Check duplicate rate
    avg_dup_rate = (dist1_info['duplicate_rate'] + dist2_info['duplicate_rate']) / 2
    if avg_dup_rate > 50:
        issues.append(f"High duplicate rate (avg {avg_dup_rate:.1f}%)")

    # Check tie rate
    if tie_pct > 20:
        issues.append(f"High tie rate ({tie_pct:.1f}%)")

    if issues:
        print("✗ K-S TEST NOT RECOMMENDED")
        print("\nReasons:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommended alternatives:")
        if n_combined_unique <= 5:
            print("  - Mann-Whitney U test (for ordinal/discrete data)")
            print("  - Chi-square test (for categorical data)")
        else:
            print("  - Mann-Whitney U test (handles ties better)")
            print("  - Permutation test (distribution-free)")
            print("  - Anderson-Darling test (more sensitive to tails)")
    else:
        print("✓ K-S TEST VALID")
        print("\nThe data appears suitable for K-S testing:")
        print(f"  - Sufficient unique values ({n_combined_unique})")
        print(f"  - Low duplicate rate ({avg_dup_rate:.1f}%)")
        print(f"  - Low tie rate ({tie_pct:.1f}%)")

    return {
        'n_combined_unique': n_combined_unique,
        'tie_pct': tie_pct,
        'valid': len(issues) == 0,
        'issues': issues
    }

def main():
    print("="*70)
    print("K-S Test Validity Check: WikiBio Dataset")
    print("="*70)

    dataset = 'wikibio'
    model = 'gpt2-xl'

    # Load analyzer
    print(f"\nLoading {dataset} analyzer...")
    analyzer, config = load_analyzer(dataset, model)

    # Get metric
    print(f"Available metrics in analyzer: {list(analyzer.results.keys())}")

    metric = 'post_rewrite_acc'  # WikiBio uses post_rewrite_acc
    if metric not in analyzer.results:
        # Try alternative metric names
        if 'post_rewrite' in analyzer.results:
            metric = 'post_rewrite'
        elif 'post_overall' in analyzer.results:
            metric = 'post_overall'
        else:
            print(f"Error: Metric {metric} not found")
            print(f"Available: {list(analyzer.results.keys())}")
            return

    print(f"Using metric: {metric}")

    print(f"Metric: {metric}")
    print(f"Total samples: {analyzer.n_samples}")
    print(f"Proxy samples: {config['proxy_size']}")
    print(f"Test samples: {len(analyzer.test_indices)}")

    # Get test indices
    test_indices = analyzer.test_indices
    test_means = analyzer.test_means[metric]

    # Find worst layer
    worst_layer = np.argmin(test_means).item()
    worst_score = test_means[worst_layer].item()

    print(f"\nLayers to analyze:")
    print(f"  Layer 19: WikiBio proxy-selected (best on proxy)")
    print(f"  Layer 15: ZSRE proxy-selected (cross-dataset comparison)")
    print(f"  Layer 12: WikiCounterfact proxy-selected (cross-dataset comparison)")
    print(f"  Layer {worst_layer}: WikiBio worst-performing (test mean: {worst_score:.4f})")

    # Extract test set performance for each layer
    layer_19_data = analyzer.results[metric][test_indices, 19].numpy()
    layer_15_data = analyzer.results[metric][test_indices, 15].numpy()
    layer_12_data = analyzer.results[metric][test_indices, 12].numpy()
    layer_worst_data = analyzer.results[metric][test_indices, worst_layer].numpy()

    # Analyze each distribution
    dist_19 = analyze_distribution(layer_19_data, "WikiBio Layer 19 (Proxy-selected)")
    dist_15 = analyze_distribution(layer_15_data, "WikiBio Layer 15 (ZSRE proxy)")
    dist_12 = analyze_distribution(layer_12_data, "WikiBio Layer 12 (WikiCounterfact proxy)")
    dist_worst = analyze_distribution(layer_worst_data, f"WikiBio Layer {worst_layer} (Worst)")

    # Compare pairs
    print("\n" + "="*70)
    print("PAIRWISE COMPARISONS")
    print("="*70)

    comp1 = compare_distributions(dist_19, dist_15,
                                   "Layer 19 (WikiBio)", "Layer 15 (ZSRE)")

    comp2 = compare_distributions(dist_19, dist_worst,
                                   "Layer 19 (WikiBio)", f"Layer {worst_layer} (Worst)")

    comp3 = compare_distributions(dist_19, dist_12,
                                   "Layer 19 (WikiBio)", "Layer 12 (WikiCounterfact)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    comparisons = [
        ("Layer 19 vs Layer 15", comp1),
        (f"Layer 19 vs Layer {worst_layer} (Worst)", comp2),
        ("Layer 19 vs Layer 12", comp3)
    ]

    print(f"\n{'Comparison':<35} {'Valid?':<10} {'Unique':<10} {'Tie Rate':<10}")
    print("-"*70)
    for name, comp in comparisons:
        valid_str = "✓ Yes" if comp['valid'] else "✗ No"
        print(f"{name:<35} {valid_str:<10} {comp['n_combined_unique']:<10} {comp['tie_pct']:.1f}%")

    # Overall recommendation
    print("\n" + "="*70)
    print("OVERALL RECOMMENDATION")
    print("="*70)

    n_valid = sum(1 for _, comp in comparisons if comp['valid'])

    if n_valid == 0:
        print("✗ K-S test NOT recommended for any comparison")
        print("\nUse Mann-Whitney U test or permutation test instead.")
    elif n_valid == len(comparisons):
        print("✓ K-S test VALID for all comparisons")
    else:
        print(f"⚠ K-S test valid for {n_valid}/{len(comparisons)} comparisons")
        print("\nValid comparisons:")
        for name, comp in comparisons:
            if comp['valid']:
                print(f"  ✓ {name}")
        print("\nInvalid comparisons (use alternatives):")
        for name, comp in comparisons:
            if not comp['valid']:
                print(f"  ✗ {name}")

if __name__ == '__main__':
    main()
