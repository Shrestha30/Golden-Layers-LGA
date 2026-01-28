#!/usr/bin/env python3
"""
Run K-S Test on WikiBio Dataset

Performs Kolmogorov-Smirnov tests for three layer comparisons:
1. Layer 19 vs Layer 15
2. Layer 19 vs Layer 0
3. Layer 19 vs Layer 12
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
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

def run_ks_test(data1, data2, name1, name2):
    """Run K-S test and return results."""
    # Remove NaN values
    data1_clean = data1[~np.isnan(data1)]
    data2_clean = data2[~np.isnan(data2)]

    # Run K-S test
    statistic, pvalue = stats.ks_2samp(data1_clean, data2_clean)

    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'n1': len(data1_clean),
        'n2': len(data2_clean),
        'name1': name1,
        'name2': name2
    }

def main():
    print("="*70)
    print("K-S Test Results: WikiBio Dataset")
    print("="*70)

    dataset = 'wikibio'
    model = 'gpt2-xl'

    # Load analyzer
    print(f"\nLoading {dataset} analyzer...")
    analyzer, config = load_analyzer(dataset, model)

    # Get metric
    metric = 'post_rewrite_acc'
    test_indices = analyzer.test_indices

    print(f"Metric: {metric}")
    print(f"Test samples: {len(test_indices)}")

    # Extract test set performance for each layer
    layer_19_data = analyzer.results[metric][test_indices, 19].numpy()
    layer_15_data = analyzer.results[metric][test_indices, 15].numpy()
    layer_17_data = analyzer.results[metric][test_indices, 17].numpy()
    layer_18_data = analyzer.results[metric][test_indices, 18].numpy()
    layer_12_data = analyzer.results[metric][test_indices, 12].numpy()
    layer_0_data = analyzer.results[metric][test_indices, 0].numpy()

    print("\n" + "="*70)
    print("K-S TEST RESULTS")
    print("="*70)

    # Run three K-S tests
    tests = [
        (layer_19_data, layer_15_data, "Layer 19", "Layer 15"),
        (layer_19_data, layer_0_data, "Layer 19", "Layer 0"),
        (layer_19_data, layer_17_data, "Layer 19", "Layer 17"),
        (layer_19_data, layer_18_data, "Layer 19", "Layer 18"),
        (layer_19_data, layer_12_data, "Layer 19", "Layer 12")
    ]

    results = []
    for data1, data2, name1, name2 in tests:
        result = run_ks_test(data1, data2, name1, name2)
        results.append(result)

    # Print results in table format
    print(f"\n{'Comparison':<25} {'KS Statistic':<15} {'P-value':<15} {'N1':<8} {'N2':<8} {'Significant?':<12}")
    print("-"*90)

    for r in results:
        comparison = f"{r['name1']} vs {r['name2']}"
        significant = "Yes (p<0.05)" if r['pvalue'] < 0.05 else "No (p≥0.05)"
        print(f"{comparison:<25} {r['statistic']:<15.6f} {r['pvalue']:<15.6e} {r['n1']:<8} {r['n2']:<8} {significant:<12}")

    # Detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['name1']} vs {r['name2']}")
        print(f"   K-S Statistic: {r['statistic']:.6f}")
        print(f"   P-value: {r['pvalue']:.6e}")
        print(f"   Sample sizes: n1={r['n1']}, n2={r['n2']}")

        if r['pvalue'] < 0.001:
            print(f"   Result: Highly significant (p < 0.001) ***")
        elif r['pvalue'] < 0.01:
            print(f"   Result: Very significant (p < 0.01) **")
        elif r['pvalue'] < 0.05:
            print(f"   Result: Significant (p < 0.05) *")
        else:
            print(f"   Result: Not significant (p ≥ 0.05)")

        # Interpretation
        if r['pvalue'] < 0.05:
            print(f"   → The distributions are significantly different")
        else:
            print(f"   → Cannot reject null hypothesis (distributions may be the same)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    significant_count = sum(1 for r in results if r['pvalue'] < 0.05)
    print(f"\nSignificant differences found: {significant_count}/3")

    if significant_count > 0:
        print("\nSignificant comparisons:")
        for r in results:
            if r['pvalue'] < 0.05:
                print(f"  • {r['name1']} vs {r['name2']} (p={r['pvalue']:.6e})")

    if significant_count < 3:
        print("\nNon-significant comparisons:")
        for r in results:
            if r['pvalue'] >= 0.05:
                print(f"  • {r['name1']} vs {r['name2']} (p={r['pvalue']:.6e})")

if __name__ == '__main__':
    main()
