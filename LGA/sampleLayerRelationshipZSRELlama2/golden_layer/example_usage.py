#!/usr/bin/env python3
"""
Example usage of the Golden Layer Analysis API
"""

from golden_layer_api import GoldenLayerAnalyzer


def example_basic_usage():
    """
    Basic usage example - complete analysis pipeline
    """
    print("="*70)
    print("EXAMPLE 1: Basic Usage - Complete Analysis Pipeline")
    print("="*70)

    # Create analyzer
    analyzer = GoldenLayerAnalyzer(data_dir='../allLayerPerformance')

    # Method chaining for complete pipeline
    (analyzer
        .load_data(verbose=True)
        .compute_overall_metric(verbose=True)
        .split_dataset(proxy_size=100, random_seed=42, verbose=True)
        .compute_proxy_test_means(verbose=True)
        .analyze(verbose=True)
        .save_bar_charts(output_dir='./figures', dpi=150)
        .save_results(output_dir='./results')
        .print_summary())


def example_custom_analysis():
    """
    Custom analysis with different parameters
    """
    print("\n\n")
    print("="*70)
    print("EXAMPLE 2: Custom Analysis with Different Proxy Size")
    print("="*70)

    # Create analyzer with custom parameters
    analyzer = GoldenLayerAnalyzer(data_dir='../allLayerPerformance')

    # Load and analyze with different proxy size
    (analyzer
        .load_data(verbose=False)
        .compute_overall_metric(verbose=False)
        .split_dataset(proxy_size=200, random_seed=42, verbose=True)
        .compute_proxy_test_means(verbose=False)
        .analyze(verbose=False))

    # Access results programmatically
    print("\nAccessing results programmatically:")
    overall_info = analyzer.best_layers_comparison['post_overall']
    print(f"  Overall metric - Proxy best layers: {overall_info['proxy_best_layers']}")
    print(f"  Overall metric - Test best layers: {overall_info['test_best_layers']}")
    print(f"  Overall metric - Correlation: {overall_info['correlation']:.4f}")

    if overall_info['overlap']:
        print(f"\n  ✓ Golden layer(s) found: {overall_info['overlap']}")
    else:
        print(f"\n  ✗ No common golden layer")

    # Save with custom output directories
    analyzer.save_bar_charts(output_dir='./figures_proxy200', dpi=200)
    analyzer.save_results(output_dir='./results_proxy200')


def example_inspect_data():
    """
    Example of inspecting loaded data
    """
    print("\n\n")
    print("="*70)
    print("EXAMPLE 3: Inspecting Loaded Data")
    print("="*70)

    analyzer = GoldenLayerAnalyzer(data_dir='../allLayerPerformance')
    analyzer.load_data(verbose=False)
    analyzer.compute_overall_metric(verbose=False)

    # Access raw data
    print("\nData structure:")
    print(f"  Total samples: {analyzer.n_samples}")
    print(f"  Total layers: {analyzer.n_layers}")

    # Check specific metrics
    for metric_key, metric_name in analyzer.metrics_info:
        tensor = analyzer.results[metric_key]
        non_nan = (~torch.isnan(tensor)).sum().item()
        total = tensor.numel()
        print(f"  {metric_name:20s}: {tensor.shape} ({non_nan}/{total} non-NaN)")

    # Access specific layer performance
    layer_17_rewrite = analyzer.results['post_rewrite'][:, 17]
    print(f"\nLayer 17 rewrite accuracy:")
    print(f"  Mean: {torch.nanmean(layer_17_rewrite).item():.4f}")
    print(f"  Std: {torch.std(layer_17_rewrite[~torch.isnan(layer_17_rewrite)]).item():.4f}")


def example_layer_specific_analysis():
    """
    Example of analyzing specific layers
    """
    print("\n\n")
    print("="*70)
    print("EXAMPLE 4: Layer-Specific Analysis")
    print("="*70)

    analyzer = GoldenLayerAnalyzer(data_dir='../allLayerPerformance')
    (analyzer
        .load_data(verbose=False)
        .compute_overall_metric(verbose=False)
        .split_dataset(proxy_size=100, random_seed=42, verbose=False)
        .compute_proxy_test_means(verbose=False)
        .analyze(verbose=False))

    # Analyze specific layers
    layers_of_interest = [15, 16, 17, 18, 19, 20]

    print(f"\nAnalyzing layers {layers_of_interest}:")
    for layer in layers_of_interest:
        overall_proxy = analyzer.proxy_means['post_overall'][layer].item()
        overall_test = analyzer.test_means['post_overall'][layer].item()
        print(f"\nLayer {layer}:")
        print(f"  Proxy overall score: {overall_proxy:.4f}")
        print(f"  Test overall score:  {overall_test:.4f}")
        print(f"  Difference: {abs(overall_proxy - overall_test):.4f}")


if __name__ == '__main__':
    import torch

    # Run all examples
    example_basic_usage()

    # Uncomment to run additional examples:
    # example_custom_analysis()
    # example_inspect_data()
    # example_layer_specific_analysis()
