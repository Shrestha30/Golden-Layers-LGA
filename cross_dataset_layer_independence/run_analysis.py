#!/usr/bin/env python3
"""
Runner script for Cross-Dataset Layer Independence Analysis

This script provides easy-to-use interfaces for running the analysis.
"""

import argparse
from cross_dataset_analyzer import CrossDatasetLayerAnalyzer
from dataset_model_registry import get_common_models, print_registry_summary


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Layer Independence Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare ZSRE and Counterfact for GPT-2 XL
  python run_analysis.py --datasets zsre counterfact --model gpt2-xl

  # Compare all GPT-2 XL datasets
  python run_analysis.py --datasets zsre counterfact wikicounterfact --model gpt2-xl

  # Show available dataset-model pairs
  python run_analysis.py --show-registry

  # Custom seed
  python run_analysis.py --datasets zsre counterfact --model llama2-7b --seed 123
        """
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        help='List of datasets to compare (e.g., zsre counterfact)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model name (e.g., gpt2-xl, llama2-7b, gemma3-12b)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='post_overall',
        help='Metric to analyze (default: post_overall)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving results to files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )

    parser.add_argument(
        '--show-registry',
        action='store_true',
        help='Show available dataset-model pairs and exit'
    )

    args = parser.parse_args()

    # Show registry if requested
    if args.show_registry:
        print_registry_summary()
        return

    # Validate required arguments
    if not args.datasets:
        parser.error("--datasets is required (use --show-registry to see options)")
    if not args.model:
        parser.error("--model is required (use --show-registry to see options)")

    # Check if model is available for all datasets
    common = get_common_models(args.datasets)
    if args.model not in common:
        print(f"✗ Error: Model '{args.model}' is not available for all specified datasets")
        print(f"  Available models for {args.datasets}: {common}")
        return

    # Create and run analyzer
    print(f"\nStarting analysis...")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Model: {args.model}")
    print(f"  Metric: {args.metric}")

    analyzer = CrossDatasetLayerAnalyzer(
        datasets=args.datasets,
        model=args.model,
        metric=args.metric,
        random_seed=args.seed
    )

    analyzer.run_full_analysis(
        save_plots=not args.no_plots,
        save_results=not args.no_save
    )

    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
