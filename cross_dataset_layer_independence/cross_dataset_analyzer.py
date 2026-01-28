#!/usr/bin/env python3
"""
Cross-Dataset Layer Independence Analyzer

This module tests whether layer performance is independent of the dataset
by conducting paired t-tests between best and worst performing layers
across different datasets.

Research Question:
"Is the layer performance independent of the dataset?"

Methodology:
For datasets D1, D2 and model M:
1. Identify Lp1, Lt1 (proxy/test best layers for D1)
2. Identify Lp2, Lt2 (proxy/test best layers for D2)
3. Find Lmax = max_performance(Lp1, Lp2, Lt1, Lt2)
4. Find Lmin = min_performance(Lp1, Lp2, Lt1, Lt2)
5. For each dataset, conduct paired sample t-test: Lmax(i) vs Lmin(i)
6. If t≈0 and p-value is high, conclude "layer performance is dataset-independent"
"""

import sys
import importlib.util
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Import registry
from dataset_model_registry import (
    DATASET_MODEL_REGISTRY,
    get_datasets,
    get_models,
    get_common_models,
    get_common_metrics,
    get_config
)


class CrossDatasetLayerAnalyzer:
    """
    Analyzes layer performance independence across different datasets.
    """

    def __init__(self, datasets: List[str], model: str, metric: str = 'post_overall',
                 random_seed: int = 42):
        """
        Initialize the cross-dataset analyzer.

        Args:
            datasets: List of dataset names (e.g., ['zsre', 'counterfact'])
            model: Model name (e.g., 'gpt2-xl')
            metric: Metric to analyze (default: 'post_overall')
            random_seed: Random seed for reproducibility

        Note:
            Proxy sizes are dataset-specific and loaded from the registry.
        """
        self.datasets = datasets
        self.model = model
        self.metric = metric
        self.random_seed = random_seed

        # Validate inputs
        self._validate_inputs()

        # Storage for analyzers and results
        self.analyzers = {}
        self.best_layers = {}
        self.layer_performance = {}
        self.test_indices = {}
        self.results = {}
        self.proxy_sizes = {}  # Store dataset-specific proxy sizes

        # Load proxy sizes from registry
        for dataset in datasets:
            config = get_config(dataset, model)
            self.proxy_sizes[dataset] = config.get('proxy_size', 100)

        print(f"Initialized Cross-Dataset Analyzer:")
        print(f"  Datasets: {datasets}")
        print(f"  Model: {model}")
        print(f"  Metric: {metric}")
        print(f"  Proxy sizes: {self.proxy_sizes}")

    def _validate_inputs(self):
        """Validate that all dataset-model pairs exist and metric is available."""
        # Check dataset-model pairs exist
        for dataset in self.datasets:
            key = (dataset, self.model)
            if key not in DATASET_MODEL_REGISTRY:
                raise ValueError(f"Dataset-model pair {key} not found in registry")

        # Check metric is available for all datasets
        common_metrics = get_common_metrics(self.datasets, self.model)
        if self.metric not in common_metrics:
            raise ValueError(
                f"Metric '{self.metric}' is not available for all specified datasets.\n"
                f"Available metrics: {common_metrics}"
            )

    def _load_analyzer(self, dataset: str):
        """
        Dynamically load the GoldenLayerAnalyzer for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Loaded GoldenLayerAnalyzer instance
        """
        config = get_config(dataset, self.model)
        api_path = config['api_path']
        base_dir = config['base_dir']
        data_dir = config['data_dir']
        proxy_size = self.proxy_sizes[dataset]  # Use dataset-specific proxy size

        # Dynamically import the API module
        spec = importlib.util.spec_from_file_location(
            f"golden_layer_api_{dataset}_{self.model}",
            api_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the analyzer class name (default to 'GoldenLayerAnalyzer')
        api_class_name = config.get('api_class', 'GoldenLayerAnalyzer')

        # Get the class from the module
        analyzer_class = getattr(module, api_class_name)

        # Create analyzer instance
        full_data_dir = base_dir / data_dir
        analyzer = analyzer_class(data_dir=str(full_data_dir))

        # Load data and compute metrics
        print(f"\nLoading data for {dataset}-{self.model} (proxy_size={proxy_size})...")
        analyzer.load_data(verbose=False)
        analyzer.compute_overall_metric(verbose=False)
        analyzer.split_dataset(
            proxy_size=proxy_size,  # Use dataset-specific proxy size
            random_seed=self.random_seed,
            verbose=False
        )
        analyzer.compute_proxy_test_means(verbose=False)

        print(f"  Loaded {analyzer.n_samples} samples, {analyzer.n_layers} layers")

        return analyzer

    def load_all_datasets(self):
        """Load analyzers for all datasets."""
        print("\n" + "=" * 80)
        print("Loading All Datasets")
        print("=" * 80)

        for dataset in self.datasets:
            self.analyzers[dataset] = self._load_analyzer(dataset)

        print(f"\n✓ Loaded {len(self.analyzers)} datasets")

    def identify_best_layers(self):
        """
        Identify best layers (proxy and test) for each dataset.
        """
        print("\n" + "=" * 80)
        print("Identifying Best Layers")
        print("=" * 80)

        for dataset, analyzer in self.analyzers.items():
            # Get proxy best layer
            proxy_means = analyzer.proxy_means[self.metric]
            proxy_best_layer = torch.argmax(proxy_means).item()

            # Get test best layer
            test_means = analyzer.test_means[self.metric]
            test_best_layer = torch.argmax(test_means).item()

            # Store
            self.best_layers[dataset] = {
                'proxy': proxy_best_layer,
                'test': test_best_layer,
                'proxy_score': proxy_means[proxy_best_layer].item(),
                'test_score': test_means[test_best_layer].item()
            }

            # Store test indices for paired t-test
            self.test_indices[dataset] = analyzer.test_indices

            print(f"\n{dataset}:")
            print(f"  Proxy best layer: {proxy_best_layer} (score: {self.best_layers[dataset]['proxy_score']:.4f})")
            print(f"  Test best layer: {test_best_layer} (score: {self.best_layers[dataset]['test_score']:.4f})")

    def find_extrema_layers(self) -> Tuple[int, int]:
        """
        Find Lmax and Lmin layers across all datasets.

        Returns:
            Tuple of (Lmax layer index, Lmin layer index)
        """
        print("\n" + "=" * 80)
        print("Finding Extrema Layers (Lmax, Lmin)")
        print("=" * 80)

        # Collect all candidate layers
        all_layers = []
        for dataset, layers_info in self.best_layers.items():
            all_layers.append(('proxy', dataset, layers_info['proxy'], layers_info['proxy_score']))
            all_layers.append(('test', dataset, layers_info['test'], layers_info['test_score']))

        # Sort by score
        all_layers_sorted = sorted(all_layers, key=lambda x: x[3], reverse=True)

        # Lmax is the layer with highest score
        lmax_type, lmax_dataset, lmax_layer, lmax_score = all_layers_sorted[0]

        # Lmin is the layer with lowest score
        lmin_type, lmin_dataset, lmin_layer, lmin_score = all_layers_sorted[-1]

        print(f"\nLmax (Best Performance):")
        print(f"  Dataset: {lmax_dataset}")
        print(f"  Type: {lmax_type}")
        print(f"  Layer: {lmax_layer}")
        print(f"  Score: {lmax_score:.4f}")

        print(f"\nLmin (Worst Performance):")
        print(f"  Dataset: {lmin_dataset}")
        print(f"  Type: {lmin_type}")
        print(f"  Layer: {lmin_layer}")
        print(f"  Score: {lmin_score:.4f}")

        # Show all candidates
        print(f"\nAll candidate layers (sorted by score):")
        print(f"  {'Type':<8} {'Dataset':<20} {'Layer':<10} {'Score':<10}")
        print("  " + "-" * 50)
        for layer_type, dataset, layer, score in all_layers_sorted:
            print(f"  {layer_type:<8} {dataset:<20} {layer:<10} {score:<10.4f}")

        return lmax_layer, lmin_layer

    def extract_layer_performance_per_sample(self, dataset: str, layer: int) -> np.ndarray:
        """
        Extract performance scores for a specific layer across all test samples.

        Args:
            dataset: Dataset name
            layer: Layer index

        Returns:
            Array of scores for test samples at specified layer
        """
        analyzer = self.analyzers[dataset]
        test_indices = analyzer.test_indices

        # Get performance tensor: shape (n_samples, n_layers)
        performance_tensor = analyzer.results[self.metric]

        # Extract test samples, specific layer
        layer_scores = performance_tensor[test_indices, layer].numpy()

        return layer_scores

    def conduct_paired_ttest(self, lmax_layer: int, lmin_layer: int):
        """
        Conduct paired t-test between Lmax and Lmin for each dataset.

        Args:
            lmax_layer: Layer index with maximum performance
            lmin_layer: Layer index with minimum performance
        """
        print("\n" + "=" * 80)
        print("Conducting Paired T-Tests")
        print("=" * 80)
        print(f"\nComparing Layer {lmax_layer} (Lmax) vs Layer {lmin_layer} (Lmin)")

        for dataset in self.datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}")
            print(f"{'='*60}")

            # Extract performance for each layer
            lmax_scores = self.extract_layer_performance_per_sample(dataset, lmax_layer)
            lmin_scores = self.extract_layer_performance_per_sample(dataset, lmin_layer)

            # Remove NaN values (paired)
            mask = ~(np.isnan(lmax_scores) | np.isnan(lmin_scores))
            lmax_clean = lmax_scores[mask]
            lmin_clean = lmin_scores[mask]

            n_samples = len(lmax_clean)

            # Conduct paired t-test
            t_statistic, p_value = stats.ttest_rel(lmax_clean, lmin_clean)

            # Compute effect size (Cohen's d for paired samples)
            differences = lmax_clean - lmin_clean
            cohen_d = np.mean(differences) / np.std(differences, ddof=1)

            # Descriptive statistics
            lmax_mean = np.mean(lmax_clean)
            lmax_std = np.std(lmax_clean, ddof=1)
            lmin_mean = np.mean(lmin_clean)
            lmin_std = np.std(lmin_clean, ddof=1)
            mean_diff = lmax_mean - lmin_mean

            # Store results
            self.results[dataset] = {
                'lmax_layer': lmax_layer,
                'lmin_layer': lmin_layer,
                'n_samples': n_samples,
                't_statistic': t_statistic,
                'p_value': p_value,
                'cohen_d': cohen_d,
                'lmax_mean': lmax_mean,
                'lmax_std': lmax_std,
                'lmin_mean': lmin_mean,
                'lmin_std': lmin_std,
                'mean_difference': mean_diff,
                'lmax_scores': lmax_clean.tolist(),
                'lmin_scores': lmin_clean.tolist()
            }

            # Print results
            print(f"  Sample size: {n_samples}")
            print(f"\n  Lmax (Layer {lmax_layer}):")
            print(f"    Mean: {lmax_mean:.4f} ± {lmax_std:.4f}")
            print(f"\n  Lmin (Layer {lmin_layer}):")
            print(f"    Mean: {lmin_mean:.4f} ± {lmin_std:.4f}")
            print(f"\n  Mean Difference: {mean_diff:.4f}")
            print(f"  T-statistic: {t_statistic:.4f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Cohen's d: {cohen_d:.4f}")

            # Interpretation
            print(f"\n  Interpretation:")
            if p_value > 0.05:
                print(f"    ✓ No significant difference (p > 0.05)")
                if abs(t_statistic) < 1.0:
                    print(f"    ✓ t-statistic close to 0 (|t| < 1.0)")
                    print(f"    → Evidence SUPPORTS dataset independence")
                else:
                    print(f"    ⚠ t-statistic not close to 0 (|t| >= 1.0)")
                    print(f"    → Weak evidence for dataset independence")
            else:
                print(f"    ✗ Significant difference detected (p <= 0.05)")
                print(f"    → Evidence AGAINST dataset independence")

            if abs(cohen_d) < 0.2:
                print(f"    ✓ Negligible effect size (|d| < 0.2)")
            elif abs(cohen_d) < 0.5:
                print(f"    ⚠ Small effect size (0.2 <= |d| < 0.5)")
            else:
                print(f"    ✗ Medium/Large effect size (|d| >= 0.5)")

    def create_summary_report(self) -> pd.DataFrame:
        """
        Create a summary report of all t-tests.

        Returns:
            DataFrame with summary statistics
        """
        print("\n" + "=" * 80)
        print("Summary Report")
        print("=" * 80)

        rows = []
        for dataset, result in self.results.items():
            rows.append({
                'Dataset': dataset,
                'N': result['n_samples'],
                'Lmax': result['lmax_layer'],
                'Lmin': result['lmin_layer'],
                'Lmax Mean': f"{result['lmax_mean']:.4f}",
                'Lmin Mean': f"{result['lmin_mean']:.4f}",
                'Mean Diff': f"{result['mean_difference']:.4f}",
                'T-stat': f"{result['t_statistic']:.4f}",
                'P-value': f"{result['p_value']:.6f}",
                "Cohen's d": f"{result['cohen_d']:.4f}",
                'Significant': 'Yes' if result['p_value'] <= 0.05 else 'No',
                'Independent': 'Yes' if (result['p_value'] > 0.05 and abs(result['t_statistic']) < 1.0) else 'No'
            })

        df = pd.DataFrame(rows)
        print("\n" + df.to_string(index=False))

        return df

    def plot_distributions(self, output_dir: str = './figures'):
        """
        Plot distributions of Lmax vs Lmin for each dataset.

        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        for dataset, result in self.results.items():
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            lmax_scores = result['lmax_scores']
            lmin_scores = result['lmin_scores']
            lmax_layer = result['lmax_layer']
            lmin_layer = result['lmin_layer']

            # Histogram
            axes[0].hist(lmax_scores, bins=30, alpha=0.6, label=f'Lmax (Layer {lmax_layer})', color='blue')
            axes[0].hist(lmin_scores, bins=30, alpha=0.6, label=f'Lmin (Layer {lmin_layer})', color='red')
            axes[0].axvline(result['lmax_mean'], color='blue', linestyle='--', linewidth=2)
            axes[0].axvline(result['lmin_mean'], color='red', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Performance Score')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'{dataset}: Distribution Comparison')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Paired differences
            differences = np.array(lmax_scores) - np.array(lmin_scores)
            axes[1].hist(differences, bins=30, alpha=0.7, color='green')
            axes[1].axvline(0, color='black', linestyle='--', linewidth=2, label='Zero')
            axes[1].axvline(np.mean(differences), color='red', linestyle='-', linewidth=2, label='Mean Diff')
            axes[1].set_xlabel('Lmax - Lmin')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'{dataset}: Paired Differences (t={result["t_statistic"]:.2f}, p={result["p_value"]:.4f})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            filename = output_path / f'{dataset}_{self.model}_lmax_vs_lmin.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close()

    def save_results(self, output_dir: str = './results'):
        """
        Save all results to JSON and CSV files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save detailed JSON
        json_file = output_path / f'cross_dataset_independence_{self.model}.json'
        output_json = {
            'metadata': {
                'model': self.model,
                'datasets': self.datasets,
                'metric': self.metric,
                'proxy_size': self.proxy_size,
                'random_seed': self.random_seed,
                'timestamp': datetime.now().isoformat()
            },
            'best_layers': self.best_layers,
            'results': self.results
        }

        with open(json_file, 'w') as f:
            json.dump(output_json, f, indent=2)
        print(f"\n✓ Saved detailed results: {json_file}")

        # Save summary CSV
        df = self.create_summary_report()
        csv_file = output_path / f'cross_dataset_independence_{self.model}_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved summary CSV: {csv_file}")

    def run_full_analysis(self, save_plots: bool = True, save_results: bool = True):
        """
        Run the complete cross-dataset independence analysis.

        Args:
            save_plots: Whether to save visualization plots
            save_results: Whether to save results to files
        """
        print("\n" + "=" * 80)
        print("CROSS-DATASET LAYER INDEPENDENCE ANALYSIS")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"Metric: {self.metric}")

        # Step 1: Load all datasets
        self.load_all_datasets()

        # Step 2: Identify best layers for each dataset
        self.identify_best_layers()

        # Step 3: Find Lmax and Lmin
        lmax_layer, lmin_layer = self.find_extrema_layers()

        # Step 4: Conduct paired t-tests
        self.conduct_paired_ttest(lmax_layer, lmin_layer)

        # Step 5: Create summary report
        summary_df = self.create_summary_report()

        # Step 6: Save results
        if save_plots:
            self.plot_distributions()

        if save_results:
            self.save_results()

        # Final conclusion
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)

        all_independent = all(
            r['p_value'] > 0.05 and abs(r['t_statistic']) < 1.0
            for r in self.results.values()
        )

        if all_independent:
            print("\n✓ Layer performance appears to be DATASET-INDEPENDENT")
            print("  All datasets show no significant difference between Lmax and Lmin")
            print("  (p > 0.05 and |t| < 1.0)")
        else:
            print("\n✗ Layer performance appears to be DATASET-DEPENDENT")
            print("  Some datasets show significant differences between Lmax and Lmin")

        print("\n" + "=" * 80)

        return summary_df


if __name__ == "__main__":
    # Example usage
    analyzer = CrossDatasetLayerAnalyzer(
        datasets=['zsre', 'counterfact'],
        model='gpt2-xl',
        metric='post_overall',
        proxy_size=100,
        random_seed=42
    )

    analyzer.run_full_analysis(save_plots=True, save_results=True)
