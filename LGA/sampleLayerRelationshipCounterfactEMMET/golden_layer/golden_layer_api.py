#!/usr/bin/env python3
"""
Golden Layer Analysis API

This module provides API functions to analyze layer performance and identify
the optimal "golden layer" for model editing.

Usage:
    from golden_layer_api import GoldenLayerAnalyzer

    analyzer = GoldenLayerAnalyzer(data_dir='../allLayerPerformance')
    analyzer.load_data()
    analyzer.compute_overall_metric()
    analyzer.split_dataset(proxy_size=100, random_seed=42)
    analyzer.compute_proxy_test_means()
    results = analyzer.analyze()
    analyzer.save_bar_charts(output_dir='./figures')
    analyzer.save_results(output_dir='./results')
"""

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GoldenLayerAnalyzer:
    """
    Analyzer class for identifying golden layers in model editing.
    """

    def __init__(self, data_dir: str = '../allLayerPerformance'):
        """
        Initialize the analyzer.

        Args:
            data_dir: Path to directory containing layer performance JSON files
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        self.n_samples = 1031
        self.n_layers = 48

        self.metrics_info = [
            ('post_rewrite', 'Rewrite Accuracy'),
            ('post_rephrase', 'Rephrase Accuracy'),
            ('post_neighborhood', 'Locality'),
            ('post_one_hop', 'Portability'),
            ('post_overall', 'Overall')
        ]

        self.proxy_indices = None
        self.test_indices = None
        self.proxy_means = {}
        self.test_means = {}
        self.best_layers_comparison = {}
        self.correlations_dict = {}

        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')

    def _get_sample_range_info(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get information about sample ranges.

        Returns:
            Dict mapping sample_range to (start_idx, end_idx, count)
        """
        return {
            '0-1031': (0, 1031, 1031)
        }

    def _load_layer_performance_file(self, sample_range: str, layer: int) -> Optional[List[dict]]:
        """
        Load a single JSON file for a specific sample range and layer.

        Args:
            sample_range: String like '25', '25-100', etc.
            layer: Layer number (0-47)

        Returns:
            List of case dictionaries or None if file doesn't exist
        """
        filename = f"EMMET_results{sample_range}_layer{layer}.json"
        filepath = self.data_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Filter out summary entries
        cases = [case for case in data if isinstance(case, dict) and not case.get('summary', False)]

        return cases

    def _extract_metric_value(self, case: dict, metric_path: List[str]) -> Optional[float]:
        """
        Extract a metric value from a case using a path.

        Args:
            case: Case dictionary
            metric_path: List of keys, e.g., ['post', 'rewrite_acc']

        Returns:
            Metric value or None
        """
        obj = case
        for key in metric_path:
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]

        # Handle list format
        if isinstance(obj, list) and len(obj) > 0:
            return sum(obj)/len(obj)
        elif isinstance(obj, (int, float)):
            return obj

        return None

    def load_data(self, verbose: bool = True) -> 'GoldenLayerAnalyzer':
        """
        Load all layer performance data into tensors.

        Args:
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        # Define metrics to extract
        metric_definitions = {
            'pre_rewrite': ['pre', 'rewrite_acc'],
            'pre_rephrase': ['pre', 'rephrase_acc'],
            'pre_one_hop': ['pre', 'portability', 'one_hop_acc'],
            'post_rewrite': ['post', 'rewrite_acc'],
            'post_rephrase': ['post', 'rephrase_acc'],
            'post_neighborhood': ['post', 'locality', 'neighborhood_acc'],
            'post_one_hop': ['post', 'portability', 'one_hop_acc'],
        }

        # Initialize tensors with NaN
        for metric_name in metric_definitions.keys():
            self.results[metric_name] = torch.full((self.n_samples, self.n_layers), float('nan'))

        # Get sample range information
        sample_range_info = self._get_sample_range_info()

        # Load data for each sample range and layer
        for sample_range, (start_idx, end_idx, count) in sample_range_info.items():
            if verbose:
                print(f"Loading sample range {sample_range}: samples {start_idx}-{end_idx-1}...")

            for layer in range(self.n_layers):
                cases = self._load_layer_performance_file(sample_range, layer)

                if cases is None:
                    if verbose:
                        print(f"  Warning: No data for {sample_range}, layer {layer}")
                    continue

                # Extract metrics for each case
                for case in cases:
                    case_id = case.get('case_id')
                    if case_id is None:
                        continue

                    # Calculate global sample index
                    sample_idx = start_idx + case_id

                    if sample_idx >= self.n_samples:
                        if verbose:
                            print(f"  Warning: sample_idx {sample_idx} out of range for case_id {case_id}")
                        continue

                    # Extract all metrics for this case
                    for metric_name, metric_path in metric_definitions.items():
                        value = self._extract_metric_value(case, metric_path)
                        if value is not None:
                            self.results[metric_name][sample_idx, layer] = value

        if verbose:
            print(f"\nLoaded {self.n_samples} samples across {self.n_layers} layers")
            for key in self.results.keys():
                if isinstance(self.results[key], torch.Tensor):
                    non_nan = (~torch.isnan(self.results[key])).sum().item()
                    total = self.results[key].numel()
                    print(f"  {key}: {non_nan}/{total} non-NaN values")

        return self

    def compute_overall_metric(self, verbose: bool = True) -> 'GoldenLayerAnalyzer':
        """
        Compute the overall metric using min-max normalization.

        Args:
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        key_metrics = ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop']

        # Min-Max Normalize each metric across layers for each sample
        normalized_metrics = {}

        for metric_name in key_metrics:
            tensor = self.results[metric_name].numpy()  # Shape: (n_samples, n_layers)

            # Min-max normalization for each sample across layers
            # min_vals = np.nanmin(tensor, axis=1, keepdims=True)
            # max_vals = np.nanmax(tensor, axis=1, keepdims=True)
            min_vals = np.nanmin(tensor)
            max_vals = np.nanmax(tensor)

            # Avoid division by zero
            denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

            # Normalize
            normalized = (tensor - min_vals) / denom
            normalized_metrics[metric_name] = normalized

            if verbose:
                print(f"Min-max normalized {metric_name}")

        # Compute overall score as mean of the four normalized metrics
        stacked = np.stack([
            normalized_metrics['post_rewrite'],
            normalized_metrics['post_rephrase'],
            normalized_metrics['post_neighborhood'],
            normalized_metrics['post_one_hop']
        ], axis=2)  # Shape: (n_samples, n_layers, 4)

        # Take mean across the 4 metrics (axis=2)
        overall_tensor = np.nanmean(stacked, axis=2)  # Shape: (n_samples, n_layers)

        # Convert back to torch tensor
        self.results['post_overall'] = torch.from_numpy(overall_tensor).float()

        if verbose:
            print(f"\nComputed post_overall metric: {overall_tensor.shape}")
            print(f"Non-NaN values: {(~np.isnan(overall_tensor)).sum()} / {overall_tensor.size}")

        return self

    def split_dataset(self, proxy_size: int = 100, random_seed: int = 42,
                     verbose: bool = True) -> 'GoldenLayerAnalyzer':
        """
        Split dataset into proxy and test sets.

        Args:
            proxy_size: Number of samples in proxy set
            random_seed: Random seed for reproducibility
            verbose: Whether to print split information

        Returns:
            Self for method chaining
        """
        # Set random seed for reproducibility
        random.seed(random_seed)
        all_indices = list(range(self.n_samples))
        random.shuffle(all_indices)

        self.proxy_indices = sorted(all_indices[:proxy_size])
        self.test_indices = sorted(all_indices[proxy_size:])

        if verbose:
            print(f"\nDataset split:")
            print(f"  Total samples: {self.n_samples}")
            print(f"  Proxy set: {len(self.proxy_indices)} samples")
            print(f"  Test set: {len(self.test_indices)} samples")

        return self

    def compute_proxy_test_means(self, verbose: bool = True) -> 'GoldenLayerAnalyzer':
        """
        Compute average performance for proxy and test sets.

        Args:
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        if self.proxy_indices is None or self.test_indices is None:
            raise ValueError("Must call split_dataset() before compute_proxy_test_means()")

        for metric_key, metric_name in self.metrics_info:
            # Proxy set: average across proxy samples for each layer
            proxy_data = self.results[metric_key][self.proxy_indices, :]
            self.proxy_means[metric_key] = torch.nanmean(proxy_data, dim=0)

            # Test set: average across test samples for each layer
            test_data = self.results[metric_key][self.test_indices, :]
            self.test_means[metric_key] = torch.nanmean(test_data, dim=0)

            if verbose:
                print(f"{metric_name:20s}: Computed proxy and test means")

        return self

    def analyze(self, verbose: bool = True) -> Dict:
        """
        Perform complete analysis including best layer identification and correlations.

        Args:
            verbose: Whether to print results

        Returns:
            Dictionary containing analysis results
        """
        if not self.proxy_means or not self.test_means:
            raise ValueError("Must call compute_proxy_test_means() before analyze()")

        for metric_key, metric_name in self.metrics_info:
            proxy_layer_means = self.proxy_means[metric_key].numpy()
            test_layer_means = self.test_means[metric_key].numpy()

            # Compute correlation
            valid_mask = ~(np.isnan(proxy_layer_means) | np.isnan(test_layer_means))
            if valid_mask.sum() > 0:
                correlation = np.corrcoef(proxy_layer_means[valid_mask], test_layer_means[valid_mask])[0, 1]
                _, p_value = stats.pearsonr(proxy_layer_means[valid_mask], test_layer_means[valid_mask])
            else:
                correlation = np.nan
                p_value = np.nan

            self.correlations_dict[metric_key] = correlation

            # Find best layers for proxy set (handle ties)
            valid_proxy = ~np.isnan(proxy_layer_means)
            if valid_proxy.any():
                proxy_max_value = np.nanmax(proxy_layer_means)
                proxy_best_layers = np.where(proxy_layer_means == proxy_max_value)[0].tolist()
            else:
                proxy_best_layers = []
                proxy_max_value = np.nan

            # Find best layers for test set (handle ties)
            valid_test = ~np.isnan(test_layer_means)
            if valid_test.any():
                test_max_value = np.nanmax(test_layer_means)
                test_best_layers = np.where(test_layer_means == test_max_value)[0].tolist()
            else:
                test_best_layers = []
                test_max_value = np.nan

            # Check if any layers match
            overlap = set(proxy_best_layers) & set(test_best_layers)

            self.best_layers_comparison[metric_key] = {
                'proxy_best_layers': proxy_best_layers,
                'proxy_best_value': proxy_max_value,
                'test_best_layers': test_best_layers,
                'test_best_value': test_max_value,
                'overlap': list(overlap),
                'correlation': correlation,
                'p_value': p_value
            }

            if verbose:
                print(f"\n{metric_name}:")
                print(f"  Proxy: Layers {proxy_best_layers} (score: {proxy_max_value:.4f})")
                print(f"  Test:  Layers {test_best_layers} (score: {test_max_value:.4f})")
                print(f"  Overlap: {list(overlap) if overlap else 'None'}")
                print(f"  Correlation: r = {correlation:.4f}, p = {p_value:.4e}")

        return self.best_layers_comparison

    def save_bar_charts(self, output_dir: str = './figures', dpi: int = 150) -> 'GoldenLayerAnalyzer':
        """
        Save separate bar charts for each metric.

        Args:
            output_dir: Directory to save figures
            dpi: DPI for saved figures

        Returns:
            Self for method chaining
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for metric_key, metric_name in self.metrics_info:
            fig, ax = plt.subplots(figsize=(14, 5))

            # Get layer-wise means for proxy and test
            proxy_layer_means = self.proxy_means[metric_key].numpy()
            test_layer_means = self.test_means[metric_key].numpy()

            # Get correlation
            correlation = self.correlations_dict.get(metric_key, np.nan)

            # Create bar chart
            x = np.arange(self.n_layers)
            width = 0.35

            ax.bar(x - width/2, proxy_layer_means, width, label='Proxy Set', alpha=0.8)
            ax.bar(x + width/2, test_layer_means, width, label='Test Set', alpha=0.8)

            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'{metric_name} - Correlation: {correlation:.3f}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(range(0, self.n_layers, 2))
            ax.set_xticklabels(range(0, self.n_layers, 2), fontsize=9)

            plt.tight_layout()

            # Save figure
            filename = f"{metric_key}_comparison.png"
            filepath = output_path / filename
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved: {filepath}")

        return self

    def save_results(self, output_dir: str = './results') -> 'GoldenLayerAnalyzer':
        """
        Save analysis results to files.

        Args:
            output_dir: Directory to save results

        Returns:
            Self for method chaining
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch tensors
        torch.save({
            'post_rewrite': self.results['post_rewrite'],
            'post_rephrase': self.results['post_rephrase'],
            'post_neighborhood': self.results['post_neighborhood'],
            'post_one_hop': self.results['post_one_hop'],
            'post_overall': self.results['post_overall'],
            'proxy_indices': self.proxy_indices,
            'test_indices': self.test_indices,
            'proxy_means': {k: v for k, v in self.proxy_means.items()},
            'test_means': {k: v for k, v in self.test_means.items()},
            'best_layers_comparison': self.best_layers_comparison,
            'correlations_dict': self.correlations_dict,
            'n_samples': self.n_samples,
            'n_layers': self.n_layers
        }, output_path / 'golden_layer_analysis.pt')
        print(f"Saved: {output_path / 'golden_layer_analysis.pt'}")

        # Create and save summary table (transposed format)
        metric_names = [name for _, name in self.metrics_info]
        metric_keys = [key for key, _ in self.metrics_info]

        table_data = {
            'Metric': metric_names,
            'Correlation': [f"{self.correlations_dict[key]:.4f}" if key in self.correlations_dict else "N/A"
                           for key in metric_keys],
            'Proxy Best Layers': [str(self.best_layers_comparison[key]['proxy_best_layers'])
                                 for key in metric_keys],
            'Test Best Layers': [str(self.best_layers_comparison[key]['test_best_layers'])
                                for key in metric_keys],
            'Proxy Best Score': [f"{self.best_layers_comparison[key]['proxy_best_value']:.4f}"
                                for key in metric_keys],
            'Test Best Score': [f"{self.best_layers_comparison[key]['test_best_value']:.4f}"
                               for key in metric_keys],
            'Layer 17 Proxy': [f"{self.proxy_means[key][17].item():.4f}"
                              for key in metric_keys],
            'Layer 17 Test': [f"{self.test_means[key][17].item():.4f}"
                             for key in metric_keys],
        }

        # Add rows for performance at layers selected by each metric (for proxy set)
        for selector_key, selector_name in self.metrics_info:
            best_layers = self.best_layers_comparison[selector_key]['proxy_best_layers']
            if best_layers:
                # Take first layer if there are ties
                layer = best_layers[0]
                row_name = f'{selector_name} Best Layer Proxy'
                table_data[row_name] = [f"{self.proxy_means[metric_key][layer].item():.4f}"
                                       for metric_key in metric_keys]

        # Add rows for performance at layers selected by each metric (for test set)
        for selector_key, selector_name in self.metrics_info:
            best_layers = self.best_layers_comparison[selector_key]['test_best_layers']
            if best_layers:
                # Take first layer if there are ties
                layer = best_layers[0]
                row_name = f'{selector_name} Best Layer Test'
                table_data[row_name] = [f"{self.test_means[metric_key][layer].item():.4f}"
                                       for metric_key in metric_keys]

        df_summary = pd.DataFrame(table_data)
        df_transposed = df_summary.set_index('Metric').T

        df_transposed.to_csv(output_path / 'golden_layer_summary_table.csv')
        print(f"Saved: {output_path / 'golden_layer_summary_table.csv'}")

        # Save detailed results
        df_detailed = pd.DataFrame({
            'Metric': metric_names,
            'Correlation': [self.correlations_dict[key] for key in metric_keys],
            'Proxy_Best_Layers': [str(self.best_layers_comparison[key]['proxy_best_layers'])
                                 for key in metric_keys],
            'Proxy_Best_Value': [self.best_layers_comparison[key]['proxy_best_value']
                                for key in metric_keys],
            'Test_Best_Layers': [str(self.best_layers_comparison[key]['test_best_layers'])
                                for key in metric_keys],
            'Test_Best_Value': [self.best_layers_comparison[key]['test_best_value']
                               for key in metric_keys],
            'Overlap': [str(self.best_layers_comparison[key]['overlap']) for key in metric_keys],
            'P_Value': [self.best_layers_comparison[key]['p_value'] for key in metric_keys]
        })

        df_detailed.to_csv(output_path / 'golden_layer_detailed.csv', index=False)
        print(f"Saved: {output_path / 'golden_layer_detailed.csv'}")

        return self

    def print_summary(self):
        """
        Print analysis summary.
        """
        print(f"\n{'='*70}")
        print("GOLDEN LAYER ANALYSIS SUMMARY")
        print(f"{'='*70}")

        # Create summary table
        metric_names = [name for _, name in self.metrics_info]
        metric_keys = [key for key, _ in self.metrics_info]

        table_data = {
            'Metric': metric_names,
            'Correlation': [f"{self.correlations_dict[key]:.4f}" if key in self.correlations_dict else "N/A"
                           for key in metric_keys],
            'Proxy Best Layers': [str(self.best_layers_comparison[key]['proxy_best_layers'])
                                 for key in metric_keys],
            'Test Best Layers': [str(self.best_layers_comparison[key]['test_best_layers'])
                                for key in metric_keys],
            'Proxy Best Score': [f"{self.best_layers_comparison[key]['proxy_best_value']:.4f}"
                                for key in metric_keys],
            'Test Best Score': [f"{self.best_layers_comparison[key]['test_best_value']:.4f}"
                               for key in metric_keys],
            'Layer 17 Proxy': [f"{self.proxy_means[key][17].item():.4f}"
                              for key in metric_keys],
            'Layer 17 Test': [f"{self.test_means[key][17].item():.4f}"
                             for key in metric_keys],
        }

        # Add rows for performance at layers selected by each metric (for proxy set)
        for selector_key, selector_name in self.metrics_info:
            best_layers = self.best_layers_comparison[selector_key]['proxy_best_layers']
            if best_layers:
                # Take first layer if there are ties
                layer = best_layers[0]
                row_name = f'{selector_name} Best Layer Proxy'
                table_data[row_name] = [f"{self.proxy_means[metric_key][layer].item():.4f}"
                                       for metric_key in metric_keys]

        # Add rows for performance at layers selected by each metric (for test set)
        for selector_key, selector_name in self.metrics_info:
            best_layers = self.best_layers_comparison[selector_key]['test_best_layers']
            if best_layers:
                # Take first layer if there are ties
                layer = best_layers[0]
                row_name = f'{selector_name} Best Layer Test'
                table_data[row_name] = [f"{self.test_means[metric_key][layer].item():.4f}"
                                       for metric_key in metric_keys]

        df_summary = pd.DataFrame(table_data)
        df_transposed = df_summary.set_index('Metric').T

        print("\n1. SUMMARY TABLE (Metrics as Columns):")
        print(df_transposed.to_string())

        print("\n\n2. GOLDEN LAYER IDENTIFICATION:")
        overall_info = self.best_layers_comparison['post_overall']
        print(f"  Proxy best layers:  {overall_info['proxy_best_layers']} (score: {overall_info['proxy_best_value']:.4f})")
        print(f"  Test best layers:   {overall_info['test_best_layers']} (score: {overall_info['test_best_value']:.4f})")

        if overall_info['overlap']:
            print(f"\n  ✓✓✓ GOLDEN LAYER(S) FOUND: {overall_info['overlap']} ✓✓✓")
            print(f"  These layers perform best on both proxy and test sets!")
        else:
            print(f"\n  ✗ No common golden layer found")
            print(f"  Proxy and test sets have different best layers")

        print("\n\n3. CORRELATION SUMMARY:")
        all_corrs = [v for v in self.correlations_dict.values() if not np.isnan(v)]
        if all_corrs:
            print(f"  Average correlation: {np.mean(all_corrs):.4f}")
            print(f"  Min correlation: {np.min(all_corrs):.4f}")
            print(f"  Max correlation: {np.max(all_corrs):.4f}")


def main():
    """
    Main function demonstrating API usage.
    """
    # Create analyzer
    analyzer = GoldenLayerAnalyzer(data_dir='../allLayerPerformance')

    # Load data
    print("Loading data...")
    analyzer.load_data(verbose=True)

    # Compute overall metric
    print("\nComputing overall metric...")
    analyzer.compute_overall_metric(verbose=True)

    # Split dataset
    print("\nSplitting dataset...")
    analyzer.split_dataset(proxy_size=100, random_seed=42, verbose=True)

    # Compute proxy and test means
    print("\nComputing proxy and test means...")
    analyzer.compute_proxy_test_means(verbose=True)

    # Analyze
    print("\nPerforming analysis...")
    results = analyzer.analyze(verbose=True)

    # Save bar charts
    print("\nSaving bar charts...")
    analyzer.save_bar_charts(output_dir='./figures', dpi=150)

    # Save results
    print("\nSaving results...")
    analyzer.save_results(output_dir='./results')

    # Print summary
    analyzer.print_summary()

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
