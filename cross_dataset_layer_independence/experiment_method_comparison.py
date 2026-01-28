#!/usr/bin/env python3
"""
Experiment: Layer Selection Method Comparison

Research Question:
"Do different layer selection methods (Proxy, Influence, CMA) result in similar
performance when their selected layers are applied across all datasets?"

Methodology:
1. For each dataset, identify the layer selected by each method:
   - Proxy: Best overall metric on proxy set
   - Influence: Based on specific.OverallNewOverallHat (if available)
   - CMA: Causal Mediation Analysis (if available)
2. For each method, collect performance across all datasets:
   Method_performance = [dataset1_perf, dataset2_perf, ..., datasetK_perf]
3. Conduct t-tests between methods:
   - Proxy vs Influence
   - Proxy vs CMA
   - Influence vs CMA

Model: gpt2-xl
Metric: Rewrite accuracy (post_rewrite or post_rewrite_acc)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dataset_model_registry import get_config, DATASET_MODEL_REGISTRY


class MethodComparisonExperiment:
    """
    Compare layer selection methods across datasets.
    """

    def __init__(self, model='gpt2-xl', metric='rewrite', random_seed=42):
        """
        Initialize experiment.

        Args:
            model: Model name
            metric: Metric to analyze (substring matching)
            random_seed: Random seed
        """
        self.model = model
        self.metric_query = metric
        self.random_seed = random_seed
        self.dataset_metrics = {}

        # Find available datasets
        self.datasets = self._find_available_datasets()
        print(f"Found {len(self.datasets)} datasets for {model}")
        print(f"Datasets: {self.datasets}")
        print(f"Metric mapping: {self.dataset_metrics}")

        # Storage
        self.analyzers = {}
        self.method_selections = {}  # method -> {dataset: layer}
        self.method_performances = {}  # method -> {dataset: performance}
        self.results = {}

    def _find_available_datasets(self):
        """Find datasets with the model and metric."""
        datasets = []
        metric_lower = self.metric_query.lower()

        for (dataset, model), config in DATASET_MODEL_REGISTRY.items():
            if model == self.model:
                available_metrics = config.get('metrics', [])
                for avail_metric in available_metrics:
                    if (self.metric_query == avail_metric or
                        metric_lower in avail_metric.lower()):
                        datasets.append(dataset)
                        self.dataset_metrics[dataset] = avail_metric
                        break
        return sorted(datasets)

    def load_datasets(self):
        """Load all datasets."""
        print("\n" + "=" * 80)
        print("STEP 1: Loading Datasets")
        print("=" * 80)

        import importlib.util
        import torch

        for dataset in self.datasets:
            metric_name = self.dataset_metrics[dataset]
            config = get_config(dataset, self.model)
            proxy_size = config.get('proxy_size', 100)

            # Load API
            api_path = config['api_path']
            spec = importlib.util.spec_from_file_location(
                f"golden_layer_api_{dataset}", api_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            api_class_name = config.get('api_class', 'GoldenLayerAnalyzer')
            analyzer_class = getattr(module, api_class_name)

            # Create analyzer
            data_dir = config['base_dir'] / config['data_dir']
            analyzer = analyzer_class(data_dir=str(data_dir))

            print(f"\nLoading {dataset}...")
            analyzer.load_data(verbose=False)

            if hasattr(analyzer, 'compute_overall_metric'):
                analyzer.compute_overall_metric(verbose=False)

            analyzer.split_dataset(proxy_size=proxy_size,
                                 random_seed=self.random_seed,
                                 verbose=False)
            analyzer.compute_proxy_test_means(verbose=False)

            print(f"  {analyzer.n_samples} samples, {analyzer.n_layers} layers")

            self.analyzers[dataset] = analyzer

        print(f"\n✓ Loaded {len(self.analyzers)} datasets")

    def identify_proxy_selected_layers(self):
        """Identify layers selected by Proxy method (best overall on proxy set)."""
        print("\n" + "=" * 80)
        print("STEP 2: Identifying Proxy-Selected Layers")
        print("=" * 80)

        # Use predefined layer selections from user-provided data
        predefined_proxy_layers = {
            'zsre': 16,
            'wikicounterfact': 19,
            'wikibio': 23,
            'wikirecent': 31,
            'counterfact': None  # Add if available
        }

        self.method_selections['proxy'] = {}

        for dataset in self.datasets:
            if dataset in predefined_proxy_layers and predefined_proxy_layers[dataset] is not None:
                best_layer = predefined_proxy_layers[dataset]
                self.method_selections['proxy'][dataset] = best_layer
                print(f"{dataset:20s}: Layer {best_layer:2d} (from predefined data)")
            else:
                # Fallback to computing if not predefined
                import torch
                analyzer = self.analyzers[dataset]
                metric_name = self.dataset_metrics[dataset]
                proxy_means = analyzer.proxy_means[metric_name]
                best_layer = torch.argmax(proxy_means).item()
                self.method_selections['proxy'][dataset] = best_layer
                print(f"{dataset:20s}: Layer {best_layer:2d} (computed from proxy means)")

    def identify_influence_selected_layers(self):
        """
        Identify layers selected by Influence method.

        Uses predefined influence layer selections.
        """
        print("\n" + "=" * 80)
        print("STEP 3: Identifying Influence-Selected Layers")
        print("=" * 80)

        # Use predefined layer selections from user-provided data
        predefined_influence_layers = {
            'zsre': 8,
            'wikicounterfact': 11,
            'wikibio': 12,
            'wikirecent': 11,
            'counterfact': None  # Add if available
        }

        self.method_selections['influence'] = {}

        for dataset in self.datasets:
            if dataset in predefined_influence_layers and predefined_influence_layers[dataset] is not None:
                best_layer = predefined_influence_layers[dataset]
                self.method_selections['influence'][dataset] = best_layer
                print(f"{dataset:20s}: Layer {best_layer:2d} (from predefined data)")
            else:
                print(f"{dataset:20s}: ✗ No predefined influence layer available")

        # Keep old code commented for reference
        """
        import torch

        self.method_selections['influence'] = {}

        # TracInInf directory path

        # Dataset name mapping for TracInInf files
        dataset_name_map = {
            'zsre': None,  # No dataset prefix for zsre
            'counterfact': 'counterfact',  # has dataset prefix
            'wikibio': 'wikibio',
            'wikicounterfact': 'counterfact',  # Note: uses 'counterfact' in filename
            'wikirecent': 'recent'
        }

        for dataset in self.datasets:
            config = get_config(dataset, self.model)
            model_name = self.model

            # Construct filename pattern - try to find the file
            dataset_prefix = dataset_name_map.get(dataset)
            influence_file = None

            # Search for matching file (handle sample count variations)
            import glob
            if dataset_prefix:
                # Has dataset prefix
                pattern = f"modelName-{model_name}_dataset-{dataset_prefix}_localIF-True_iden-True_qLoss-False_norm-False_overall-specific.OverallNewOverallHat_layers-48__nsamples-*.pt"
            else:
                # No dataset prefix (zsre) - try both norm=True and norm=False
                pattern = f"modelName-{model_name}_localIF-True_iden-True_qLoss-False_norm-*_overall-specific.OverallNewOverallHat_layers-48__nsamples-*.pt"

            matching_files = list(glob.glob(str(tracininf_dir / pattern)))
            if matching_files:
                # Use the first matching file (prefer larger sample counts)
                matching_files.sort(reverse=True)  # Sort descending by filename (larger nsamples first)
                influence_file = Path(matching_files[0])

            if influence_file and influence_file.exists():
                try:
                    # Load influence scores
                    print(f"  Loading: {influence_file.name}")
                    influence_data = torch.load(influence_file, map_location='cpu')

                    # The data structure should be: dict or tensor with shape (n_samples, n_layers)
                    # or possibly (n_layers,) if aggregated
                    if isinstance(influence_data, dict):
                        # Try to find the scores in the dict
                        if 'influence_scores' in influence_data:
                            scores = influence_data['influence_scores']
                        elif 'scores' in influence_data:
                            scores = influence_data['scores']
                        else:
                            # Use the first tensor value
                            scores = list(influence_data.values())[0]
                    else:
                        scores = influence_data

                    # Aggregate across samples if needed (take mean)
                    if len(scores.shape) == 2:
                        # Shape: (n_samples, n_layers)
                        layer_scores = scores.mean(dim=0)
                    else:
                        # Already aggregated
                        layer_scores = scores

                    # Find best layer (maximum influence score)
                    best_layer = torch.argmax(layer_scores).item()

                    self.method_selections['influence'][dataset] = best_layer
                    print(f"{dataset:20s}: Layer {best_layer:2d} (score: {layer_scores[best_layer]:.4f})")
                except Exception as e:
                    print(f"{dataset:20s}: ✗ Error loading influence file: {e}")
            else:
                print(f"{dataset:20s}: ✗ No influence file found (pattern: {pattern})")
        """

    def identify_cma_selected_layers(self):
        """
        Identify layers selected by CMA (Causal Mediation Analysis) method.

        Uses predefined CMA layer selections.
        """
        print("\n" + "=" * 80)
        print("STEP 3.5: Identifying CMA-Selected Layers")
        print("=" * 80)

        # Use predefined layer selections from user-provided data
        predefined_cma_layers = {
            'zsre': 5,
            'wikicounterfact': 6,
            'wikibio': 6,
            'wikirecent': 6,
            'counterfact': None  # Add if available
        }

        self.method_selections['cma'] = {}

        for dataset in self.datasets:
            if dataset in predefined_cma_layers and predefined_cma_layers[dataset] is not None:
                best_layer = predefined_cma_layers[dataset]
                self.method_selections['cma'][dataset] = best_layer
                print(f"{dataset:20s}: Layer {best_layer:2d} (from predefined data)")
            else:
                print(f"{dataset:20s}: ✗ No predefined CMA layer available")

    def evaluate_method_performances(self):
        """
        For each method, evaluate the selected layer's performance on each dataset's test set.
        """
        print("\n" + "=" * 80)
        print("STEP 4: Evaluating Method Performances")
        print("=" * 80)

        import torch

        for method in self.method_selections.keys():
            print(f"\n--- Method: {method.upper()} ---")
            self.method_performances[method] = {}

            for dataset in self.datasets:
                if dataset not in self.method_selections[method]:
                    print(f"{dataset:20s}: ✗ No layer selected by this method")
                    continue

                selected_layer = self.method_selections[method][dataset]
                analyzer = self.analyzers[dataset]
                metric_name = self.dataset_metrics[dataset]

                # Get test set performance
                test_indices = analyzer.test_indices
                test_performance = analyzer.results[metric_name][test_indices, selected_layer]
                mean_perf = test_performance.mean().item()

                # Filter NaN
                valid_scores = test_performance[~torch.isnan(test_performance)]
                n_valid = len(valid_scores)

                self.method_performances[method][dataset] = {
                    'layer': selected_layer,
                    'mean': mean_perf,
                    'n_samples': n_valid,
                    'scores': valid_scores.numpy()
                }

                print(f"{dataset:20s}: Layer {selected_layer:2d} → {mean_perf:.4f} ({n_valid} samples)")

    def compare_methods(self):
        """
        Compare methods using t-tests across datasets.

        For each pair of methods, conduct t-test:
        Method1_performances = [dataset1, dataset2, ..., datasetK]
        Method2_performances = [dataset1, dataset2, ..., datasetK]
        """
        print("\n" + "=" * 80)
        print("STEP 5: Comparing Methods")
        print("=" * 80)

        methods = list(self.method_performances.keys())

        # Find common datasets (datasets where all methods have selections)
        common_datasets = set(self.datasets)
        for method in methods:
            common_datasets &= set(self.method_performances[method].keys())

        common_datasets = sorted(common_datasets)
        print(f"\nCommon datasets across all methods: {common_datasets}")
        print(f"Number of datasets: {len(common_datasets)}\n")

        if len(common_datasets) < 2:
            print("✗ Not enough common datasets for comparison (need at least 2)")
            return

        self.results = {}

        # Compare each pair of methods
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]

                print(f"--- {method1.upper()} vs {method2.upper()} ---")

                # Collect performances across common datasets
                perf1 = []
                perf2 = []

                for dataset in common_datasets:
                    perf1.append(self.method_performances[method1][dataset]['mean'])
                    perf2.append(self.method_performances[method2][dataset]['mean'])

                perf1 = np.array(perf1)
                perf2 = np.array(perf2)

                # Conduct paired t-test
                t_stat, p_value = stats.ttest_rel(perf1, perf2)

                # Compute effect size (Cohen's d)
                differences = perf1 - perf2
                cohen_d = differences.mean() / (differences.std() + 1e-10)

                # Store results
                comparison_key = f"{method1}_vs_{method2}"
                self.results[comparison_key] = {
                    'method1': method1,
                    'method2': method2,
                    'n_datasets': len(common_datasets),
                    'datasets': common_datasets,
                    'method1_performances': perf1.tolist(),
                    'method2_performances': perf2.tolist(),
                    'method1_mean': float(perf1.mean()),
                    'method2_mean': float(perf2.mean()),
                    'mean_difference': float(perf1.mean() - perf2.mean()),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohen_d': float(cohen_d),
                    'significant': bool(p_value <= 0.05)
                }

                # Display
                print(f"  {method1} mean: {perf1.mean():.4f}")
                print(f"  {method2} mean: {perf2.mean():.4f}")
                print(f"  Difference: {perf1.mean() - perf2.mean():.4f}")
                print(f"  T-statistic: {t_stat:.4f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Cohen's d: {cohen_d:.4f}")

                if p_value <= 0.05:
                    winner = method1 if perf1.mean() > perf2.mean() else method2
                    print(f"  ✓ Significant difference (p ≤ 0.05): {winner} performs better")
                else:
                    print(f"  ✗ No significant difference (p > 0.05)")
                print()

    def save_results(self, output_dir='./results'):
        """Save results to files."""
        print("\n" + "=" * 80)
        print("STEP 6: Saving Results")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed JSON
        json_data = {
            'metadata': {
                'experiment': 'method_comparison',
                'model': self.model,
                'metric_query': self.metric_query,
                'dataset_metrics': self.dataset_metrics,
                'datasets': self.datasets,
                'random_seed': self.random_seed,
                'timestamp': datetime.now().isoformat()
            },
            'method_selections': {
                method: {dataset: info['layer']
                        for dataset, info in selections.items()}
                for method, selections in self.method_performances.items()
            },
            'method_performances': {
                method: {dataset: {'layer': info['layer'], 'mean': info['mean'], 'n_samples': info['n_samples']}
                        for dataset, info in perfs.items()}
                for method, perfs in self.method_performances.items()
            },
            'comparisons': self.results
        }

        json_path = output_dir / f'method_comparison_{self.model}.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Saved: {json_path}")

        # Save comparison CSV
        if self.results:
            data = []
            for comparison_key, result in self.results.items():
                data.append({
                    'Comparison': f"{result['method1']} vs {result['method2']}",
                    'N Datasets': result['n_datasets'],
                    'Method1 Mean': f"{result['method1_mean']:.4f}",
                    'Method2 Mean': f"{result['method2_mean']:.4f}",
                    'Difference': f"{result['mean_difference']:.4f}",
                    'T-stat': f"{result['t_statistic']:.4f}",
                    'P-value': f"{result['p_value']:.4f}",
                    "Cohen's d": f"{result['cohen_d']:.4f}",
                    'Significant': 'Yes' if result['significant'] else 'No'
                })

            df = pd.DataFrame(data)
            csv_path = output_dir / f'method_comparison_{self.model}_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved: {csv_path}")

    def print_conclusion(self):
        """Print final conclusion."""
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)

        if not self.results:
            print("No method comparisons available.")
            return

        for comparison_key, result in self.results.items():
            print(f"\n{result['method1'].upper()} vs {result['method2'].upper()}:")
            print(f"  Datasets: {result['datasets']}")
            print(f"  {result['method1']}: {result['method1_mean']:.4f}")
            print(f"  {result['method2']}: {result['method2_mean']:.4f}")

            if result['significant']:
                winner = result['method1'] if result['mean_difference'] > 0 else result['method2']
                print(f"  → {winner.upper()} significantly better (p={result['p_value']:.4f})")
            else:
                print(f"  → No significant difference (p={result['p_value']:.4f})")

    def run_full_experiment(self):
        """Run complete experiment."""
        print("\n" + "=" * 80)
        print("EXPERIMENT: Layer Selection Method Comparison")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Metric: {self.metric_query}")

        self.load_datasets()
        self.identify_proxy_selected_layers()
        self.identify_influence_selected_layers()
        self.identify_cma_selected_layers()
        self.evaluate_method_performances()
        self.compare_methods()
        self.save_results()
        self.print_conclusion()

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare layer selection methods across datasets'
    )
    parser.add_argument('--model', type=str, default='gpt2-xl')
    parser.add_argument('--metric', type=str, default='rewrite')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./results')

    args = parser.parse_args()

    experiment = MethodComparisonExperiment(
        model=args.model,
        metric=args.metric,
        random_seed=args.seed
    )

    experiment.run_full_experiment()


if __name__ == "__main__":
    main()
