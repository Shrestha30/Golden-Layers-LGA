#!/usr/bin/env python3
"""
Experiment: Layer Independence Analysis for GPT2-XL Rewrite Accuracy

Research Question:
"Are the layers selected for gpt2-xl independent of the dataset?"

Methodology:
1. Load all available datasets for gpt2-xl
2. Identify best layers (proxy and test) for each dataset
3. Collect candidate layer sets:
   - Proxy layers only
   - Test layers only
   - All layers (proxy + test)
4. For each dataset:
   - Find Lmax and Lmin from the candidate set based on that dataset's performance
   - Conduct paired t-test on that dataset: Lmax(i) vs Lmin(i)
5. Analyze results to determine if layer selection is dataset-independent

Metric: Rewrite accuracy (flexible matching: post_rewrite or post_rewrite_acc)
Model: gpt2-xl (48 layers)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_model_registry import get_config, DATASET_MODEL_REGISTRY


class RewriteExperiment:
    """
    Experiment to test layer independence for GPT2-XL rewrite accuracy.
    """

    def __init__(self, model='gpt2-xl', metric='rewrite', random_seed=42):
        """
        Initialize the experiment.

        Args:
            model: Model name (default: gpt2-xl)
            metric: Metric to analyze (default: rewrite)
                   Can be exact name or substring (e.g., 'rewrite' matches both 'post_rewrite' and 'post_rewrite_acc')
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.metric_query = metric  # What user requested
        self.random_seed = random_seed
        self.dataset_metrics = {}  # Will store actual metric name per dataset

        # Find all datasets available for this model
        self.datasets = self._find_available_datasets()
        print(f"Found {len(self.datasets)} datasets for {model}: {self.datasets}")
        print(f"Metric mapping: {self.dataset_metrics}")

        # Storage
        self.analyzers = {}
        self.best_layers = {}
        self.candidate_layers = {}
        self.results = {}

    def _find_available_datasets(self):
        """Find all datasets that have this model and metric (flexible matching)."""
        datasets = []
        metric_lower = self.metric_query.lower()

        for (dataset, model), config in DATASET_MODEL_REGISTRY.items():
            if model == self.model:
                # Check if metric_query matches any available metric (exact or substring)
                available_metrics = config.get('metrics', [])
                for avail_metric in available_metrics:
                    if (self.metric_query == avail_metric or  # Exact match
                        metric_lower in avail_metric.lower()):  # Substring match
                        datasets.append(dataset)
                        # Store the actual metric name for this dataset
                        self.dataset_metrics[dataset] = avail_metric
                        break
        return sorted(datasets)

    def load_datasets(self):
        """Load all datasets directly (each with its own metric name)."""
        print("\n" + "=" * 80)
        print("STEP 1: Loading All Datasets")
        print("=" * 80)

        # Import here to avoid circular dependencies
        import importlib.util
        import torch

        # Load each dataset individually with its specific metric
        for dataset in self.datasets:
            metric_name = self.dataset_metrics[dataset]
            config = get_config(dataset, self.model)

            # Get dataset-specific proxy size
            proxy_size = config.get('proxy_size', 100)

            # Dynamically load the API
            api_path = config['api_path']
            spec = importlib.util.spec_from_file_location(
                f"golden_layer_api_{dataset}",
                api_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the correct class
            api_class_name = config.get('api_class', 'GoldenLayerAnalyzer')
            analyzer_class = getattr(module, api_class_name)

            # Create and load analyzer
            data_dir = config['base_dir'] / config['data_dir']
            analyzer = analyzer_class(data_dir=str(data_dir))

            print(f"\nLoading {dataset} (metric={metric_name}, proxy_size={proxy_size})...")
            analyzer.load_data(verbose=False)

            # Check if compute_overall_metric exists
            if hasattr(analyzer, 'compute_overall_metric'):
                analyzer.compute_overall_metric(verbose=False)

            analyzer.split_dataset(proxy_size=proxy_size, random_seed=self.random_seed, verbose=False)
            analyzer.compute_proxy_test_means(verbose=False)

            print(f"  Loaded {analyzer.n_samples} samples, {analyzer.n_layers} layers")

            # Store analyzer
            self.analyzers[dataset] = analyzer

            # Identify best layers for this dataset using its specific metric
            proxy_means = analyzer.proxy_means[metric_name]
            test_means = analyzer.test_means[metric_name]
            proxy_best = torch.argmax(proxy_means).item()
            test_best = torch.argmax(test_means).item()

            self.best_layers[dataset] = {
                'proxy': proxy_best,
                'test': test_best,
                'proxy_score': proxy_means[proxy_best].item(),
                'test_score': test_means[test_best].item(),
                'metric_name': metric_name  # Store which metric was used
            }

        print(f"\n✓ Loaded {len(self.analyzers)} datasets")

        # Display best layers and filter out datasets with NaN scores
        print("\n" + "=" * 80)
        print("Best Layers Identified")
        print("=" * 80)

        valid_datasets = []
        for dataset in self.datasets:
            info = self.best_layers[dataset]
            print(f"\n{dataset}:")
            print(f"  Proxy: Layer {info['proxy']:2d} (score: {info['proxy_score']:.4f})")
            print(f"  Test:  Layer {info['test']:2d} (score: {info['test_score']:.4f})")

            # Check if scores are valid (not NaN)
            import math
            if not (math.isnan(info['proxy_score']) or math.isnan(info['test_score'])):
                valid_datasets.append(dataset)
            else:
                print(f"  ⚠ WARNING: Dataset has NaN scores, will be excluded from analysis")

        # Update datasets list to only include valid ones
        if len(valid_datasets) < len(self.datasets):
            print(f"\n⚠ Excluding {len(self.datasets) - len(valid_datasets)} dataset(s) with invalid data")
            print(f"  Valid datasets: {valid_datasets}")
            print(f"  Excluded datasets: {[d for d in self.datasets if d not in valid_datasets]}")
            self.datasets = valid_datasets

        if len(self.datasets) == 0:
            raise ValueError("No valid datasets found with non-NaN scores!")

    def collect_candidate_layers(self):
        """Collect candidate layers from all datasets."""
        print("\n" + "=" * 80)
        print("STEP 2: Collecting Candidate Layers")
        print("=" * 80)

        # Collect proxy layers
        proxy_layers = set()
        test_layers = set()

        for dataset in self.datasets:
            proxy_layers.add(self.best_layers[dataset]['proxy'])
            test_layers.add(self.best_layers[dataset]['test'])

        # Create candidate sets
        self.candidate_layers = {
            'proxy_only': sorted(proxy_layers),
            'test_only': sorted(test_layers),
            'all': sorted(proxy_layers.union(test_layers))
        }

        print(f"\nProxy layers: {self.candidate_layers['proxy_only']}")
        print(f"Test layers: {self.candidate_layers['test_only']}")
        print(f"All layers: {self.candidate_layers['all']}")

    def run_analysis_for_layer_set(self, layer_set_name):
        """
        Run analysis for a specific set of candidate layers.

        Args:
            layer_set_name: 'proxy_only', 'test_only', or 'all'
        """
        print("\n" + "=" * 80)
        print(f"STEP 3.{['proxy_only', 'test_only', 'all'].index(layer_set_name) + 1}: "
              f"Analysis for {layer_set_name.upper()} layers")
        print("=" * 80)

        candidate_layers = self.candidate_layers[layer_set_name]
        print(f"Candidate layers: {candidate_layers}")

        results = {}

        for dataset in self.datasets:
            print(f"\n--- Analyzing {dataset} ---")

            analyzer = self.analyzers[dataset]
            metric_name = self.dataset_metrics[dataset]  # Use dataset-specific metric

            # Debug: Check what's available
            if not hasattr(analyzer, 'results') or metric_name not in analyzer.results:
                print(f"  ✗ Error: Metric '{metric_name}' not found in analyzer.results")
                print(f"     Available metrics: {list(analyzer.results.keys()) if hasattr(analyzer, 'results') else 'None'}")
                continue

            # Get performance for all candidate layers on this dataset
            layer_performances = {}
            for layer in candidate_layers:
                try:
                    # Get test set performance for this layer
                    test_indices = analyzer.test_indices
                    performance_data = analyzer.results[metric_name][test_indices, layer]
                    mean_perf = performance_data.mean().item()
                    layer_performances[layer] = mean_perf
                except Exception as e:
                    print(f"  ✗ Error accessing layer {layer}: {e}")
                    print(f"     Results shape: {analyzer.results[metric_name].shape if hasattr(analyzer.results[metric_name], 'shape') else 'N/A'}")
                    print(f"     Test indices type: {type(test_indices)}")
                    raise

            # Find Lmax and Lmin for this dataset
            lmax_layer = max(layer_performances.items(), key=lambda x: x[1])[0]
            lmin_layer = min(layer_performances.items(), key=lambda x: x[1])[0]

            print(f"  Lmax: Layer {lmax_layer} (mean: {layer_performances[lmax_layer]:.4f})")
            print(f"  Lmin: Layer {lmin_layer} (mean: {layer_performances[lmin_layer]:.4f})")

            # Extract per-sample scores for Lmax and Lmin
            lmax_scores = analyzer.results[metric_name][test_indices, lmax_layer].numpy()
            lmin_scores = analyzer.results[metric_name][test_indices, lmin_layer].numpy()

            # Remove NaN values
            valid_mask = ~(np.isnan(lmax_scores) | np.isnan(lmin_scores))
            lmax_scores_clean = lmax_scores[valid_mask]
            lmin_scores_clean = lmin_scores[valid_mask]

            n_valid = len(lmax_scores_clean)
            print(f"  Valid samples: {n_valid}")

            # Conduct paired t-test
            if n_valid < 2:
                print(f"  ✗ Not enough valid samples for t-test")
                continue

            t_stat, p_value = stats.ttest_rel(lmax_scores_clean, lmin_scores_clean)

            # Compute effect size (Cohen's d)
            differences = lmax_scores_clean - lmin_scores_clean
            cohen_d = differences.mean() / (differences.std() + 1e-10)

            # Store results
            results[dataset] = {
                'lmax_layer': int(lmax_layer),
                'lmin_layer': int(lmin_layer),
                'lmax_mean': float(layer_performances[lmax_layer]),
                'lmin_mean': float(layer_performances[lmin_layer]),
                'mean_difference': float(layer_performances[lmax_layer] - layer_performances[lmin_layer]),
                'n_samples': int(n_valid),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohen_d': float(cohen_d),
                'significant': bool(p_value <= 0.05),
                'independent': bool(p_value > 0.05 and abs(t_stat) < 1.0 and abs(cohen_d) < 0.2)
            }

            # Display results
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohen_d:.4f}")

            # Interpretation
            if results[dataset]['independent']:
                print(f"  ✓ Evidence SUPPORTS dataset independence")
            else:
                print(f"  ✗ Evidence AGAINST dataset independence")

        return results

    def run_all_analyses(self):
        """Run analysis for all three layer sets."""
        print("\n" + "=" * 80)
        print("STEP 3: Conducting Analyses")
        print("=" * 80)

        for layer_set_name in ['proxy_only', 'test_only', 'all']:
            self.results[layer_set_name] = self.run_analysis_for_layer_set(layer_set_name)

    def create_summary_report(self):
        """Create summary DataFrame for each analysis."""
        print("\n" + "=" * 80)
        print("STEP 4: Summary Reports")
        print("=" * 80)

        summaries = {}

        for layer_set_name in ['proxy_only', 'test_only', 'all']:
            print(f"\n--- {layer_set_name.upper()} Summary ---")

            data = []
            for dataset, result in self.results[layer_set_name].items():
                data.append({
                    'Dataset': dataset,
                    'N': result['n_samples'],
                    'Lmax': result['lmax_layer'],
                    'Lmin': result['lmin_layer'],
                    'Lmax Mean': f"{result['lmax_mean']:.4f}",
                    'Lmin Mean': f"{result['lmin_mean']:.4f}",
                    'Mean Diff': f"{result['mean_difference']:.4f}",
                    'T-stat': f"{result['t_statistic']:.4f}",
                    'P-value': f"{result['p_value']:.4f}",
                    "Cohen's d": f"{result['cohen_d']:.4f}",
                    'Significant': 'Yes' if result['significant'] else 'No',
                    'Independent': 'Yes' if result['independent'] else 'No'
                })

            df = pd.DataFrame(data)
            summaries[layer_set_name] = df

            print(df.to_string(index=False))

        return summaries

    def save_results(self, output_dir='./results'):
        """Save results to files."""
        print("\n" + "=" * 80)
        print("STEP 5: Saving Results")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat()

        # Save detailed JSON
        json_data = {
            'metadata': {
                'experiment': 'layer_independence',
                'model': self.model,
                'metric_query': self.metric_query,
                'dataset_metrics': self.dataset_metrics,  # Show which metric was used per dataset
                'datasets': self.datasets,
                'random_seed': self.random_seed,
                'timestamp': timestamp
            },
            'best_layers': self.best_layers,
            'candidate_layers': self.candidate_layers,
            'results': self.results
        }

        json_path = output_dir / f'experiment_rewrite_{self.model}.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Saved detailed results: {json_path}")

        # Save summary CSVs
        summaries = self.create_summary_report()
        for layer_set_name, df in summaries.items():
            csv_path = output_dir / f'experiment_rewrite_{self.model}_{layer_set_name}_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved summary CSV: {csv_path}")

    def print_final_conclusion(self):
        """Print final conclusion about layer independence."""
        print("\n" + "=" * 80)
        print("FINAL CONCLUSION")
        print("=" * 80)
        print(f"\nDatasets analyzed: {self.datasets}")

        for layer_set_name in ['proxy_only', 'test_only', 'all']:
            print(f"\n--- {layer_set_name.upper()} Layers ---")
            print(f"Candidate layers: {self.candidate_layers[layer_set_name]}")

            results = self.results[layer_set_name]

            # Count independent vs dependent
            n_independent = sum(1 for r in results.values() if r['independent'])
            n_total = len(results)

            print(f"\nResults: {n_independent}/{n_total} datasets show independence")

            # Overall conclusion
            if n_independent == n_total:
                print(f"✓ STRONG EVIDENCE: Layer performance is dataset-independent")
            elif n_independent >= n_total * 0.7:
                print(f"~ MODERATE EVIDENCE: Layer performance is mostly dataset-independent")
            else:
                print(f"✗ EVIDENCE AGAINST: Layer performance is dataset-dependent")

            # Show which datasets support independence
            independent_datasets = [d for d, r in results.items() if r['independent']]
            dependent_datasets = [d for d, r in results.items() if not r['independent']]

            if independent_datasets:
                print(f"  Independent: {', '.join(independent_datasets)}")
            if dependent_datasets:
                print(f"  Dependent: {', '.join(dependent_datasets)}")

    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        print("\n" + "=" * 80)
        print("EXPERIMENT: Layer Independence for GPT2-XL Rewrite Accuracy")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Metric query: {self.metric_query}")
        print(f"Random seed: {self.random_seed}")

        self.load_datasets()
        self.collect_candidate_layers()
        self.run_all_analyses()
        self.save_results()
        self.print_final_conclusion()

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Layer Independence Experiment for GPT2-XL Rewrite Accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt2-xl',
        help='Model name (default: gpt2-xl)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='rewrite',
        help='Metric to analyze - can be substring (default: rewrite)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory (default: ./results)'
    )

    args = parser.parse_args()

    # Run experiment
    experiment = RewriteExperiment(
        model=args.model,
        metric=args.metric,
        random_seed=args.seed
    )

    experiment.run_full_experiment()


if __name__ == "__main__":
    main()
