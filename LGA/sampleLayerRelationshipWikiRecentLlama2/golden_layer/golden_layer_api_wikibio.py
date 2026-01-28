"""
WikiBio Golden Layer API

Handles WikiBio-specific data structure and metrics:
- rewrite_acc (Edit Success)
- locality_acc (Relation Specificity - averaged)
- fluency (ngram_entropy - lower is better, so NEGATED before normalization)
- overall (normalized average of the three metrics)
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


class GoldenLayerAnalyzerWikiBio:
    """Analyzer for WikiBio dataset with proper metric handling"""

    def __init__(self, data_dir='../allLayerPerformance', n_layers=48):
        self.data_dir = Path(data_dir)
        self.n_layers = n_layers
        self.results = {}
        self.n_samples = 0
        self.proxy_means = {}
        self.test_means = {}
        self.proxy_indices = None
        self.test_indices = None
        self.best_layers_comparison = {}

        # WikiBio-specific metrics
        self.metrics_info = [
            ('post_rewrite_acc', 'Rewrite Accuracy'),
            ('post_locality_acc', 'Locality'),
            ('post_fluency', 'Fluency'),
            ('post_overall', 'Overall')
        ]

    def load_data(self, verbose=True):
        """
        Load WikiBio results from nested JSON files.

        WikiBio structure:
        {
          "post": {
            "rewrite_acc": [score],
            "locality": {"Relation_Specificity_acc": [scores...]},
            "fluency": {"ngram_entropy": value}
          }
        }
        """
        all_data = defaultdict(list)

        # Load all JSON files
        for layer in range(self.n_layers):
            layer_samples = []

            # Find all files for this layer (pattern: *_layer{layer}.json)
            for json_file in sorted(self.data_dir.glob(f"*_layer{layer}.json")):
                with open(json_file, 'r') as f:
                    samples = json.load(f)

                # Remove the last sample if it exists (often a summary or incomplete)
                if len(samples) > 0:
                    samples = samples[:-1]

                for sample in samples:
                    if 'post' not in sample:
                        continue

                    post = sample['post']
                    sample_data = {}

                    # Extract rewrite_acc
                    if 'rewrite_acc' in post and isinstance(post['rewrite_acc'], list):
                        sample_data['rewrite_acc'] = post['rewrite_acc'][0]
                    else:
                        sample_data['rewrite_acc'] = np.nan

                    # Extract locality (Relation_Specificity_acc) - average the scores
                    if 'locality' in post and 'Relation_Specificity_acc' in post['locality']:
                        locality_scores = post['locality']['Relation_Specificity_acc']
                        if isinstance(locality_scores, list) and len(locality_scores) > 0:
                            sample_data['locality_acc'] = np.mean(locality_scores)
                        else:
                            sample_data['locality_acc'] = np.nan
                    else:
                        sample_data['locality_acc'] = np.nan

                    # Extract fluency (ngram_entropy)
                    if 'fluency' in post and 'ngram_entropy' in post['fluency']:
                        sample_data['fluency'] = post['fluency']['ngram_entropy']
                    else:
                        sample_data['fluency'] = np.nan

                    layer_samples.append(sample_data)

            all_data[layer] = layer_samples

        # Determine number of samples
        self.n_samples = max(len(samples) for samples in all_data.values()) if all_data else 0

        # Convert to tensors (samples x layers)
        metrics = ['rewrite_acc', 'locality_acc', 'fluency']

        for metric in metrics:
            tensor = torch.full((self.n_samples, self.n_layers), float('nan'))

            for layer in range(self.n_layers):
                layer_samples = all_data[layer]
                for sample_idx, sample in enumerate(layer_samples):
                    if sample_idx < self.n_samples:
                        tensor[sample_idx, layer] = sample.get(metric, np.nan)

            # Store with 'post_' prefix to match API expectations
            self.results[f'post_{metric}'] = tensor
        for key in self.results:
            if 'fluency' in key:
                tensor_np = self.results[key].numpy()
                
                v_min = np.nanmin(tensor_np)
                v_max = np.nanmax(tensor_np)
                denom = (v_max - v_min) if v_max != v_min else 1.0
                
                # Normalize so 0.0 is the best (lowest raw loss/entropy)
                normalized = (tensor_np - v_min) / denom
                self.results[key] = torch.from_numpy(normalized).float()
        if verbose:
            print(f"Loaded WikiBio data:")
            print(f"  Samples: {self.n_samples}")
            print(f"  Layers: {self.n_layers}")
            for metric_key, metric_name in self.metrics_info[:-1]:  # Skip overall
                if metric_key in self.results:
                    tensor = self.results[metric_key]
                    non_nan = (~torch.isnan(tensor)).sum().item()
                    total = tensor.numel()
                    print(f"  {metric_name}: {non_nan}/{total} non-NaN values")

        return self

    def compute_overall_metric(self, verbose=True):
        """
        Compute overall metric with fluency negation.

        Steps:
        1. NEGATE fluency (lower is better → higher is better)
        2. Min-max normalize all metrics to [0, 1]
        3. Average the normalized metrics
        """
        # Step 1: Negate fluency
        fluency_negated = -self.results['post_fluency']

        # Step 2: Min-max normalize each metric
        normalized = {}

        for metric_name, values in [
            ('rewrite_acc', self.results['post_rewrite_acc']),
            ('locality_acc', self.results['post_locality_acc']),
            ('fluency_negated', fluency_negated)
        ]:
            valid_mask = ~torch.isnan(values)

            if valid_mask.sum() == 0:
                normalized[metric_name] = values
                if verbose:
                    print(f"{metric_name}: No valid data for normalization")
                continue

            min_val = values[valid_mask].min()
            max_val = values[valid_mask].max()

            if max_val - min_val > 1e-10:
                norm_values = (values - min_val) / (max_val - min_val)
            else:
                norm_values = torch.ones_like(values) * 0.5

            normalized[metric_name] = norm_values

            if verbose:
                print(f"Normalized {metric_name}: [{min_val:.4f}, {max_val:.4f}] → [0, 1]")

        # Step 3: Compute overall as average
        overall = (
            normalized['rewrite_acc'] +
            normalized['locality_acc'] +
            normalized['fluency_negated']
        ) / 3.0

        self.results['post_overall'] = overall

        if verbose:
            non_nan = (~torch.isnan(overall)).sum().item()
            total = overall.numel()
            print(f"Computed overall: {non_nan}/{total} non-NaN values")

        return self

    def split_dataset(self, proxy_size=30, random_seed=42, verbose=True):
        """Split into proxy and test sets"""
        torch.manual_seed(random_seed)
        indices = torch.randperm(self.n_samples)

        self.proxy_indices = indices[:proxy_size]
        self.test_indices = indices[proxy_size:]

        if verbose:
            print(f"\nDataset split:")
            print(f"  Total samples: {self.n_samples}")
            print(f"  Proxy set: {len(self.proxy_indices)} samples")
            print(f"  Test set: {len(self.test_indices)} samples")

        return self

    def compute_proxy_test_means(self, verbose=True):
        """Compute mean performance for proxy and test sets"""
        for metric_key, _ in self.metrics_info:
            if metric_key in self.results:
                self.proxy_means[metric_key] = self.results[metric_key][self.proxy_indices].nanmean(dim=0)
                self.test_means[metric_key] = self.results[metric_key][self.test_indices].nanmean(dim=0)

        if verbose:
            print("\nComputed proxy and test means for all metrics")

        return self

    def analyze(self, verbose=True):
        """Find best layers and compute correlations"""
        self.best_layers_comparison = {}

        for metric_key, metric_name in self.metrics_info:
            if metric_key not in self.proxy_means or metric_key not in self.test_means:
                continue

            proxy_mean = self.proxy_means[metric_key]
            test_mean = self.test_means[metric_key]

            # Find best layers (handle NaN values)
            # For fluency (raw), LOWER is better, so find MIN
            valid_proxy = ~torch.isnan(proxy_mean)
            valid_test = ~torch.isnan(test_mean)

            if metric_key == 'post_fluency':
                # For fluency: lower is better
                if valid_proxy.sum() > 0:
                    proxy_best_val = proxy_mean[valid_proxy].min()
                    proxy_best = torch.where(proxy_mean == proxy_best_val)[0].tolist()
                else:
                    proxy_best_val = torch.tensor(float('nan'))
                    proxy_best = []

                if valid_test.sum() > 0:
                    test_best_val = test_mean[valid_test].min()
                    test_best = torch.where(test_mean == test_best_val)[0].tolist()
                else:
                    test_best_val = torch.tensor(float('nan'))
                    test_best = []
            else:
                # For other metrics: higher is better
                if valid_proxy.sum() > 0:
                    proxy_best_val = proxy_mean[valid_proxy].max()
                    proxy_best = torch.where(proxy_mean == proxy_best_val)[0].tolist()
                else:
                    proxy_best_val = torch.tensor(float('nan'))
                    proxy_best = []

                if valid_test.sum() > 0:
                    test_best_val = test_mean[valid_test].max()
                    test_best = torch.where(test_mean == test_best_val)[0].tolist()
                else:
                    test_best_val = torch.tensor(float('nan'))
                    test_best = []

            overlap = list(set(proxy_best) & set(test_best))

            # Compute correlation
            valid_mask = ~(torch.isnan(proxy_mean) | torch.isnan(test_mean))
            if valid_mask.sum() > 1:
                corr = torch.corrcoef(torch.stack([
                    proxy_mean[valid_mask],
                    test_mean[valid_mask]
                ]))[0, 1].item()
            else:
                corr = np.nan

            self.best_layers_comparison[metric_key] = {
                'proxy_best_layers': proxy_best,
                'test_best_layers': test_best,
                'overlap': overlap,
                'correlation': corr
            }

            if verbose:
                print(f"\n{metric_name}:")
                print(f"  Proxy best: {proxy_best} (score: {proxy_best_val:.4f})")
                print(f"  Test best: {test_best} (score: {test_best_val:.4f})")
                print(f"  Overlap: {overlap if overlap else 'None'}")
                print(f"  Correlation: r = {corr:.4f}")

        return self
