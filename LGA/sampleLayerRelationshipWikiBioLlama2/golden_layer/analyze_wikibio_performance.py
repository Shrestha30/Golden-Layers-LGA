"""
WikiBio All-Layer Performance Analysis Script

Processes WikiBio results with proper metric handling:
- rewrite_acc
- Relation_Specificity_acc (locality)
- fluency (negated - lower is better)
- Overall (average of normalized metrics)
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

class WikiBioAnalyzer:
    """Analyzer for WikiBio experiment results"""

    def __init__(self, data_dir='../allLayerPerformance'):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.n_samples = 0
        self.n_layers = 48

    def load_data(self, verbose=True):
        """Load all layer results from JSON files"""

        # Initialize storage for each metric
        metrics = {
            'rewrite_acc': [],
            'locality_acc': [],
            'fluency': []
        }

        # Load data for each layer
        for layer in range(self.n_layers):
            # Find all JSON files for this layer
            pattern = f"*_layer{layer}.json"
            layer_files = list(self.data_dir.glob(pattern))

            if not layer_files:
                if verbose:
                    print(f"Warning: No data found for layer {layer}")
                continue

            # Load and process each file for this layer
            layer_data = {
                'rewrite_acc': [],
                'locality_acc': [],
                'fluency': []
            }

            for file_path in sorted(layer_files):
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Process each sample in the file
                for sample in data:
                    if 'post' not in sample:
                        continue

                    post = sample['post']

                    # Extract rewrite accuracy
                    if 'rewrite_acc' in post and isinstance(post['rewrite_acc'], list):
                        layer_data['rewrite_acc'].append(post['rewrite_acc'][0])

                    # Extract locality (Relation_Specificity)
                    if 'locality' in post and 'Relation_Specificity_acc' in post['locality']:
                        locality_scores = post['locality']['Relation_Specificity_acc']
                        if isinstance(locality_scores, list) and len(locality_scores) > 0:
                            # Average locality scores
                            avg_locality = np.mean(locality_scores)
                            layer_data['locality_acc'].append(avg_locality)
                        else:
                            layer_data['locality_acc'].append(np.nan)

                    # Extract fluency (ngram_entropy)
                    if 'fluency' in post and 'ngram_entropy' in post['fluency']:
                        layer_data['fluency'].append(post['fluency']['ngram_entropy'])

            # Store layer data
            for metric in metrics:
                if layer_data[metric]:
                    metrics[metric].append(layer_data[metric])
                else:
                    # Fill with NaN if no data
                    metrics[metric].append([np.nan] * (self.n_samples or 100))

        # Determine number of samples (from first non-empty layer)
        for metric_data in metrics.values():
            for layer_data in metric_data:
                if layer_data:
                    self.n_samples = len(layer_data)
                    break
            if self.n_samples > 0:
                break

        # Convert to tensors (samples x layers)
        self.results = {}
        for metric, data in metrics.items():
            # Pad or truncate to ensure consistent size
            processed_data = []
            for layer_samples in data:
                if len(layer_samples) < self.n_samples:
                    # Pad with NaN
                    layer_samples.extend([np.nan] * (self.n_samples - len(layer_samples)))
                elif len(layer_samples) > self.n_samples:
                    # Truncate
                    layer_samples = layer_samples[:self.n_samples]
                processed_data.append(layer_samples)

            # Convert to tensor (samples x layers)
            self.results[metric] = torch.tensor(processed_data, dtype=torch.float32).T

        if verbose:
            print(f"\nLoaded WikiBio data:")
            print(f"  Samples: {self.n_samples}")
            print(f"  Layers: {self.n_layers}")
            for metric, tensor in self.results.items():
                non_nan = (~torch.isnan(tensor)).sum().item()
                total = tensor.numel()
                print(f"  {metric}: {non_nan}/{total} non-NaN values")

        return self

    def compute_overall_metric(self, verbose=True):
        """
        Compute overall metric with proper handling:
        1. Negate fluency (lower is better → higher is better)
        2. Min-max normalize all metrics to [0, 1]
        3. Average normalized metrics
        """

        # Negate fluency first (lower is better → higher is better)
        fluency_negated = -self.results['fluency']

        # Min-max normalize each metric
        normalized = {}

        for metric_name, values in [
            ('rewrite_acc', self.results['rewrite_acc']),
            ('locality_acc', self.results['locality_acc']),
            ('fluency', fluency_negated)
        ]:
            # Get min and max across all samples and layers
            valid_mask = ~torch.isnan(values)
            if valid_mask.sum() == 0:
                normalized[metric_name] = values
                continue

            min_val = values[valid_mask].min()
            max_val = values[valid_mask].max()

            if max_val - min_val > 1e-10:
                norm_values = (values - min_val) / (max_val - min_val)
            else:
                norm_values = values

            normalized[metric_name] = norm_values

            if verbose:
                print(f"Normalized {metric_name}: [{min_val:.4f}, {max_val:.4f}] → [0, 1]")

        # Compute overall as average
        overall = (
            normalized['rewrite_acc'] +
            normalized['locality_acc'] +
            normalized['fluency']
        ) / 3.0

        self.results['overall'] = overall

        if verbose:
            non_nan = (~torch.isnan(overall)).sum().item()
            total = overall.numel()
            print(f"\nComputed overall metric: {non_nan}/{total} non-NaN values")

        return self

    def split_dataset(self, proxy_size=100, random_seed=42, verbose=True):
        """Split into proxy and test sets"""
        torch.manual_seed(random_seed)
        indices = torch.randperm(self.n_samples)

        self.proxy_indices = indices[:proxy_size]
        self.test_indices = indices[proxy_size:]

        if verbose:
            print(f"\nDataset split:")
            print(f"  Proxy set: {len(self.proxy_indices)} samples")
            print(f"  Test set: {len(self.test_indices)} samples")

        return self

    def compute_proxy_test_means(self, verbose=True):
        """Compute mean performance for proxy and test sets"""
        self.proxy_means = {}
        self.test_means = {}

        for metric, values in self.results.items():
            # Proxy means (average over proxy samples for each layer)
            proxy_values = values[self.proxy_indices]
            self.proxy_means[metric] = proxy_values.nanmean(dim=0)

            # Test means (average over test samples for each layer)
            test_values = values[self.test_indices]
            self.test_means[metric] = test_values.nanmean(dim=0)

            if verbose:
                print(f"{metric}: Computed proxy and test means")

        return self

    def find_best_layers(self, verbose=True):
        """Find best performing layers for each metric"""
        self.best_layers = {}

        for metric in ['rewrite_acc', 'locality_acc', 'overall']:
            # Find best layers on proxy set
            proxy_mean = self.proxy_means[metric]
            max_score = proxy_mean.nanmax()
            best_layers_proxy = torch.where(proxy_mean == max_score)[0].tolist()

            # Find best layers on test set
            test_mean = self.test_means[metric]
            max_score_test = test_mean.nanmax()
            best_layers_test = torch.where(test_mean == max_score_test)[0].tolist()

            # Compute correlation
            valid_mask = ~(torch.isnan(proxy_mean) | torch.isnan(test_mean))
            if valid_mask.sum() > 1:
                corr = torch.corrcoef(torch.stack([
                    proxy_mean[valid_mask],
                    test_mean[valid_mask]
                ]))[0, 1].item()
            else:
                corr = np.nan

            self.best_layers[metric] = {
                'proxy_best': best_layers_proxy,
                'proxy_score': max_score.item(),
                'test_best': best_layers_test,
                'test_score': max_score_test.item(),
                'correlation': corr,
                'overlap': list(set(best_layers_proxy) & set(best_layers_test))
            }

            if verbose:
                info = self.best_layers[metric]
                print(f"\n{metric}:")
                print(f"  Proxy best layers: {info['proxy_best']} (score: {info['proxy_score']:.4f})")
                print(f"  Test best layers: {info['test_best']} (score: {info['test_score']:.4f})")
                print(f"  Overlap: {info['overlap'] if info['overlap'] else 'None'}")
                print(f"  Correlation: {info['correlation']:.4f}")

        return self

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("WIKIBIO LAYER ANALYSIS SUMMARY")
        print("="*70)

        # Golden layer identification
        overall_info = self.best_layers['overall']
        print("\nGOLDEN LAYER IDENTIFICATION:")
        print(f"  Proxy best layers:  {overall_info['proxy_best']} (score: {overall_info['proxy_score']:.4f})")
        print(f"  Test best layers:   {overall_info['test_best']} (score: {overall_info['test_score']:.4f})")

        if overall_info['overlap']:
            print(f"\n  ✓✓✓ GOLDEN LAYER(S) FOUND: {overall_info['overlap']} ✓✓✓")
            print("  These layers perform best on both proxy and test sets!")
        else:
            print("\n  ✗ No perfect overlap between proxy and test best layers")

        # Correlation summary
        print("\nCORRELATION SUMMARY:")
        correlations = [info['correlation'] for info in self.best_layers.values()
                       if not np.isnan(info['correlation'])]
        if correlations:
            print(f"  Average correlation: {np.mean(correlations):.4f}")
            print(f"  Min correlation: {np.min(correlations):.4f}")
            print(f"  Max correlation: {np.max(correlations):.4f}")


if __name__ == "__main__":
    # Example usage
    analyzer = WikiBioAnalyzer(data_dir='../allLayerPerformance')

    (analyzer
        .load_data(verbose=True)
        .compute_overall_metric(verbose=True)
        .split_dataset(proxy_size=100, random_seed=42, verbose=True)
        .compute_proxy_test_means(verbose=True)
        .find_best_layers(verbose=True))

    analyzer.print_summary()
