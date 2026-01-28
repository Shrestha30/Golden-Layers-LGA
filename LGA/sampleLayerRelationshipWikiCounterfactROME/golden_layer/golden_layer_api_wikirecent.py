"""
WikiRecent Golden Layer API

Handles WikiRecent-specific data structure and metrics:
- rewrite_acc (Edit Success)
- locality_acc (averaged across all locality keys)
- portability_acc (averaged across all portability keys)
- fluency (ngram_entropy - lower is better, so NEGATED before normalization)
- overall (normalized average of the four metrics)

NOTE: WikiRecent does NOT have rephrase_acc
IMPORTANT: Normalization is done PER-SAMPLE across layers (not global)
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


class GoldenLayerAnalyzerWikiRecent:
    """Analyzer for WikiRecent dataset with proper metric handling"""

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

        # WikiRecent-specific metrics (4 metrics, NO rephrase_acc)
        self.metrics_info = [
            ('post_rewrite_acc', 'Rewrite Accuracy'),
            ('post_locality_acc', 'Locality'),
            ('post_portability_acc', 'Portability'),
            ('post_fluency', 'Fluency'),
            ('overall', 'Overall')
        ]

    def load_data(self, verbose=True):
        """
        Load WikiRecent results from nested JSON files.

        WikiRecent structure:
        {
          "post": {
            "rewrite_acc": [score],
            "locality": {"Relation_Specificity_acc": [scores...], ...},
            "portability": {
              "Subject_Aliasing_acc": [scores...],
              "reasoning_acc": [scores...],
              "Logical_Generalization_acc": [scores...]
            },
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

                    # Extract locality (average across all locality keys) - EXACT eval structure
                    if 'locality' in post:
                        case_list = []
                        for key in post['locality'].keys():
                            if isinstance(post['locality'][key], list) and len(post['locality'][key]) > 0:
                                case_list.append(sum(post['locality'][key]) / len(post['locality'][key]) * 100)
                        if len(case_list) != 0:
                            sample_data['locality_acc'] = np.mean(case_list) / 100  # Convert back to [0,1]
                        else:
                            sample_data['locality_acc'] = np.nan
                    else:
                        sample_data['locality_acc'] = np.nan

                    # Extract portability (average across all portability keys) - EXACT eval structure
                    if 'portability' in post:
                        case_list = []
                        for key in post['portability'].keys():
                            if isinstance(post['portability'][key], list) and len(post['portability'][key]) > 0:
                                case_list.append(sum(post['portability'][key]) / len(post['portability'][key]) * 100)
                        if len(case_list) != 0:
                            sample_data['portability_acc'] = np.mean(case_list) / 100  # Convert back to [0,1]
                        else:
                            sample_data['portability_acc'] = np.nan
                    else:
                        sample_data['portability_acc'] = np.nan

                    # Extract fluency (ngram_entropy)
                    if 'fluency' in post and 'ngram_entropy' in post['fluency']:
                        sample_data['fluency'] = post['fluency']['ngram_entropy']
                    else:
                        sample_data['fluency'] = np.nan

                    layer_samples.append(sample_data)

            all_data[layer] = layer_samples

        # Determine number of samples
        self.n_samples = max(len(samples) for samples in all_data.values()) if all_data else 0

        # Convert to tensors (samples x layers) - WikiRecent has 4 metrics
        metrics = ['rewrite_acc', 'locality_acc', 'portability_acc', 'fluency']

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
            print(f"Loaded WikiRecent data:")
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
        Compute overall metric with PER-SAMPLE normalization (WikiRecent has 4 metrics).

        Steps:
        1. NEGATE fluency (lower is better → higher is better)
        2. Min-max normalize each metric PER-SAMPLE across layers to [0, 1]
        3. Average the normalized metrics for each sample
        """
        # Step 1: Negate fluency
        fluency_negated = -self.results['post_fluency']

        # Step 2: Min-max normalize each metric PER-SAMPLE (across layers)
        normalized = {}

        for metric_name, values in [
            ('rewrite_acc', self.results['post_rewrite_acc']),
            ('locality_acc', self.results['post_locality_acc']),
            ('portability_acc', self.results['post_portability_acc']),
            ('fluency_negated', fluency_negated)
        ]:
            tensor = values.numpy()  # Shape: (n_samples, n_layers)

            # Min-max normalization for each sample across layers
            # min_vals = np.nanmin(tensor, axis=1, keepdims=True)
            # max_vals = np.nanmax(tensor, axis=1, keepdims=True)
            min_vals = np.nanmin(tensor)
            max_vals = np.nanmax(tensor)

            # Avoid division by zero
            denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

            # Normalize
            normalized_tensor = (tensor - min_vals) / denom
            normalized[metric_name] = normalized_tensor

            if verbose:
                print(f"Min-max normalized {metric_name} per-sample across layers")

        # Step 3: Compute overall as mean of the four normalized metrics
        stacked = np.stack([
            normalized['rewrite_acc'],
            normalized['locality_acc'],
            normalized['portability_acc'],
            normalized['fluency_negated']
        ], axis=2)  # Shape: (n_samples, n_layers, 4)

        # Take mean across the 4 metrics (axis=2)
        overall_tensor = np.nanmean(stacked, axis=2)  # Shape: (n_samples, n_layers)

        # Convert back to torch tensor
        self.results['overall'] = torch.from_numpy(overall_tensor).float()

        if verbose:
            non_nan = (~np.isnan(overall_tensor)).sum()
            total = overall_tensor.size
            print(f"\nComputed post_overall metric: {non_nan} / {total} non-NaN values")

        return self

    def split_dataset(self, proxy_size=80, random_seed=42, verbose=True):
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
                'name': metric_name,
                'proxy_best_layers': proxy_best,
                'proxy_score': proxy_best_val.item(),
                'test_best_layers': test_best,
                'test_score': test_best_val.item(),
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


def eval(result_path):
    """
    Evaluate WikiRecent dataset results at a specific path.
    This is the EXACT eval function for WikiRecent (4 metrics only).
    """
    from os import path

    if path.exists(result_path):

        with open(result_path, 'r') as file:
            datas = json.load(file)
        datas = datas[:-1]

        # Rewrite Accuracy
        Edit_Succ_list = [data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ = sum(Edit_Succ_list) / len(Edit_Succ_list) * 100
        print('Edit_Succ:', Edit_Succ)

        # Portability (average across all portability metrics)
        Portability_list = []
        for data_rome_counterfact in datas:
            case_list = []
            for key in data_rome_counterfact['post']['portability'].keys():
                case_list.append(sum(data_rome_counterfact['post']['portability'][key]) / len(data_rome_counterfact['post']['portability'][key]) * 100)
            if len(case_list) != 0:
                Portability_list.append(np.mean(case_list))
        Overall_portability = np.mean(Portability_list)
        print('Overall_portability:', Overall_portability)

        # Locality (average across all locality keys)
        Locality_list = []
        for data_rome_counterfact in datas:
            case_list = []
            for key in data_rome_counterfact['post']['locality'].keys():
                case_list.append(sum(data_rome_counterfact['post']['locality'][key]) / len(data_rome_counterfact['post']['locality'][key]) * 100)
            if len(case_list) != 0:
                Locality_list.append(np.mean(case_list))
        Overall_locality = np.mean(Locality_list)
        print('Overall_locality:', Overall_locality)

        # Fluency (ngram_entropy)
        Fluency_list = [x['post']['fluency']['ngram_entropy'] for x in datas]
        Fluency = sum(Fluency_list) / len(Fluency_list) * 100
        print('Fluency:', Fluency)

        return {
            'Edit_Succ': Edit_Succ,
            'Portability': Overall_portability,
            'Locality': Overall_locality,
            'Fluency': Fluency
        }
    else:
        print(f"Error: File not found: {result_path}")
        return None


def main():
    """
    Example usage of WikiRecent Golden Layer Analyzer
    """
    # Create analyzer
    analyzer = GoldenLayerAnalyzerWikiRecent(data_dir='../allLayerPerformance', n_layers=32)

    # Load data
    print("Loading WikiRecent data...")
    analyzer.load_data(verbose=True)

    # Compute overall metric
    print("\nComputing overall metric...")
    analyzer.compute_overall_metric(verbose=True)

    # Split dataset
    print("\nSplitting dataset...")
    analyzer.split_dataset(proxy_size=120, random_seed=42, verbose=True)

    # Compute proxy and test means
    print("\nComputing proxy and test means...")
    analyzer.compute_proxy_test_means(verbose=True)

    # Analyze
    print("\nPerforming analysis...")
    analyzer.analyze(verbose=True)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
