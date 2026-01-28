#!/usr/bin/env python3
"""
Golden Layer Analyzers - Unified Module

This module contains all analyzer classes for different dataset types:
1. GoldenLayerAnalyzer - for ZSRE and Counterfact datasets
2. GoldenLayerAnalyzerWikiBio - for WikiBio dataset
3. GoldenLayerAnalyzerWikiRecent - for WikiRecent and WikiCounterfact datasets

Usage:
    from analyzers import GoldenLayerAnalyzer, GoldenLayerAnalyzerWikiBio, GoldenLayerAnalyzerWikiRecent
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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# GoldenLayerAnalyzer - For ZSRE and Counterfact datasets
# =============================================================================

class GoldenLayerAnalyzer:
    """
    Analyzer class for identifying golden layers in model editing.
    Used for ZSRE and Counterfact datasets.
    """

    def __init__(self, data_dir: str = '../allLayerPerformance', n_layers: int = 48):
        """
        Initialize the analyzer.

        Args:
            data_dir: Path to directory containing layer performance JSON files
            n_layers: Number of layers in the model
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        self.n_samples = 1037
        self.n_layers = n_layers

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

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')

    def _get_sample_range_info(self) -> Dict[str, Tuple[int, int, int]]:
        """Get information about sample ranges."""
        return {
            '25': (0, 25, 25),
            '25-100': (25, 100, 75),
            '100-200': (100, 200, 100),
            '200-300': (200, 300, 100),
            '300-1037': (300, 1037, 737)
        }

    def _load_layer_performance_file(self, sample_range: str, layer: int) -> Optional[List[dict]]:
        """Load a single JSON file for a specific sample range and layer."""
        filename = f"R-ROME_results{sample_range}_layer{layer}.json"
        filepath = self.data_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        cases = [case for case in data if isinstance(case, dict) and not case.get('summary', False)]
        return cases

    def _extract_metric_value(self, case: dict, metric_path: List[str]) -> Optional[float]:
        """Extract a metric value from a case using a path."""
        obj = case
        for key in metric_path:
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]

        if isinstance(obj, list) and len(obj) > 0:
            return obj[0]
        elif isinstance(obj, (int, float)):
            return obj
        return None

    def load_data(self, verbose: bool = True) -> 'GoldenLayerAnalyzer':
        """Load all layer performance data into tensors."""
        metric_definitions = {
            'pre_rewrite': ['pre', 'rewrite_acc'],
            'pre_rephrase': ['pre', 'rephrase_acc'],
            'pre_one_hop': ['pre', 'portability', 'one_hop_acc'],
            'post_rewrite': ['post', 'rewrite_acc'],
            'post_rephrase': ['post', 'rephrase_acc'],
            'post_neighborhood': ['post', 'locality', 'neighborhood_acc'],
            'post_one_hop': ['post', 'portability', 'one_hop_acc'],
        }

        for metric_name in metric_definitions.keys():
            self.results[metric_name] = torch.full((self.n_samples, self.n_layers), float('nan'))

        sample_range_info = self._get_sample_range_info()

        for sample_range, (start_idx, end_idx, count) in sample_range_info.items():
            if verbose:
                print(f"Loading sample range {sample_range}: samples {start_idx}-{end_idx-1}...")

            for layer in range(self.n_layers):
                cases = self._load_layer_performance_file(sample_range, layer)

                if cases is None:
                    if verbose:
                        print(f"  Warning: No data for {sample_range}, layer {layer}")
                    continue

                for case in cases:
                    case_id = case.get('case_id')
                    if case_id is None:
                        continue

                    sample_idx = start_idx + case_id

                    if sample_idx >= self.n_samples:
                        continue

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
        """Compute the overall metric using min-max normalization."""
        key_metrics = ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop']
        normalized_metrics = {}

        for metric_name in key_metrics:
            tensor = self.results[metric_name].numpy()
            min_vals = np.nanmin(tensor, axis=1, keepdims=True)
            max_vals = np.nanmax(tensor, axis=1, keepdims=True)
            denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
            normalized = (tensor - min_vals) / denom
            normalized_metrics[metric_name] = normalized
            if verbose:
                print(f"Min-max normalized {metric_name}")

        stacked = np.stack([
            normalized_metrics['post_rewrite'],
            normalized_metrics['post_rephrase'],
            normalized_metrics['post_neighborhood'],
            normalized_metrics['post_one_hop']
        ], axis=2)

        overall_tensor = np.nanmean(stacked, axis=2)
        self.results['post_overall'] = torch.from_numpy(overall_tensor).float()

        if verbose:
            print(f"\nComputed post_overall metric: {overall_tensor.shape}")

        return self

    def split_dataset(self, proxy_size: int = 100, random_seed: int = 42,
                     verbose: bool = True) -> 'GoldenLayerAnalyzer':
        """Split dataset into proxy and test sets."""
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
        """Compute average performance for proxy and test sets."""
        if self.proxy_indices is None or self.test_indices is None:
            raise ValueError("Must call split_dataset() before compute_proxy_test_means()")

        for metric_key, metric_name in self.metrics_info:
            proxy_data = self.results[metric_key][self.proxy_indices, :]
            self.proxy_means[metric_key] = torch.nanmean(proxy_data, dim=0)

            test_data = self.results[metric_key][self.test_indices, :]
            self.test_means[metric_key] = torch.nanmean(test_data, dim=0)

            if verbose:
                print(f"{metric_name:20s}: Computed proxy and test means")

        return self

    def analyze(self, verbose: bool = True) -> Dict:
        """Perform complete analysis including best layer identification and correlations."""
        if not self.proxy_means or not self.test_means:
            raise ValueError("Must call compute_proxy_test_means() before analyze()")

        for metric_key, metric_name in self.metrics_info:
            proxy_layer_means = self.proxy_means[metric_key].numpy()
            test_layer_means = self.test_means[metric_key].numpy()

            valid_mask = ~(np.isnan(proxy_layer_means) | np.isnan(test_layer_means))
            if valid_mask.sum() > 0:
                correlation = np.corrcoef(proxy_layer_means[valid_mask], test_layer_means[valid_mask])[0, 1]
                _, p_value = stats.pearsonr(proxy_layer_means[valid_mask], test_layer_means[valid_mask])
            else:
                correlation = np.nan
                p_value = np.nan

            self.correlations_dict[metric_key] = correlation

            valid_proxy = ~np.isnan(proxy_layer_means)
            if valid_proxy.any():
                proxy_max_value = np.nanmax(proxy_layer_means)
                proxy_best_layers = np.where(proxy_layer_means == proxy_max_value)[0].tolist()
            else:
                proxy_best_layers = []
                proxy_max_value = np.nan

            valid_test = ~np.isnan(test_layer_means)
            if valid_test.any():
                test_max_value = np.nanmax(test_layer_means)
                test_best_layers = np.where(test_layer_means == test_max_value)[0].tolist()
            else:
                test_best_layers = []
                test_max_value = np.nan

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


# =============================================================================
# GoldenLayerAnalyzerWikiBio - For WikiBio dataset
# =============================================================================

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

        self.metrics_info = [
            ('post_rewrite_acc', 'Rewrite Accuracy'),
            ('post_locality_acc', 'Locality'),
            ('post_fluency', 'Fluency'),
            ('post_overall', 'Overall')
        ]

    def load_data(self, verbose=True):
        """Load WikiBio results from nested JSON files."""
        all_data = defaultdict(list)

        for layer in range(self.n_layers):
            layer_samples = []

            for json_file in sorted(self.data_dir.glob(f"*_layer{layer}.json")):
                with open(json_file, 'r') as f:
                    samples = json.load(f)

                if len(samples) > 0:
                    samples = samples[:-1]

                for sample in samples:
                    if 'post' not in sample:
                        continue

                    post = sample['post']
                    sample_data = {}

                    if 'rewrite_acc' in post and isinstance(post['rewrite_acc'], list):
                        sample_data['rewrite_acc'] = post['rewrite_acc'][0]
                    else:
                        sample_data['rewrite_acc'] = np.nan

                    if 'locality' in post and 'Relation_Specificity_acc' in post['locality']:
                        locality_scores = post['locality']['Relation_Specificity_acc']
                        if isinstance(locality_scores, list) and len(locality_scores) > 0:
                            sample_data['locality_acc'] = np.mean(locality_scores)
                        else:
                            sample_data['locality_acc'] = np.nan
                    else:
                        sample_data['locality_acc'] = np.nan

                    if 'fluency' in post and 'ngram_entropy' in post['fluency']:
                        sample_data['fluency'] = post['fluency']['ngram_entropy']
                    else:
                        sample_data['fluency'] = np.nan

                    layer_samples.append(sample_data)

            all_data[layer] = layer_samples

        self.n_samples = max(len(samples) for samples in all_data.values()) if all_data else 0
        metrics = ['rewrite_acc', 'locality_acc', 'fluency']

        for metric in metrics:
            tensor = torch.full((self.n_samples, self.n_layers), float('nan'))

            for layer in range(self.n_layers):
                layer_samples = all_data[layer]
                for sample_idx, sample in enumerate(layer_samples):
                    if sample_idx < self.n_samples:
                        tensor[sample_idx, layer] = sample.get(metric, np.nan)

            self.results[f'post_{metric}'] = tensor

        if verbose:
            print(f"Loaded WikiBio data:")
            print(f"  Samples: {self.n_samples}")
            print(f"  Layers: {self.n_layers}")

        return self

    def compute_overall_metric(self, verbose=True):
        """Compute overall metric with PER-SAMPLE normalization and fluency negation."""
        fluency_negated = -self.results['post_fluency']
        normalized = {}

        for metric_name, values in [
            ('rewrite_acc', self.results['post_rewrite_acc']),
            ('locality_acc', self.results['post_locality_acc']),
            ('fluency_negated', fluency_negated)
        ]:
            tensor = values.numpy()
            min_vals = np.nanmin(tensor, axis=1, keepdims=True)
            max_vals = np.nanmax(tensor, axis=1, keepdims=True)
            denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
            normalized_tensor = (tensor - min_vals) / denom
            normalized[metric_name] = normalized_tensor

        stacked = np.stack([
            normalized['rewrite_acc'],
            normalized['locality_acc'],
            normalized['fluency_negated']
        ], axis=2)

        overall_tensor = np.nanmean(stacked, axis=2)
        self.results['post_overall'] = torch.from_numpy(overall_tensor).float()

        if verbose:
            print(f"Computed post_overall metric")

        return self

    def split_dataset(self, proxy_size=30, random_seed=42, verbose=True):
        """Split into proxy and test sets"""
        torch.manual_seed(random_seed)
        indices = torch.randperm(self.n_samples)

        self.proxy_indices = indices[:proxy_size]
        self.test_indices = indices[proxy_size:]

        if verbose:
            print(f"\nDataset split: {len(self.proxy_indices)} proxy, {len(self.test_indices)} test")

        return self

    def compute_proxy_test_means(self, verbose=True):
        """Compute mean performance for proxy and test sets"""
        for metric_key, _ in self.metrics_info:
            if metric_key in self.results:
                self.proxy_means[metric_key] = self.results[metric_key][self.proxy_indices].nanmean(dim=0)
                self.test_means[metric_key] = self.results[metric_key][self.test_indices].nanmean(dim=0)

        if verbose:
            print("Computed proxy and test means")

        return self

    def analyze(self, verbose=True):
        """Find best layers and compute correlations"""
        self.best_layers_comparison = {}

        for metric_key, metric_name in self.metrics_info:
            if metric_key not in self.proxy_means or metric_key not in self.test_means:
                continue

            proxy_mean = self.proxy_means[metric_key]
            test_mean = self.test_means[metric_key]

            valid_proxy = ~torch.isnan(proxy_mean)
            valid_test = ~torch.isnan(test_mean)

            if metric_key == 'post_fluency':
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

        return self


# =============================================================================
# GoldenLayerAnalyzerWikiRecent - For WikiRecent and WikiCounterfact datasets
# =============================================================================

class GoldenLayerAnalyzerWikiRecent:
    """Analyzer for WikiRecent/WikiCounterfact dataset with proper metric handling"""

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

        self.metrics_info = [
            ('post_rewrite_acc', 'Rewrite Accuracy'),
            ('post_locality_acc', 'Locality'),
            ('post_portability_acc', 'Portability'),
            ('post_fluency', 'Fluency'),
            ('overall', 'Overall')
        ]

    def load_data(self, verbose=True):
        """Load WikiRecent results from nested JSON files."""
        all_data = defaultdict(list)

        for layer in range(self.n_layers):
            layer_samples = []

            for json_file in sorted(self.data_dir.glob(f"*_layer{layer}.json")):
                with open(json_file, 'r') as f:
                    samples = json.load(f)

                if len(samples) > 0:
                    samples = samples[:-1]

                for sample in samples:
                    if 'post' not in sample:
                        continue

                    post = sample['post']
                    sample_data = {}

                    if 'rewrite_acc' in post and isinstance(post['rewrite_acc'], list):
                        sample_data['rewrite_acc'] = post['rewrite_acc'][0]
                    else:
                        sample_data['rewrite_acc'] = np.nan

                    if 'locality' in post:
                        case_list = []
                        for key in post['locality'].keys():
                            if isinstance(post['locality'][key], list) and len(post['locality'][key]) > 0:
                                case_list.append(sum(post['locality'][key]) / len(post['locality'][key]) * 100)
                        if len(case_list) != 0:
                            sample_data['locality_acc'] = np.mean(case_list) / 100
                        else:
                            sample_data['locality_acc'] = np.nan
                    else:
                        sample_data['locality_acc'] = np.nan

                    if 'portability' in post:
                        case_list = []
                        for key in post['portability'].keys():
                            if isinstance(post['portability'][key], list) and len(post['portability'][key]) > 0:
                                case_list.append(sum(post['portability'][key]) / len(post['portability'][key]) * 100)
                        if len(case_list) != 0:
                            sample_data['portability_acc'] = np.mean(case_list) / 100
                        else:
                            sample_data['portability_acc'] = np.nan
                    else:
                        sample_data['portability_acc'] = np.nan

                    if 'fluency' in post and 'ngram_entropy' in post['fluency']:
                        sample_data['fluency'] = post['fluency']['ngram_entropy']
                    else:
                        sample_data['fluency'] = np.nan

                    layer_samples.append(sample_data)

            all_data[layer] = layer_samples

        self.n_samples = max(len(samples) for samples in all_data.values()) if all_data else 0
        metrics = ['rewrite_acc', 'locality_acc', 'portability_acc', 'fluency']

        for metric in metrics:
            tensor = torch.full((self.n_samples, self.n_layers), float('nan'))

            for layer in range(self.n_layers):
                layer_samples = all_data[layer]
                for sample_idx, sample in enumerate(layer_samples):
                    if sample_idx < self.n_samples:
                        tensor[sample_idx, layer] = sample.get(metric, np.nan)

            self.results[f'post_{metric}'] = tensor

        if verbose:
            print(f"Loaded WikiRecent data:")
            print(f"  Samples: {self.n_samples}")
            print(f"  Layers: {self.n_layers}")

        return self

    def compute_overall_metric(self, verbose=True):
        """Compute overall metric with PER-SAMPLE normalization."""
        fluency_negated = -self.results['post_fluency']
        normalized = {}

        for metric_name, values in [
            ('rewrite_acc', self.results['post_rewrite_acc']),
            ('locality_acc', self.results['post_locality_acc']),
            ('portability_acc', self.results['post_portability_acc']),
            ('fluency_negated', fluency_negated)
        ]:
            tensor = values.numpy()
            min_vals = np.nanmin(tensor, axis=1, keepdims=True)
            max_vals = np.nanmax(tensor, axis=1, keepdims=True)
            denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
            normalized_tensor = (tensor - min_vals) / denom
            normalized[metric_name] = normalized_tensor

        stacked = np.stack([
            normalized['rewrite_acc'],
            normalized['locality_acc'],
            normalized['portability_acc'],
            normalized['fluency_negated']
        ], axis=2)

        overall_tensor = np.nanmean(stacked, axis=2)
        self.results['overall'] = torch.from_numpy(overall_tensor).float()

        if verbose:
            print(f"Computed overall metric")

        return self

    def split_dataset(self, proxy_size=120, random_seed=42, verbose=True):
        """Split into proxy and test sets"""
        torch.manual_seed(random_seed)
        indices = torch.randperm(self.n_samples)

        self.proxy_indices = indices[:proxy_size]
        self.test_indices = indices[proxy_size:]

        if verbose:
            print(f"\nDataset split: {len(self.proxy_indices)} proxy, {len(self.test_indices)} test")

        return self

    def compute_proxy_test_means(self, verbose=True):
        """Compute mean performance for proxy and test sets"""
        for metric_key, _ in self.metrics_info:
            if metric_key in self.results:
                self.proxy_means[metric_key] = self.results[metric_key][self.proxy_indices].nanmean(dim=0)
                self.test_means[metric_key] = self.results[metric_key][self.test_indices].nanmean(dim=0)

        if verbose:
            print("Computed proxy and test means")

        return self

    def analyze(self, verbose=True):
        """Find best layers and compute correlations"""
        self.best_layers_comparison = {}

        for metric_key, metric_name in self.metrics_info:
            if metric_key not in self.proxy_means or metric_key not in self.test_means:
                continue

            proxy_mean = self.proxy_means[metric_key]
            test_mean = self.test_means[metric_key]

            valid_proxy = ~torch.isnan(proxy_mean)
            valid_test = ~torch.isnan(test_mean)

            if metric_key == 'post_fluency':
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

        return self
