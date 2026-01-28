#!/usr/bin/env python3
"""
Dataset-Model Configuration for Cross-Dataset Layer Independence Analysis

This file defines the configuration for all dataset-model combinations.
Update the DATA_ROOT path to point to your data directory.
"""

from pathlib import Path
from typing import Dict, Tuple

# =============================================================================
# CONFIGURE THIS: Set the root path to your data directory
# =============================================================================
# Option 1: Relative path (if data is in a subdirectory)
DATA_ROOT = Path(__file__).parent / 'data'

# Option 2: Absolute path (uncomment and modify)
# DATA_ROOT = Path('/path/to/your/data/directory')

# =============================================================================
# Dataset-Model Registry
# =============================================================================
# Each entry defines:
#   - analyzer_class: Which analyzer to use ('default', 'wikibio', 'wikirecent')
#   - data_subdir: Subdirectory under DATA_ROOT containing the JSON files
#   - n_samples: Expected number of samples
#   - n_layers: Number of layers in the model
#   - proxy_size: Size of proxy set for analysis
#   - metrics: List of metric keys available for this dataset

DATASET_MODEL_REGISTRY: Dict[Tuple[str, str], Dict] = {
    # ==== ZSRE Dataset (proxy_size=100) ====
    ('zsre', 'gpt2-xl'): {
        'analyzer_class': 'default',
        'data_subdir': 'zsre_gpt2xl',
        'n_samples': 1037,
        'n_layers': 48,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('zsre', 'llama2-7b'): {
        'analyzer_class': 'default',
        'data_subdir': 'zsre_llama2',
        'n_samples': 1037,
        'n_layers': 32,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('zsre', 'gemma3-12b'): {
        'analyzer_class': 'default',
        'data_subdir': 'zsre_gemma3',
        'n_samples': 1037,
        'n_layers': 48,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('zsre', 'qwen3-32b'): {
        'analyzer_class': 'default',
        'data_subdir': 'zsre_qwen3',
        'n_samples': 1037,
        'n_layers': 64,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },

    # ==== Counterfact Dataset (proxy_size=100) ====
    ('counterfact', 'gpt2-xl'): {
        'analyzer_class': 'default',
        'data_subdir': 'counterfact_gpt2xl',
        'n_samples': 1266,
        'n_layers': 48,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('counterfact', 'llama2-7b'): {
        'analyzer_class': 'default',
        'data_subdir': 'counterfact_llama2',
        'n_samples': 1266,
        'n_layers': 32,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('counterfact', 'gemma3-12b'): {
        'analyzer_class': 'default',
        'data_subdir': 'counterfact_gemma3',
        'n_samples': 1031,
        'n_layers': 48,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('counterfact', 'qwen3-32b'): {
        'analyzer_class': 'default',
        'data_subdir': 'counterfact_qwen3',
        'n_samples': 1031,
        'n_layers': 64,
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },

    # ==== WikiBio Dataset (proxy_size=30) ====
    ('wikibio', 'gpt2-xl'): {
        'analyzer_class': 'wikibio',
        'data_subdir': 'wikibio_gpt2xl',
        'n_samples': 306,
        'n_layers': 48,
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_fluency', 'post_overall']
    },
    ('wikibio', 'llama2-7b'): {
        'analyzer_class': 'wikibio',
        'data_subdir': 'wikibio_llama2',
        'n_samples': 306,
        'n_layers': 32,
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_fluency', 'post_overall']
    },
    ('wikibio', 'gemma3-12b'): {
        'analyzer_class': 'wikibio',
        'data_subdir': 'wikibio_gemma3',
        'n_samples': 306,
        'n_layers': 48,
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_fluency', 'post_overall']
    },
    ('wikibio', 'qwen3-32b'): {
        'analyzer_class': 'wikibio',
        'data_subdir': 'wikibio_qwen3',
        'n_samples': 306,
        'n_layers': 64,
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_fluency', 'post_overall']
    },

    # ==== WikiRecent Dataset (proxy_size=120) ====
    ('wikirecent', 'gpt2-xl'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikirecent_gpt2xl',
        'n_samples': 1266,
        'n_layers': 48,
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },
    ('wikirecent', 'llama2-7b'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikirecent_llama2',
        'n_samples': 1266,
        'n_layers': 32,
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },
    ('wikirecent', 'gemma3-12b'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikirecent_gemma3',
        'n_samples': 1266,
        'n_layers': 48,
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },
    ('wikirecent', 'qwen3-32b'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikirecent_qwen3',
        'n_samples': 1266,
        'n_layers': 64,
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },

    # ==== WikiCounterfact Dataset (proxy_size=80) ====
    ('wikicounterfact', 'gpt2-xl'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikicounterfact_gpt2xl',
        'n_samples': 839,
        'n_layers': 48,
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },
    ('wikicounterfact', 'llama2-7b'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikicounterfact_llama2',
        'n_samples': 839,
        'n_layers': 32,
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },
    ('wikicounterfact', 'gemma3-12b'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikicounterfact_gemma3',
        'n_samples': 839,
        'n_layers': 48,
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },
    ('wikicounterfact', 'qwen3-32b'): {
        'analyzer_class': 'wikirecent',
        'data_subdir': 'wikicounterfact_qwen3',
        'n_samples': 839,
        'n_layers': 64,
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'overall']
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_config(dataset: str, model: str) -> Dict:
    """Get configuration for a specific dataset-model pair."""
    key = (dataset, model)
    if key not in DATASET_MODEL_REGISTRY:
        raise KeyError(f"Dataset-model pair {key} not found in registry")

    config = DATASET_MODEL_REGISTRY[key].copy()
    config['data_dir'] = DATA_ROOT / config['data_subdir']
    return config


def get_datasets() -> list:
    """Get list of unique dataset names."""
    return sorted(list(set([k[0] for k in DATASET_MODEL_REGISTRY.keys()])))


def get_models() -> list:
    """Get list of unique model names."""
    return sorted(list(set([k[1] for k in DATASET_MODEL_REGISTRY.keys()])))


def get_available_pairs() -> list:
    """Get all available (dataset, model) pairs."""
    return sorted(list(DATASET_MODEL_REGISTRY.keys()))


def print_registry_summary():
    """Print a summary of the dataset-model registry."""
    print("=" * 80)
    print("Dataset-Model Registry Summary")
    print("=" * 80)

    print(f"\nDATA_ROOT: {DATA_ROOT}")
    print(f"\nAvailable Datasets: {get_datasets()}")
    print(f"Available Models: {get_models()}")
    print(f"\nTotal pairs: {len(DATASET_MODEL_REGISTRY)}")

    print("\nDataset-Model Matrix:")
    datasets = get_datasets()
    models = get_models()

    header = f"{'Dataset':<20} " + " ".join([f"{m:<12}" for m in models])
    print(header)
    print("-" * len(header))

    for dataset in datasets:
        row = f"{dataset:<20} "
        for model in models:
            if (dataset, model) in DATASET_MODEL_REGISTRY:
                config = DATASET_MODEL_REGISTRY[(dataset, model)]
                row += f"L={config['n_layers']:<8} "
            else:
                row += f"{'--':<12} "
        print(row)


if __name__ == "__main__":
    print_registry_summary()
