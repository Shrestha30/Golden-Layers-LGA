#!/usr/bin/env python3
"""
Dataset-Model Registry for Cross-Dataset Layer Independence Analysis

UPDATED: Added proxy_size and metrics for each dataset-model pair
- zsre: proxy_size=100
- counterfact: proxy_size=100
- wikibio: proxy_size=30
- wikirecent: proxy_size=120
- wikicounterfact: proxy_size=80
"""

from pathlib import Path
from typing import Dict, Tuple

# Base directory - points to LGA folder
BASE_DIR = Path(__file__).parent.parent / 'LGA'


# Dataset-Model Registry with proxy sizes and metrics
DATASET_MODEL_REGISTRY: Dict[Tuple[str, str], Dict] = {
    # ==== ZSRE Dataset (proxy_size=100) ====
    ('zsre', 'gpt2-xl'): {
        'api_path': BASE_DIR / 'sampleLayerRelationship' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationship',
        'n_samples': 1037,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('zsre', 'llama2-7b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipZSRELlama2' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipZSRELlama2',
        'n_samples': 1037,
        'n_layers': 32,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'meta-llama/Llama-2-7b-hf',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('zsre', 'gemma3-12b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipZSREGemma31' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipZSREGemma31',
        'n_samples': 1037,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'google/gemma-3-12b-pt',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('zsre', 'qwen3-32b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipZSREQwen31' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipZSREQwen31',
        'n_samples': 1037,
        'n_layers': 64,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'Qwen/Qwen3-32B',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },

    # ==== Counterfact Dataset (proxy_size=100) ====
    ('counterfact', 'gpt2-xl'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipCounterfact' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipCounterfact',
        'n_samples': 1266,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('counterfact', 'llama2-7b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipCounterfactLlama2' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipCounterfactLlama2',
        'n_samples': 1266,
        'n_layers': 32,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'meta-llama/Llama-2-7b-hf',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('counterfact', 'gemma3-12b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipCounterfactGemma31' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipCounterfactGemma31',
        'n_samples': 1031,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'google/gemma-3-12b-pt',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },
    ('counterfact', 'qwen3-32b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipCounterfactQwen31' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipCounterfactQwen31',
        'n_samples': 1031,
        'n_layers': 64,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'Qwen/Qwen3-32B',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall']
    },

    # ==== WikiBio Dataset (proxy_size=30) ====
    ('wikibio', 'gpt2-xl'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiBio' / 'golden_layer' / 'golden_layer_api_wikibio.py',
        'api_class': 'GoldenLayerAnalyzerWikiBio',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiBio',
        'n_samples': 306,  # Actual sample count
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'fluency', 'post_overall']  # WikiBio metrics
    },
    ('wikibio', 'llama2-7b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiBioLlama2' / 'golden_layer' / 'golden_layer_api_wikibio.py',
        'api_class': 'GoldenLayerAnalyzerWikiBio',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiBioLlama2',
        'n_samples': 306,
        'n_layers': 32,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'meta-llama/Llama-2-7b-hf',
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_fluency', 'post_overall']
    },
    ('wikibio', 'qwen3-32b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiBioQwen31' / 'golden_layer' / 'golden_layer_api_wikibio.py',
        'api_class': 'GoldenLayerAnalyzerWikiBio',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiBioQwen31',
        'n_samples': 306,
        'n_layers': 64,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'Qwen/Qwen3-32B',
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_fluency', 'post_overall']
    },
    ('wikibio', 'gemma3-12b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiBioGemma31' / 'golden_layer' / 'golden_layer_api_wikibio.py',
        'api_class': 'GoldenLayerAnalyzerWikiBio',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiBioGemma31',
        'n_samples': 306,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'google/gemma-3-12b-pt',
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_fluency', 'post_overall']
    },

    # ==== WikiCounterfact Dataset (proxy_size=80) ====
    ('wikicounterfact', 'gpt2-xl'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiCounterfact' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiCounterfact',
        'n_samples': 839,  # Actual sample count
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall']
    },
    ('wikicounterfact', 'llama2-7b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactLlama2' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactLlama2',
        'n_samples': 839,
        'n_layers': 32,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'meta-llama/Llama-2-7b-hf',
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall']
    },
    ('wikicounterfact', 'gemma3-12b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactGemma31' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactGemma31',
        'n_samples': 839,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'google/gemma-3-12b-pt',
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall']
    },

    # ==== WikiRecent Dataset (proxy_size=120) ====
    ('wikirecent', 'gpt2-xl'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiRecent' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiRecent',
        'n_samples': 1266,  # Actual sample count
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'fluency', 'post_overall']  # WikiRecent metrics
    },
    ('wikirecent', 'llama2-7b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiRecentLlama2' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiRecentLlama2',
        'n_samples': 1266,
        'n_layers': 32,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'meta-llama/Llama-2-7b-hf',
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall']
    },
    ('wikirecent', 'qwen3-32b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiRecentQwen31' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiRecentQwen31',
        'n_samples': 1266,
        'n_layers': 64,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'Qwen/Qwen3-32B',
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall']
    },
    ('wikicounterfact', 'qwen3-32b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactQwen31' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactQwen31',
        'n_samples': 839,
        'n_layers': 64,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'Qwen/Qwen3-32B',
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall']
    },
    ('wikirecent', 'gemma3-12b'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiRecentGemma31' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiRecentGemma31',
        'n_samples': 1266,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'google/gemma-3-12b-pt',
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall']
    },

    # ==== EMMET (gpt2-xl model with EMMET editing method) ====
    # NOTE: EMMET uses gpt2-xl influence scores, only performance differs
    ('zsre', 'emmet'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipZSREEMMET' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipZSREEMMET',
        'n_samples': 1037,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall'],
        'influence_model': 'gpt2-xl',  # Use gpt2-xl influence scores
    },
    ('counterfact', 'emmet'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipCounterfactEMMET' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipCounterfactEMMET',
        'n_samples': 1266,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },
    ('wikibio', 'emmet'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiBioEMMET' / 'golden_layer' / 'golden_layer_api_wikibio.py',
        'api_class': 'GoldenLayerAnalyzerWikiBio',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiBioEMMET',
        'n_samples': 306,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'fluency', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },
    ('wikicounterfact', 'emmet'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactEMMET' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactEMMET',
        'n_samples': 839,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },
    ('wikirecent', 'emmet'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiRecentEMMET' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiRecentEMMET',
        'n_samples': 1266,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'fluency', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },

    # ==== EMMET (gpt2-xl model with EMMET editing method) ====
    # NOTE: EMMET uses gpt2-xl influence scores, only performance differs
    ('zsre', 'rome'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipZSREROME' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipZSREROME',
        'n_samples': 1037,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall'],
        'influence_model': 'gpt2-xl',  # Use gpt2-xl influence scores
    },
    ('counterfact', 'rome'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipCounterfactROME' / 'golden_layer' / 'golden_layer_api.py',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipCounterfactROME',
        'n_samples': 1266,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 100,
        'metrics': ['post_rewrite', 'post_rephrase', 'post_neighborhood', 'post_one_hop', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },
    ('wikibio', 'rome'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiBioROME' / 'golden_layer' / 'golden_layer_api_wikibio.py',
        'api_class': 'GoldenLayerAnalyzerWikiBio',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiBioROME',
        'n_samples': 306,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 30,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'fluency', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },
    ('wikicounterfact', 'rome'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactROME' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiCounterfactROME',
        'n_samples': 839,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 80,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'post_portability_acc', 'post_fluency', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },
    ('wikirecent', 'rome'): {
        'api_path': BASE_DIR / 'sampleLayerRelationshipWikiRecentROME' / 'golden_layer' / 'golden_layer_api_wikirecent.py',
        'api_class': 'GoldenLayerAnalyzerWikiRecent',
        'base_dir': BASE_DIR / 'sampleLayerRelationshipWikiRecentROME',
        'n_samples': 1266,
        'n_layers': 48,
        'data_dir': 'allLayerPerformance',
        'model_full_name': 'gpt2-xl',
        'proxy_size': 120,
        'metrics': ['post_rewrite_acc', 'post_locality_acc', 'fluency', 'post_overall'],
        'influence_model': 'gpt2-xl',
    },
}


def get_datasets() -> list:
    """Get list of unique dataset names."""
    return sorted(list(set([k[0] for k in DATASET_MODEL_REGISTRY.keys()])))


def get_models() -> list:
    """Get list of unique model names."""
    return sorted(list(set([k[1] for k in DATASET_MODEL_REGISTRY.keys()])))


def get_dataset_model_pairs() -> list:
    """Get all (dataset, model) pairs."""
    return sorted(list(DATASET_MODEL_REGISTRY.keys()))


def get_config(dataset: str, model: str) -> Dict:
    """
    Get configuration for a specific dataset-model pair.

    Args:
        dataset: Dataset name (e.g., 'zsre', 'counterfact')
        model: Model name (e.g., 'gpt2-xl', 'llama2-7b')

    Returns:
        Configuration dictionary

    Raises:
        KeyError: If dataset-model pair not found
    """
    key = (dataset, model)
    if key not in DATASET_MODEL_REGISTRY:
        raise KeyError(f"Dataset-model pair {key} not found in registry")
    return DATASET_MODEL_REGISTRY[key]


def get_common_models(datasets: list) -> list:
    """
    Get models that are available for all specified datasets.

    Args:
        datasets: List of dataset names

    Returns:
        List of model names available for all datasets
    """
    if not datasets:
        return []

    # Get models for first dataset
    common = set([k[1] for k in DATASET_MODEL_REGISTRY.keys() if k[0] == datasets[0]])

    # Intersect with models from other datasets
    for dataset in datasets[1:]:
        dataset_models = set([k[1] for k in DATASET_MODEL_REGISTRY.keys() if k[0] == dataset])
        common = common.intersection(dataset_models)

    return sorted(list(common))


def get_common_metrics(datasets: list, model: str) -> list:
    """
    Get metrics that are available for all specified dataset-model pairs.

    Args:
        datasets: List of dataset names
        model: Model name

    Returns:
        List of metric names available for all datasets with this model
    """
    if not datasets:
        return []

    # Get metrics for first dataset
    try:
        config = get_config(datasets[0], model)
        common = set(config.get('metrics', []))
    except KeyError:
        return []

    # Intersect with metrics from other datasets
    for dataset in datasets[1:]:
        try:
            config = get_config(dataset, model)
            dataset_metrics = set(config.get('metrics', []))
            common = common.intersection(dataset_metrics)
        except KeyError:
            return []

    return sorted(list(common))


def print_registry_summary():
    """Print a summary of the dataset-model registry."""
    print("=" * 100)
    print("Dataset-Model Registry Summary (UPDATED with proxy_size)")
    print("=" * 100)

    datasets = get_datasets()
    models = get_models()

    print(f"\nAvailable Datasets: {len(datasets)}")
    for dataset in datasets:
        # Get proxy size (assume same across models for a dataset)
        sample_config = None
        for model in models:
            if (dataset, model) in DATASET_MODEL_REGISTRY:
                sample_config = DATASET_MODEL_REGISTRY[(dataset, model)]
                break
        proxy_size = sample_config.get('proxy_size', '?') if sample_config else '?'
        print(f"  - {dataset:<20} (proxy_size={proxy_size})")

    print(f"\nAvailable Models: {len(models)}")
    for model in models:
        print(f"  - {model}")

    print(f"\nTotal Dataset-Model Pairs: {len(DATASET_MODEL_REGISTRY)}")

    # Show matrix
    print("\nDataset-Model Matrix:")
    header = f"{'Dataset':<20} " + " ".join([f"{m:<15}" for m in models])
    print(header)
    print("-" * len(header))

    for dataset in datasets:
        row = f"{dataset:<20} "
        for model in models:
            if (dataset, model) in DATASET_MODEL_REGISTRY:
                config = DATASET_MODEL_REGISTRY[(dataset, model)]
                n_layers = config['n_layers']
                row += f"{'✓ (L=' + str(n_layers) + ')':<15} "
            else:
                row += f"{'✗':<15} "
        print(row)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    print_registry_summary()

    # Test getting common models
    print("\nCommon models for ['zsre', 'counterfact']:")
    common = get_common_models(['zsre', 'counterfact'])
    print(f"  {common}")

    # Test getting common metrics
    print("\nCommon metrics for ['zsre', 'counterfact'] with gpt2-xl:")
    common_metrics = get_common_metrics(['zsre', 'counterfact'], 'gpt2-xl')
    print(f"  {common_metrics}")

    # Test proxy sizes
    print("\nProxy sizes:")
    for dataset in get_datasets():
        for model in get_models():
            if (dataset, model) in DATASET_MODEL_REGISTRY:
                config = get_config(dataset, model)
                print(f"  {dataset:<20} {model:<15}: proxy_size={config.get('proxy_size', 'N/A')}")
                break  # Just show one model per dataset
