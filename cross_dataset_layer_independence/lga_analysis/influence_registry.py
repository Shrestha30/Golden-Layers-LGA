#!/usr/bin/env python3
"""
Influence Data Registry

Contains paths to influence score files for different model/dataset combinations.

!!! UPDATE THESE PATHS FOR YOUR SYSTEM !!!

Each entry maps (model, dataset) -> dict with:
    - base_dir: Path to directory containing influence files
    - pattern: Glob pattern for finding influence files
    - n_samples: Expected number of samples
"""

from pathlib import Path


# =============================================================================
# Base Directories - Local Setup
# =============================================================================

# Compute project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LGA_SCORES_DIR = PROJECT_ROOT / 'LGA' / 'lga_scores'

# Base directories for influence files
INFLUENCE_BASE_DIRS = {
    'gpt2-xl': LGA_SCORES_DIR,
    'llama2-7b': LGA_SCORES_DIR,
    'qwen3-32b': LGA_SCORES_DIR,
    'gemma3-12b': LGA_SCORES_DIR,
}


# =============================================================================
# Influence File Registry
# =============================================================================

INFLUENCE_REGISTRY = {
    # ==== GPT2-XL ====
    ('gpt2-xl', 'zsre'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1037.pt',
        'n_samples': 1037,
    },
    ('gpt2-xl', 'counterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1031.pt',
        'n_samples': 1031,
    },
    ('gpt2-xl', 'wikibio'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-306.pt',
        'n_samples': 306,
    },
    ('gpt2-xl', 'wikirecent'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1266.pt',
        'n_samples': 1266,
    },
    ('gpt2-xl', 'wikicounterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-839.pt',
        'n_samples': 839,
    },

    # ==== LLaMA2-7B ====
    ('llama2-7b', 'zsre'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('llama2-7b'),
        'pattern': 'Llama-2-7b*_nsamples-1037.pt',
        'n_samples': 1037,
    },
    ('llama2-7b', 'counterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('llama2-7b'),
        'pattern': 'Llama-2-7b*_nsamples-1031.pt',
        'n_samples': 1031,
    },
    ('llama2-7b', 'wikibio'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('llama2-7b'),
        'pattern': 'Llama-2-7b*_nsamples-306.pt',
        'n_samples': 306,
    },
    ('llama2-7b', 'wikirecent'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('llama2-7b'),
        'pattern': 'Llama-2-7b*_nsamples-1266.pt',
        'n_samples': 1266,
    },
    ('llama2-7b', 'wikicounterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('llama2-7b'),
        'pattern': 'Llama-2-7b*_nsamples-839.pt',
        'n_samples': 839,
    },

    # ==== Qwen3-32B ====
    ('qwen3-32b', 'zsre'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('qwen3-32b'),
        'pattern': 'Qwen3-32B*_nsamples-1037.pt',
        'n_samples': 1037,
    },
    ('qwen3-32b', 'counterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('qwen3-32b'),
        'pattern': 'Qwen3-32B_*_nsamples-1031.pt',
        'n_samples': 1031,
    },
    ('qwen3-32b', 'wikibio'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('qwen3-32b'),
        'pattern': 'Qwen3-32B_*_nsamples-306.pt',
        'n_samples': 306,
    },
    ('qwen3-32b', 'wikirecent'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('qwen3-32b'),
        'pattern': 'Qwen3-32B_*recent*_nsamples-1266.pt',
        'n_samples': 1266,
    },
    ('qwen3-32b', 'wikicounterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('qwen3-32b'),
        'pattern': 'Qwen3-32B_*counterfact*_nsamples-839.pt',
        'n_samples': 839,
    },

    # ==== Gemma3-12B ====
    ('gemma3-12b', 'zsre'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gemma3-12b'),
        'pattern': 'gemma-3-12b-pt*_nsamples-1037.pt',
        'n_samples': 1037,
    },
    ('gemma3-12b', 'counterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gemma3-12b'),
        'pattern': 'gemma-3-12b-pt*_nsamples-1031.pt',
        'n_samples': 1031,
    },
    ('gemma3-12b', 'wikibio'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gemma3-12b'),
        'pattern': 'gemma-3-12b-pt*_nsamples-306.pt',
        'n_samples': 306,
    },
    ('gemma3-12b', 'wikirecent'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gemma3-12b'),
        'pattern': 'gemma-3-12b-pt*_nsamples-1266.pt',
        'n_samples': 1266,
    },
    ('gemma3-12b', 'wikicounterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gemma3-12b'),
        'pattern': 'gemma-3-12b-pt*_nsamples-839.pt',
        'n_samples': 839,
    },

    # ==== EMMET (uses gpt2-xl influence scores - same model, different editing method) ====
    ('emmet', 'zsre'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1037.pt',
        'n_samples': 1037,
    },
    ('emmet', 'counterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1031.pt',
        'n_samples': 1031,
    },
    ('emmet', 'wikibio'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-306.pt',
        'n_samples': 306,
    },
    ('emmet', 'wikirecent'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1266.pt',
        'n_samples': 1266,
    },
    ('emmet', 'wikicounterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-839.pt',
        'n_samples': 839,
    },
    # ==== ROME (uses gpt2-xl influence scores - same model, different editing method) ====
    ('rome', 'zsre'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1037.pt',
        'n_samples': 1037,
    },
    ('rome', 'counterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1031.pt',
        'n_samples': 1031,
    },
    ('rome', 'wikibio'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-306.pt',
        'n_samples': 306,
    },
    ('rome', 'wikirecent'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-1266.pt',
        'n_samples': 1266,
    },
    ('rome', 'wikicounterfact'): {
        'base_dir': INFLUENCE_BASE_DIRS.get('gpt2-xl'),
        'pattern': 'modelName-gpt2-xl_*_nsamples-839.pt',
        'n_samples': 839,
    },
}


# =============================================================================
# Registry Access Functions
# =============================================================================

def get_influence_config(model, dataset):
    """
    Get influence file configuration for a model/dataset pair.

    Args:
        model: Model name (e.g., 'gpt2-xl', 'llama2-7b')
        dataset: Dataset name (e.g., 'zsre', 'counterfact')

    Returns:
        Config dict with base_dir, pattern, n_samples

    Raises:
        KeyError if model/dataset pair not found
    """
    key = (model, dataset)
    if key not in INFLUENCE_REGISTRY:
        available = list(INFLUENCE_REGISTRY.keys())
        raise KeyError(f"No influence config for {key}. Available: {available}")
    return INFLUENCE_REGISTRY[key]


def find_influence_files(model, dataset):
    """
    Find all influence score files for a model/dataset pair.

    Args:
        model: Model name
        dataset: Dataset name

    Returns:
        List of (filepath, influence_type) tuples
    """
    config = get_influence_config(model, dataset)
    base_dir = config['base_dir']
    pattern = config['pattern']

    if base_dir is None or not base_dir.exists():
        print(f"Warning: Base directory not found: {base_dir}")
        return []

    files = list(base_dir.glob(pattern))

    results = []
    for f in sorted(files):
        # Extract influence type from filename
        from influence_api import get_influence_type_from_filename
        inf_type = get_influence_type_from_filename(f.name)
        results.append((f, inf_type))

    return results


def list_available_pairs():
    """List all available (model, dataset) pairs."""
    return list(INFLUENCE_REGISTRY.keys())


def print_registry_summary():
    """Print summary of the influence registry."""
    print("=" * 80)
    print("Influence Data Registry")
    print("=" * 80)

    # Group by model
    models = sorted(set(k[0] for k in INFLUENCE_REGISTRY.keys()))

    for model in models:
        print(f"\n{model.upper()}:")
        datasets = [k[1] for k in INFLUENCE_REGISTRY.keys() if k[0] == model]
        for dataset in sorted(datasets):
            config = INFLUENCE_REGISTRY[(model, dataset)]
            base_dir = config['base_dir']
            exists = base_dir.exists() if base_dir else False
            status = "✓" if exists else "✗ (not found)"
            print(f"  {dataset}: {status}")
            if exists:
                files = list(base_dir.glob(config['pattern']))
                print(f"    Found {len(files)} influence files")


if __name__ == '__main__':
    print_registry_summary()
