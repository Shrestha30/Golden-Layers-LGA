#!/usr/bin/env python3
"""
Influence-based Layer Selection API

Provides reusable functions for:
1. Loading influence scores from various methods
2. Finding best layers based on influence scores
3. Evaluating performance at influence-selected layers
4. Generating comparison tables
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import heatmap_api
import dataset_model_registry


# =============================================================================
# Core Layer Selection Functions
# =============================================================================

def find_best_layers(influence_scores, influence_type=None, mode='min'):
    """
    Find the best layer(s) based on influence scores.

    Args:
        influence_scores: Tensor/array of shape (n_samples, n_layers)
        influence_type: String name of influence type (used to auto-detect mode)
        mode: 'max' or 'min' (overridden if influence_type contains 'CausalTraceProba')

    Returns:
        dict with:
            - best_layers: List of best layer indices (handles ties)
            - best_value: The best mean influence value
            - layer_means: Array of mean influence per layer
            - mode_used: The mode that was actually used
    """
    # Convert to numpy if needed
    if hasattr(influence_scores, 'numpy'):
        influence_scores = influence_scores.numpy()
    elif torch.is_tensor(influence_scores):
        influence_scores = influence_scores.cpu().numpy()

    # Average across samples
    layer_means = np.nanmean(influence_scores, axis=0)
    if "OverallNewOverallHat" in influence_type:
        print(f"influence Type:{influence_type} : {layer_means}")
        layer_count = 0
        for i in layer_means:
            print(f"Layer {layer_count}: {i}")
            layer_count+=1

    # if "OverallNewOverallHat" in influence_type:
    #     print("Layer means: ", layer_means)

    # Determine mode based on influence type
    if influence_type is not None and 'causaltrace' in influence_type.lower():
        mode_used = 'max'
    else:
        mode_used = mode

    # Find best value
    if mode_used == 'max':
        best_value = np.nanmax(layer_means)
    else:  # min
        best_value = np.nanmin(layer_means)

    # Find all layers with best value (handle ties)
    best_layers = np.where(layer_means == best_value)[0].tolist()

    if "OverallNewOverallHat" in influence_type:
        print(f"influence Type:{influence_type} : {layer_means}")
        layer_count = 0
        for i in layer_means:
            print(f"Layer {layer_count}: {i}")
            layer_count+=1

    return {
        'best_layers': best_layers,
        'best_value': float(best_value),
        'layer_means': layer_means,
        'mode_used': mode_used
    }

def find_best_layers_outlier(
    influence_scores: torch.Tensor,
    influence_type: str,
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1,
    verbose: bool = True,
    mode = "min"
):
    """
    Find best layer(s) while reporting removed layers.

    Returns:
        best_layers: List[int]
        diagnostics: dict
    """

    if not isinstance(influence_scores, torch.Tensor):
        influence_scores = torch.tensor(influence_scores)
    
    diagnostics = {
        "invalid_layers": [],
        "outlier_layers": [],
        "kept_layers": [],
        "outlier_bounds": None,
        "best_value": None,
    }

    # 1. Mean across samples
    layer_means = torch.nanmean(influence_scores, dim=0).cpu().numpy()
    # print("layer Means: ", layer_means)

    # 2. Remove invalid values
    valid_mask = np.isfinite(layer_means)
    invalid_indices = np.where(~valid_mask)[0].tolist()
    diagnostics["invalid_layers"] = invalid_indices

    if verbose and invalid_indices:
        print(f"Removed invalid layers (NaN/±inf): {invalid_indices}")

    valid_indices = np.where(valid_mask)[0]
    valid_means = layer_means[valid_mask]

    if valid_means.size == 0:
        raise ValueError("All layers invalid after NaN/inf removal.")

    # 3. Outlier removal
    if outlier_method == "iqr" and valid_means.size > 2 and ("CausalTraceProba" not in influence_type):
        q1 = np.percentile(valid_means, 25)
        q3 = np.percentile(valid_means, 75)
        iqr = q3 - q1

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        diagnostics["outlier_bounds"] = (lower, upper)

        inlier_mask = (valid_means >= lower) & (valid_means <= upper)

        outlier_indices = valid_indices[~inlier_mask].tolist()
        diagnostics["outlier_layers"] = outlier_indices

        if verbose and outlier_indices:
            print(
                f"Removed outlier layers (IQR): {outlier_indices} "
                f"[bounds: {lower:.4g}, {upper:.4g}]"
            )

        valid_indices = valid_indices[inlier_mask]
        valid_means = valid_means[inlier_mask]

        if valid_means.size == 0:
            raise ValueError("All layers removed as outliers.")

    diagnostics["kept_layers"] = valid_indices.tolist()

    # 4. Select best
    if ("CausalTraceProba" in influence_type) or (mode =="max"):
        best_value = np.max(valid_means)
        mode_used = 'max'
    else:
        best_value = np.min(valid_means)
        mode_used = 'min'

    diagnostics["best_value"] = float(best_value)

    best_layers = valid_indices[valid_means == best_value].tolist()

    if verbose:
        print(f"Selected best layers: {best_layers} (value={best_value:.4g})")
    
    return {
            'best_layers': best_layers,
            'best_value': float(best_value),
            'layer_means': layer_means,
            'mode_used': mode_used
        }


def find_best_layers_subset(influence_scores, sample_indices, influence_type=None, mode='min', outlier=False):
    """
    Find best layers using only a subset of samples (e.g., proxy set).

    Args:
        influence_scores: Tensor/array of shape (n_samples, n_layers)
        sample_indices: Indices of samples to use (e.g., proxy_indices)
        influence_type: String name of influence type
        mode: 'max' or 'min'

    Returns:
        Same as find_best_layers()
    """
    # Convert to numpy if needed
    if hasattr(influence_scores, 'numpy'):
        influence_scores = influence_scores.numpy()
    elif torch.is_tensor(influence_scores):
        influence_scores = influence_scores.cpu().numpy()

    # Convert indices if needed
    if hasattr(sample_indices, 'numpy'):
        sample_indices = sample_indices.numpy()
    elif torch.is_tensor(sample_indices):
        sample_indices = sample_indices.cpu().numpy()
    sample_indices = np.array(sample_indices)

    # Select subset
    subset_scores = influence_scores[sample_indices]

    if outlier==False:
        return find_best_layers(subset_scores, influence_type=influence_type, mode=mode)
    else:
        return find_best_layers_outlier(subset_scores, influence_type=influence_type, mode=mode)



# =============================================================================
# Performance Evaluation Functions
# =============================================================================

def evaluate_layer_performance(analyzer, layer_idx, use_test_set=True):
    """
    Evaluate performance at a specific layer for all metrics.

    Args:
        analyzer: Loaded analyzer with proxy_means and test_means
        layer_idx: Layer index to evaluate
        use_test_set: If True, use test_means; else use proxy_means

    Returns:
        dict mapping metric_name -> performance value
    """
    means = analyzer.test_means if use_test_set else analyzer.proxy_means

    performance = {}
    for metric_key, metric_name in analyzer.metrics_info:
        if metric_key in means:
            layer_means = means[metric_key]
            if hasattr(layer_means, 'numpy'):
                layer_means = layer_means.numpy()
            elif torch.is_tensor(layer_means):
                layer_means = layer_means.cpu().numpy()

            if layer_idx < len(layer_means):
                performance[metric_name] = float(layer_means[layer_idx])
            else:
                performance[metric_name] = np.nan
        else:
            performance[metric_name] = np.nan

    return performance


def get_metric_short_names():
    """Get mapping of full metric names to short names for display."""
    return {
        'Rewrite Accuracy': 'Rewrite',
        'Rephrase Accuracy': 'Rephrase',
        'Locality': 'Locality',
        'Portability': 'Portability',
        'Fluency': 'Fluency',
        'Overall': 'Overall'
    }


# =============================================================================
# Influence File Loading
# =============================================================================

def load_influence_scores(filepath, expected_shape=None):
    """
    Load influence scores from a .pt file.

    Handles multiple storage formats:
    - Direct tensor
    - List of tensors (e.g., CausalTraceProba.last returns list)
    - Dict with various keys ('scores', 'influence', 'data', 'influences')

    Args:
        filepath: Path to the .pt file
        expected_shape: Optional tuple (n_samples, n_layers) for validation

    Returns:
        tuple: (scores_tensor, error_message)
            - scores_tensor: Tensor of shape (n_samples, n_layers) or None if error
            - error_message: None if success, otherwise error string
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None, f"File not found: {filepath}"

    try:
        data = torch.load(filepath, map_location='cpu')

        scores = None

        # Case 1: Direct tensor
        if isinstance(data, torch.Tensor):
            scores = data

        # Case 2: List of tensors
        elif isinstance(data, list):
            if len(data) > 0:
                if isinstance(data[0], torch.Tensor):
                    if data[0].dim() == 1:
                        # Each element is 1D (n_layers,) - stack them
                        scores = torch.stack(data)
                    elif data[0].dim() == 2:
                        # Each element is 2D (n_tokens, n_layers) - take last token
                        last_token_scores = [sample_scores[-1] for sample_scores in data]
                        scores = torch.stack(last_token_scores)

        # Case 3: Dictionary with various keys
        elif isinstance(data, dict):
            for key in ['scores', 'influence', 'data', 'influences', 'influence_scores', 'results']:
                if key in data:
                    val = data[key]
                    if isinstance(val, torch.Tensor):
                        scores = val
                        break
                    elif isinstance(val, list) and len(val) > 0:
                        if isinstance(val[0], torch.Tensor):
                            if val[0].dim() == 1:
                                scores = torch.stack(val)
                            elif val[0].dim() == 2:
                                last_token_scores = [s[-1] for s in val]
                                scores = torch.stack(last_token_scores)
                        break

        if scores is None:
            return None, f"Could not extract tensor from {type(data)}"

        # Convert to float32 if needed (handles BFloat16 from Gemma3, etc.)
        if scores.dtype == torch.bfloat16 or scores.dtype == torch.float16:
            scores = scores.float()

        # Validate shape if expected_shape provided
        if expected_shape is not None:
            if scores.shape != expected_shape:
                return None, f"Unexpected shape {scores.shape}, expected {expected_shape}"

        return scores, None

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_influence_scores_simple(filepath):
    """
    Simple wrapper that raises exception on error (for backward compatibility).

    Args:
        filepath: Path to the .pt file

    Returns:
        Tensor of shape (n_samples, n_layers)

    Raises:
        Exception if loading fails
    """
    scores, error = load_influence_scores(filepath)
    if error:
        raise Exception(error)
    return scores


def get_influence_type_from_filename(filename):
    """
    Extract influence type from filename.

    Handles filename patterns like:
    - 'modelName-gpt2-xl_localIF-True_iden-True_qLoss-False_norm-True_overall-specific.TNewOverallNew_layers-48__nsamples-1037.pt'

    The method name is in the 'overall-{method_name}' part.

    Returns:
        Influence type string (e.g., 'specific.TNewOverallNew')
    """
    stem = Path(filename).stem
    parts = stem.split('_')

    # Find the 'overall-' part which contains the method name
    for part in parts:
        if part.startswith('overall-'):
            # Extract method name after 'overall-'
            method_name = part[len('overall-'):]
            return method_name if method_name else 'Unknown'

    return 'Unknown'


# =============================================================================
# Summary Table Generation
# =============================================================================

def generate_method_comparison_table(
    analyzer,
    influence_methods,  # List of dicts with 'name', 'scores', 'mode'
    baseline_method_name='CausalTraceProba.last',
    include_layer_17=True,
    mode='min',
    outlier=False
):
    """
    Generate comparison table for multiple influence methods.

    Args:
        analyzer: Loaded analyzer with proxy/test means and indices
        influence_methods: List of dicts, each with:
            - name: str, method name
            - scores: Tensor (n_samples, n_layers) of influence scores
            - mode: 'max' or 'min' (optional, auto-detected from name)
        baseline_method_name: Name of baseline method for comparison
        include_layer_17: If True, include layer 17 performance for comparison

    Returns:
        DataFrame with method comparison
    """
    short_names = get_metric_short_names()
    proxy_indices = analyzer.proxy_indices

    # First pass: compute best layers for all methods using proxy set
    method_results = []
    baseline_overall_perf = None

    # Add "Proxy" baseline: best layer from proxy means, evaluated on test set
    # Find the proxy set's best layer based on Overall metric
    proxy_best_layer = None
    proxy_overall_key = None
    for key in ['Overall', 'post_overall', 'overall']:
        if key in analyzer.proxy_means:
            proxy_overall_key = key
            proxy_means_overall = analyzer.proxy_means[key]
            if hasattr(proxy_means_overall, 'numpy'):
                proxy_means_overall = proxy_means_overall.numpy()
            proxy_best_perf = np.nanmax(proxy_means_overall)
            proxy_best_layers = np.where(proxy_means_overall == proxy_best_perf)[0].tolist()
            proxy_best_layer = proxy_best_layers[0] if proxy_best_layers else None
            break

    if proxy_best_layer is not None:
        proxy_performance = evaluate_layer_performance(analyzer, proxy_best_layer, use_test_set=True)
        method_results.append({
            'name': 'Proxy (Golden)',
            'mode': 'Max',
            'best_layers': proxy_best_layers,
            'best_layer': proxy_best_layer,
            'performance': proxy_performance
        })

    for method in influence_methods:
        name = method['name']
        scores = method['scores']

        # if mode == 'min':
        #     mode = method.get('mode', 'min')
        # else:
        #     mode = 'max'

        # Find best layers using proxy set
        result = find_best_layers_subset(scores, proxy_indices, influence_type=name, mode=mode, outlier=outlier)

        # Get first best layer for evaluation
        best_layer = result['best_layers'][0] if result['best_layers'] else None

        # Evaluate performance at this layer on test set
        if best_layer is not None:
            performance = evaluate_layer_performance(analyzer, best_layer, use_test_set=True)
        else:
            performance = {name: np.nan for name in short_names.values()}

        # Track baseline
        if name == baseline_method_name and 'Overall' in performance:
            baseline_overall_perf = performance.get('Overall', np.nan)

        method_results.append({
            'name': name,
            'mode': result['mode_used'].capitalize(),
            'best_layers': result['best_layers'],
            'best_layer': best_layer,
            'performance': performance
        })

    # Build summary rows
    summary_rows = []
    for mr in method_results:
        row = {
            'Method': mr['name'],
            'Mode': mr['mode'],
            'Layer (Proxy)': str(mr['best_layers'])
        }

        # Add performance metrics
        for full_name, short_name in short_names.items():
            perf = mr['performance'].get(full_name, np.nan)
            if not np.isnan(perf):
                row[short_name] = f"{perf:.4f}"
            else:
                row[short_name] = 'N/A'

        summary_rows.append(row)

    # Create DataFrame and sort by Overall (descending)
    df = pd.DataFrame(summary_rows)

    # Sort by Overall (convert to float for sorting)
    def safe_float(x):
        try:
            return float(x)
        except:
            return -999

    if 'Overall' in df.columns:
        df['_sort'] = df['Overall'].apply(safe_float)
        df = df.sort_values('_sort', ascending=False).drop(columns=['_sort'])

    return df, baseline_overall_perf


def print_comparison_table(df, baseline_perf, model_name, dataset_name):
    """
    Print formatted comparison table.

    Args:
        df: DataFrame from generate_method_comparison_table
        baseline_perf: Baseline overall performance value
        model_name: Model name for display
        dataset_name: Dataset name for display
    """
    print("=" * 200)
    print("INFLUENCE-BASED LAYER SELECTION: METHOD COMPARISON")
    print("=" * 200)
    print(f"\nModel: {model_name}, Dataset: {dataset_name}")
    if baseline_perf is not None:
        print(f"Baseline Overall Performance: {baseline_perf:.4f}")
    print("\nLayer Selection: Test set evaluated at the layer selected by PROXY set")
    print("\n")

    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))

    # Print LaTeX formatted table
    print_latex_table(df, model_name, dataset_name)


def print_latex_table(df, model_name, dataset_name):
    """
    Print LaTeX formatted table for paper.

    Prints TWO formats:

    Format 1 (Dataset multirow, 3 methods):
    & \\multirow{3}{*}{Dataset}
    & Proxy & Layer & Rewrite & Rephrase & Locality & Portability & Fluency & Overall \\\\
    & CMA & ... \\\\
    & Inf & ... \\\\
    \\cmidrule(lr){2-10}

    Format 2 (Model multirow, 5 methods):
    \\multirow{5}{*}{\\shortstack[c]{GPT2\\\\XL}}
    & Proxy & Layer & Rewrite & Rephrase & Locality & Portability & Overall \\\\
    & CMA & ... \\\\
    & Inf(ONH) & ... \\\\
    & Inf(OH) & ... \\\\
    & Inf(Q) & ... \\\\
    \\midrule

    Best values are bolded (\\textbf{}), second best are underlined (\\underline{}).
    For Fluency, lowest is best. For all others, highest is best.
    Columns with all N/A values are removed in Format 2.
    """
    import re

    # Pretty names
    dataset_pretty = {
        'zsre': 'ZSRE',
        'counterfact': 'CounterFact',
        'wikibio': 'WikiBio',
        'wikirecent': 'WikiRecent',
        'wikicounterfact': 'WikiCounterFact'
    }.get(dataset_name.lower(), dataset_name)

    model_pretty = {
        'gpt2-xl': 'GPT-2 XL',
        'llama2-7b': 'LLaMA2-7B',
        'qwen3-32b': 'Qwen3-32B',
        'gemma3-12b': 'Gemma3-12B'
    }.get(model_name.lower(), model_name)

    # Model short names for LaTeX (for Format 2)
    model_latex = {
        'gpt2-xl': 'GPT2\\\\XL',
        'llama2-7b': 'LLaMA2\\\\7B',
        'qwen3-32b': 'Qwen3\\\\32B',
        'gemma3-12b': 'Gemma3\\\\12B'
    }.get(model_name.lower(), model_name)

    # For Fluency, lowest is best; for others, highest is best
    lower_is_better = {'Fluency'}

    # All possible metrics columns in order
    all_metric_cols = ['Rewrite', 'Rephrase', 'Locality', 'Portability', 'Fluency', 'Overall']

    # ============================================================
    # FORMAT 1: Dataset multirow, 3 methods (Proxy, CMA, Inf)
    # ============================================================
    print("\n")
    print("=" * 100)
    print("LATEX TABLE FORMAT 1 (Dataset multirow, 3 methods)")
    print("=" * 100)
    print("% Columns: Model & Dataset & Method & Layer & Rewrite & Rephrase & Locality & Portability & Fluency & Overall")
    print("% Best values are bolded, second best are underlined")
    print()

    # Method short names for Format 1
    method_short_names_f1 = {
        'Proxy (Golden)': 'Proxy',
        'specific.OverallNewOverallHat': 'Inf',
        'CausalTraceProba.last': 'CMA',
    }

    metric_cols_f1 = all_metric_cols.copy()
    key_methods_f1 = ['Proxy (Golden)', 'CausalTraceProba.last', 'specific.OverallNewOverallHat']

    # Get rows for key methods (Format 1)
    latex_rows_f1 = []
    for method in key_methods_f1:
        row_data = df[df['Method'] == method]
        if len(row_data) > 0:
            row = row_data.iloc[0]
            latex_rows_f1.append((method, row))

    n_rows_f1 = len(latex_rows_f1)
    if n_rows_f1 > 0:
        # Collect metric values for best/second best
        metric_values_f1 = {col: [] for col in metric_cols_f1}
        for method_idx, (method, row) in enumerate(latex_rows_f1):
            for col in metric_cols_f1:
                val = row.get(col, 'N/A')
                if val != 'N/A' and not pd.isna(val):
                    try:
                        numeric_val = float(val)
                        metric_values_f1[col].append((method_idx, numeric_val))
                    except (ValueError, TypeError):
                        pass

        # Find best and second best (handling ties with sets)
        best_indices_f1 = {}  # col -> set of method indices with best value
        second_best_indices_f1 = {}  # col -> set of method indices with second best value
        for col in metric_cols_f1:
            values = metric_values_f1[col]
            if len(values) == 0:
                continue
            if col in lower_is_better:
                sorted_vals = sorted(values, key=lambda x: x[1])
            else:
                sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)

            # Find all indices with the best value
            best_val = sorted_vals[0][1]
            best_indices_f1[col] = {idx for idx, val in values if val == best_val}

            # Find all indices with the second best value (different from best)
            second_best_val = None
            for idx, val in sorted_vals:
                if val != best_val:
                    second_best_val = val
                    break
            if second_best_val is not None:
                second_best_indices_f1[col] = {idx for idx, val in values if val == second_best_val}

        # Print Format 1
        print(f"& \\multirow{{{n_rows_f1}}}{{*}}{{{dataset_pretty}}}")

        for method_idx, (method, row) in enumerate(latex_rows_f1):
            method_short = method_short_names_f1.get(method, method[:10])

            # Extract layers
            layer_str = str(row.get('Layer (Proxy)', ''))
            layer_matches = re.findall(r'\d+', layer_str)
            layer = ', '.join(layer_matches) if layer_matches else '--'

            # Build metric values with formatting
            metric_vals = []
            for col in metric_cols_f1:
                val = row.get(col, 'N/A')
                if val == 'N/A' or pd.isna(val):
                    metric_vals.append('--')
                else:
                    val_str = str(val)
                    if col in best_indices_f1 and method_idx in best_indices_f1[col]:
                        val_str = f'\\textbf{{{val_str}}}'
                    elif col in second_best_indices_f1 and method_idx in second_best_indices_f1[col]:
                        val_str = f'\\underline{{{val_str}}}'
                    metric_vals.append(val_str)

            metrics_str = ' & '.join(metric_vals)

            if method_idx == 0:
                print(f"& {method_short} & {layer} & {metrics_str} \\\\")
            else:
                print(f"&      & {method_short} & {layer} & {metrics_str} \\\\")

        print("\\cmidrule(lr){2-10}")
    else:
        print("% No key methods found in dataframe for Format 1")

    # ============================================================
    # FORMAT 2: Model multirow, 5 methods (Proxy, CMA, Inf(ONH), Inf(OH), Inf(Q))
    # ============================================================
    print("\n")
    print("=" * 100)
    print("LATEX TABLE FORMAT 2 (Model multirow, 5 methods)")
    print("=" * 100)
    print("% Best values are bolded, second best are underlined")
    print("% Columns with all N/A values are removed")
    print()

    # Method short names for Format 2
    method_short_names_f2 = {
        'Proxy (Golden)': 'Proxy',
        'CausalTraceProba.last': 'CMA',
        'specific.OverallNewOverallHat': 'Inf(ONH)',
        'l2normSpecific.OverallHat': 'Inf(OH)',
        'l2normSpecific.Query': 'Inf(Q)',
    }

    key_methods_f2 = ['Proxy (Golden)', 'CausalTraceProba.last', 'specific.OverallNewOverallHat',
                      'l2normSpecific.OverallHat', 'l2normSpecific.Query']

    # Get rows for key methods (Format 2)
    latex_rows_f2 = []
    for method in key_methods_f2:
        row_data = df[df['Method'] == method]
        if len(row_data) > 0:
            row = row_data.iloc[0]
            latex_rows_f2.append((method, row))

    n_rows_f2 = len(latex_rows_f2)
    if n_rows_f2 > 0:
        # Determine which metric columns have at least one valid value
        valid_metric_cols = []
        for col in all_metric_cols:
            has_valid = False
            for method, row in latex_rows_f2:
                val = row.get(col, 'N/A')
                if val != 'N/A' and not pd.isna(val):
                    has_valid = True
                    break
            if has_valid:
                valid_metric_cols.append(col)

        metric_cols_f2 = valid_metric_cols
        print(f"% Columns: Model & Method & Layer & {' & '.join(metric_cols_f2)}")

        # Collect metric values for best/second best
        metric_values_f2 = {col: [] for col in metric_cols_f2}
        for method_idx, (method, row) in enumerate(latex_rows_f2):
            for col in metric_cols_f2:
                val = row.get(col, 'N/A')
                if val != 'N/A' and not pd.isna(val):
                    try:
                        numeric_val = float(val)
                        metric_values_f2[col].append((method_idx, numeric_val))
                    except (ValueError, TypeError):
                        pass

        # Find best and second best (handling ties with sets)
        best_indices_f2 = {}  # col -> set of method indices with best value
        second_best_indices_f2 = {}  # col -> set of method indices with second best value
        for col in metric_cols_f2:
            values = metric_values_f2[col]
            if len(values) == 0:
                continue
            if col in lower_is_better:
                sorted_vals = sorted(values, key=lambda x: x[1])
            else:
                sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)

            # Find all indices with the best value
            best_val = sorted_vals[0][1]
            best_indices_f2[col] = {idx for idx, val in values if val == best_val}

            # Find all indices with the second best value (different from best)
            second_best_val = None
            for idx, val in sorted_vals:
                if val != best_val:
                    second_best_val = val
                    break
            if second_best_val is not None:
                second_best_indices_f2[col] = {idx for idx, val in values if val == second_best_val}

        # Print Format 2 - Model multirow
        print(f"\\multirow{{{n_rows_f2}}}{{*}}{{\\shortstack[c]{{{model_latex}}}}}")

        for method_idx, (method, row) in enumerate(latex_rows_f2):
            method_short = method_short_names_f2.get(method, method[:10])

            # Extract layers
            layer_str = str(row.get('Layer (Proxy)', ''))
            layer_matches = re.findall(r'\d+', layer_str)
            layer = ', '.join(layer_matches) if layer_matches else '--'

            # Build metric values with formatting
            metric_vals = []
            for col in metric_cols_f2:
                val = row.get(col, 'N/A')
                if val == 'N/A' or pd.isna(val):
                    metric_vals.append('--')
                else:
                    val_str = str(val)
                    if col in best_indices_f2 and method_idx in best_indices_f2[col]:
                        val_str = f'\\textbf{{{val_str}}}'
                    elif col in second_best_indices_f2 and method_idx in second_best_indices_f2[col]:
                        val_str = f'\\underline{{{val_str}}}'
                    metric_vals.append(val_str)

            metrics_str = ' & '.join(metric_vals)
            print(f"& {method_short} & {layer} & {metrics_str} \\\\")

        print("\\midrule")
    else:
        print("% No key methods found in dataframe for Format 2")

    print()
