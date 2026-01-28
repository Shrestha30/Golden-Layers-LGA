import matplotlib.patches as mpatches
import importlib.util
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ks_2samp
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind


# =============================================================================
# Centralized Metric Mappings
# =============================================================================
# Maps (dataset, metric_display_name) -> actual_metric_key_in_api
# metric_display_name: Rewrite, Rephrase, Locality, Portability, Fluency, Overall

METRIC_MAPPINGS = {
    'zsre': {
        'Rewrite': 'post_rewrite',
        'Rephrase': 'post_rephrase',
        'Locality': 'post_neighborhood',
        'Portability': 'post_one_hop',
        'Fluency': None,
        'Overall': 'post_overall',
    },
    'counterfact': {
        'Rewrite': 'post_rewrite',
        'Rephrase': 'post_rephrase',
        'Locality': 'post_neighborhood',
        'Portability': 'post_one_hop',
        'Fluency': None,
        'Overall': 'post_overall',
    },
    'wikibio': {
        'Rewrite': 'post_rewrite_acc',
        'Rephrase': None,
        'Locality': 'post_locality_acc',
        'Portability': None,
        'Fluency': 'post_fluency',
        'Overall': 'post_overall',
    },
    'wikirecent': {
        'Rewrite': 'post_rewrite_acc',
        'Rephrase': None,
        'Locality': 'post_locality_acc',
        'Portability': 'post_portability_acc',
        'Fluency': 'post_fluency',
        'Overall': 'overall',  # WikiRecent uses 'overall' not 'post_overall'
    },
    'wikicounterfact': {
        'Rewrite': 'post_rewrite_acc',
        'Rephrase': None,
        'Locality': 'post_locality_acc',
        'Portability': 'post_portability_acc',
        'Fluency': 'post_fluency',
        'Overall': 'overall',  # WikiCounterfact uses 'overall' not 'post_overall'
    },
}

METRIC_COLUMNS = ['Rewrite', 'Rephrase', 'Locality', 'Portability', 'Fluency', 'Overall']


def get_metric_key(dataset, metric_name):
    """Get the actual metric key for a dataset and metric display name."""
    if dataset not in METRIC_MAPPINGS:
        return None
    return METRIC_MAPPINGS[dataset].get(metric_name)


# =============================================================================
# Analyzer Loading Utilities
# =============================================================================

def load_analyzer(dataset, model, registry_module):
    """
    Load the golden layer analyzer for a dataset.

    Args:
        dataset: Dataset name (e.g., 'zsre', 'wikibio')
        model: Model name (e.g., 'gpt2-xl', 'llama2-7b')
        registry_module: The dataset_model_registry module

    Returns:
        (analyzer, config) tuple
    """
    config = registry_module.get_config(dataset, model)

    # Load the API module
    api_path = config['api_path']
    spec = importlib.util.spec_from_file_location("analyzer_module", api_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get analyzer class
    api_class_name = config.get('api_class', 'GoldenLayerAnalyzer')
    analyzer_class = getattr(module, api_class_name)

    # Create analyzer with correct n_layers
    data_dir = config['base_dir'] / config.get('data_dir', 'allLayerPerformance')
    n_layers = config.get('n_layers', 48)
    try:
        analyzer = analyzer_class(data_dir=str(data_dir), n_layers=n_layers)
    except TypeError:
        analyzer = analyzer_class(data_dir=str(data_dir))

    # Load data
    analyzer.load_data(verbose=False)

    # Compute overall metric if available
    if hasattr(analyzer, 'compute_overall_metric'):
        analyzer.compute_overall_metric(verbose=False)

    # Split dataset
    proxy_size = config.get('proxy_size', 100)
    analyzer.split_dataset(proxy_size=proxy_size, random_seed=42, verbose=False)
    analyzer.compute_proxy_test_means(verbose=False)

    return analyzer, config


def compute_correlation(proxy_means, test_means):
    """
    Compute Pearson correlation between proxy and test layer means.

    Args:
        proxy_means: Array of proxy set means per layer
        test_means: Array of test set means per layer

    Returns:
        correlation coefficient (or NaN if not computable)
    """
    # Convert to numpy if needed
    if hasattr(proxy_means, 'numpy'):
        proxy_means = proxy_means.numpy()
    if hasattr(test_means, 'numpy'):
        test_means = test_means.numpy()

    # Filter out NaN values
    valid_mask = ~(np.isnan(proxy_means) | np.isnan(test_means))

    if valid_mask.sum() < 3:  # Need at least 3 points for meaningful correlation
        return np.nan

    proxy_valid = proxy_means[valid_mask]
    test_valid = test_means[valid_mask]

    # Compute Pearson correlation
    corr, _ = stats.spearmanr(proxy_valid, test_valid)

    return corr


# =============================================================================
# Oracle vs Golden Layer Analysis
# =============================================================================

def compute_oracle_vs_golden(analyzer, metric_key, higher_is_better=True, golden_layer_proxy=True):
    """
    Compare oracle (sample-wise best) vs golden layer (fixed from proxy) performance.

    Args:
        analyzer: Loaded analyzer with results and test_indices
        metric_key: The metric key in analyzer.results (e.g., 'post_overall')
        higher_is_better: If True, oracle = max; if False (e.g., fluency), oracle = min

    Returns:
        dict with:
            - golden_layer: int, the first layer selected from proxy set (used for calculation)
            - golden_layers: list, all layers with best proxy performance (handles ties)
            - oracle_perf: array of oracle performance per test sample
            - golden_perf: array of golden layer performance per test sample
            - gaps: array of (oracle - golden) per sample
            - mean_gap: float
            - median_gap: float
            - n_samples: int
    """
    # Get raw data tensor: shape (n_samples, n_layers)
    data = analyzer.results[metric_key]
    if hasattr(data, 'numpy'):
        data = data.numpy()

    # Get test indices
    import numpy as np
    test_indices = analyzer.test_indices
    if hasattr(test_indices, 'numpy'):
        test_indices = test_indices.numpy()
    test_indices = np.array(test_indices)

    # Get test data: shape (n_test_samples, n_layers)
    test_data = data[test_indices]

    # Find golden layer(s) from proxy set means - use np.where to handle ties
    # (consistent with heatmap scripts)
    if golden_layer_proxy:
        proxy_means = analyzer.proxy_means[metric_key]
    else:
        proxy_means = analyzer.test_means[metric_key]
        
    if hasattr(proxy_means, 'numpy'):
        proxy_means = proxy_means.numpy()

    if higher_is_better:
        best_perf = np.nanmax(proxy_means)
    else:
        best_perf = np.nanmin(proxy_means)

    # Find ALL layers with best performance (handles ties like heatmap script)
    golden_layers = np.where(proxy_means == best_perf)[0].tolist()
    golden_layer = golden_layers[0]  # Use first for calculation

    # Compute oracle performance (sample-wise best)
    if higher_is_better:
        oracle_perf = np.nanmax(test_data, axis=1)
        oracle_layers = np.nanargmax(test_data, axis=1)
        
    else:
        oracle_perf = np.nanmin(test_data, axis=1)
        oracle_layers = np.nanargmin(test_data, axis=1)

    # Compute golden layer performance
    golden_perf = test_data[:, golden_layer]
    oracle_perf_all = test_data[:, oracle_layers]

    # Compute gaps (oracle - golden)
    # For higher_is_better: positive gap means oracle is better
    # For lower_is_better: we negate so positive gap still means oracle is better
    if higher_is_better:
        gaps = oracle_perf - golden_perf
    else:
        gaps = golden_perf - oracle_perf  # Flip sign for lower-is-better

    # Filter out NaN values for statistics
    valid_mask = ~(np.isnan(oracle_perf) | np.isnan(golden_perf))
    valid_gaps = gaps[valid_mask]

    # oracle_perf and golden_perf are arrays of shape (n_test_samples,)
    valid_mask = ~(np.isnan(oracle_perf) | np.isnan(golden_perf))
    if valid_mask.sum() >= 3:  # need at least 3 samples
        corr, _ = pearsonr(oracle_perf[valid_mask], golden_perf[valid_mask])
    else:
        corr = np.nan

    # =========================
    # Correlation diagnostics
    # =========================
    oracle_valid = oracle_perf[valid_mask]
    golden_valid = golden_perf[valid_mask]

    oracle_var = np.nanvar(oracle_valid)
    golden_var = np.nanvar(golden_valid)

    oracle_unique = np.unique(oracle_valid).size
    golden_unique = np.unique(golden_valid).size

    # if oracle_var == 0 or golden_var == 0:
    #     print("[INFO] Pearson correlation undefined (NaN)")
    #     print(f"       Oracle variance: {oracle_var:.6f}")
    #     print(f"       Golden variance: {golden_var:.6f}")
    #     print(f"       Unique oracle values: {oracle_unique}")
    #     print(f"       Unique golden values: {golden_unique}")
        
    # =========================
    # KS test (distributional comparison)
    # =========================
    if valid_mask.sum() >= 3:
        ks_stat, ks_pvalue = ks_2samp(
            oracle_perf[valid_mask],
            golden_perf[valid_mask],
            alternative='two-sided',
            mode='auto'
        )
    else:
        ks_stat, ks_pvalue = np.nan, np.nan

    ks_significant = (ks_pvalue <= 0.05)


    t_stat, p_two_tailed = ttest_rel(oracle_perf, golden_perf)

    print("*******One-tailed test (Oracle > Golden):")
    #One-tailed test (IF > Alpha)
    if np.mean(oracle_perf) > np.mean(golden_perf):
        p_one_tailed = p_two_tailed / 2
    else:
        p_one_tailed = 1 - (p_two_tailed / 2)

    print(f"{metric_key} Mean Oracle: {np.mean(oracle_perf):.4f}, Mean Golden: {np.mean(golden_perf):.4f}")
    print(f"{metric_key}: {float(np.nanmean(valid_gaps)) if len(valid_gaps) > 0 else np.nan:.4f} gap over {int(valid_mask.sum())} samples")
    print(
        f"{metric_key} | "
        f"Oracle: {np.mean(oracle_perf):.4f} ± {np.std(oracle_perf):.4f}, "
        f"Golden: {np.mean(golden_perf):.4f} ± {np.std(golden_perf):.4f}"
    )

    print(f'{metric_key} oracle: {oracle_perf}')
    print(f'{metric_key} golden: {golden_perf}')

    print(f"One-tailed p-value (Oracle > Golden): {p_one_tailed:.5f}")
    print(f"Two-tailed p-value (Oracle vs Golden): {p_two_tailed:.5f}")


    import numpy as np

    # --- Step 1: Identify golden and oracle layers for the chosen metric_key ---
    test_means = analyzer.test_means[metric_key]
    if hasattr(test_means, 'numpy'):
        test_means = test_means.numpy()

    if higher_is_better:
        best_perf = np.nanmax(test_means)
    else:
        best_perf = np.nanmin(test_means)

    golden_layers = np.where(test_means == best_perf)[0].tolist()
    golden_layer = golden_layers[0]  # first golden layer

    # Sample-wise oracle
      # [n_samples, n_layers] for selected metric
    if higher_is_better:
        oracle_layers = np.nanargmax(test_data, axis=1)
    else:
        oracle_layers = np.nanargmin(test_data, axis=1)

    print("Proxy set golden layers:", np.argmax(analyzer.proxy_means[metric_key].numpy()), "performance on test:", analyzer.test_means[metric_key].numpy()[np.argmax(analyzer.proxy_means[metric_key].numpy())])
    print("Test set golden layers:", np.argmax(analyzer.test_means[metric_key].numpy()), "performance on test:", analyzer.test_means[metric_key].numpy()[np.argmax(analyzer.test_means[metric_key].numpy())])

    # --- Step 2: Collect performances for all metrics based on selected layers ---
    results = {}
    for metric in analyzer.results.keys():
        data = analyzer.results[metric][test_indices]  # [n_samples, n_layers]
        
        # Golden layer performance
        golden_perf = data[:, golden_layer]
        golden_mean = np.nanmean(golden_perf)
        golden_std = np.nanstd(golden_perf)
        
        # Oracle layers performance (sample-wise)
        oracle_perf = data[np.arange(data.shape[0]), oracle_layers]
        oracle_mean = np.nanmean(oracle_perf)
        oracle_std = np.nanstd(oracle_perf)

        print()

        print("Length of oracle_perf:", len(oracle_perf))
        print("Length of golden_perf:", len(golden_perf))
        print("any NaNs in oracle_perf:", np.isnan(oracle_perf).any())
        print("any NaNs in golden_perf:", np.isnan(golden_perf).any())
        t_stat, p_two_tailed = ttest_ind(oracle_perf, golden_perf, nan_policy= 'omit')
        print("*******One-tailed test (Oracle > Golden):")
        print(f"{metric} p-value (two-tailed): {p_two_tailed:.5f}. Significance(Yes/No): {p_two_tailed <= 0.05}")
        
        # Store results
        results[metric] = {
            'golden_mean': golden_mean,
            'golden_std': golden_std,
            'oracle_mean': oracle_mean,
            'oracle_std': oracle_std
        }

    # --- Step 3: results dict now has mean & std for all metrics ---
    print(f"*******Oracle vs Golden Performance for All Metrics || {metric_key} ||:")
    print(results)

    print("================================")
    def printer(results):
        def fmt(mean, std):
            return f"{mean:.4f}$_{{\\pm{std:.4f}}}$"
        cols = [
            "post_rewrite",
            "post_rephrase",
            "post_neighborhood",
            "post_one_hop",
            "post_overall",
        ]
        #WikiBio
        cols = [
            "post_rewrite_acc",
            "post_locality_acc",
            "post_fluency",
            "post_overall",
        ]

        #wikiCounterfact
        cols = [
            "post_rewrite_acc",
            "post_locality_acc",
            "post_portability_acc",
            "post_fluency",
            "overall",
        ]

        # Oracle row
        oracle_row = " & ".join(
            fmt(results[c]["oracle_mean"], results[c]["oracle_std"])
            for c in cols
        )

        # Golden row
        golden_row = " & ".join(
            fmt(results[c]["golden_mean"], results[c]["golden_std"])
            for c in cols
        )

        print(
        f"""&Sample-Wise Optimal Layer & {oracle_row}\\\\
        &Golden Layer & {golden_row}\\\\
        \\cmidrule{{2-7}}"""
        )
        
        
        data = analyzer.results[metric_key][test_indices]  # [n_samples, n_layers]
        
        # Golden layer performance
        golden_perf = data[:, golden_layer]
        golden_mean = np.nanmean(golden_perf)
        golden_std = np.nanstd(golden_perf)
        
        # Oracle layers performance (sample-wise)
        oracle_perf = data[np.arange(data.shape[0]), oracle_layers]
        oracle_mean = np.nanmean(oracle_perf)
        oracle_std = np.nanstd(oracle_perf)

        print()

        # print("Length of oracle_perf:", len(oracle_perf))
        # print("Length of golden_perf:", len(golden_perf))
        # print("any NaNs in oracle_perf:", np.isnan(oracle_perf).any())
        # print("any NaNs in golden_perf:", np.isnan(golden_perf).any())
        t_stat, p_two_tailed = ttest_ind(oracle_perf, golden_perf, nan_policy= 'omit')
        # print("*******One-tailed test (Oracle > Golden):")
        print(f"{metric} p-value (two-tailed): {p_two_tailed:.5f}. Significance(Yes/No): {p_two_tailed <= 0.05}")
        
        results = {
            "oracle_mean": oracle_mean,
            'oracle_std': oracle_std,
            'golden_mean': golden_mean,
            'golden_std': golden_std,
            'stat_diff': 'Yes' if p_two_tailed <= 0.05 else 'No'
        }
    return results





    # print("*******Two-tailed test (Oracle > Golden):")
    # t_stat, p_two_tailed = ttest_rel(if_all, mola_all)

    # #One-tailed test (IF > MOLA)
    # if np.mean(if_all) > np.mean(mola_all):
    #     p_one_tailed = p_two_tailed / 2
    # else:
    #     p_one_tailed = 1 - (p_two_tailed / 2)

    # print(f"Mean IF: {np.mean(if_all):.2f}, Mean MOLA: {np.mean(mola_all):.2f}")
    # print(f"One-tailed p-value (IF > MOLA): {p_one_tailed:.5f}")

    # return {
    #     'golden_layer': golden_layer,
    #     'golden_layers': golden_layers,  # All tied best layers
    #     'oracle_perf': oracle_perf,
    #     'golden_perf': golden_perf,
    #     'gaps': gaps,
    #     'mean_gap': float(np.nanmean(valid_gaps)) if len(valid_gaps) > 0 else np.nan,
    #     'median_gap': float(np.nanmedian(valid_gaps)) if len(valid_gaps) > 0 else np.nan,
    #     'n_samples': int(valid_mask.sum()),
    #     'higher_is_better': higher_is_better,
    #     'std_gap': float(np.nanstd(valid_gaps)) if len(valid_gaps) > 0 else np.nan,
    #     'correlation': float(corr),
    #     'ks_stat': float(ks_stat),
    #     'ks_pvalue': float(ks_pvalue),
    #     'ks_significant': bool(ks_significant),
    #     "t_stat": float(t_stat),
    #     "p_two_tailed": float(p_two_tailed),

    # }


# =============================================================================
# Heatmap Utilities
# =============================================================================

def get_segments(indices):
    """Groups [14, 15, 16, 20, 21] into [(14, 16), (20, 21)]"""
    if not indices: return []
    indices = sorted(list(set(indices)))
    segments = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            segments.append((start, indices[i-1]))
            start = indices[i]
    segments.append((start, indices[-1]))
    return segments

def highlight_mark_layers(GOLDEN_LAYERS, DATASET_GOLDEN_LAYERS, dataset_names, best_layers, ax):
    # 1. Highlight GOLDEN_LAYERS (Vertical blocks)
    golden_segments = get_segments(GOLDEN_LAYERS)
    lw_gold = 2
    # Small offset to keep the border inside the cell (units of data)
    offset_gold = 0.02 

    for start, end in golden_segments:
        width = (end - start) + 1
        # Inset x and y slightly, and reduce width/height by double that offset
        rect = mpatches.Rectangle((start - 0.5 + offset_gold, -0.5 + offset_gold), 
                                   width - 2*offset_gold, len(dataset_names) - 2*offset_gold,
                                   fill=False, edgecolor='gold',
                                   linewidth=lw_gold, linestyle='-', zorder=4)
        ax.add_patch(rect)

    # 2. Highlight DATASET_GOLDEN_LAYERS (Horizontal blocks)
    lw_green = 1
    offset_green = 0.02 # Larger offset for thicker line

    for row, dataset in enumerate(dataset_names):
        if dataset in DATASET_GOLDEN_LAYERS:
            dataset_segments = get_segments(DATASET_GOLDEN_LAYERS[dataset])
            for start, end in dataset_segments:
                width = (end - start) + 1
                rect = mpatches.Rectangle((start - 0.5 + offset_green, row - 0.5 + offset_green), 
                                           width - 2*offset_green, 1 - 2*offset_green,
                                           fill=False, edgecolor='limegreen',
                                           linewidth=lw_green, linestyle='-', zorder=5)
                ax.add_patch(rect)

    # 3. Add best layer markers (star)
    for row, dataset in enumerate(dataset_names):
        if dataset in best_layers:
            layers = best_layers[dataset]
            ax.plot(layers, [row]*len(layers), marker='*', color='white', 
                    markersize=10, markeredgecolor='black', markeredgewidth=0.8, 
                    linestyle='None', zorder=6) # Highest zorder