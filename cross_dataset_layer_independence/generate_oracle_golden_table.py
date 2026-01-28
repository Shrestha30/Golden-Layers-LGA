#!/usr/bin/env python3
"""
Generate Oracle vs Golden Layer LaTeX Table

Creates a LaTeX table comparing oracle (sample-wise best) vs golden layer (fixed from proxy)
performance across models and metrics.

Columns: Model | Metric | Golden Layer(s) | Used | Mean | Median | KS Stat | KS p-value

** indicates statistical significance (p <= 0.05)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import dataset_model_registry
import heatmap_api


# Model configurations
MODEL_CONFIGS = {
    'llama2-7b': {
        'display_name': 'LLaMA2-7B',
        'datasets': ['zsre'],  # Using zsre for the table
    },
    'gpt2-xl': {
        'display_name': 'GPT2-XL',
        'datasets': ['zsre'],
    },
    'gemma3-12b': {
        'display_name': 'Gemma3-12B',
        'datasets': ['zsre'],
    },
    'qwen3-32b': {
        'display_name': 'Qwen3-32B',
        'datasets': ['zsre'],
    },
}

# Metrics to analyze (in order)
METRICS = ['Rewrite', 'Rephrase', 'Locality', 'Portability', 'Overall']


def format_golden_layers(layers):
    """
    Format golden layers list for LaTeX.
    Groups consecutive layers into ranges like [38--51].

    Args:
        layers: list of layer indices

    Returns:
        formatted string like "[12,13,14,24]" or "[23,25,35,38--51]"
    """
    if not layers:
        return "[]"

    layers = sorted(layers)

    if len(layers) == 1:
        return f"[{layers[0]}]"

    # Group consecutive layers into ranges
    groups = []
    start = layers[0]
    end = layers[0]

    for i in range(1, len(layers)):
        if layers[i] == end + 1:
            end = layers[i]
        else:
            # Save current group
            if end - start >= 2:  # Range of 3+ consecutive
                groups.append(f"{start}--{end}")
            elif end - start == 1:  # 2 consecutive
                groups.append(f"{start},{end}")
            else:  # Single
                groups.append(str(start))
            start = layers[i]
            end = layers[i]

    # Save last group
    if end - start >= 2:
        groups.append(f"{start}--{end}")
    elif end - start == 1:
        groups.append(f"{start},{end}")
    else:
        groups.append(str(start))

    return "[" + ",".join(groups) + "]"


def compute_model_metrics(model, dataset='zsre'):
    """
    Compute oracle vs golden statistics for all metrics for a model.

    Returns:
        dict mapping metric name to result dict
    """
    results = {}

    print(f"\n  Loading {model} / {dataset}...")

    try:
        analyzer, config = heatmap_api.load_analyzer(dataset, model, dataset_model_registry)
    except Exception as e:
        print(f"    Error loading: {e}")
        return results

    for metric_name in METRICS:
        try:
            metric_key = heatmap_api.get_metric_key(dataset, metric_name)
            if metric_key is None:
                print(f"    Warning: {metric_name} not supported for {dataset}")
                continue

            # Fluency is lower-is-better
            higher_is_better = metric_name.lower() != 'fluency'

            result = heatmap_api.compute_oracle_vs_golden(
                analyzer, metric_key, higher_is_better, golden_layer_proxy=False
            )
            results[metric_name] = result

            print(f"    {metric_name}: golden={result['golden_layers']}, "
                  f"mean_gap={result['mean_gap']:.4f}, "
                  f"ks_stat={result['ks_stat']:.4f}, ks_p={result['ks_pvalue']:.4f}")

        except Exception as e:
            print(f"    Error computing {metric_name}: {e}")

    return results


def generate_latex_table(all_results, dataset='zsre'):
    """
    Generate LaTeX table from results.

    Args:
        all_results: dict mapping model -> (dict mapping metric -> result)
        dataset: dataset name for caption

    Returns:
        LaTeX table string
    """
    lines = []

    # Table header
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Metric & Golden Layer(s) & Used & Mean & Median & KS Stat & KS $p$ \\")
    lines.append(r"\midrule")

    model_order = ['llama2-7b', 'gpt2-xl', 'gemma3-12b', 'qwen3-32b']

    for model in model_order:
        if model not in all_results:
            continue

        model_results = all_results[model]
        config = MODEL_CONFIGS[model]
        display_name = config['display_name']

        # Count valid metrics for multirow
        n_metrics = len([m for m in METRICS if m in model_results])

        if n_metrics == 0:
            continue

        # First metric row with multirow
        first = True
        for metric_name in METRICS:
            if metric_name not in model_results:
                continue

            result = model_results[metric_name]

            # Format golden layers
            golden_layers_str = format_golden_layers(result['golden_layers'])
            used_layer = result['golden_layer']
            mean_gap = result['mean_gap']
            median_gap = result['median_gap']
            ks_stat = result['ks_stat']
            ks_pvalue = result['ks_pvalue']

            # Format KS stat with ** if significant
            if not np.isnan(ks_stat):
                ks_stat_str = f"{ks_stat:.4f}"
                if ks_pvalue <= 0.05:
                    ks_stat_str += "**"
            else:
                ks_stat_str = "--"

            # Format p-value
            if not np.isnan(ks_pvalue):
                if ks_pvalue < 0.0001:
                    ks_p_str = "$<$0.0001"
                else:
                    ks_p_str = f"{ks_pvalue:.4f}"
            else:
                ks_p_str = "--"

            # Format mean and median
            mean_str = f"{mean_gap:.4f}" if not np.isnan(mean_gap) else "--"
            median_str = f"{median_gap:.4f}" if not np.isnan(median_gap) else "--"

            if first:
                # Multirow for model name
                line = f"\\multirow{{{n_metrics}}}{{*}}{{{display_name}}}"
                first = False
            else:
                line = ""

            line += f" & {metric_name:11s} & {golden_layers_str:17s} & {used_layer:2d} & {mean_str} & {median_str} & {ks_stat_str} & {ks_p_str} \\\\"
            lines.append(line)

        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{Oracle vs. Golden-layer comparison across models and metrics ({dataset.upper()} dataset). ** indicates statistical significance ($p \\leq 0.05$).}}")
    lines.append(r"\label{tab:oracle_vs_golden}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_ttest_latex_table(all_results, dataset='zsre'):
    """
    Generate LaTeX table with t-test statistics.

    Args:
        all_results: dict mapping model -> (dict mapping metric -> result)
        dataset: dataset name for caption

    Returns:
        LaTeX table string
    """
    lines = []

    # Table header
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\caption{Oracle vs. Golden-layer (GL) comparison across models and metrics (" + dataset.upper() + r" dataset).")
    lines.append(r"Two-sided paired $t$-test is reported. ** indicates statistical significance ($p \leq 0.05$).}")
    lines.append(r"\begin{tabular}{l l c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Metric & GL & $t$-stat & $p$-value \\")
    lines.append(r"\midrule")

    model_order = ['llama2-7b', 'gpt2-xl', 'gemma3-12b', 'qwen3-32b']

    for model in model_order:
        if model not in all_results:
            continue

        model_results = all_results[model]
        config = MODEL_CONFIGS[model]
        display_name = config['display_name']

        # Count valid metrics for multirow
        n_metrics = len([m for m in METRICS if m in model_results])

        if n_metrics == 0:
            continue

        # First metric row with multirow
        first = True
        for metric_name in METRICS:
            if metric_name not in model_results:
                continue

            result = model_results[metric_name]

            # Get golden layer (first one used)
            used_layer = result['golden_layer']

            # Get t-test statistics
            t_stat = result.get('t_stat', np.nan)
            p_two_tailed = result.get('p_two_tailed', np.nan)

            # Format t-stat (2 decimal places)
            if not np.isnan(t_stat):
                t_stat_str = f"{t_stat:.2f}"
            else:
                t_stat_str = "--"

            # Format p-value (3 decimal places, with ** if significant)
            if not np.isnan(p_two_tailed):
                if p_two_tailed < 0.001:
                    p_str = "0.000"
                else:
                    p_str = f"{p_two_tailed:.3f}"
                if p_two_tailed <= 0.05:
                    p_str += "**"
            else:
                p_str = "--"

            if first:
                # Multirow for model name with shortstack
                line = f"\\multirow{{{n_metrics}}}{{*}}{{\\shortstack[c]{{{display_name}}}}}"
                first = False
            else:
                line = ""

            line += f" & {metric_name:11s} & {used_layer:2d} & {t_stat_str:5s} & {p_str} \\\\"
            lines.append(line)

        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:oracle_vs_golden_ttest}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Oracle vs Golden Layer LaTeX Table')
    parser.add_argument('--dataset', type=str, default='zsre',
                        help='Dataset name (zsre, counterfact, wikibio)')
    parser.add_argument('--output', type=str, default='results/oracle_vs_golden_table.tex',
                        help='Output path for LaTeX table')
    parser.add_argument('--table-type', type=str, default='both', choices=['ks', 'ttest', 'both'],
                        help='Type of table to generate: ks (KS test), ttest (t-test), or both')
    args = parser.parse_args()

    print("=" * 80)
    print("Oracle vs Golden Layer LaTeX Table Generator")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Table type: {args.table_type}")
    print("=" * 80)

    # Compute results for all models
    all_results = {}
    for model in MODEL_CONFIGS.keys():
        results = compute_model_metrics(model, args.dataset)
        if results:
            all_results[model] = results

    # Generate LaTeX tables based on table-type
    if args.table_type in ['ks', 'both']:
        print("\n" + "=" * 80)
        print("LATEX TABLE (KS Test)")
        print("=" * 80)

        latex_table_ks = generate_latex_table(all_results, args.dataset)
        print(latex_table_ks)

        # Save KS table to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_table_ks)
        print(f"\nSaved KS table to: {output_path}")

    if args.table_type in ['ttest', 'both']:
        print("\n" + "=" * 80)
        print("LATEX TABLE (t-test)")
        print("=" * 80)

        latex_table_ttest = generate_ttest_latex_table(all_results, args.dataset)
        print(latex_table_ttest)

        # Save t-test table to file
        ttest_output_path = Path(args.output).with_name(
            Path(args.output).stem + '_ttest.tex'
        )
        with open(ttest_output_path, 'w') as f:
            f.write(latex_table_ttest)
        print(f"\nSaved t-test table to: {ttest_output_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
