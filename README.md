# Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis

Code for reproducing Golden Layers and LGA in knowledge editing.

## Installation

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token_here"
```

## Configuration

Update model paths in hyperparameter files located in `hparams/`:
- `hparams/R-ROME/` - Contains YAML configs for supported models:
  - [gpt2-xl.yaml](hparams/R-ROME/gpt2-xl.yaml)
  - [llama-7b.yaml](hparams/R-ROME/llama-7b.yaml) (for LLaMA2-7B)
  - [gemma3-12b.yaml](hparams/R-ROME/gemma3-12b.yaml)
- `hparams/EMMET/` - GPT2-XL only
- `hparams/ROME/` - GPT2-XL only

## Collecting Layer scores by LGA (ours) and CMA (Baseline)

Three scripts collect layer-wise attrobution scores:

### 1. ZSRE / CounterFact Scores

```bash
# ZSRE
python zsre_collect_all_layer_scores.py \
    --editing_method "R-ROME" \
    --hparams_dir "./hparams/R-ROME/gpt2-xl.yaml" \
    --overall "LGA"

# CounterFact
python counterfact_collect_all_layer_scores.py \
    --editing_method "R-ROME" \
    --hparams_dir "./hparams/R-ROME/gpt2-xl.yaml" \
    --overall "LGA"
```

### 2. WikiBio, WikiCounterfact and WikiBio Datasets Scores

```bash
python wiki_collect_all_layer_scores.py \
    --editing_method "R-ROME" \
    --hparams_dir "./hparams/R-ROME/gpt2-xl.yaml" \
    --data_dir ./data/knowedit \
    --datatype "recent" \
    --overall "LGA"
```

**`--datatype` options** (wiki_collect_all_layer_scores.py only):
- `recent` - WikiRecent
- `wikibio` - WikiBio
- `counterfact` - WikiCounterfact

### 3. Key Arguments

- **`--overall`**: Layer selection method to score the layers
  - `LGA` - Our method (default)
  - `CMA` - Causal Mediation Analysis baseline
<!-- - **`--model_name`**: Model identifier used in scripts
  - `gpt2-xl` - GPT2-XL (48 layers)
  - `llama2-7b` - LLaMA2-7B (32 or 64 layers, depending on script)
  - `gemma3-12b` - Gemma3-12B (48 layers) -->

## Collecting All Layers' Editing Performance

Collect editing performance for every layer of the model across datasets:

### 1. ZSRE Layer Performance

```bash
python zsre_collect_all_layer_performance.py \
    --editing_method "R-ROME" \
    --hparams_dir "./hparams/R-ROME/gpt2-xl.yaml" \
    --model_name "gpt2-xl"
```

### 2. CounterFact Layer Performance

```bash
python counterfact_collect_all_layer_performance.py \
    --editing_method "R-ROME" \
    --hparams_dir "./hparams/R-ROME/gpt2-xl.yaml" \
    --model_name "gpt2-xl"
```

### 3. Wiki Datasets Layer Performance

```bash
python wiki_collect_all_layer_performance.py \
    --editing_method "R-ROME" \
    --hparams_dir "./hparams/R-ROME/gpt2-xl.yaml" \
    --model_name "gpt2-xl" \
    --datatype "recent"
```

**Arguments:**
- **`--editing_method`**: Editing method (e.g., "R-ROME", "EMMET", "ROME")
- **`--hparams_dir`**: Path to hyperparameter YAML file
<!-- - **`--data_dir`**: Path to data directory -->
- **`--model_name`**: Model identifier (`gpt2-xl`, `llama2-7b`, `gemma3-12b`)
- **`--datatype`**: Dataset type for wiki script (`counterfact`: WikiCounterfact, `recent`: WikiRecent, `wikibio`: WikiBio)
<!-- - **`--run_count`**: Starting layer index for parallel processing (default: 0)
- **`--total_scripts`**: Number of parallel scripts running (default: 1) -->
<!-- - **`--metrics_save_dir`**: Override default save directory (optional) -->
<!-- - **`--ds_size`**: Limit dataset size (optional) -->

## Analysis

After collecting scores and layer performances, analyze the results:

```bash
cd cross_dataset_layer_independence/lga_analysis

# Analyze results for a model/dataset combination
python run_method_comparison.py \
    --model gpt2-xl \
    --dataset zsre \
    --top-n 19 \
    --baseline CausalTraceProba.last
```

**Arguments for [run_method_comparison.py](cross_dataset_layer_independence/lga_analysis/run_method_comparison.py):**
- **`--model`**: Model name
  - `gpt2-xl` - GPT2-XL
  - `llama2-7b` - LLaMA2-7B
  - `qwen3-32b` - Qwen3-32B
  - `gemma3-12b` - Gemma3-12B
  - `emmet` - EMMET
- **`--dataset`**: Dataset name
  - `zsre` - ZSRE
  - `counterfact` - CounterFact
  - `wikibio` - WikiBio
  - `wikirecent` - WikiRecent
  - `wikicounterfact` - WikiCounterFact
<!-- - **`--top-n`**: Number of top methods to display in results (default: 19) -->
<!-- - **`--baseline`**: Baseline method name for comparison (default: "CausalTraceProba.last") -->

See notebooks in `cross_dataset_layer_independence/lga_analysis/` for detailed analysis examples.

## Output

Layer scores are saved as `.pt` files in `LGA/lga_scores/`:
```
modelName-{model}_overall-{method}_layers-{n}_nsamples-{n}.pt
```
