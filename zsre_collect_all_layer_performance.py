# %%
# KN, DINM, MEMIT, EMMET, and R-ROME
import os

# Use SLURM temporary directory (gets cleaned up automatically after job ends)
if 'SLURM_TMPDIR' in os.environ:
    cache_dir = os.path.join(os.environ['SLURM_TMPDIR'], 'hf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    print(f"Using temporary cache: {cache_dir}")
else:
    raise RuntimeError("SLURM_TMPDIR not set. This script must be run as a SLURM job.")

from huggingface_hub import login

# Load Hugging Face token from environment variable
hf_token = os.getenv('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it with your Hugging Face API token.")
login(token=hf_token)

# %%
# KN, DINM, MEMIT, EMMET, and R-ROME
import os.path
import sys
sys.path.append('..')
sys.path.append('cross_dataset_layer_independence')
import json
import random
from pathlib import Path
from easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    EMMETHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    R_ROMEHyperParams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset
from easyeditor import ZsreDataset
import numpy as np
from cross_dataset_layer_independence.dataset_model_registry import get_config

import argparse

# %%


parser = argparse.ArgumentParser()
parser.add_argument('--editing_method', required=True, type=str)
parser.add_argument('--hparams_dir', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--model_name', default='gpt2-xl', type=str, help='Model name: gpt2-xl, llama2-7b, gemma3-12b, rome, etc.')
parser.add_argument('--ds_size', default=None, type=int)
parser.add_argument('--metrics_save_dir', default=None, type=str, help='Override default save directory from registry')
parser.add_argument('--datatype', default=None,type=str)
parser.add_argument('--train_data_path', type=str)
parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
parser.add_argument('--run_count', default=0, type=int, help='Which script to run (0-5)')
parser.add_argument('--total_scripts', default=1, type=int, help='Total number of parallel scripts')

args = parser.parse_args()

# Get save directory from dataset_model_registry if not provided
if args.metrics_save_dir is None:
    try:
        # Get configuration from registry for zsre dataset
        config = get_config('zsre', args.model_name)
        args.metrics_save_dir = str(config['base_dir'] / config['data_dir'])
        print(f"Using save directory from registry: {args.metrics_save_dir}")
    except KeyError as e:
        print(f"Warning: Model '{args.model_name}' not found in registry for zsre dataset.")
        print(f"Falling back to default: ./allLayerPerformance")
        args.metrics_save_dir = './allLayerPerformance'

if args.editing_method == 'FT':
    editing_hparams = FTHyperParams
elif args.editing_method == 'IKE':
    editing_hparams = IKEHyperParams
elif args.editing_method == 'KN':
    editing_hparams = KNHyperParams
elif args.editing_method == 'MEMIT':
    editing_hparams = MEMITHyperParams
elif args.editing_method == 'ROME':
    editing_hparams = ROMEHyperParams
elif args.editing_method == 'LoRA':
    editing_hparams = LoRAHyperParams
elif args.editing_method == 'R-ROME':
    editing_hparams = R_ROMEHyperParams
elif args.editing_method == 'EMMET':
    editing_hparams = EMMETHyperParams
else:
    raise NotImplementedError

datas = json.load(open(os.path.join(args.data_dir, 'zsre_mend_eval_portability_gpt4.json'), 'r', encoding='utf-8'))
print(len(datas))
# print(datas.shape)
# datas = datas[:sample_count]
print(len(datas))

# %%
start, end = None, None
# datas = KnowEditDataset(args.data_dir,size=args.ds_size)
if (start is not None) and (end is not None):
    datas = datas[start:end]
else:
    start, end = 0, len(datas)
print(f"Processing data range: {start} to {end}")
print(f"Run count: {args.run_count}, Total scripts: {args.total_scripts}")

total_scripts = args.total_scripts
run_count = args.run_count

prompts = [test_data_['src'] for test_data_ in datas]
rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in datas]
target_new = [edit_data_['alt'] for edit_data_ in datas]
locality_prompts = [edit_data_['loc'] for edit_data_ in datas]
locality_ans = [edit_data_['loc_ans'] for edit_data_ in datas]
portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in datas]
portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in datas]
actual_target = [edit_data_['answers'][0] for edit_data_ in datas]

locality_inputs = {
    'neighborhood':{
        'prompt': locality_prompts,
        'ground_truth': locality_ans
    },
}
portability_inputs = {
    'one_hop':{
        'prompt': portability_prompts,
        'ground_truth': portability_ans
    },
}
subject = [edit_data_['subject'] for edit_data_ in datas]
hparams = editing_hparams.from_hparams(args.hparams_dir)

if args.editing_method == 'IKE':
    train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
    train_ds = ZsreDataset(train_data_path)
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts(sentence_model, train_ds, hparams)
else:
    train_ds = None

editor = BaseEditor.from_hparams(hparams, localLayerIF=False, identityEnabled=True, queryLoss=False, maxMode=False, norm = True, overall="finegrainedTOldTNew",
                                 rangeMode = "mid", allLayerScores=True)
import os
import json
from tqdm import tqdm

tot_layer_counts = {
    'gpt2-xl': 48,
    'llama2-7b': 64,
    'gemma3-12b': 48,
}

layer = run_count
pbar = tqdm(total=tot_layer_counts[args.model_name], desc="Editing layers")
while layer < tot_layer_counts[args.model_name]:
    save_path = os.path.join(
        args.metrics_save_dir,
        f"{args.editing_method}_results{start}-{end}_layer{layer}.json"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Skipping layers {layer} and {layer + 1} because file exists for layer {layer}: {save_path}")
        # Skip current and next layer by increasing layer by 2
        layer += total_scripts
        pbar.update(total_scripts)
        continue

    # Do the editing and saving
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        target_truth=actual_target,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True,
        predefinedLayer=[layer],
    )

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    pbar.update(total_scripts)
    layer += total_scripts

pbar.close()




