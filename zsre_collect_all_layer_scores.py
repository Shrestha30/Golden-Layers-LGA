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

import argparse

# %%
sample_count = None

parser = argparse.ArgumentParser()
parser.add_argument('--editing_method', required=True, type=str)
parser.add_argument('--hparams_dir', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--ds_size', default=None, type=int)
parser.add_argument('--metrics_save_dir', default='./tempstorage', type=str)
parser.add_argument('--datatype', default=None,type=str)
parser.add_argument('--train_data_path', type=str)
parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
parser.add_argument('--overall', default="LGA", type=str, help='Overall metric to use')
parser.add_argument('--maxmode', required=True, type=str, help='Max mode: True or False')

args = parser.parse_args()

# Convert maxmode string to boolean
args.maxmode = args.maxmode.lower() == 'true'

if args.overall == "LGA":
    args.overall = "specific.OverallNewOverallHat"
else:
    args.overall = "CausalTraceProba.last"

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
# Load WikiBio dataset using KnowEditDataset
if sample_count is not None:
    datas = datas[:sample_count]
    print("Limiting the length of data to:", len(datas))
print("Total data length:", len(datas))
sample_count = len(datas)

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

editor = BaseEditor.from_hparams(hparams, localLayerIF=True, identityEnabled=True, queryLoss=False, maxMode=args.maxmode, norm=False, overall=args.overall, endAfterInfluence=True, use_lossBackward=True)

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    rephrase_prompts=rephrase_prompts,
    target_new=target_new,
    target_truth=actual_target,
    subject=subject,
    train_ds=train_ds,
    locality_inputs=locality_inputs,
    portability_inputs=portability_inputs,
    keep_original_weight=True
)

# No save as endafterinfluence is true