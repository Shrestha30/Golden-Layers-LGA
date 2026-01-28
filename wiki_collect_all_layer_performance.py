# %%
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
def eval(result_path):
    if path.exists(result_path):
        
        with open(result_path,'r') as file:
            datas=json.load(file)
        datas = datas[:-1]
        #data_rome_counterfact['post'].keys()  dict_keys(['rewrite_acc', 'locality', 'portability'])
        Edit_Succ_list=[data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ=sum(Edit_Succ_list)/len(Edit_Succ_list)*100
        print('Edit_Succ:',Edit_Succ)
        
        Portability_list=[]
        for data_rome_counterfact in datas:
            case_list=[]
            for key in data_rome_counterfact['post']['portability'].keys():
                case_list.append(sum(data_rome_counterfact['post']['portability'][key])/len(data_rome_counterfact['post']['portability'][key])*100)
            if len(case_list) != 0:
                Portability_list.append(np.mean(case_list))
        Overall_portability = np.mean(Portability_list)
        print('Overall_portability:',Overall_portability)

        Locality_list=[]
        for data_rome_counterfact in datas:
            case_list=[]
            for key in data_rome_counterfact['post']['locality'].keys():
                case_list.append(sum(data_rome_counterfact['post']['locality'][key])/len(data_rome_counterfact['post']['locality'][key])*100)
            if len(case_list) != 0:
                Locality_list.append(np.mean(case_list))
        Overall_locality = np.mean(Locality_list)
        print('Overall_locality:',Overall_locality)
        
        Fluency_list=[x['post']['fluency']['ngram_entropy'] for x in datas]
        Fluency=sum(Fluency_list)/len(Fluency_list)*100
        print('Fluency:',Fluency)

# %%
sample_count = 25

parser = argparse.ArgumentParser()
parser.add_argument('--editing_method', required=True, type=str)
parser.add_argument('--hparams_dir', required=True, type=str)
parser.add_argument('--data_dir', default=None, type=str, help='Override default data directory from registry')
parser.add_argument('--model_name', default='gpt2-xl', type=str, help='Model name: gpt2-xl, llama2-7b, gemma3-12b emmet, rome, etc.')
parser.add_argument('--ds_size', default=None, type=int)
parser.add_argument('--metrics_save_dir', default=None, type=str, help='Override default save directory from registry')
parser.add_argument('--datatype', required=True, type=str, help='Dataset type: counterfact, recent, zsre, wikibio')
parser.add_argument('--train_data_path', type=str)
parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
parser.add_argument('--run_count', default=0, type=int, help='Which script to run (0-5)')
parser.add_argument('--total_scripts', default=1, type=int, help='Total number of parallel scripts')

args = parser.parse_args()

# Map datatype argument to registry dataset name
datatype_to_dataset = {
    'counterfact': 'wikicounterfact',
    'recent': 'wikirecent',
    'wikibio': 'wikibio'
}

# Get dataset name for registry lookup
registry_dataset = datatype_to_dataset.get(args.datatype, args.datatype)

# Get paths from dataset_model_registry if not provided
if args.data_dir is None or args.metrics_save_dir is None:
    try:
        config = get_config(registry_dataset, args.model_name)

        if args.metrics_save_dir is None:
            args.metrics_save_dir = str(config['base_dir'] / config['data_dir'])
            print(f"Using save directory from registry: {args.metrics_save_dir}")

        # Set data_dir based on datatype (relative paths)
        if args.data_dir is None:
            project_root = Path(__file__).resolve().parent
            data_base = project_root / 'data' / 'knowedit'

            if args.datatype == 'counterfact':
                args.data_dir = str(data_base / 'wiki_counterfact' / 'test_cf.json')
            elif args.datatype == 'recent':
                args.data_dir = str(data_base / 'wiki_recent' / 'recent_test.json')
            elif args.datatype == 'zsre':
                args.data_dir = str(data_base / 'zsre' / 'zsre_test.json')
            elif args.datatype == 'wikibio':
                args.data_dir = str(data_base / 'WikiBio' / 'wikibio-test-all.json')

            print(f"Using data directory: {args.data_dir}")

    except KeyError as e:
        print(f"Warning: Model '{args.model_name}' not found in registry for {registry_dataset} dataset.")
        print(f"Falling back to defaults")

        if args.metrics_save_dir is None:
            args.metrics_save_dir = './allLayerPerformance'

        if args.data_dir is None:
            project_root = Path(__file__).resolve().parent
            data_base = project_root / 'data' / 'knowedit'

            if args.datatype == 'counterfact':
                args.data_dir = str(data_base / 'wiki_counterfact' / 'test_cf.json')
            elif args.datatype == 'recent':
                args.data_dir = str(data_base / 'wiki_recent' / 'recent_test.json')
            elif args.datatype == 'zsre':
                args.data_dir = str(data_base / 'zsre' / 'zsre_test.json')
            elif args.datatype == 'wikibio':
                args.data_dir = str(data_base / 'WikiBio' / 'wikibio-test-all.json')

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

# %%
start, end = None, None
datas = KnowEditDataset(args.data_dir,size=args.ds_size)
if (start is not None) and (end is not None):
    datas = datas[start:end]
else:
    start, end = 0, len(datas)
print(f"Processing data range: {start} to {end}")
print(f"Run count: {args.run_count}, Total scripts: {args.total_scripts}")

total_scripts = args.total_scripts
run_count = args.run_count

count_changed=0
for data in datas:
    if data['ground_truth'] is None:
        data['ground_truth'] = data['target_new']
        count_changed+=1
print(f"Number of ground truth entries changed from None to target_new: {count_changed}")

if args.datatype == 'counterfact' or args.datatype == 'recent' or args.datatype == 'zsre':
    prompts=[data['prompt'] for data in datas]
    subjects=[data['subject'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    actual_target = [data['ground_truth'] for data in datas]
    
    portability_r =[data['portability_r'] for data in datas]
    portability_s =[data['portability_s'] for data in datas]
    portability_l =[data['portability_l'] for data in datas]

    portability_reasoning_prompts=[]
    portability_reasoning_ans=[]
    portability_Logical_Generalization_prompts=[]
    portability_Logical_Generalization_ans=[]
    portability_Subject_Aliasing_prompts=[]
    portability_Subject_Aliasing_ans=[]
    
    portability_data = [portability_r,portability_s,portability_l]
    portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts]
    portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans]
    for data, portable_prompts, portable_answers in zip(portability_data,portability_prompts,portability_answers):
        for item in data:
            if item is None:
                portable_prompts.append(None)
                portable_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                portable_prompts.append(temp_prompts)
                portable_answers.append(temp_answers)
    assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
    
    locality_rs = [data['locality_rs'] for data in datas]
    locality_f = [data['locality_f'] for data in datas]
    locality_Relation_Specificity_prompts=[]
    locality_Relation_Specificity_ans=[]
    locality_Forgetfulness_prompts=[]        
    locality_Forgetfulness_ans=[]
    
    locality_data = [locality_rs, locality_f]
    locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
    locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
    for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
        for item in data:
            if item is None:
                local_prompts.append(None)
                local_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                local_prompts.append(temp_prompts)
                local_answers.append(temp_answers)
    assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
    locality_inputs = {}
    portability_inputs = {}
    
    locality_inputs = {
        'Relation_Specificity':{
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        },
        'Forgetfulness':{
            'prompt':locality_Forgetfulness_prompts,
            'ground_truth':locality_Forgetfulness_ans
        }
    }
    portability_inputs = {
        'Subject_Aliasing':{
            'prompt': portability_Subject_Aliasing_prompts,
            'ground_truth': portability_Subject_Aliasing_ans
        },
        'reasoning':{
            'prompt': portability_reasoning_prompts,
            'ground_truth': portability_reasoning_ans           
        },
        'Logical_Generalization':{
            'prompt': portability_Logical_Generalization_prompts,
            'ground_truth': portability_Logical_Generalization_ans           
        }
    }
if args.datatype == 'wikibio':
    prompts=[data['prompt'] for data in datas]
    subjects=[data['subject'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    actual_target = [data['ground_truth'] if data['ground_truth'] is not None else " " for data in datas]
    
    locality_rs = [data['locality_rs'] for data in datas]
    locality_f = [data['locality_f'] for data in datas]
    locality_Relation_Specificity_prompts=[]
    locality_Relation_Specificity_ans=[]
    
    locality_data = [locality_rs]
    locality_prompts = [locality_Relation_Specificity_prompts]
    locality_answers = [locality_Relation_Specificity_ans]
    for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
        for item in data:
            if item is None:
                local_prompts.append(None)
                local_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                local_prompts.append(temp_prompts)
                local_answers.append(temp_answers)
    assert len(prompts) == len(locality_Relation_Specificity_prompts)
    portability_inputs = None
    locality_inputs = {}
    locality_inputs = {
        'Relation_Specificity':{
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        }
    }
    
results_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_resultsFull_R-ROME_{args.datatype}.json')
# print(locality_inputs)
hparams = editing_hparams.from_hparams(args.hparams_dir)
args.pre_file = f"./{hparams.model_name.split('/')[-1]}_{args.datatype}_pre_edit.json"
print(args.pre_file)
if args.pre_file is not None and os.path.exists(args.pre_file):
    pre_edit = json.load(open(args.pre_file,'r'))
    assert len(pre_edit) == len(prompts)
else:
    pre_edit = None
if args.editing_method == 'IKE':
    train_ds = KnowEditDataset(args.train_data_path)
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts(sentence_model, train_ds, hparams)
elif args.editing_method == 'ICE':
    hparams.use_icl_examples = False
    train_ds = None
else:
    train_ds = None

editor = BaseEditor.from_hparams(hparams, dataset_name=args.datatype, localLayerIF=False, identityEnabled=True, queryLoss=False, maxMode=False, norm = True, overall="finegrainedTOldTNew",
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
        target_new=target_new,
        subject=subjects,
        target_truth=actual_target,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True,
        pre_file=args.pre_file,
        pre_edit=pre_edit,  # Generate fresh pre-edit metrics for each layer to avoid shared data structure issues
        test_generation=True,
        predefinedLayer=[layer],
    )

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    pbar.update(total_scripts)
    layer += total_scripts

pbar.close()




