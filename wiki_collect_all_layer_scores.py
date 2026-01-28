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

# %%
# Load WikiBio dataset using KnowEditDataset
datas = KnowEditDataset(args.data_dir, size=args.ds_size)
if sample_count is not None:
    datas = datas[:sample_count]
    print("Limiting the length of data to:", len(datas))
print("Total data length:", len(datas))
sample_count = len(datas)

replace_count=0
for data in datas:
    if data['ground_truth'] is None:
        data['ground_truth'] = data['target_new']
        replace_count+=1
print(f"Replaced {replace_count} None ground truth values with target_new.")

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

editor = BaseEditor.from_hparams(hparams, dataset_name=args.datatype, localLayerIF=True, identityEnabled=True, queryLoss=False, maxMode=args.maxmode, norm=False, overall=args.overall, endAfterInfluence=True, use_lossBackward=True)
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
    pre_edit=pre_edit,
    test_generation=True
)

# No save as endafterinfluence is true