from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
import copy
import torch
import numpy as np
import random
from pathlib import Path
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from transformers import LlamaTokenizer,PreTrainedTokenizerFast, LlamaTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from ..util.causal_tracing import plot_all_flow, ModelAndTokenizer, collect_embedding_std, plot_all_flow_last_subject
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric, compute_edit_quality_target_truth
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality
from sklearn.decomposition import IncrementalPCAxf
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from scipy import sparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams, localLayerIF=False, identityEnabled = True, queryLoss = True,
                     maxMode=True, norm = True, overall="False", layer_count=1, endAfterInfluence = False, rangeMode="all",
                     absolute_influence=False, influence_block = "mlp", allLayerScores = False, save_inf = True,
                     predefinedPath = None, dataset_name = "zsre", debugInf=False, use_lossBackward=False):
        return cls(hparams, localLayerIF=localLayerIF, identityEnabled = identityEnabled, queryLoss=queryLoss, maxMode=maxMode,
                   norm = norm, overall = overall, layer_count=layer_count, endAfterInfluence = endAfterInfluence, rangeMode=rangeMode,
                   absolute_influence=absolute_influence, influence_block=influence_block, allLayerScores = allLayerScores, save_inf = save_inf,
                   predefinedPath = predefinedPath, dataset_name = dataset_name, debugInf=debugInf, use_lossBackward=use_lossBackward)
    def __init__(self, hparams: HyperParams, localLayerIF=False, identityEnabled = True, queryLoss = True, maxMode=True, norm = True,
                 overall = "False", layer_count=1, endAfterInfluence = False, rangeMode="all", absolute_influence=False, influence_block = "mlp",
                 allLayerScores = False, save_inf = True, predefinedPath = None, dataset_name = "zsre", debugInf=False, use_lossBackward=False):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        self.localLayerIF = localLayerIF
        self.identityEnabled = identityEnabled
        self.queryLoss = queryLoss
        self.use_lossBackward = use_lossBackward
        self.maxMode = maxMode
        self.norm = norm
        self.overall = overall
        self.layer_count = layer_count
        self.endAfterInfluence = endAfterInfluence
        self.rangeMode = rangeMode
        self.absolute_influence = absolute_influence
        self.influence_block = influence_block
        self.allLayerScores = allLayerScores
        self.debugInf = debugInf
        self.prevProbGrad = []
        self.supportedLocalInfMethods = ['R-ROME', 'MEMIT', 'EMMET']
        self.save_inf = save_inf
        self.dataset_name = dataset_name
        self.attn_outputs = []
        self.mlp_outputs = []
        self.hooks_registered = False
        self.predefinedPath = predefinedPath
        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            
            # QLoRA configuration
            if hparams.alg_name == 'QLoRA':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(hparams.quantization_bit == 4),
                    bnb_4bit_use_double_quant=hparams.double_quant,
                    bnb_4bit_quant_type=hparams.quant_type,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "torch_dtype": torch_dtype,
                    "device_map": {'': hparams.device}
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "device_map": device_map
                }

            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'chatglm-api' in self.model_name.lower():
                self.model, self.tok = None, None
                self.hparams = hparams
                return
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs, trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                if 'chatglm2'in self.model_name.lower(): 
                    self.tok.unk_token_id = 64787
                else: 
                    self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen3' in self.model_name.lower():
                # Qwen3 uses same architecture as Qwen2, use bfloat16 for better performance
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16,
                    device_map=device_map
                )
                # Use official Qwen3 tokens: eos=<|im_end|>, pad=<|endoftext|>
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                # Ensure pad_token is set (Qwen3 uses <|endoftext|> for padding)
                if self.tok.pad_token is None:
                    self.tok.pad_token = self.tok.eos_token
            elif 'gemma3' in self.model_name.lower() or 'gemma-3' in self.model_name.lower():
                # Gemma3: Load vision-language model as text-only using Gemma3ForCausalLM (omits vision tower)
                from transformers import Gemma3ForCausalLM
                self.model = Gemma3ForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map
                )
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                # Ensure pad_token is set
                if self.tok.pad_token is None:
                    self.tok.pad_token = self.tok.eos_token
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit','CORE']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower() or 'gemma' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit', 'CORE']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel: 
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device') and hparams.alg_name != 'QLoRA':
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def initialize_PCA(self, n_components=50, batch_size=50):
        self.grad_pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.pca_grad_buffer = []
        self.pca_batch_size = batch_size
        self.grad_scaler = StandardScaler()

    def save_scaler(self):
        scaler_path = f'scaler_modelName-{self.model_name}_overall-{self.overall}.pt'
        joblib.dump(self.grad_scaler, scaler_path)
        print(f'Saved scaler to {scaler_path}')
        
    def load_scalar(self):
        scaler_path = f'scaler_modelName-{self.model_name}_overall-{self.overall}.pt'
        self.grad_scaler = joblib.load(scaler_path)
        print(f'Loaded scaler from {scaler_path}')
    
    def save_pca(self):
        scaler_path = f'pca_modelName-{self.model_name}_overall-{self.overall}.pt'
        joblib.dump(self.grad_pca, scaler_path)
        print(f'Saved PCA to {scaler_path}')
    
    def load_pca(self):
        scaler_path = f'pca_modelName-{self.model_name}_overall-{self.overall}.pt'
        self.grad_pca = joblib.load(scaler_path)
        print(f'Loaded PCA from {scaler_path}')

    def load_fitted_pca_rc_scaler(self):
        # Use environment variable if set, otherwise use relative path
            # Go up 3 levels to project root, then into LGA/models
        project_root = Path(__file__).resolve().parent.parent.parent
        data_dir = str(project_root / 'LGA' / 'models')

        grad_scaler_path = f"{data_dir}/scaler_modelName-gpt2-xl_overall-PCAScalarFit.origTarget.pt"
        grad_pca_path = f"{data_dir}/pca_modelName-gpt2-xl_overall-PCAPartialFit.origTarget.cpn50.pt"
        self.grad_scaler = joblib.load(grad_scaler_path)
        self.grad_pca = joblib.load(grad_pca_path)
        rf_scaler_path = f"{data_dir}/pcascaler.origTarget.joblib"
        rf_path = f"{data_dir}/random_forest.origTarget.joblib"
        self.rf_scaler = joblib.load(rf_scaler_path)
        self.rf_model = joblib.load(rf_path)

    def set_save_inf(self, overall, status = True):
        self.save_inf = status
        self.overall = overall

    def preprocess_model_for_influence(self, device=None):
        """
        Prepares the model for influence computation:
        - Moves to device
        - Sets eval mode
        - FREEZES all parameters (requires_grad=False) for memory efficiency

        NOTE: Gradients will be enabled selectively per-layer in grad_dict().
        This saves memory by not storing intermediate activations for frozen layers.

        Returns:
        - Dictionary mapping parameter names -> original requires_grad
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device).eval()

        prev_requires_grad = {}
        for name, p in self.model.named_parameters():
            prev_requires_grad[name] = p.requires_grad
            p.requires_grad_(False)  # Freeze all - will enable selectively per layer

        return prev_requires_grad


    def revert_model_after_influence(self, prev_requires_grad, set_eval=True):
        """
        Restores model state after influence computation:
        - Restores original requires_grad flags
        - Optionally sets model back to eval mode
        """
        for name, p in self.model.named_parameters():
            if name in prev_requires_grad:
                p.requires_grad_(prev_requires_grad[name])
        
        if self.hparams.model_parallel: 
            self.hparams.device = str(self.model.device).split(":")[1]
        if not self.hparams.model_parallel and hasattr(self.hparams, 'device') and self.hparams.alg_name != 'QLoRA':
            self.model.to(f'cuda:{self.hparams.device}')

        if set_eval:
            self.model.eval()


    def normalize_influences(self):
        if isinstance(self.influences, list):
            self.influences = torch.stack(self.influences, dim=0)

        # -----------------------------
        # 2. Normalize each row (per sample across layers)
        # -----------------------------
        def normalize_rows(x: torch.Tensor) -> torch.Tensor:
            """
            Normalizes each row of a [num_samples, num_layers] tensor
            so that values are between 0 and 1.
            """
            x_min, _ = x.min(dim=1, keepdim=True)
            x_max, _ = x.max(dim=1, keepdim=True)
            return (x - x_min) / (x_max - x_min + 1e-8)
        
        if self.norm:
            self.influences_norm = normalize_rows(self.influences)
        else:
            self.influences_norm = self.influences
    
    def get_max_influence_layer_for_sample(self, sample_idx: int):
        """
        Given a sample index, return the indices of the top-k layers 
        with the highest (or lowest, if maxMode=False) normalized influence.
        Returns:
            topk_layers (List[int]): list of layer indices sorted by influence strength.
            topk_values (List[float]): corresponding influence values.
        """
        # for predefined:
        import pickle

        def load_results_pickle(filepath="HSminPoolMLPContinuous.pkl"):
            with open(filepath, "rb") as f:
                return pickle.load(f)

        def build_index(results):
            """
            Converts list → dict for O(1) lookups.
            Example: index[17] = {"sample_id":17, "best_layer":..., "scores":[...]}
            """
            return {entry["sample_id"]: entry for entry in results}


        def get_best_layer_fast(sample_id, index_dict):
            if sample_id not in index_dict:
                raise KeyError(f"sample_id {sample_id} not found.")

            return index_dict[sample_id]["best_layer"]
        
        def get_best_layer(sample_id, results):
            for entry in results:
                if entry["sample_id"] == sample_id:
                    return entry["best_layer"]
            raise KeyError(f"sample_id {sample_id} not found.")

        if self.predefinedPath is not None:
            results = load_results_pickle(self.predefinedPath)
            print("Using predefined results of total samples: ", len(results))
            best_layer = get_best_layer(sample_idx, results)

            # For HiddenStates with 49 dims, layer indices are 1-based (index 0 is embedding)
            # Need to add 1 to convert from performance layer index to HiddenStates index
            if "HiddenStates" in self.overall and any(x in self.overall for x in ["HSHs", "HSDiffHs"]):
                # Check if influences have 49 layers (embedding + 48 transformer layers)
                if hasattr(self, "influences") and self.influences.shape[1] == 49:
                    best_layer = best_layer + 1  # Convert to 1-based index
                    print(f"HiddenStates mode: Adjusted layer {best_layer-1} -> {best_layer} (1-based for 49-dim)")

            return [best_layer], [1]

        if not hasattr(self, "influences_norm"):
            raise ValueError("Influences not yet normalized. Call self.normalize_influences() first.")

        # Get the row corresponding to the sample
        if self.absolute_influence:
            sample_influences = torch.abs(self.influences_norm[sample_idx])  # shape: [num_layers]
        else:
            sample_influences = self.influences_norm[sample_idx]  # shape: [num_layers]

        # For HiddenStates with 49 dims, we need to skip index 0 (embedding layer)
        # The performance metrics use 0-47 for the 48 transformer layers
        # But HiddenStates indices are 1-48 for the same layers (0 is embedding)
        hs_offset = 0
        if "HiddenStates" in self.overall and any(x in self.overall for x in ["HSHs", "HSDiffHs"]):
            if sample_influences.shape[0] == 49:
                # Skip the first element (embedding layer)
                sample_influences = sample_influences[1:]
                hs_offset = 1
                print(f"HiddenStates mode: Using layers 1-48 from 49-dim tensor (skipping embedding at index 0)")

        # Get top-k layers based on mode
        if self.rangeMode == "mid":
            # define layer index range (e.g., 20% to 75%)
            start_idx = int(0.20 * self.total_layers)
            end_idx = int(0.75 * self.total_layers)

            # restrict to mid-range layers
            mid_range_influences = sample_influences[start_idx:end_idx]
            print("the length k is: ", len(mid_range_influences))

            # compute top-k *within* this range
            if self.maxMode:
                topk_values, topk_indices_local = torch.topk(mid_range_influences, self.layer_count)
            else:
                topk_values, topk_indices_local = torch.topk(-mid_range_influences, self.layer_count)
                topk_values = -topk_values  # restore sign

            # adjust indices back to global layer indices
            topk_indices = topk_indices_local + start_idx

        else:
            # Original behavior: use all layers
            if self.maxMode:
                topk_values, topk_indices = torch.topk(sample_influences, self.layer_count)
            else:
                topk_values, topk_indices = torch.topk(-sample_influences, self.layer_count)
                topk_values = -topk_values  # restore sign


        # Convert to Python lists
        topk_layers = topk_indices.cpu().tolist()
        topk_values = topk_values.cpu().tolist()

        # Add back the offset for HiddenStates
        if hs_offset > 0:
            topk_layers = [layer + hs_offset for layer in topk_layers]
            print(f"HiddenStates mode: Adjusted selected layers by +{hs_offset}: {topk_layers}")

        # ---- sort top-k by ascending index ----
        # sorted_vals, sorted_idx = zip(*sorted(zip(topk_indices.tolist(), topk_values.tolist())))
        # topk_layers = list(sorted_vals)
        # topk_values = list(sorted_idx)

        return topk_layers, topk_values


    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             target_truth: Optional[Union[str, List[str]]] = None,
             target_neg: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             sequential_edit=False,
             verbose=True,
             predefinedLayer = [0],
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)
        self.predefinedLayer = predefinedLayer

        if self.allLayerScores:
            self.hparams.layers = self.predefinedLayer

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
            subjects = kwargs.get('subject', None)
            if "CausalTraceProba" in self.overall:
                print("CausalTraceProba requires subject inputs checking...")
                assert subjects is not None, "CausalTraceProba requires subject inputs."
                assert len(subjects) == len(prompts), "Length of subjects must match length of prompts."
        else:
            prompts, target_new, subjects = [prompts,], [target_new,], [kwargs.get('subject', None),]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(self.alg_name):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
        else:# Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>'] * (len(prompts))
            
        if target_truth is not None:
            target_truth = [target_truth,] if isinstance(target_truth, str) else target_truth
        else:# Default ground truth is <|endoftext|>
            target_truth = ['dummy'] * (len(prompts))

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, subjects, target_new, ground_truth, target_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        self.influences = None
        if self.localLayerIF == True and self.alg_name in self.supportedLocalInfMethods:
            if self.hparams.model_name == 'gpt2-xl':
                self.total_layers = 48 #TODO: dynamically set for each model, now set for gpt2-xl
            elif 'llama' in self.hparams.model_name.lower() or 'gemma' in self.hparams.model_name.lower() or 'qwen' in self.hparams.model_name.lower():
                self.total_layers = self.model.config.num_hidden_layers
                print("Total layers set to: ", self.total_layers)
            total_layers = self.total_layers
            num_samples = len(requests)
            cache_path = self._build_cache_path(num_samples, total_layers)

            prev_requires_grad = self.preprocess_model_for_influence()
            # influences = torch.tensor([get_prompt_to_attribute_influence(mt, i, 0, knowns[:500]) for i in range(48)])
            # Try to load from cache first
            if os.path.exists(cache_path) and self.debugInf==False:
                if "SparseCompleteFit" not in self.overall:
                    print(f"[CACHE] Loading cached influences from {cache_path}")
                    self.influences = torch.load(cache_path)
            else:
                if self.predefinedPath is not None:
                    print("[Cache] no cache needed. Will be using predefined path.")
                    return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, **kwargs)
                
                print(f"[CACHE] No cached influences from {cache_path}")
                if "SparseCompleteFit" in self.overall:
                    self.influences = [
                        self.get_prompt_to_attribute_influence(layer, requests)
                        for layer in tqdm(range(total_layers), desc="Computing influences per layer")
                    ]
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.influences, f)
                    print(f"[CACHE] Saved influences to {cache_path}")
                    return None, None, None
                elif "HiddenStates" in self.overall:
                    # Check if we're using a specific subcategory that returns layer scores
                    if any(x in self.overall for x in ["HSMlp", "HSAttn", "HSHs", 
                                                        "HSDiffMlp", "HSDiffAttn", "HSDiffHs"]):
                        # These subcategories return tensors of shape [num_layers] per sample
                        # We need to stack them into shape [num_samples, num_layers]
                        influences_list = self.get_prompt_to_attribute_influence(0, requests)
                        # influences_list is a list of tensors, each of shape [num_layers]
                        self.influences = torch.stack(influences_list, dim=0)  # shape: [num_samples, num_layers]
                        print(f"Stacked influences shape: {self.influences.shape}")
                    else:
                        # Original HiddenStates behavior - just save and return
                        self.influences = self.get_prompt_to_attribute_influence(0, requests)
                        torch.save(self.influences, cache_path)
                        print(f"Saved {len(self.influences)} samples to {cache_path}")
                        print(f"[CACHE] Saved influences to {cache_path}")
                        return None, None, None
                elif "CausalTraceProba" in self.overall:
                    self.influences = self.get_prompt_to_attribute_influence(0, requests)

                if self.influences is None:
                    self.influences = torch.stack([
                        self.get_prompt_to_attribute_influence(layer, requests)
                        for layer in tqdm(range(total_layers), desc="Computing influences per layer")
                    ], dim=1)  # shape: [num_samples, num_layers]

                # Save computed influences for future use
                if self.save_inf and self.debugInf==False:
                    if isinstance(self.influences, torch.Tensor):
                        print("Shaping influences for saving: ", self.influences.shape)
                    elif isinstance(self.influences, list):
                        # list of tensors: print lengths and individual shapes
                        print("Shaping influences for saving: list of", len(self.influences), "samples")
                        print("Example shapes:", [t.shape for t in self.influences[:5]])  # first 5 for brevity
                    else:
                        print("Unknown type for influences:", type(self.influences))

                    torch.save(self.influences, cache_path)
                    print(f"[CACHE] Saved influences to {cache_path}")
                else: print(f"[CACHE] Did not save influences as save_inf is set to False or debugInf is True.")


            if "CausalTraceProba.default" in self.overall:
                print("Preprocessing influences for Causal Trace Proba...")
                print("Pre len influences: ", len(self.influences), " shape[0]: ", self.influences[0].shape)
                # self.influences: list of tensors, each shape (n_tokens, n_layers)
                collapsed = []

                for sample in self.influences:
                    # sum across tokens (dim=0) → shape (n_layers,)
                    collapsed.append(sample.sum(dim=0))

                # stack all samples → shape (n_samples, n_layers)
                self.influences = torch.stack(collapsed)  # shape: (n_samples, n_layers)
                print("Post shape influences: ", self.influences.shape)
            if "CausalTraceProba.last" in self.overall:
                self.influences = torch.stack(self.influences)
                print("Post shape influences: ", self.influences.shape)

            self.normalize_influences()
            self.revert_model_after_influence(prev_requires_grad=prev_requires_grad, set_eval=True)

            if self.endAfterInfluence:
                return None, None, None

        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, **kwargs)
    
    def _get_cache_dir(self):
        """Return (and create if missing) the default cache directory."""
        # Use environment variable if set, otherwise use relative path
        # This file is at: EasyEditEdited/easyeditor/editors/editor.py
        # Go up 3 levels to project root, then into LGA/lga_scores
        project_root = Path(__file__).resolve().parent.parent.parent
        base_dir = str(project_root / 'LGA' / 'lga_scores')
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _build_cache_path(self, num_samples, num_layers):
        """Build descriptive cache file path from configuration."""
        # Only include dataset_name in filename if it's not the default "zsre"
        dataset_part = "" if self.dataset_name == "zsre" else f"dataset-{self.dataset_name}_"
        
        if self.influence_block == "mlp":
            filename = (
                f"modelName-{self.model_name}_"
                f"{dataset_part}"
                f"localIF-{self.localLayerIF}_iden-{self.identityEnabled}_"
                f"qLoss-{self.queryLoss}_norm-{self.norm}_"
                f"overall-{self.overall}_layers-{num_layers}_"
                f"_nsamples-{num_samples}.pt"
            )
        else:
            filename = (
                f"modelName-{self.model_name}_"
                f"{dataset_part}"
                f"localIF-{self.localLayerIF}_iden-{self.identityEnabled}_"
                f"qLoss-{self.queryLoss}_norm-{self.norm}_"
                f"overall-{self.overall}_layers-{num_layers}_"
                f"_nsamples-{num_samples}_{self.influence_block}.pt"
            )
        cache_path = os.path.join(self._get_cache_dir(), filename)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        return cache_path

    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   target_neg: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }

                chunk_metrics.append(metrics)

            if sequential_edit:
                self.model = edited_model
            else:
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)

                if 'locality' in chunk_metrics[i]['post'].keys():
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        if hasattr(self.hparams, 'evaluation_type') and self.hparams.evaluation_type == "LLM-judge":
                            locality_result.append(float(chunk_metrics[i]['post']['locality'][f'{locality_key}_output']==chunk_metrics[i]['pre']['locality'][f'{locality_key}_output']))
                        else:
                            for ans, label in zip(chunk_metrics[i]['post']['locality'][f'{locality_key}_output'], chunk_metrics[i]['pre']['locality'][f'{locality_key}_output']):
                                locality_result.append(np.mean(np.equal(ans, label)))
                        chunk_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        chunk_metrics[i]['post']['locality'].pop(f'{locality_key}_output')
                    chunk_metrics[i]['pre'].pop('locality')

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy

    from collections import defaultdict
    import torch

    def get_prompt_to_attribute_influence(self, layer, requests, lambda_const_param=10):
        from IF import IFEngineGeneration as IFEngine  # assuming IFEngine code is in IFEngine.py
        print("Computing for layer: ", layer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(device).train()
        
        def make_prompt_and_attribute_sample_orig(input_ids, enc_prompt, attn_mask):
            # ---- Generate model predicted tokens for "original target" ----
            with torch.no_grad():
                # Length of target part
                seq_len = (input_ids != -100).sum().item() - len(enc_prompt['input_ids'])
                query_len = len(enc_prompt['input_ids'])

                # Handle case where target_truth is missing (WikiBio doesn't have ground_truth)
                # In this case, seq_len will be 0 or negative
                if seq_len <= 0:
                    # Use a reasonable default (generate up to 5 tokens)
                    print("Token generation length <= 0, using default seq_len=5")
                    seq_len = 5

                generated = self.model.generate(
                    input_ids=input_ids[:, :query_len],
                    attention_mask=attn_mask[:, :query_len],
                    max_new_tokens=seq_len,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                    use_cache=False,
                )

            # predicted tokens after prompt prefix
            y_hat = generated[:, query_len:]

            # build input_ids_pred = prompt + predicted
            input_ids_pred = torch.cat([input_ids[:, :query_len], y_hat], dim=1)

            # Build attention mask
            attn_mask_pred = torch.ones_like(input_ids_pred)

            # Labels for test_pred → mask prompt, only supervise predicted tokens
            labels_test_pred = input_ids_pred.clone()
            labels_test_pred[0, :query_len] = -100
            z_test_orig = {"input_ids": input_ids_pred, "labels": labels_test_pred, "attention_mask": attn_mask_pred, "propmt_len": query_len}

            # Labels for overall_pred → supervise entire generated sequence
            labels_overall_pred = input_ids_pred.clone()
            z_overall_orig = {"input_ids": input_ids_pred, "labels": labels_overall_pred, "attention_mask": attn_mask_pred, "propmt_len": query_len}
            return z_test_orig, z_overall_orig


        def make_prompt_and_attribute_samples(known):
            prompt, target = known['prompt'], known['target_truth']
            target_new = known['target_new']
            
            text = prompt + " " + target
            text_new = prompt + " " + target_new
            
            enc_all = self.tok(text, add_special_tokens=False)
            enc_all_new = self.tok(text_new, add_special_tokens=False)
            
            enc_prompt = self.tok(prompt, add_special_tokens=False)
            input_ids = torch.tensor([enc_all['input_ids']], dtype=torch.long, device=device)
            input_ids_new = torch.tensor([enc_all_new['input_ids']], dtype=torch.long, device=device)
            attn_mask = torch.tensor([enc_all["attention_mask"]], dtype=torch.long, device=device)
            
            if  self.overall == "finegrainedTOrig" or self.overall == "finegrainedTOrigTNew":
                z_test_orig, z_overall_orig = make_prompt_and_attribute_sample_orig(input_ids, enc_prompt, attn_mask)
                # TEST: attribute only (Target Loss)
                labels_test = input_ids_new.clone()
                labels_test[0, :len(enc_prompt['input_ids'])] = -100
                z_test_new = {"input_ids": input_ids_new, "labels": labels_test, "attention_mask": attn_mask, "propmt_len": len(enc_prompt['input_ids'])}
                
                return None, None, None, z_test_new, None, z_test_orig, z_overall_orig
            
            # TRAIN: prompt only (Query Loss)
            labels_train = input_ids.clone()
            labels_train[0, len(enc_prompt['input_ids']):] = -100
            z_train = {"input_ids": input_ids, "labels": labels_train, "attention_mask": attn_mask, "propmt_len": len(enc_prompt['input_ids'])}

            # TEST: attribute only (Target Loss)
            labels_test = input_ids.clone()
            labels_test[0, :len(enc_prompt['input_ids'])] = -100
            z_test = {"input_ids": input_ids, "labels": labels_test, "attention_mask": attn_mask, "propmt_len": len(enc_prompt['input_ids'])}

            # TEST: attribute only (Target Loss)
            labels_test = input_ids_new.clone()
            labels_test[0, :len(enc_prompt['input_ids'])] = -100
            z_test_new = {"input_ids": input_ids_new, "labels": labels_test, "attention_mask": attn_mask, "propmt_len": len(enc_prompt['input_ids'])}
            
            #overall loss
            labels_test = input_ids.clone()
            z_overall = {"input_ids": input_ids, "labels": labels_test, "attention_mask": attn_mask, "propmt_len": len(enc_prompt['input_ids'])}
            
            #overall with target_new loss
            labels_test = input_ids_new.clone()
            z_overall_new = {"input_ids": input_ids_new, "labels": labels_test, "attention_mask": attn_mask, "propmt_len": len(enc_prompt['input_ids'])}
            
            # Optimize: avoid computing orig variants unless needed by current mode
            if ("specific" in self.overall or "specificl2norm" in self.overall):
                needs_orig = ("THAT" in self.overall) or ("OverallHat" in self.overall)
                if needs_orig:
                    z_test_orig, z_overall_orig = make_prompt_and_attribute_sample_orig(input_ids, enc_prompt, attn_mask)
                else:
                    z_test_orig, z_overall_orig = None, None
            else:
                z_test_orig, z_overall_orig = make_prompt_and_attribute_sample_orig(input_ids, enc_prompt, attn_mask)

            return z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig

        def safe_dot_product(v1, v2=None):
            """
            Compute dot product in float64 for maximum precision and to avoid overflow.

            Args:
                v1: First vector (or only vector for v1·v1)
                v2: Second vector (optional, if None computes v1·v1)

            Returns:
                Dot product as a scalar tensor (0-D tensor) in float64.
                Always uses float64 regardless of input dtype to prevent overflow
                (float16 max ~65k, float32 max ~3.4e38, float64 max ~1.8e308).
            """
            # Always convert to float64 for accurate computation
            v1_64 = v1.to(torch.float64)

            if v2 is None:
                # Compute v1·v1
                result = torch.dot(v1_64, v1_64)
            else:
                # Compute v1·v2
                v2_64 = v2.to(torch.float64)
                result = torch.dot(v1_64, v2_64)

            return result

        def debug_influence(influence, method_name, v=None, g=None, sample=None, v1=None, v2=None, g1=None, g2=None):
            """
            Debug helper to print detailed information when influence is NaN or Inf.

            Args:
                influence: The computed influence value
                method_name: Name of the method (e.g., "Query", "TTruth", "THAT")
                v: Single gradient vector (for single-gradient methods)
                g: Single gradient dict (for single-gradient methods)
                sample: The input sample dict
                v1, v2: Two gradient vectors (for two-gradient methods)
                g1, g2: Two gradient dicts (for two-gradient methods)
            """
            
            # if v is not None:
            #     influenceNotSafe = torch.dot(v, v)
            # elif v1 is not None and v2 is not None:
            #     influenceNotSafe = torch.dot(v1, v2)
            
            if not (torch.isinf(influence).any() or torch.isnan(influence).any()): #or torch.isinf(influenceNotSafe).any() or torch.isnan(influenceNotSafe).any()):
                return

            print(f"\n{'='*60}")
            print(f"DEBUG: Influence is {influence.item()} for layer {layer}")
            print(f"  Method: {method_name}")

            # if v is not None:            
            #     acc = torch.zeros((), dtype=torch.float32, device=v.device)
            #     for i in range(v.numel()):
            #         product = v[i] * v[i]
            #         acc += product
                    
            #         # Print periodically to see where things blow up
            #         if i % 1_000_000 == 0:
            #             print(f"i={i}, product={product}, acc={acc}")
            
            if v1 is not None and g1 is not None:
                v=v1
                g=g1
                print("  (===============Using first gradient vector/dict for single-gradient debug info)")
            # For single gradient methods
            if v is not None and g is not None:
                print(f"  Gradient dict keys: {list(g.keys())}")
                print(f"  Concatenated gradient vector stats:")
                print(f"    shape: {v.shape}")
                print(f"    mean: {v.mean().item():.6e}")
                print(f"    std: {v.std().item():.6e}")
                print(f"    min: {v.min().item():.6e}")
                print(f"    max: {v.max().item():.6e}")
                print(f"    has_nan: {torch.isnan(v).any()}")
                print(f"    has_inf: {torch.isinf(v).any()}")
                print(f"    all_zero: {(v == 0).all()}")
                # Check for very large values that might cause overflow
                abs_v = torch.abs(v)
                print(f"    |v| max: {abs_v.max().item():.6e}")
                print(f"    |v| > 1e6 count: {(abs_v > 1e6).sum().item()}")
                print(f"    |v| > 1e10 count: {(abs_v > 1e10).sum().item()}")
                print(f"    dot product (v·v): {torch.dot(v, v).item()}")
                # Try alternative calculations to diagnose
                print(f"    sum(v*v): {torch.sum(v * v).item()}")
                print(f"    norm^2: {(torch.norm(v)**2).item()}")
                # Try with float64 to see if it helps
                v_64 = v.to(torch.float64)
                dot_64 = torch.dot(v_64, v_64).item()
                print(f"    dot product in float64: {dot_64:.6e}")
                print(f"    dot64 converted to float32: {torch.tensor(dot_64, dtype=torch.float32).item()}")


            if v2 is not None and g2 is not None:
                v=v2
                g=g2
                print("  (===============Using second gradient vector/dict for single-gradient debug info)")
            # For single gradient methods
            if v is not None and g is not None:
                print(f"  Gradient dict keys: {list(g.keys())}")
                print(f"  Concatenated gradient vector stats:")
                print(f"    shape: {v.shape}")
                print(f"    mean: {v.mean().item():.6e}")
                print(f"    std: {v.std().item():.6e}")
                print(f"    min: {v.min().item():.6e}")
                print(f"    max: {v.max().item():.6e}")
                print(f"    has_nan: {torch.isnan(v).any()}")
                print(f"    has_inf: {torch.isinf(v).any()}")
                print(f"    all_zero: {(v == 0).all()}")
                # Check for very large values that might cause overflow
                abs_v = torch.abs(v)
                print(f"    |v| max: {abs_v.max().item():.6e}")
                print(f"    |v| > 1e6 count: {(abs_v > 1e6).sum().item()}")
                print(f"    |v| > 1e10 count: {(abs_v > 1e10).sum().item()}")
                print(f"    dot product (v·v): {torch.dot(v, v).item()}")
                # Try alternative calculations to diagnose
                print(f"    sum(v*v): {torch.sum(v * v).item()}")
                print(f"    norm^2: {(torch.norm(v)**2).item()}")
                # Try with float64 to see if it helps
                v_64 = v.to(torch.float64)
                dot_64 = torch.dot(v_64, v_64).item()
                print(f"    dot product in float64: {dot_64:.6e}")
                print(f"    dot64 converted to float32: {torch.tensor(dot_64, dtype=torch.float32).item()}")
                
            
            # if v1 is not None:            
            #     acc = torch.zeros((), dtype=torch.float32, device=v.device)
            #     for i in range(v1.numel()):
            #         product = v1[i] * v2[i]
            #         acc += product
                    
            #         # Print periodically to see where things blow up
            #         if i % 1_000_000 == 0:
            #             print(f"i={i}, product={product}, acc={acc}")
            # For two gradient methods
            if v1 is not None and v2 is not None:
                print(f"  Gradient vector 1 stats:")
                print(f"    shape: {v1.shape}, mean: {v1.mean().item():.6e}, std: {v1.std().item():.6e}")
                print(f"    has_nan: {torch.isnan(v1).any()}, has_inf: {torch.isinf(v1).any()}, all_zero: {(v1 == 0).all()}")
                print(f"  Gradient vector 2 stats:")
                print(f"    shape: {v2.shape}, mean: {v2.mean().item():.6e}, std: {v2.std().item():.6e}")
                print(f"    has_nan: {torch.isnan(v2).any()}, has_inf: {torch.isinf(v2).any()}, all_zero: {(v2 == 0).all()}")
                print(f"  Dot product (v1·v2): {torch.dot(v1, v2).item()}")

            # Sample info if provided
            if sample is not None:
                print(f"  Sample info:")
                print(f"    input_ids shape: {sample['input_ids'].shape}")
                print(f"    labels shape: {sample['labels'].shape}")
                supervised = (sample['labels'] != -100).sum().item()
                total_tokens = sample['labels'].numel()
                print(f"    supervised tokens: {supervised}/{total_tokens}")

            print(f"{'='*60}\n")

        def get_block_layer_for_layer(target_layer):
            """Get the block layer module for a given layer index."""
            if "llama" in self.hparams.model_name.lower() or "gemma" in self.hparams.model_name.lower() or "qwen" in self.hparams.model_name.lower():
                layers = self.model.model.layers
                if self.influence_block == "mlp":
                    return layers[target_layer].mlp
                elif self.influence_block == "attn":
                    return layers[target_layer].self_attn
                elif self.influence_block == "block":
                    return layers[target_layer]
                elif self.influence_block == "mlpLast":
                    return layers[target_layer].mlp.down_proj
                else:
                    raise NotImplementedError
            else:
                # For GPT-2 and others
                if self.influence_block == "mlp":
                    return self.model.transformer.h[target_layer].mlp
                elif self.influence_block == "attn":
                    return self.model.transformer.h[target_layer].attn
                elif self.influence_block == "block":
                    return self.model.transformer.h[target_layer]
                elif self.influence_block == "mlpLast":
                    return self.model.transformer.h[target_layer].mlp.c_proj
                else:
                    raise NotImplementedError

        def enable_layer_gradients(target_layer):
            """Enable gradients only for target layer's block parameters. Stores params in self._current_layer_params."""
            block_layer = get_block_layer_for_layer(target_layer)
            for p in block_layer.parameters():
                p.requires_grad_(True)
            self._current_layer_params = list(block_layer.parameters())
            return self._current_layer_params

        def disable_layer_gradients(target_layer):
            """Disable gradients for target layer's block parameters."""
            block_layer = get_block_layer_for_layer(target_layer)
            for p in block_layer.parameters():
                p.requires_grad_(False)
            self._current_layer_params = None

        def grad_dict(sample):
            """Compute gradients for a sample. Params must be pre-enabled via enable_layer_gradients."""
            # Use pre-enabled params from enable_layer_gradients (for specific/l2normspecific modes)
            # or enable them here for backward compatibility with other modes
            if hasattr(self, '_current_layer_params') and self._current_layer_params is not None:
                params = self._current_layer_params
            else:
                # Backward compatibility: enable gradients for this call only
                params = enable_layer_gradients(layer)
                self._grad_dict_enabled_grads = True  # Flag to disable after this call

            if "llama" in self.hparams.model_name.lower() or "gemma" in self.hparams.model_name.lower() or "qwen" in self.hparams.model_name.lower():
                layers = self.model.model.layers
                block_layer = get_block_layer_for_layer(layer)

                if "HiddenStates" in self.overall:
                    # --- HOOK FUNCTIONS ---
                    def attn_hook(module, input, output):
                        self.attn_outputs.append(output)

                    def mlp_hook(module, input, output):
                        self.mlp_outputs.append(output)

                    # --- REGISTER HOOKS ACROSS ALL LAYERS ---
                    attn_hooks = []
                    mlp_hooks = []
                    for lay in layers:
                        h1 = lay.self_attn.register_forward_hook(attn_hook)
                        h2 = lay.mlp.register_forward_hook(mlp_hook)
                        attn_hooks.append(h1)
                        mlp_hooks.append(h2)
            else:
                # For GPT-2 and others using AutoModelForCausalLM
                block_layer = get_block_layer_for_layer(layer)

                if "HiddenStates" in self.overall:
                    def attn_hook(module, input, output):
                        self.attn_outputs.append(output)
                    def mlp_hook(module, input, output):
                        self.mlp_outputs.append(output)

                    attn_hooks = []
                    mlp_hooks = []

                    if "llama" in self.hparams.model_name.lower() or "gemma" in self.hparams.model_name.lower() or "qwen" in self.hparams.model_name.lower():
                        for layerr in self.model.model.layers:
                            h1 = layerr.self_attn.register_forward_hook(attn_hook)
                            h2 = layerr.mlp.register_forward_hook(mlp_hook)
                            attn_hooks.append(h1)
                            mlp_hooks.append(h2)
                    else:
                        for layerr in self.model.transformer.h:
                            h1 = layerr.attn.register_forward_hook(attn_hook)
                            h2 = layerr.mlp.register_forward_hook(mlp_hook)
                            attn_hooks.append(h1)
                            mlp_hooks.append(h2)

            # Clear any existing gradients to prevent leakage between samples
            self.model.zero_grad()
            
            # TODO: Uncomment if memory errors occur - but note this is expensive when called per sample
            # Memory optimization: clear CUDA cache before gradient computation
            # if ("llama" in self.hparams.model_name.lower() or "gemma" in self.hparams.model_name.lower() or "qwen" in self.hparams.model_name.lower()) and "specific" in self.overall.lower():
            #     torch.cuda.empty_cache()
            
            with torch.enable_grad():
                out = self.model(input_ids=sample["input_ids"], labels=sample["labels"],
                                 output_hidden_states=True)
                loss = out.loss
                
                if "HiddenStates" in self.overall:
                    record = {
                            "prompt_len": sample["propmt_len"],

                            # move all hidden states to CPU
                            "hidden_states": [hs.detach().cpu() for hs in out.hidden_states],

                            # attention outputs: some are tuples — you used hs[0]
                            "attn_outputs": [
                                (hs[0].detach().cpu() if isinstance(hs, tuple) else hs.detach().cpu())
                                for hs in self.attn_outputs
                            ],

                            # mlp outputs
                            "mlp_outputs": [hs.detach().cpu() for hs in self.mlp_outputs],
                        }
                    # print("len of mlp outputs: ", len(record["mlp_outputs"]))
                    # print("len of attn outputs: ", len(record["attn_outputs"]))
                    # print("len of hidden states: ", len(record["hidden_states"]))
                    # print("shape of one hidden state: ", record["hidden_states"][0].shape)
                    # print("shape of one attn output: ", record["attn_outputs"][0].shape)
                    # print("shape of one mlp output: ", record["mlp_outputs"][0].shape)
                    def tensor_size_mb(t):
                        """Return memory size of a tensor in MB."""
                        return t.element_size() * t.nelement() / (1024**2)

                    def list_tensor_size_mb(tensor_list):
                        """Return total MB of all tensors in a list."""
                        total = 0.0
                        for t in tensor_list:
                            if isinstance(t, np.ndarray):
                                total += t.nbytes / (1024**2)
                            elif hasattr(t, "nelement"):
                                total += tensor_size_mb(t)
                            elif isinstance(t, (list, tuple)):
                                for x in t:
                                    if hasattr(x, "nelement"):
                                        total += tensor_size_mb(x)
                            # ignore None
                        return total

                    # print("size in mb")
                    # hs_mb = list_tensor_size_mb(record["hidden_states"])
                    # attn_mb = list_tensor_size_mb(record["attn_outputs"])
                    # mlp_mb = list_tensor_size_mb(record["mlp_outputs"])

                    # print(f"Hidden states memory:   {hs_mb:.2f} MB")
                    # print(f"Attention outputs:      {attn_mb:.2f} MB")
                    # print(f"MLP outputs:            {mlp_mb:.2f} MB")
                    # print(f"Total record memory:    {hs_mb + attn_mb + mlp_mb:.2f} MB")
                    # input("continue?")
                    
                    self.attn_outputs = []
                    self.mlp_outputs = []

                    for h in attn_hooks:
                        h.remove()
                    for h in mlp_hooks:
                        h.remove()
                    return record

                # Memory optimization: delete outputs immediately after extracting loss for Llama specific methods
                # if "llama" in self.hparams.model_name.lower() and "specific" in self.overall.lower():
                #     del out
                #     torch.cuda.empty_cache()

            # Check if params list is empty - this would cause issues
            if len(params) == 0:
                print(f"WARNING: No trainable parameters found for layer {layer}!")
                return {}

            # Count supervised tokens for debugging
            non_masked = (sample['labels'] != -100).sum().item()
            total = sample['labels'].numel()
            
            # TODO comment this whole section
            # print("sample: ", sample)
            # print(f"[debug] Sample input_ids shape: {sample['input_ids'].shape}")
            # print(f"[debug] Sample labels shape: {sample['labels'].shape}")
            # print(f"[debug] Sample input_ids: {sample['input_ids']}")
            # print(f"[debug] Sample labels: {sample['labels']}")
            # Check for inf/nan in loss before gradient computation
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is {'NaN' if torch.isnan(loss) else 'Inf'} for layer {layer}!")
                print(f"Sample input_ids shape: {sample['input_ids'].shape}")
                print(f"Sample labels shape: {sample['labels'].shape}")
                print(f"Sample input_ids: {sample['input_ids']}")
                print(f"Sample labels: {sample['labels']}")
                # Count how many labels are NOT -100 (i.e., actually supervised)
                non_masked = (sample['labels'] != -100).sum().item()
                total = sample['labels'].numel()
                print(f"Labels: {non_masked}/{total} tokens are supervised (not -100)")
                if non_masked == 0:
                    print(f"ERROR: ALL labels are -100! No tokens to supervise, loss will be NaN!")
                # Return zero gradients for this case
                grad_dict_result = {f"param_{i}": torch.zeros_like(p).cpu() for i, p in enumerate(params)}
                return grad_dict_result

            # Check for inf/nan in loss before gradient computation
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'='*60}")
                print(f"WARNING: Loss is {'NaN' if torch.isnan(loss) else 'Inf'} for layer {layer}!")
                print(f"Sample input_ids shape: {sample['input_ids'].shape}")
                print(f"Sample labels: {sample['labels']}")
                print(f"Labels: {non_masked}/{total} tokens are supervised (not -100)")
                print(f"{'='*60}\n")

            # Compute gradients using either loss.backward() or torch.autograd.grad()
            if self.use_lossBackward:
                # Use loss.backward() - standard PyTorch approach
                loss.backward()

                # Extract gradients from parameters
                grads = []
                for p in params:
                    if p.grad is not None:
                        grads.append(p.grad.clone())  # Clone to prevent modification
                    else:
                        grads.append(torch.zeros_like(p))
            else:
                # Use torch.autograd.grad() - functional approach (current default)
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

            # Check for inf/nan in gradients
            has_nan_inf = False
            for i, g in enumerate(grads):
                if torch.isnan(g).any() or torch.isinf(g).any():
                    print(f"WARNING: Gradient {i} for layer {layer} contains {'NaN' if torch.isnan(g).any() else 'Inf'}!")
                    has_nan_inf = True

            # Debug: print gradient statistics if there are issues
            if has_nan_inf or non_masked == 0:
                print(f"\n[DEBUG] Layer {layer} gradient statistics (showing first 3):")
                for i, g in enumerate(grads[:3]):
                    print(f"  Grad {i}: shape={g.shape}, mean={g.mean().item():.6e}, std={g.std().item():.6e}, "
                          f"has_nan={torch.isnan(g).any()}, has_inf={torch.isinf(g).any()}, "
                          f"all_zero={(g == 0).all()}")
                if len(grads) > 3:
                    print(f"  ... and {len(grads)-3} more gradients\n")

            grad_dict_result = {f"param_{i}": g.detach().cpu() for i, g in enumerate(grads)}

            # If using loss.backward(), clear gradients to prevent memory buildup
            if self.use_lossBackward:
                for p in params:
                    p.grad = None

            # TODO: Uncomment if memory errors occur - but note this is expensive when called per sample
            # Memory optimization: delete tensors and clear cache AFTER gradient computation for Llama/Gemma/Qwen
            # if ("llama" in self.hparams.model_name.lower() or "gemma" in self.hparams.model_name.lower() or "qwen" in self.hparams.model_name.lower()) and "specific" in self.overall.lower():
            #     del out, loss, grads
            #     torch.cuda.empty_cache()

            # Backward compatibility: disable gradients if we enabled them in this call
            if hasattr(self, '_grad_dict_enabled_grads') and self._grad_dict_enabled_grads:
                disable_layer_gradients(layer)
                self._grad_dict_enabled_grads = False

            return grad_dict_result
        
        def grad2_dict_prob(sample, i=0):
            """
            Compute diagonal second derivatives of p(target|prompt) w.r.t. parameters
            of the selected influence block.

            Args:
                sample: dictionary with keys "input_ids" and "labels" (e.g., z_test)

            Returns:
                dict: {param_i: tensor} where each tensor has same shape as the parameter
                    containing the diagonal second derivatives ∂²p(target|prompt)/∂θ²
            """
            # 1️⃣ Select the block to compute gradients for
            if "llama" in self.hparams.model_name.lower() or "gemma" in self.hparams.model_name.lower() or "qwen" in self.hparams.model_name.lower():
                # For LLaMA, Gemma3, Qwen3 models
                layers = self.model.model.layers
                if self.influence_block == "mlp":
                    block_layer = layers[layer].mlp
                elif self.influence_block == "attn":
                    block_layer = layers[layer].self_attn
                elif self.influence_block == "block":
                    block_layer = layers[layer]
                elif self.influence_block == "mlpLast":
                    block_layer = layers[layer].mlp.down_proj
                else:
                    raise NotImplementedError
            else:
                # For GPT-2 and similar models
                if self.influence_block == "mlp":
                    block_layer = self.model.transformer.h[layer].mlp
                elif self.influence_block == "attn":
                    block_layer = self.model.transformer.h[layer].attn
                elif self.influence_block == "block":
                    block_layer = self.model.transformer.h[layer]
                elif self.influence_block == "mlpLast":
                    block_layer = self.model.transformer.h[layer].mlp.c_proj
                else:
                    raise NotImplementedError

            # --- ENABLE GRADIENTS ONLY FOR TARGET BLOCK (memory optimization) ---
            for p in block_layer.parameters():
                p.requires_grad_(True)
            params = list(block_layer.parameters())

            input_ids = sample["input_ids"]
            labels = sample["labels"]  # contains -100 for masked positions (prompt)

            # Clear any existing gradients to prevent leakage between samples
            self.model.zero_grad()

            # 2️⃣ Forward pass
            with torch.enable_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits  # [batch, seq_len, vocab]

                # Log-softmax over vocab
                log_probs = torch.log_softmax(logits, dim=-1)

                # Mask for positions to include (target positions only)
                mask = (labels != -100)
                # print("log_probs shape: ", log_probs.shape)
                # print("labels shape: ", labels.shape)
                # input("continue?")
                # Gather log-probabilities corresponding to true labels
                masked_labels = labels.clone()
                masked_labels[masked_labels == -100] = 0  # safe placeholder
                token_log_probs = log_probs.gather(-1, masked_labels.unsqueeze(-1)).squeeze(-1)

                # Apply mask so prompt tokens do not contribute
                token_log_probs = token_log_probs * mask

                # Joint log-probability of target tokens
                joint_log_p = token_log_probs.sum()

                # Probability p(target|prompt)
                p_y_given_x = joint_log_p.exp()

                # Inside your loop or per case:
                case_entry = {
                    "case_id": i,
                    "prob": p_y_given_x.detach().cpu().item()  # convert to float
                }
                self.prevProbGrad.append(case_entry)

            # 3️⃣ First derivative ∂p/∂θ
            grads = torch.autograd.grad(
                p_y_given_x, params,
                create_graph=True, retain_graph=True, allow_unused=True
            )

            if self.overall == "probGreedy":
                # Disable gradients before returning (cleanup)
                for p in params:
                    p.requires_grad_(False)
                return {f"param_{i}": g.detach().cpu() for i, g in enumerate(grads)}

            # 4️⃣ Exact diagonal second derivative ∂²p/∂θ_i²
            diag_hessian = []
            for g, p in zip(grads, params):
                if g is None:
                    diag_hessian.append(torch.zeros_like(p).cpu())
                    continue

                # Allocate tensor for diagonal second derivatives
                h_diag = torch.zeros_like(p)

                # Flatten tensor for iteration
                p_flat = p.view(-1)
                g_flat = g.view(-1)
                h_diag_flat = h_diag.view(-1)

                for i in range(p_flat.numel()):
                    g_i = g_flat[i]
                    h_i = torch.autograd.grad(
                        g_i, p, retain_graph=True, allow_unused=True
                    )[0]
                    if h_i is not None:
                        h_diag_flat[i] = h_i.view(-1)[i].detach()  # pick the diagonal element
                    else:
                        print(f"Warning: second derivative is None for parameter index {i} in param of shape {p.shape}")

                diag_hessian.append(h_diag.cpu())

            # Disable gradients before returning (cleanup)
            for p in params:
                p.requires_grad_(False)

            # 5️⃣ Return as dictionary
            return {f"param_{i}": diag_hessian[i] for i in range(len(diag_hessian))}

        def grad2_dict_prob_entropy(sample, sample_new, i=0):
            """
            Compute diagonal second derivatives of p(target|prompt) w.r.t. parameters
            of the selected influence block.

            Args:
                sample: dictionary with keys "input_ids" and "labels" (e.g., z_test)

            Returns:
                dict: {param_i: tensor} where each tensor has same shape as the parameter
                    containing the diagonal second derivatives ∂²p(target|prompt)/∂θ²
            """
            # 1️⃣ Select the block to compute gradients for
            if "llama" in self.hparams.model_name.lower() or "gemma" in self.hparams.model_name.lower() or "qwen" in self.hparams.model_name.lower():
                # For LLaMA, Gemma3, Qwen3 models
                layers = self.model.model.layers
                if self.influence_block == "mlp":
                    block_layer = layers[layer].mlp
                elif self.influence_block == "attn":
                    block_layer = layers[layer].self_attn
                elif self.influence_block == "block":
                    block_layer = layers[layer]
                elif self.influence_block == "mlpLast":
                    block_layer = layers[layer].mlp.down_proj
                else:
                    raise NotImplementedError
            else:
                # For GPT-2 and similar models
                if self.influence_block == "mlp":
                    block_layer = self.model.transformer.h[layer].mlp
                elif self.influence_block == "attn":
                    block_layer = self.model.transformer.h[layer].attn
                elif self.influence_block == "block":
                    block_layer = self.model.transformer.h[layer]
                elif self.influence_block == "mlpLast":
                    block_layer = self.model.transformer.h[layer].mlp.c_proj
                else:
                    raise NotImplementedError

            # --- ENABLE GRADIENTS ONLY FOR TARGET BLOCK (memory optimization) ---
            for p in block_layer.parameters():
                p.requires_grad_(True)
            params = list(block_layer.parameters())

            input_ids = sample["input_ids"]
            labels = sample["labels"]  # contains -100 for masked positions (prompt)

            input_ids_new = sample_new["input_ids"]
            labels_new = sample_new["labels"]

            seq_len = (labels != -100).sum().item()  # number of target tokens
            query_len = (labels == -100).sum().item()

            # Clear any existing gradients to prevent leakage between samples
            self.model.zero_grad()

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids[:, :query_len],
                    attention_mask=sample['attention_mask'][:, :query_len],
                    max_new_tokens=seq_len,
                    do_sample=False,  # or True if you want stochastic sampling
                    pad_token_id=self.tok.eos_token_id,
                    use_cache=False,
                )   
            y_hat = generated[:, input_ids[:, :query_len].shape[1]:]  # shape: [batch, seq_len]

            # Forward pass again for input_ids (prompt + predicted tokens)
            outputs_pred = self.model(input_ids=generated)
            logits_pred = outputs_pred.logits  # [batch, total_len, vocab]

            # Take the last seq_len positions corresponding to predicted tokens
            logits_pred = logits_pred[:, -seq_len:, :]  # [batch, seq_len, vocab]

            # Compute log-probs
            log_probs_pred = torch.log_softmax(logits_pred, dim=-1)

            # Gather log-probs for each predicted token
            token_log_probs_pred = log_probs_pred.gather(-1, y_hat.unsqueeze(-1)).squeeze(-1)

            # Joint log-probability
            joint_log_p_yhat = token_log_probs_pred.sum(dim=-1)  # sum across sequence

            # Probability (optional)
            p_yhat_given_x = joint_log_p_yhat.exp()


            # 2️⃣ Forward pass
            with torch.enable_grad():
                outputs = self.model(input_ids=input_ids)
                outputs_new = self.model(input_ids=input_ids_new)
                logits = outputs.logits  # [batch, seq_len, vocab]
                logits_new = outputs_new.logits

                # Log-softmax over vocab
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs_new = torch.log_softmax(logits_new, dim=-1)

                # Mask for positions to include (target positions only)
                mask = (labels != -100)
                mask_new = (labels_new != -100)
                # print("log_probs shape: ", log_probs.shape)
                # print("labels shape: ", labels.shape)
                # input("continue?")
                # Gather log-probabilities corresponding to true labels
                masked_labels = labels.clone()
                masked_labels[masked_labels == -100] = 0  # safe placeholder
                token_log_probs = log_probs.gather(-1, masked_labels.unsqueeze(-1)).squeeze(-1)
                masked_labels_new = labels_new.clone()
                masked_labels_new[masked_labels_new == -100] = 0  # safe placeholder
                token_log_probs_new = log_probs_new.gather(-1, masked_labels_new.unsqueeze(-1)).squeeze(-1)

                probs = torch.softmax(logits, dim=-1)
                entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # [batch, seq_len]
                mask = (labels != -100)
                entropy_masked = entropy_per_token * mask
                entropy_sample = entropy_masked.sum() / mask.sum()

                # Apply mask so prompt tokens do not contribute
                token_log_probs = token_log_probs * mask
                token_log_probs_new = token_log_probs_new * mask_new

                # Joint log-probability of target tokens
                joint_log_p = token_log_probs.sum()
                joint_log_p_new = token_log_probs_new.sum()

                # Probability p(target|prompt)
                p_y_given_x = joint_log_p.exp()
                p_y_given_x_new = joint_log_p_new.exp()

                # Inside your loop or per case:
                case_entry = {
                    "case_id": i,
                    "prob": p_y_given_x.detach().cpu().item()  # convert to float
                }
                self.prevProbGrad.append(case_entry)

            # 3️⃣ First derivative ∂p/∂θ
            grads = torch.autograd.grad(
                # p_y_given_x_new/p_yhat_given_x/(1+entropy_sample), params, #probEntropy
                # p_y_given_x_new/p_yhat_given_x, params, #propEntropySimple
                p_y_given_x_new/p_y_given_x/(1+entropy_sample), params, #probEntropyPrev
                create_graph=True, retain_graph=True, allow_unused=True
            )

            if self.overall == "probEntropy" or self.overall == "probEntropySimple" or self.overall == "probEntropyPrev":
                # Disable gradients before returning (cleanup)
                for p in params:
                    p.requires_grad_(False)
                return {f"param_{i}": g.detach().cpu() for i, g in enumerate(grads)}

        if self.identityEnabled:
            self_influences = []
            i=0
            if "CausalTraceProba" in self.overall: noise_level = 3 * collect_embedding_std(ModelAndTokenizer(self.model, self.tok), [k["subject"] for k in requests])
            else: noise_level = None

            # For specific/l2normSpecific modes: enable gradients once before sample loop (memory optimization)
            # This avoids enabling/disabling gradients for each sample
            if "specific" in self.overall or "l2normSpecific" in self.overall:
                enable_layer_gradients(layer)

            for k in tqdm(requests):
                # Ensure first-time print registry exists
                if not hasattr(self, "firstTimePrinted") or self.firstTimePrinted is None:
                    self.firstTimePrinted = {}

                def print_once(key: str):
                    if key not in self.firstTimePrinted:
                        print(f"[Influence] Using subcategory: {key}")
                        self.firstTimePrinted[key] = True
                if self.overall=="False":
                    if self.queryLoss:
                        z_train, _, _, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, _, _, _ = make_prompt_and_attribute_samples(k)
                    grad = grad_dict(z_train)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = -safe_dot_product(grad_vector)
                elif self.overall == "overall":
                    if self.queryLoss:
                        z_train, _, z_overall, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, z_overall, _, _ = make_prompt_and_attribute_samples(k)
                    grad = grad_dict(z_overall)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    print(len(grad_vector))
                    influence = -torch.dot(grad_vector, grad_vector)
                elif self.overall == "overallNew":
                    if self.queryLoss:
                        z_train, _, _, _, z_overall = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, _, _, z_overall = make_prompt_and_attribute_samples(k)
                    grad = grad_dict(z_overall)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = -torch.dot(grad_vector, grad_vector)
                elif self.overall == "finegrainedOverall":
                    if self.queryLoss: #train queryloss else test query loss
                        z_train, _, z_overall, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, z_overall, _, _ = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_train)
                    grad_overall = grad_dict(z_overall)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall)
                elif self.overall == "finegrainedOverallNew":
                    if self.queryLoss: #train queryloss else test query loss
                        z_train, _, _, _, z_overall = make_prompt_and_attribute_samples(k)
                    else:
                        _, _, _, z_train, z_overall = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_train)
                    grad_overall = grad_dict(z_overall)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall)
                elif self.overall == "overallNewOld":
                    if self.queryLoss: #train queryloss else test query loss
                        _, _, z_overall, _, z_overall_new = make_prompt_and_attribute_samples(k)
                    else:
                        _, _, z_overall, _, z_overall_new = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_overall)
                    grad_overall = grad_dict(z_overall_new)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall)
                elif self.overall == "finegrainedOverallNewTargetOld":
                    if self.queryLoss: #train queryloss else test query loss
                        z_train, _, _, _, z_overall = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, _, _, z_overall = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_train)
                    grad_overall = grad_dict(z_overall)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall)
                elif "specific" in self.overall:
                    z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    if "TTruthTNew" in self.overall:
                        print_once("specific:TTruthTNew")
                        grad_train = grad_dict(z_test)
                        grad_overall = grad_dict(z_test_new)
                        grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                        grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                        influence = -safe_dot_product(grad_vector_train, grad_vector_overall)
                        debug_influence(influence, "TTruthTNew (specific)", v1=grad_vector_train, v2=grad_vector_overall, g1=grad_train, g2=grad_overall)
                    elif "QueryTTruth" in self.overall:
                        print_once("specific:QueryTTruth")
                        g1 = grad_dict(z_train)
                        g2 = grad_dict(z_test)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "QueryTTruth (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "QueryOverallTruth" in self.overall:
                        print_once("specific:QueryOverallTruth")
                        g1 = grad_dict(z_train)
                        g2 = grad_dict(z_overall)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "QueryOverallTruth (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "QueryTNew" in self.overall:
                        print_once("specific:QueryTNew")
                        g1 = grad_dict(z_train)
                        g2 = grad_dict(z_test_new)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "QueryTNew (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "QueryOverallNew" in self.overall:
                        print_once("specific:QueryOverallNew")
                        g1 = grad_dict(z_train)
                        g2 = grad_dict(z_overall_new)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "QueryOverallNew (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "QueryTHAT" in self.overall:
                        print_once("specific:QueryTHAT")
                        g1 = grad_dict(z_train)
                        g2 = grad_dict(z_test_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "QueryTHAT (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "QueryOverallHat" in self.overall:
                        print_once("specific:QueryOverallHat")
                        g1 = grad_dict(z_train)
                        g2 = grad_dict(z_overall_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "QueryOverallHat (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "TTruthOverallTruth" in self.overall:
                        print_once("specific:TTruthOverallTruth")
                        g1 = grad_dict(z_test)
                        g2 = grad_dict(z_overall)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "TTruthOverallTruth (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "TTruthOverallNew" in self.overall:
                        print_once("specific:TTruthOverallNew")
                        g1 = grad_dict(z_test)
                        g2 = grad_dict(z_overall_new)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "TTruthOverallNew (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "TTruthTHAT" in self.overall:
                        print_once("specific:TTruthTHAT")
                        g1 = grad_dict(z_test)
                        g2 = grad_dict(z_test_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "TTruthTHAT (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "TTruthOverallHat" in self.overall:
                        print_once("specific:TTruthOverallHat")
                        g1 = grad_dict(z_test)
                        g2 = grad_dict(z_overall_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "TTruthOverallHat (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "OverallTruthTNew" in self.overall:
                        print_once("specific:OverallTruthTNew")
                        g1 = grad_dict(z_overall)
                        g2 = grad_dict(z_test_new)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "OverallTruthTNew (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "OverallTruthOverallNew" in self.overall:
                        print_once("specific:OverallTruthOverallNew")
                        g1 = grad_dict(z_overall)
                        g2 = grad_dict(z_overall_new)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "OverallTruthOverallNew (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "OverallTruthTHAT" in self.overall:
                        print_once("specific:OverallTruthTHAT")
                        g1 = grad_dict(z_overall)
                        g2 = grad_dict(z_test_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "OverallTruthTHAT (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "OverallTruthOverallHat" in self.overall:
                        print_once("specific:OverallTruthOverallHat")
                        g1 = grad_dict(z_overall)
                        g2 = grad_dict(z_overall_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "OverallTruthOverallHat (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "TNewOverallNew" in self.overall:
                        print_once("specific:TNewOverallNew")
                        g1 = grad_dict(z_test_new)
                        g2 = grad_dict(z_overall_new)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "TNewOverallNew (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "TNewTHAT" in self.overall:
                        print_once("specific:TNewTHAT")
                        g1 = grad_dict(z_test_new)
                        g2 = grad_dict(z_test_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "TNewTHAT (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "TNewOverallHat" in self.overall:
                        print_once("specific:TNewOverallHat")
                        g1 = grad_dict(z_test_new)
                        g2 = grad_dict(z_overall_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "TNewOverallHat (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "OverallNewTHAT" in self.overall:
                        print_once("specific:OverallNewTHAT")
                        g1 = grad_dict(z_overall_new)
                        g2 = grad_dict(z_test_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "OverallNewTHAT (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "OverallNewOverallHat" in self.overall:
                        print_once("specific:OverallNewOverallHat")
                        g1 = grad_dict(z_overall_new)
                        g2 = grad_dict(z_overall_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        # debug_influence(influence, "OverallNewOverallHat (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                    elif "THATOverallHat" in self.overall:
                        print_once("specific:THATOverallHat")
                        g1 = grad_dict(z_test_orig)
                        g2 = grad_dict(z_overall_orig)
                        v1 = torch.cat([t.flatten() for t in g1.values()])
                        v2 = torch.cat([t.flatten() for t in g2.values()])
                        influence = -safe_dot_product(v1, v2)
                        debug_influence(influence, "THATOverallHat (specific)", v1=v1, v2=v2, g1=g1, g2=g2)
                elif "l2normSpecific" in self.overall:
                    z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    if "Query" in self.overall:
                        print_once("specificl2norm:Query")
                        g = grad_dict(z_train)
                        v = torch.cat([t.flatten() for t in g.values()])
                        influence = -safe_dot_product(v)
                        # debug_influence(influence, "Query (l2normSpecific)", v=v, g=g, sample=z_train)
                    elif "TTruth" in self.overall:
                        print_once("specificl2norm:TTruth")
                        g = grad_dict(z_test)
                        v = torch.cat([t.flatten() for t in g.values()])
                        influence = -safe_dot_product(v)
                        debug_influence(influence, "TTruth (l2normSpecific)", v=v, g=g, sample=z_test)
                    elif "OverallTruth" in self.overall:
                        print_once("specificl2norm:OverallTruth")
                        g = grad_dict(z_overall)
                        v = torch.cat([t.flatten() for t in g.values()])
                        influence = -safe_dot_product(v)
                        debug_influence(influence, "OverallTruth (l2normSpecific)", v=v, g=g, sample=z_overall)
                    elif "TNew" in self.overall:
                        print_once("specificl2norm:TNew")
                        g = grad_dict(z_test_new)
                        v = torch.cat([t.flatten() for t in g.values()])
                        influence = -safe_dot_product(v)
                        debug_influence(influence, "TNew (l2normSpecific)", v=v, g=g, sample=z_test_new)
                    elif "OverallNew" in self.overall:
                        print_once("specificl2norm:OverallNew")
                        g = grad_dict(z_overall_new)
                        v = torch.cat([t.flatten() for t in g.values()])
                        influence = -safe_dot_product(v)
                        debug_influence(influence, "OverallNew (l2normSpecific)", v=v, g=g, sample=z_overall_new)
                    elif "THAT" in self.overall:
                        print_once("specificl2norm:THAT")
                        g = grad_dict(z_test_orig)
                        v = torch.cat([t.flatten() for t in g.values()])
                        influence = -safe_dot_product(v)
                        debug_influence(influence, "THAT (l2normSpecific)", v=v, g=g, sample=z_test_orig)
                    elif "OverallHat" in self.overall:
                        print_once("specificl2norm:OverallHat")
                        g = grad_dict(z_overall_orig)
                        v = torch.cat([t.flatten() for t in g.values()])
                        influence = -safe_dot_product(v)
                        # debug_influence(influence, "OverallHat (l2normSpecific)", v=v, g=g, sample=z_overall_orig)
                    else: print("No specific l2norm match")
                elif self.overall == "finegrainedTOldTNew":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        _, z_train, _, z_overall, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, _, z_overall, _ = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_train)
                    grad_overall = grad_dict(z_overall)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall)
                elif self.overall == "finegrainedTOldTNew2":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        _, z_train, _, z_overall, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, _, z_overall, _ = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_train)
                    grad_overall = grad_dict(z_overall)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall)
                elif self.overall=="prob":
                    if self.queryLoss:
                        z_test, _, _, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_test, _, _, _ = make_prompt_and_attribute_samples(k)
                    grad = grad2_dict_prob(z_test) 
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = grad_vector
                    # print("influence shape: ", influence.shape)
                    # print("influence length: ", len(influence))
                    # input("continue?")
                elif self.overall=="probGreedy":
                    if self.queryLoss:
                        z_test, _, _, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_test, _, _, _ = make_prompt_and_attribute_samples(k)
                    grad = grad2_dict_prob(z_test, i)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = grad_vector.sum()
                elif self.overall=="probEntropy":
                    if self.queryLoss:
                        z_test, _, _, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_test, _, z_test_new, _ = make_prompt_and_attribute_samples(k)
                    grad = grad2_dict_prob_entropy(z_test, z_test_new, i)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = -torch.dot(grad_vector, grad_vector)
                elif self.overall=="probEntropySimple":
                    if self.queryLoss:
                        z_test, _, _, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_test, _, z_test_new, _ = make_prompt_and_attribute_samples(k)
                    grad = grad2_dict_prob_entropy(z_test, z_test_new, i)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = -torch.dot(grad_vector, grad_vector)
                elif self.overall=="probEntropyPrev":
                    if self.queryLoss:
                        z_test, _, _, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_test, _, z_test_new, _ = make_prompt_and_attribute_samples(k)
                    grad = grad2_dict_prob_entropy(z_test, z_test_new, i)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = -torch.dot(grad_vector, grad_vector)
                elif self.overall == "l2norm":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    def l2norm(z):
                        grad = grad_dict(z)
                        grad_vector = torch.cat([g.flatten() for g in grad.values()])
                        return torch.norm(grad_vector, p=2)
                    
                    norm_train = l2norm(z_train)
                    norm_test = l2norm(z_test)
                    norm_test_new = l2norm(z_test_new)
                    norm_overall = l2norm(z_overall)
                    norm_overall_new = l2norm(z_overall_new)
                    norm_test_orig = l2norm(z_test_orig)
                    norm_overall_orig = l2norm(z_overall_orig)
                    influence = torch.stack([norm_train, norm_test, norm_test_new,
                                             norm_overall, norm_overall_new, norm_test_orig, norm_overall_orig])
                elif "PCAScalarFit" in self.overall:
                    if self.queryLoss:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)

                    def fit_scaler(z):
                        grad = grad_dict(z)
                        grad_vector = torch.cat([g.flatten() for g in grad.values()]).detach().cpu().to(torch.float16).numpy()
                        # print("grad_vector shape: ", grad_vector.shape)
                        # print("grad_vector 1,-1 shape", grad_vector.reshape(1, -1).shape)
                        self.grad_scaler.partial_fit(grad_vector.reshape(1, -1))
                        return grad_vector

                    # You only use one gradient type per run
                    if "query" in self.overall: fit_scaler(z_train)
                    elif "testTarget" in self.overall: fit_scaler(z_test)
                    elif "testNewTarget" in self.overall: fit_scaler(z_test_new)
                    elif "overall" in self.overall: fit_scaler(z_overall)
                    elif "overNew" in self.overall: fit_scaler(z_overall_new)
                    elif "origTarget" in self.overall: fit_scaler(z_test_orig)
                    elif "origO" in self.overall: fit_scaler(z_overall_orig)

                    influence = torch.tensor(0.0)

                elif "PCAPartialFit" in self.overall:
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    def l2norm(z, pca):
                        grad = grad_dict(z)
                        grad_vector = torch.cat([g.flatten() for g in grad.values()]).detach().cpu().to(torch.float16).numpy()
                        
                        # # Check for NaNs or Infs
                        # if np.isnan(grad_vector).any():
                        #     print("NaN detected in gradient vector!")
                        # if np.isinf(grad_vector).any():
                        #     print("Inf detected in gradient vector!")
                        
                        # Scale using pretrained scaler
                        grad_scaled = self.grad_scaler.transform(grad_vector.reshape(1, -1))
                        # print("received+1 sample")
                        self.pca_grad_buffer.append(grad_scaled.flatten())
    
                        # Perform partial fit once buffer reaches batch size
                        if len(self.pca_grad_buffer) >= self.pca_batch_size:
                            X_batch = np.stack(self.pca_grad_buffer)
                            # print(X_batch.shape)
                            pca.partial_fit(X_batch)
                            self.pca_grad_buffer.clear()
                        return torch.tensor(grad_vector, dtype=torch.float16)
                    
                    if "query" in self.overall: norm = l2norm(z_train, self.grad_pca)
                    elif "testTarget" in self.overall: norm = l2norm(z_test, self.grad_pca)
                    elif "testNewTarget" in self.overall: norm = l2norm(z_test_new, self.grad_pca)
                    elif "overall" in self.overall: norm = l2norm(z_overall, self.grad_pca)
                    elif "overNew" in self.overall: norm = l2norm(z_overall_new, self.grad_pca)
                    elif "origTarget" in self.overall: norm = l2norm(z_test_orig, self.grad_pca)
                    elif "origO" in self.overall: norm = l2norm(z_overall_orig, self.grad_pca)
                    
                    # print("Completed for one sample")
                    
                    influence = norm
                elif "PCACompleteFit" in self.overall:
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    def l2norm(z, pca):
                        grad = grad_dict(z)
                        grad_vector = torch.cat([g.flatten() for g in grad.values()]).detach().cpu().to(torch.float16).numpy()
                        
                        # Scale before transform
                        grad_scaled = self.grad_scaler.transform(grad_vector.reshape(1, -1))
                        # Reduce dimension
                        reduced = pca.transform(grad_scaled)
                        return torch.tensor(reduced, dtype=torch.float16)  # shape: [1, n_components]
                    
                    if "query" in self.overall: norm = l2norm(z_train, self.grad_pca)
                    elif "testTarget" in self.overall: norm = l2norm(z_test, self.grad_pca)
                    elif "testNewTarget" in self.overall: norm = l2norm(z_test_new, self.grad_pca)
                    elif "overall" in self.overall: norm = l2norm(z_overall, self.grad_pca)
                    elif "overNew" in self.overall: norm = l2norm(z_overall_new, self.grad_pca)
                    elif "origTarget" in self.overall: norm = l2norm(z_test_orig, self.grad_pca)
                    elif "origO" in self.overall: norm = l2norm(z_overall_orig, self.grad_pca)
                    
                    influence = norm
                elif "HiddenStates" in self.overall:
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    
                    # Check for subcategory: mlp, attn, hs, or difference-based
                    if "HSMlp" in self.overall:
                        # Calculate L2 norm of MLP outputs for each layer
                        def l2norm_mlp(z):
                            record = grad_dict(z)  # contains mlp_outputs, attn_outputs, hidden_states
                            mlp_outputs = record["mlp_outputs"]
                            # Calculate L2 norm for each layer
                            layer_norms = []
                            for layer_output in mlp_outputs:
                                # layer_output shape: (batch_size, seq_len, hidden_dim)
                                # Calculate L2 norm across the last dimension
                                norm = torch.norm(layer_output, p=2, dim=-1).mean()  # average over batch and seq
                                layer_norms.append(norm.item())
                            return torch.tensor(layer_norms)
                        
                        if "OverallNew" in self.overall: norm = l2norm_mlp(z_overall_new)
                        elif "OverallOrig" in self.overall: norm = l2norm_mlp(z_overall_orig)
                        else: 
                            print_once("HSMlp requires 'OverallNew' or 'OverallOrig' in overall")
                            norm = l2norm_mlp(z_overall_new)
                        
                    elif "HSAttn" in self.overall:
                        # Calculate L2 norm of attention outputs for each layer
                        def l2norm_attn(z):
                            record = grad_dict(z)
                            attn_outputs = record["attn_outputs"]
                            layer_norms = []
                            for layer_output in attn_outputs:
                                # layer_output shape: (batch_size, seq_len, hidden_dim)
                                norm = torch.norm(layer_output, p=2, dim=-1).mean()
                                layer_norms.append(norm.item())
                            return torch.tensor(layer_norms)
                        
                        if "OverallNew" in self.overall: norm = l2norm_attn(z_overall_new)
                        elif "OverallOrig" in self.overall: norm = l2norm_attn(z_overall_orig)
                        else:
                            print_once("HSAttn requires 'OverallNew' or 'OverallOrig' in overall")
                            norm = l2norm_attn(z_overall_new)
                        
                    elif "HSHs" in self.overall:
                        # Calculate L2 norm of hidden states for each layer
                        def l2norm_hs(z):
                            record = grad_dict(z)
                            hidden_states = record["hidden_states"]
                            layer_norms = []
                            for layer_hs in hidden_states:
                                # layer_hs shape: (batch_size, seq_len, hidden_dim)
                                norm = torch.norm(layer_hs, p=2, dim=-1).mean()
                                layer_norms.append(norm.item())
                            return torch.tensor(layer_norms)
                        
                        if "OverallNew" in self.overall: norm = l2norm_hs(z_overall_new)
                        elif "OverallOrig" in self.overall: norm = l2norm_hs(z_overall_orig)
                        else:
                            print_once("HSHs requires 'OverallNew' or 'OverallOrig' in overall")
                            norm = l2norm_hs(z_overall_new)
                        
                    elif "HSDiffMlp" in self.overall:
                        # Calculate L2 norm of difference (new - orig) for MLP outputs
                        def l2norm_diff_mlp(z_new, z_orig):
                            record_new = grad_dict(z_new)
                            record_orig = grad_dict(z_orig)
                            mlp_new = record_new["mlp_outputs"]
                            mlp_orig = record_orig["mlp_outputs"]
                            layer_norms = []
                            for mlp_n, mlp_o in zip(mlp_new, mlp_orig):
                                # Handle different sequence lengths by taking minimum
                                min_len = min(mlp_n.shape[1], mlp_o.shape[1])
                                diff = mlp_n[:, :min_len, :] - mlp_o[:, :min_len, :]
                                norm = torch.norm(diff, p=2, dim=-1).mean()
                                layer_norms.append(norm.item())
                            return torch.tensor(layer_norms)
                        
                        norm = l2norm_diff_mlp(z_overall_new, z_overall_orig)
                        
                    elif "HSDiffAttn" in self.overall:
                        # Calculate L2 norm of difference (new - orig) for attention outputs
                        def l2norm_diff_attn(z_new, z_orig):
                            record_new = grad_dict(z_new)
                            record_orig = grad_dict(z_orig)
                            attn_new = record_new["attn_outputs"]
                            attn_orig = record_orig["attn_outputs"]
                            layer_norms = []
                            for attn_n, attn_o in zip(attn_new, attn_orig):
                                # Handle different sequence lengths by taking minimum
                                min_len = min(attn_n.shape[1], attn_o.shape[1])
                                diff = attn_n[:, :min_len, :] - attn_o[:, :min_len, :]
                                norm = torch.norm(diff, p=2, dim=-1).mean()
                                layer_norms.append(norm.item())
                            return torch.tensor(layer_norms)
                        
                        norm = l2norm_diff_attn(z_overall_new, z_overall_orig)
                        
                    elif "HSDiffHs" in self.overall:
                        # Calculate L2 norm of difference (new - orig) for hidden states
                        def l2norm_diff_hs(z_new, z_orig):
                            record_new = grad_dict(z_new)
                            record_orig = grad_dict(z_orig)
                            hs_new = record_new["hidden_states"]
                            hs_orig = record_orig["hidden_states"]
                            layer_norms = []
                            for hs_n, hs_o in zip(hs_new, hs_orig):
                                # Take the smallest length to handle mismatched sequence lengths
                                min_len = min(hs_n.shape[1], hs_o.shape[1])
                                diff = hs_n[:, :min_len] - hs_o[:, :min_len]
                                norm = torch.norm(diff, p=2, dim=-1).mean()
                                layer_norms.append(norm.item())
                            return torch.tensor(layer_norms)
                        
                        norm = l2norm_diff_hs(z_overall_new, z_overall_orig)
                    else:
                        # Original HiddenStates behavior - return full record
                        def l2norm(z):
                            grad = grad_dict(z)
                            return grad  # contains hidden states and attn/mlp outputs
                        
                        if "query" in self.overall: norm = l2norm(z_train)
                        elif "testTarget" in self.overall: norm = l2norm(z_test)
                        elif "testNewTarget" in self.overall: norm = l2norm(z_test_new)
                        elif "overall" in self.overall: norm = l2norm(z_overall)
                        elif "overNew" in self.overall: norm = l2norm(z_overall_new)
                        elif "origTarget" in self.overall: norm = l2norm(z_test_orig)
                        elif "origO" in self.overall: norm = l2norm(z_overall_orig)
                    
                    influence = norm
                elif "CausalTraceProba" in self.overall:
                    mt = ModelAndTokenizer(self.model, self.tok)
                    if "default" in self.overall:
                        scores = plot_all_flow(mt, k["prompt"], k["subject"], noise=noise_level, index = i, title =
                                    "Prompt: "+ k['prompt'] + "; subject: " + k["subject"]) # shape: (n_toks, n_layers)
                    elif "last" in self.overall:
                        print("Calculating the CMA for last subject token")
                        scores = plot_all_flow_last_subject(mt, k["prompt"], k["subject"], noise=noise_level, index = i, title =
                                    "Prompt: "+ k['prompt'] + "; subject: " + k["subject"]) # shape: (n_layers,)
                    influence = scores
                elif "RFPredictionProba" in self.overall:
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    def l2norm(z):
                        grad = grad_dict(z)
                        grad_vector = torch.cat([g.flatten() for g in grad.values()]).detach().cpu().to(torch.float16).numpy()
                        
                        # Scale before transform
                        grad_scaled = self.grad_scaler.transform(grad_vector.reshape(1, -1))
                        reduced = self.grad_pca.transform(grad_scaled)
                        rf_scaled = self.rf_scaler.transform(reduced)
                        proba = self.rf_model.predict_proba(rf_scaled)[:, 1] # probability for best layer class
                        return torch.tensor(proba, dtype=torch.float32)
                    
                    if "query" in self.overall: norm = l2norm(z_train)
                    elif "testTarget" in self.overall: norm = l2norm(z_test)
                    elif "testNewTarget" in self.overall: norm = l2norm(z_test_new)
                    elif "overall" in self.overall: norm = l2norm(z_overall)
                    elif "overNew" in self.overall: norm = l2norm(z_overall_new)
                    elif "origTarget" in self.overall: norm = l2norm(z_test_orig)
                    elif "origO" in self.overall: norm = l2norm(z_overall_orig)
                    
                    influence = norm
                elif self.overall == "finegrainedTOrig":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        _, _, _, _, _, z_tOrig, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, _, _, _, _, z_tOrig, _ = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_tOrig)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_train)
                elif self.overall == "finegrainedTOrigTNew":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        _, z_train, _, z_overall, _, z_orig, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, _, z_overall, _, z_orig, _ = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_orig)
                    grad_overall = grad_dict(z_overall)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall)
                elif "SparseCompleteFit" in self.overall:
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    def l2norm(z):
                        grad = grad_dict(z)
                        grad_vector = torch.cat([g.flatten() for g in grad.values()])
                        
                        # 2. Compute the threshold for the lowest 99.9% elements (by absolute value)
                        k = int(0.999 * grad_vector.numel())  # number of elements to zero out
                        abs_vals = torch.abs(grad_vector)
                        threshold = torch.kthvalue(abs_vals, k).values  # kth largest absolute value

                        # 3. Set lowest 99.9% to zero
                        grad_vector_thresholded = grad_vector.clone()
                        grad_vector_thresholded[abs_vals <= threshold] = 0

                        grad_vector_thresholded = grad_vector_thresholded.detach().cpu().numpy()
                        sparse_csr = sparse.csr_matrix(grad_vector_thresholded)

                        
                        return sparse_csr  # shape: [1, n_components]
                    
                    if "query" in self.overall: norm = l2norm(z_train)
                    elif "testTarget" in self.overall: norm = l2norm(z_test)
                    elif "testNewTarget" in self.overall: norm = l2norm(z_test_new)
                    elif "overall" in self.overall: norm = l2norm(z_overall)
                    elif "overNew" in self.overall: norm = l2norm(z_overall_new)
                    elif "origTarget" in self.overall: norm = l2norm(z_test_orig)
                    elif "origO" in self.overall: norm = l2norm(z_overall_orig)
                    
                    influence = norm
                elif self.overall == "finegrainedTOrigTQueryToldV1":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_test_orig)
                    grad_overall = grad_dict(z_test_new)
                    grad_overall2 = grad_dict(z_train)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    grad_vector_overall2 = torch.cat([g.flatten() for g in grad_overall2.values()])
                    influence = -torch.dot(grad_vector_train, grad_vector_overall * grad_vector_overall2)
                elif self.overall == "finegrainedTOrigSubQueryTNewV1":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_test_orig)
                    grad_overall = grad_dict(z_test_new)
                    grad_overall2 = grad_dict(z_train)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    grad_vector_overall2 = torch.cat([g.flatten() for g in grad_overall2.values()])
                    grad_vector_overall2 = grad_vector_train - grad_vector_overall2
                    influence = -torch.dot(grad_vector_overall, grad_vector_overall2)
                elif self.overall == "finegrainedTNewSubQueryTOrigV1":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_test_orig)
                    grad_overall = grad_dict(z_test_new)
                    grad_overall2 = grad_dict(z_train)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    grad_vector_overall2 = torch.cat([g.flatten() for g in grad_overall2.values()])
                    grad_vector_overall2 = grad_vector_overall - grad_vector_overall2
                    influence = -torch.dot(grad_vector_train, grad_vector_overall2)
                elif self.overall == "finegrainedTNewSubQueryTOldV1":
                    if self.queryLoss: #queyloss doesnt matter as both are target
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    else:
                        z_train, z_test, z_overall, z_test_new, z_overall_new, z_test_orig, z_overall_orig = make_prompt_and_attribute_samples(k)
                    grad_train = grad_dict(z_test)
                    grad_overall = grad_dict(z_test_new)
                    grad_overall2 = grad_dict(z_train)
                    grad_vector_train = torch.cat([g.flatten() for g in grad_train.values()])
                    grad_vector_overall = torch.cat([g.flatten() for g in grad_overall.values()])
                    grad_vector_overall2 = torch.cat([g.flatten() for g in grad_overall2.values()])
                    grad_vector_overall2 = grad_vector_overall - grad_vector_overall2
                    influence = -torch.dot(grad_vector_train, grad_vector_overall2)
                if self.overall == "l2norm":
                    self_influences.append(influence)
                elif ("PCACompleteFit" in self.overall) or ("PCAPartialFit" in self.overall) or ("RPCompleteFit" in self.overall) or ("SparseCompleteFit" in self.overall):
                    # influence is shape [1, n_components]
                    # Store it directly (so later torch.stack works)
                    self_influences.append(influence)
                elif ("HiddenStates" in self.overall) or ("CausalTraceProba" in self.overall):
                    # influence is a dict of hidden states
                    self_influences.append(influence)
                else: self_influences.append(float(influence))
                i+=1

            # For specific/l2normSpecific modes: disable gradients after sample loop (memory optimization)
            if "specific" in self.overall or "l2normSpecific" in self.overall:
                disable_layer_gradients(layer)

            # if "PCAPartialFit" in self.overall: print("Completed one PCA partial fit epoch", i)
            # Save to file (e.g., "prev_prob_grad.json")
            # with open("prev_prob_grad.json", "w") as f:
            #     json.dump(self.prevProbGrad, f, indent=4)

            if (self.overall == "l2norm") or ("PCACompleteFit" in self.overall) or ("PCAPartialFit" in self.overall) or ("RPCompleteFit" in self.overall):
                return torch.stack(self_influences)
            elif ("SparseCompleteFit" in self.overall) or ("HiddenStates" in self.overall) or ("CausalTraceProba" in self.overall):
                return self_influences
            else: return torch.tensor(self_influences)

        # --- collect grads ---
        tr_grad_dict = {}
        for i, k in enumerate(requests):
            if not self.queryLoss:
                _, z_train, _ = make_prompt_and_attribute_samples(k)
            else:
                z_train, _, _ = make_prompt_and_attribute_samples(k)
            tr_grad_dict[i] = grad_dict(z_train)

        # validation: just the attribute-only sample for sample_index
        # _, z_test = make_prompt_and_attribute_samples(knowns[sample_index])
        # val_grad_dict = {0: grad_dict(z_test)}

        # --- run IFEngine ---
        engine = IFEngine()
        engine.preprocess_gradients(tr_grad_dict, tr_grad_dict)
        engine.compute_hvps(lambda_const_param=lambda_const_param)
        engine.compute_IF()

        # grab DataInf influences (their 'proposed' method)
        influences = engine.IF_dict['proposed']

        # return influence of the chosen training sample (sample_index) wrt attribute
        # return float(influences[sample_index][sample_index])
        self_influences = [
            float(influences[sample_idx][sample_idx])
            for sample_idx in range(len(requests))
        ]
        return torch.tensor(self_influences)

    def edit_requests(self,
             requests,
             sequential_edit=False,
             verbose=True,
             test_generation=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        print("Using Model: " + self.model.config.name_or_path)
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = copy.deepcopy(metrics)  # Deep copy to prevent mutation across iterations
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {"pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''], request, self.hparams.device, pre_edit=True)}
                else:
                    metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation, 
                                                           selfTemp=self),
                                "preTruth": compute_edit_quality_target_truth(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation, 
                                                           selfTemp=self)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE' or self.alg_name == 'ICE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
            eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
            if self.alg_name == 'IKE':
                all_metrics[idx].update({
                    'case_id': idx,
                    'layers': self.hparams.layers,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device ,test_generation=test_generation),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
    
            else:
                all_metrics[idx].update({
                    'case_id': idx,
                    'layers': self.hparams.layers if self.alg_name != "KN" else None,
                    "requested_rewrite": request,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation,
                                                 selfTemp=self),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
                if 'locality' in all_metrics[idx]['post'].keys() and not hasattr(self.hparams, 'evaluation_type'):
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        if hasattr(self.hparams, 'evaluation_type'):
                            locality_result.append(float(all_metrics[idx]['post']['locality'][f'{locality_key}_output']==all_metrics[idx]['pre']['locality'][f'{locality_key}_output']))
                        else:
                            for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                                locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[idx]['pre'].pop('locality')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")


        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                if self.localLayerIF and self.alg_name in self.supportedLocalInfMethods:
                    max_layer, max_value = self.get_max_influence_layer_for_sample(i)
                    print(f"Sample {i} - Selected Layer: {max_layer} with influence value: {max_value}")
                    LOG.info(f"Sample {i} - Selected Layer: {max_layer} with influence value: {max_value}")
                    self.hparams.layers = max_layer
                if self.allLayerScores:
                    self.hparams.layers = self.predefinedLayer
                edited_model, weights_copy, icl_examples = edit_func(request)
            if self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                self.model = edited_model
            if self.alg_name == 'WISE' and hasattr(self.hparams, 'save_path') and self.hparams.save_path:
                print("Start saving the WISE model!")
                edited_model.save(self.hparams.save_path)
            for i, request in enumerate(requests):
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                if self.localLayerIF and self.alg_name in self.supportedLocalInfMethods:
                    max_layer, max_value = self.get_max_influence_layer_for_sample(i)
                    print(f"Sample {i} - Selected Layer: {max_layer} with influence value: {max_value}")
                    LOG.info(f"Sample {i} - Selected Layer: {max_layer} with influence value: {max_value}")
                    self.hparams.layers = max_layer
                if self.allLayerScores:
                    self.hparams.layers = self.predefinedLayer
                edited_model, weights_copy, icl_examples = edit_func(request)
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

                # Clear CUDA cache after each edit to prevent OOM from memory fragmentation
                torch.cuda.empty_cache()


        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if not hasattr(self.hparams, 'evaluation_type') or self.hparams.evaluation_type != "generate-text":
            summary_metrics(all_metrics)

        return all_metrics, edited_model, weights_copy

    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        sequential_edit=False,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy
    
    def generate_edit(
        self,
        prompts: Union[str, List[str]],
        target_new: Union[str, List[str]],
        ground_truth: Optional[Union[str, List[str]]] = None,
        target_neg: Optional[Union[str, List[str]]] = None,
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        locality_inputs:  Optional[Dict] = None,
        portability_inputs: Optional[Dict] = None,
        sequential_edit=False,
        verbose=True,
        **kwargs
    ):
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        test_generation = kwargs.pop('test_generation', False)

        assert len(prompts) == len(target_new)

        if hasattr(self.hparams, 'batch_size'):
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        
        def text_generate(
            model,
            model_name,
            hparams: HyperParams,
            tok: AutoTokenizer,
            query,
            device,
            eval_metric: str = 'token_em',
            test_generation = False
        ):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tok.encode(text, return_tensors="pt").to(f"cuda:{device}")
            template_length = len(model_inputs[0])
            generated_ids = model.generate(
                input_ids=model_inputs,
                max_new_tokens=512
            )
            trimmed_generated_ids = generated_ids[0][template_length:]
            response = tok.decode(trimmed_generated_ids, skip_special_tokens=True)
            return response

        all_results = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            results = kwargs['pre_edit']
            all_results = results
        else:
            for i, request in enumerate(tqdm(requests)):
                results = {}
                results['pre'] = {}
                results['pre']['rewrite_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results['pre']['rephrase_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results['pre']['locality_ans'] = loc_results
                results['pre']['portability_ans'] = por_results
                all_results.append(results)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_results, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples
        
        def post_edit_results(all_results, request, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs):
            if self.alg_name == 'IKE':
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device),
                })
            else:
                results_post = {}
                results_post['rewrite_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results_post['rephrase_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results_post['locality_ans'] = loc_results
                results_post['portability_ans'] = por_results
                if test_generation:
                    if self.hparams.alg_name == 'GRACE':
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=True)
                    else:
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=False)
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": results_post
                })
            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_results) != 0:
            summary_metrics(all_results)

        return all_results, edited_model, weights_copy

    def deep_edit(
        self,
        datasets
    ):
        metrics = self.apply_algo(datasets, self.hparams)
        return metrics



