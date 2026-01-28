"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_prediction_acc_LLM_judge,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    OOD_PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)
import json

def get_prompt_to_attribute_influence(self, requests, edited_model, lambda_const_param=10, layer = 0):
        from IF import IFEngineGeneration as IFEngine  # assuming IFEngine code is in IFEngine.py
        print("Computing for layer: ", layer)

        for p in edited_model.parameters():
            p.requires_grad_(True)

        self.tempProb = -1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            # TRAIN: prompt only (Query Loss)
            labels_train = input_ids.clone()
            labels_train[0, len(enc_prompt['input_ids']):] = -100
            z_train = {"input_ids": input_ids, "labels": labels_train}

            # TEST: attribute only (Target Loss)
            labels_test = input_ids.clone()
            labels_test[0, :len(enc_prompt['input_ids'])] = -100
            z_test = {"input_ids": input_ids, "labels": labels_test}

            # TEST: attribute only (Target Loss)
            labels_test = input_ids_new.clone()
            labels_test[0, :len(enc_prompt['input_ids'])] = -100
            z_test_new = {"input_ids": input_ids_new, "labels": labels_test}
            
            #overall loss
            labels_test = input_ids.clone()
            z_overall = {"input_ids": input_ids, "labels": labels_test}
            
            #overall with target_new loss
            labels_test = input_ids_new.clone()
            z_overall_new = {"input_ids": input_ids_new, "labels": labels_test}

            return z_train, z_test, z_overall, z_test_new, z_overall_new

        def grad_dict(sample):
            if self.influence_block == "mlp":
                block_layer = edited_model.transformer.h[layer].mlp
            elif self.influence_block == "attn":
                block_layer = edited_model.transformer.h[layer].attn
            elif self.influence_block == "block":
                block_layer = edited_model.transformer.h[layer]
            elif self.influence_block == "mlpLast":
                block_layer = edited_model.transformer.h[layer].mlp.c_proj
            else:
                raise NotImplementedError
            params = [p for p in block_layer.parameters() if p.requires_grad]
            with torch.enable_grad():
                out = edited_model(input_ids=sample["input_ids"], labels=sample["labels"])
                loss = out.loss
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
            return {f"param_{i}": g.detach().cpu() for i, g in enumerate(grads)}
        
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
            if self.influence_block == "mlp":
                block_layer = edited_model.transformer.h[layer].mlp
            elif self.influence_block == "attn":
                block_layer = edited_model.transformer.h[layer].attn
            elif self.influence_block == "block":
                block_layer = edited_model.transformer.h[layer]
            elif self.influence_block == "mlpLast":
                block_layer = edited_model.transformer.h[layer].mlp.c_proj
            else:
                raise NotImplementedError

            # Parameters of this block
            params = [p for p in block_layer.parameters() if p.requires_grad]

            input_ids = sample["input_ids"]
            labels = sample["labels"]  # contains -100 for masked positions (prompt)

            # 2️⃣ Forward pass
            with torch.enable_grad():
                outputs = edited_model(input_ids=input_ids)
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
                # case_entry = {
                #     "case_id": i,
                #     "prob": p_y_given_x.detach().cpu().item()  # convert to float
                # }
                # self.prevProbGrad.append(case_entry)
                self.tempProb = p_y_given_x.detach().cpu().item()  # convert to float

            # 3️⃣ First derivative ∂p/∂θ
            grads = torch.autograd.grad(
                p_y_given_x, params,
                create_graph=True, retain_graph=True, allow_unused=True
            )

            if self.overall == "probGreedy" or self.overall == "probEntropy":
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

            # 5️⃣ Return as dictionary
            return {f"param_{i}": diag_hessian[i] for i in range(len(diag_hessian))}


        if self.identityEnabled:
            self_influences = []
            i=0
            for k in requests:
                if self.overall=="False":
                    if self.queryLoss:
                        z_train, _, _, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, _, _, _ = make_prompt_and_attribute_samples(k)
                    grad = grad_dict(z_train)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = -torch.dot(grad_vector, grad_vector)
                elif self.overall == "overall":
                    if self.queryLoss:
                        z_train, _, z_overall, _, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, z_train, z_overall, _, _ = make_prompt_and_attribute_samples(k)
                    grad = grad_dict(z_overall)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
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
                        _, _, _, z_test, _ = make_prompt_and_attribute_samples(k)
                    else:
                        _, _, _, z_test, _ = make_prompt_and_attribute_samples(k)
                    grad = grad2_dict_prob(z_test, i)
                    grad_vector = torch.cat([g.flatten() for g in grad.values()])
                    influence = grad_vector.sum()
                self_influences.append(float(influence))
                i+=1
            
            # Save to file (e.g., "prev_prob_grad.json")
            # with open("prev_prob_grad.json", "w") as f:
            #     json.dump(self.prevProbGrad, f, indent=4)

            return torch.tensor(self_influences).item(), self.tempProb

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

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    selfTemp = None
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", 'target_truth']
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if selfTemp is not None and selfTemp.overall == "probGreedy" or selfTemp.overall == "probEntropy":
        ret['infGrad'], ret['prob'] = get_prompt_to_attribute_influence(selfTemp, [record], model, lambda_const_param=10, layer = hparams.layers[0])
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_edit_quality_target_truth(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    selfTemp = None
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", 'target_truth']
    )
    target_new = ground_truth

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if selfTemp is not None and selfTemp.overall == "probGreedy":
        ret['infGrad'], ret['prob'] = get_prompt_to_attribute_influence(selfTemp, [record], model, lambda_const_param=10, layer = hparams.layers[0])
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:

    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'

    # Handle multiple rephrase prompts: replicate target_new to match prompt list length
    if isinstance(prompt, list) and isinstance(target_new, str):
        target_new = [target_new] * len(prompt)

    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        acc, gen_content = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_acc": acc,
            f"{key}_gen_content": gen_content
        }
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "generate-text":
        gen_content_model = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_gen_content": gen_content_model
        }
    else:  # traditional evaluation 
        if eval_metric == 'ppl':
            ppl = PPL(model, tok, prompt, target_new, device)
            ret = {
                f"{key}_ppl": ppl
            }
        elif eval_metric == 'ood_ppl':
            ans = OOD_PPL(model, tok, prompt, target_new, device)
            ret = {
                f"ood_acc": ans
            }
        elif hparams.alg_name=="GRACE":
            # ppl = PPL(model, tok, prompt, target_new, device)
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device, vanilla_generation=True)
            f1 = F1(model,tok,hparams,prompt,target_new,device, vanilla_generation=True)
            ret = {
                f"{key}_acc": acc,
                # f"{key}_PPL": ppl,
                f"{key}_F1":f1     
            }        
        else:  # teacher-forcing evaluation
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
            ret = {
                f"{key}_acc": acc
            }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type'):
        loc_tokens = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:  # traditional evaluation 
        if 't5' in model_name.lower():
            loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
        else:
            loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True, vanilla_generation=hparams.alg_name=='GRACE')
        if type(loc_tokens) is not list:
            loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:
    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        portability_correct = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, ground_truth, device, locality=False)
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "generate-text":
        portability_correct = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, ground_truth, device, locality=False)
    else:  # traditional evaluation
        if 't5' in model_name.lower():
            portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
        else:
            portability_correct = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')

    ret = {
        f"{portability_key}_acc": portability_correct
    }
    return ret

def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False,
        test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, new_fact)
    ret = {
        f"rewrite_acc": [edit_acc]
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                   target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'], list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],
                                    record['locality'][locality_key]['prompt']):
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                                   f"{x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                    f"New Fact: {prompt} {target_new}\nPrompt: {x_p}",
                                                    neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans, label in zip(pre_neighbor, post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
                pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''],
                                           record['locality'][locality_key]['ground_truth'],
                                           f"{record['locality'][locality_key]['prompt']}",
                                           neighborhood=True)
                post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                            record['locality'][locality_key]['ground_truth'],
                                            f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                            neighborhood=True)
                if type(pre_neighbor) is not list:
                    pre_neighbor = [pre_neighbor, ]
                if type(post_neighbor) is not list:
                    post_neighbor = [post_neighbor, ]
                assert len(pre_neighbor) == len(post_neighbor)

                ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                icl_input = ['']
                x_prefix = ""
            else:
                icl_input = icl_examples
                x_prefix = f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'], list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],
                                    record['portability'][portability_key]['prompt']):
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input, x_a,
                                                      f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input,
                                              record['portability'][portability_key]['ground_truth'],
                                              f"{x_prefix}{record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok, prefixes=new_fact if isinstance(new_fact,list) else [new_fact,], max_out_len=100, vanilla_generation=False)
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
