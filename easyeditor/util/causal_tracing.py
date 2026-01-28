# This corresponding code blocks are implementations by the original ROME algorithm: https://github.com/kmeng01/rome

import re
import torch
import numpy
from collections import defaultdict
from . import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer

def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level

# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

# def find_token_range(tokenizer, token_array, substring):
#     toks = decode_tokens(tokenizer, token_array)
#     whole_string = "".join(toks)
#     print("whole string",whole_string)
#     print("substring", substring)
#     # input()
#     char_loc = whole_string.index(substring)
#     loc = 0
#     tok_start, tok_end = None, None
#     for i, t in enumerate(toks):
#         loc += len(t)
#         if tok_start is None and loc > char_loc:
#             tok_start = i
#         if tok_end is None and loc >= char_loc + len(substring):
#             tok_end = i + 1
#             break
#     return (tok_start, tok_end)

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


import difflib

def fuzzy_locate(sub, text):
    """
    Return (start, end) best fuzzy match of sub inside text.
    """
    # 1. Exact match
    if sub in text:
        start = text.index(sub)
        return start, start + len(sub)

    # 2. Medium fuzzy match
    matcher = difflib.SequenceMatcher(None, text, sub)
    match = matcher.find_longest_match(0, len(text), 0, len(sub))

    # Accept if at least 60% of substring length matches
    if match.size >= 0.6 * len(sub):
        start = match.a
        return start, start + match.size

    # 3. Lower fuzziness fallback
    matcher = difflib.SequenceMatcher(lambda x: x == " ", text, sub)
    match = matcher.find_longest_match(0, len(text), 0, len(sub))

    if match.size >= 0.4 * len(sub):
        start = match.a
        return start, start + match.size

    # 4. Handle encoding issues: Try matching with unicode replacement chars treated as wildcards
    # When special unicode chars (σ, π, etc) get corrupted to � during tokenization,
    # we need to match positions where � in text could correspond to any char in substring
    import re
    if '\ufffd' in text or '�' in text:
        # Character-by-character matching where � in text acts as wildcard
        for i in range(len(text) - len(sub) + 1):
            match = True
            for j in range(len(sub)):
                # If text has replacement char at this position, treat it as a match for any char
                if text[i+j] in ['\ufffd', '�']:
                    continue  # Wildcard matches anything
                # Otherwise, characters must match exactly
                if text[i+j] != sub[j]:
                    match = False
                    break
            if match:
                return i, i + len(sub)
    
    # 5. Ultra-lenient wildcard-span matching
    if '\ufffd' in text or '�' in text:
        import re

        # Split text into fixed segments around wildcard runs
        segments = re.split(r'[�\ufffd]+', text)
        segments = [s for s in segments if s]  # remove empty

        if not segments:
            return None

        pos = 0
        start = None

        for seg in segments:
            idx = text.find(seg, pos)
            if idx == -1:
                return None
            if start is None:
                start = idx
            pos = idx + len(seg)

        return start, pos

    return None


def find_token_range(tokenizer, token_array, substring, sub_ids = None):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    
    # For Llama tokenizers, individual token decodes don't include spaces
    # So we need to normalize the substring by removing spaces to match
    is_llama = "llama" in tokenizer.__class__.__name__.lower() or \
               (hasattr(tokenizer, 'name_or_path') and 'llama' in tokenizer.name_or_path.lower())
    
    search_substring = substring.replace(" ", "") if is_llama else substring

    print("whole string", whole_string)
    print("substring (original)", substring)
    print("substring (search)", search_substring)

    # Minimal crash safeguard: treat runs of replacement chars as wildcards
    import re
    # Collapse runs of � to a single wildcard
    text_c = re.sub(r'[�\ufffd]+', '¤', whole_string)
    sub_c  = re.sub(r'[^\x00-\x7F]+', '¤', search_substring)
    
    # # ------------------------------------------------------------
    # # ✅ HARD SWITCH: corrupted decode → token-ID matching (robust version)
    # # ------------------------------------------------------------
    # if ('\ufffd' in whole_string or '�' in whole_string) and sub_ids is not None:
    #     # Convert to plain lists
    #     tokens = token_array.tolist() if hasattr(token_array, "tolist") else list(token_array)
    #     sub_ids_a = sub_ids.tolist() if hasattr(sub_ids, "tolist") else list(sub_ids)
    #     n = len(sub_ids_a)

    #     # First try exact match (fast path)
    #     for i in range(len(tokens) - n + 1):
    #         if tokens[i:i+n] == sub_ids_a:
    #             return i, i + n

    #     # If exact match fails, fallback to fuzzy token match using difflib
    #     import difflib
    #     matcher = difflib.SequenceMatcher(None, tokens, sub_ids_a)
    #     match = matcher.find_longest_match(0, len(tokens), 0, len(sub_ids_a))

    #     # Accept if at least 80% of tokens match
    #     if match.size >= 0.7 * n:
    #         return match.a, match.a + match.size

    #     # If still fails, raise error
    #     raise ValueError(f"Substring '{substring}' not found via token-ID matching.")
    #     # ------------------------------------------------------------

    # # ---------- NEW: fuzzy-aware substring location ----------
    # if '\ufffd' in whole_string or '�' in whole_string:
    #     res = fuzzy_locate(sub_c, text_c)
    # else: res = fuzzy_locate(search_substring, whole_string)
    
    res = fuzzy_locate(search_substring, whole_string)
    if res is None:
        raise ValueError(f"Substring '{search_substring}' not found even with fuzzy matching.")
    char_loc, char_end = res
    # ----------------------------------------------------------

    loc = 0
    tok_start, tok_end = None, None

    for i, t in enumerate(toks):
        loc += len(t)

        if tok_start is None and loc > char_loc:
            tok_start = i

        # ---------- MODIFIED: use fuzzy end index instead of len(substring) ----------
        if tok_end is None and loc >= char_end:
            tok_end = i + 1
            break
        # ---------------------------------------------------------------------------

    return (tok_start, tok_end)

def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    # LLaMA / LLaMA-2 structure (HuggingFace)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f"model.layers.{num}" + ("" if kind is None else f".{kind}")
    assert False, "unknown transformer structure"

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    # print('inp: ', inp)
    # input()
    # print("Answert_t: ", answer_t)
    # input()
    # print("base_score: ", base_score)
    # input()
    # print("answer", answer)
    # input()
    # print("prompt: ", prompt, "; Subject: ", subject)
    # input()

    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject,
                               sub_ids = make_inputs(mt.tokenizer, [subject]* (2))["input_ids"][0])
    # print("e_range", e_range)
    
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    # print("low_score: ", low_score)
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    print("-----number of tokens: ", ntoks)
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()

def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
    index = None,
    title = None
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    # if kind == "mlp":
        # differences = []
        # differences.append( result["scores"][result['subject_range'][-1]-1] )
        # print("grabbed scores of mlp erange: ", result['subject_range'][-1]-1)
        
        # indices = [1,4,13,17,20,24]
        # if index in indices:
        #     plot_trace_heatmap(result, savepdf, modelname=modelname, title = title)
    return result["scores"] # shape (tokens, layers) or list of tokens with shape (layers,)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None, index = None, title = None):
    # for kind in [None, "mlp", "attn"]:
    return plot_hidden_flow(
        mt, prompt, subject, modelname=modelname, noise=noise, kind="mlp", index = index, title = title
    )

def plot_hidden_flow_lastSubject(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
    index = None,
    title = None
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    return result["scores"][result['subject_range'][-1]-1] # shape (tokens, layers) or list of tokens with shape (layers,)

def plot_all_flow_last_subject(mt, prompt, subject=None, noise=0.1, modelname=None, index = None, title = None):
    # for kind in [None, "mlp", "attn"]:
    return plot_hidden_flow_lastSubject(
        mt, prompt, subject, modelname=modelname, noise=noise, kind="mlp", index = index, title = title
    )

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        set_requires_grad_change=True
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype,
            )
            if set_requires_grad_change:
                nethook.set_requires_grad(False, model)
                model.eval().cuda()

        self.tokenizer = tokenizer
        self.model = model

        # ---------------------------------------------------------
        # Add LLaMA detection: model.model.layers.N
        # ---------------------------------------------------------
        llama_regex = r"^model\.layers\.\d+$"          # LLaMA, LLaMA-2
        gpt2_regex   = r"^transformer\.h\.\d+$"        # GPT-2, GPT-2-XL
        neox_regex   = r"^gpt_neox\.layers\.\d+$"      # GPT-NeoX

        combined_regex = f"({gpt2_regex})|({neox_regex})|({llama_regex})"

        self.layer_names = [
            n
            for n, m in model.named_modules()
            if re.match(combined_regex, n)
        ]

        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )
