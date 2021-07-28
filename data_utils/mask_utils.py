import json
import random
import numpy as np

def load_head_mask_file(path):
    with open(path) as f:
        t = json.load(f)
    heads_to_mask = {}
    for idx, v in t.items():
        heads_to_mask[int(idx)] = {}
        for layer, heads in v.items():
            heads_to_mask[int(idx)][int(layer)] = set(heads)
    return heads_to_mask

def load_ffn_mask_file(path):
    with open(path) as f:
        t = json.load(f)
    ffn_to_mask = {}
    for idx, v in t.items():
        ffn_to_mask[int(idx)] = set(v)
    return ffn_to_mask

def gen_random_head(heads_to_mask, n_heads, n_layers, preserve_layer_ratio=True):
    if preserve_layer_ratio:
        for task_id, v in heads_to_mask.items():
            for layer, heads in heads_to_mask[task_id].items():
                heads_to_mask[task_id][layer] = set(np.random.choice(n_heads, len(heads)))
    else:
        # import ipdb; ipdb.set_trace()
        tot_n_heads = {}
        for task_id, v in heads_to_mask.items():
            for layer, heads in heads_to_mask[task_id].items():
                if task_id not in tot_n_heads:
                    tot_n_heads[task_id] = 0
                tot_n_heads[task_id] += len(heads)

        for task_id, v in heads_to_mask.items():
            heads_to_mask[task_id] = {}
            random_heads = np.random.choice(n_heads * n_layers, tot_n_heads[task_id])
            for head_id in random_heads:
                if head_id // n_layers not in heads_to_mask[task_id]:
                    heads_to_mask[task_id][head_id // n_layers] = set()
                heads_to_mask[task_id][head_id // n_layers].add(head_id % n_heads)
    return heads_to_mask

def gen_random_ffn(ffn_to_mask, n_layers):
    for task_id, v in ffn_to_mask.items():
        random_ffn = np.random.choice(n_layers, len(ffn_to_mask[task_id]))
        ffn_to_mask[task_id] = set([ffn_id for ffn_id in random_ffn])
    return ffn_to_mask
