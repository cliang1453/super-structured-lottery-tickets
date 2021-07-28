from math import sqrt
from logger import logger
import torch
import math

def determine_masking_sequence(args, n_heads=0, n_layers=0):
    mask_number = args.mask_number
    if mask_number is None:
        # Compute the number of heads to prune on percentage if needed
        mask_number = []
        for prune_percent in args.mask_percent:
            total_heads = n_heads * n_layers
            n_to_mask = int(total_heads * prune_percent / 100)
            # Make sure we keep at least one head per layer
            if args.min_number_attention_heads > 0:
                if n_to_mask > total_heads - args.min_number_attention_heads * n_layers:
                    n_to_mask = total_heads - args.min_number_attention_heads * n_layers
                    mask_number.append(n_to_mask)
                    break
            mask_number.append(n_to_mask)

    # We'll incrementally prune layers and evaluate
    mask_number = sorted(mask_number)
    mask_sequence = mask_number[:]
    for idx in range(1, len(mask_number)):
        mask_sequence[idx] = mask_number[idx] - mask_number[idx-1]

    # Verify that the total number of heads pruned stayed the same
    assert mask_number[-1] == sum(mask_sequence)
    return mask_sequence

def what_to_threshold(
    head_importance,
    mask_threshold,
    heads_to_mask=None,
    min_number_attention_heads=0,
    n_heads=0,
    n_layers=0,
):
    for idx, score in head_importance.items():

        if idx not in heads_to_mask:
            heads_to_mask[idx] = {}

        score = (score < mask_threshold).nonzero()
        thresholded_heads = [(score[i,0].item(), score[i,1].item()) for i in range(score.size()[0])]

        # layer/heads that were already pruned
        # Prune the lowest scoring heads
        thresholded_heads = [
            (layer, head)
            for (layer, head) in thresholded_heads
            if layer not in heads_to_mask[idx] or head not in heads_to_mask[idx][layer]
        ]

        # Update heads to prune
        for layer, head in thresholded_heads:
            if layer not in heads_to_mask[idx]:
                heads_to_mask[idx][layer] = set()
            heads_to_mask[idx][layer].add(head)

    return heads_to_mask

def what_to_mask(
    head_importance,
    n_to_mask,
    heads_to_mask=None,
    min_number_attention_heads=0,
    n_heads=0,
    n_layers=0,
    reverse=False
):
    # Sort heads by score
    for idx, score in head_importance.items():

        if idx not in heads_to_mask:
            heads_to_mask[idx] = {}

        heads_and_score = [
            ((layer, head), score[layer, head])
            for layer in range(n_layers)
            for head in range(n_heads)
        ]

        heads_and_score = sorted(heads_and_score, key=lambda x: x[1], reverse=reverse)
        sorted_heads = [head_and_score[0] for head_and_score in heads_and_score]

        # Ensure we don't delete all heads in a layer
        if min_number_attention_heads:
            # Remove the top scoring head in each layer
            to_protect = {l: 0 for l in range(n_layers)}
            filtered_sorted_heads = []
            for layer, head in reversed(sorted_heads):
                if layer in to_protect:
                    if to_protect[layer] < min_number_attention_heads:
                        to_protect[layer] += 1
                        continue
                    else:
                        to_protect.pop(layer)
                filtered_sorted_heads.insert(0, (layer, head))
            sorted_heads = filtered_sorted_heads

        # layer/heads that were already pruned
        # Prune the lowest scoring heads
        sorted_heads = [
            (layer, head)
            for (layer, head) in sorted_heads
            if layer not in heads_to_mask[idx] or head not in heads_to_mask[idx][layer]
        ]

        # Update heads to prune
        for layer, head in sorted_heads[:n_to_mask]:
            if layer not in heads_to_mask[idx]:
                heads_to_mask[idx][layer] = set()
            heads_to_mask[idx][layer].add(head)

    return heads_to_mask

def what_to_mask_iterative(
    head_importance,
    n_to_mask,
    curr_n_to_mask,
    heads_to_mask=None,
    min_number_attention_heads=0,
    n_heads=0,
    n_layers=0
):
    # Sort heads by score
    for idx, score in head_importance.items():

        if idx not in heads_to_mask:
            heads_to_mask[idx] = {}

        heads_and_score = [
            ((layer, head), score[layer, head])
            for layer in range(n_layers)
            for head in range(n_heads)
        ]
        heads_and_score = sorted(heads_and_score, key=lambda x: x[1])
        sorted_heads = [head_and_score[0] for head_and_score in heads_and_score]

        # Ensure we don't delete all heads in a layer
        if min_number_attention_heads:
            # Remove the top scoring head in each layer
            to_protect = {l: 0 for l in range(n_layers)}
            filtered_sorted_heads = []
            for layer, head in reversed(sorted_heads):
                if layer in to_protect:
                    if to_protect[layer] < min_number_attention_heads:
                        to_protect[layer] += 1
                        continue
                    else:
                        to_protect.pop(layer)
                filtered_sorted_heads.insert(0, (layer, head))
            sorted_heads = filtered_sorted_heads

        # layer/heads that were already pruned
        # Prune the lowest scoring heads
        sorted_heads = [
            (layer, head)
            for (layer, head) in sorted_heads
            if layer not in heads_to_mask[idx] or head not in heads_to_mask[idx][layer]
        ]

        # Update heads to prune
        for layer, head in sorted_heads[:curr_n_to_mask]:
            if layer not in heads_to_mask[idx]:
                heads_to_mask[idx][layer] = set()
            heads_to_mask[idx][layer].add(head)

    return heads_to_mask
