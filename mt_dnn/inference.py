# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from data_utils.metrics import calc_metrics
from mt_dnn.batcher import Collater
from data_utils.task_def import TaskType
import torch
from tqdm import tqdm

def extract_encoding(model, data, use_cuda=True):
    if use_cuda:
        model.cuda()
    sequence_outputs = []
    max_seq_len = 0
    for idx, (batch_info, batch_data) in enumerate(data):
        batch_info, batch_data = Collater.patch_data(use_cuda, batch_info, batch_data)
        sequence_output = model.encode(batch_info, batch_data)
        sequence_outputs.append(sequence_output)
        max_seq_len = max(max_seq_len, sequence_output.shape[1])

    new_sequence_outputs = []
    for sequence_output in sequence_outputs:
        new_sequence_output = torch.zeros(sequence_output.shape[0], max_seq_len, sequence_output.shape[2])
        new_sequence_output[:, :sequence_output.shape[1], :] = sequence_output
        new_sequence_outputs.append(new_sequence_output)

    return torch.cat(new_sequence_outputs)

def eval_model(model, data, metric_meta, use_cuda=True, with_label=True, label_mapper=None, task_type=TaskType.Classification):
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for (batch_info, batch_data) in tqdm(data, total=len(data)):
        batch_info, batch_data = Collater.patch_data(use_cuda, batch_info, batch_data)
        score, pred, gold = model.predict(batch_info, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_info['uids'])

    if task_type == TaskType.Span:
        from experiments.squad import squad_utils
        golds = squad_utils.merge_answers(ids, golds)
        predictions, scores = squad_utils.select_answers(ids, predictions, scores)
    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    return metrics, predictions, scores, golds, ids

def calculate_importance(args, model, data_list, n_heads=0, n_layers=0):

    if args.cuda:
        model.cuda()
    head_importance = {}
    ffn_importance = {}
    for idx, dataset in enumerate(args.train_datasets):
        if idx not in head_importance:
            if args.cuda:
                head_importance[idx] = torch.zeros(n_layers, n_heads).cuda()
                ffn_importance[idx] = torch.zeros(n_layers).cuda()
            else:
                head_importance[idx] = torch.zeros(n_layers, n_heads)
                ffn_importance[idx] = torch.zeros(n_layers)
        data = data_list[idx] # data_list[idx] is a DataLoader object

        tot_len = 0
        for (batch_meta, batch_data) in tqdm(data, total=len(data)):

            batch_meta, batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)
            loss = model.predict_loss(batch_meta, batch_data)
            if args.fp16:
                with amp.scale_loss(loss, model.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            for layer in range(n_layers):
                model_layer = model.network.bert.encoder.layer[layer]
                head_importance[idx][layer] += model_layer.attention.self.heads_mask.grad
                ffn_importance[idx][layer] += model_layer.output.ffn_mask.grad[0]
            tot_len += 1

        head_importance[idx] /= tot_len
        ffn_importance[idx] /= tot_len

    # Layerwise importance normalization
    if args.normalize_scores_by_layer:
        exp = 2
        for idx, score in head_importance.items():
            norm_by_layer = torch.pow(torch.pow(score, exp).sum(-1), 1/exp)
            head_importance[idx] /= norm_by_layer.unsqueeze(-1) + 1e-20
        for idx, score in ffn_importance.items():
            norm = torch.pow(torch.pow(ffn_importance[idx], exp).sum(), 1/exp)
            ffn_importance[idx] /= norm.unsqueeze(-1) + 1e-20


    return head_importance, ffn_importance
