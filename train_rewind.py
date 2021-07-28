# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pretrained_models import *
from tensorboardX import SummaryWriter
from experiments.exp_def import TaskDefs
from mt_dnn.inference import eval_model, extract_encoding, calculate_importance
from data_utils.log_wrapper import create_logger
from data_utils.task_def import EncoderModelType
from data_utils.utils import set_environment
from data_utils.mask_utils import load_head_mask_file, load_ffn_mask_file, gen_random_head, gen_random_ffn
from mt_dnn.batcher import SingleTaskDataset, MultiTaskDataset, Collater, MultiTaskBatchSampler
from mt_dnn.model import MTDNNModel
from masking import what_to_mask, what_to_threshold, determine_masking_sequence

def model_config(parser):
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--max_answer_len', type=int, default=10)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=EncoderModelType.BERT)
    parser.add_argument('--num_hidden_layers', type=int, default=-1)

    # BERT pre-training
    parser.add_argument('--bert_model_type', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--short_seq_prob', type=float, default=0.2)
    parser.add_argument('--max_predictions_per_seq', type=int, default=128)
    return parser


def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tensorboard_logdir', default='tensorboard_logdir')
    parser.add_argument("--init_checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str)
    parser.add_argument("--eval_checkpoint", default='', type=str)
    parser.add_argument('--data_dir', default='data/canonical_data/bert_uncased_lower')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default="experiments/glue/glue_task_def.yml")
    parser.add_argument('--train_datasets', default='mnli')
    parser.add_argument('--test_datasets', default='mnli_matched,mnli_mismatched')
    parser.add_argument('--glue_format_on', default=1)
    parser.add_argument('--mkd-opt', type=int, default=0,
                        help=">0 to turn on knowledge distillation, requires 'softlabel' column in input data")
    parser.add_argument('--do_padding', action='store_true')
    return parser


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # loading
    parser.add_argument("--model_ckpt", default='checkpoints/model_0.pt', type=str)
    parser.add_argument("--resume", action='store_true')

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)

    # fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # adv training
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--adv_opt', default=0, type=int)
    parser.add_argument('--adv_p_norm', default='inf', type=str)
    parser.add_argument('--adv_alpha', default=1, type=float)
    parser.add_argument('--adv_k', default=1, type=int)
    parser.add_argument('--adv_step_size', default=1e-3, type=float)
    parser.add_argument('--adv_noise_var', default=1e-5, type=float)
    parser.add_argument('--adv_epsilon', default=1e-6, type=float)
    return parser

def mask_config(parser):
    # head masking
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_mask', action='store_true')
    parser.add_argument('--do_rewind', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument("--trained_checkpoint", default='', type=str)
    parser.add_argument('--save_rewind_model', action='store_true')

    parser.add_argument('--normalize_scores_by_layer', action='store_true')
    parser.add_argument('--random_rewind', action='store_true')
    parser.add_argument('--preserve_layer_ratio', action='store_true', help='only used in random rewind mode.')
    parser.add_argument('--mask_number', default=None, nargs="*", type=int)
    parser.add_argument('--mask_percent', default=[50], nargs="*", type=float)
    parser.add_argument('--mask_threshold', default=[], nargs="*", type=float)
    parser.add_argument('--mask_reverse', action='store_true')
    parser.add_argument('--min_number_attention_heads', default=1, type=int)
    parser.add_argument("--head_mask_file", default=None, type=str)
    parser.add_argument("--ffn_mask_file", default=None, type=str)
    return parser

parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
parser = mask_config(parser)
parser.add_argument('--encode_mode', action='store_true', help="only encode test data")

args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir
args.train_datasets = args.train_datasets.split(',')
args.test_datasets = args.test_datasets.split(',')
pprint(args)

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(args.seed, args.cuda)
log_path = args.log_file
logger = create_logger(__name__, to_disk=True, log_file=log_path)
logger.info(args.answer_opt)

task_defs = TaskDefs(args.task_def)
encoder_type = args.encoder_type

def dump(path, data, convert_key_type=False):

    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj.item(), np.integer):
            return int(obj.item())
        elif isinstance(obj.item(), float):
            return obj.item()
        raise TypeError

    if convert_key_type:
        d = {}
        for k, v1 in data.items():
            v1 =  {int(k):v2 for k,v2 in v1.items()}
            d[int(k)] = v1
    else:
        d = data
    with open(path, 'w') as f:
        json.dump(d, f, default=set_default)

def load_init_model(args=None, init_model=None, opt=None, num_all_batches=0):
    '''
    load the initial model/config and reset the optimizers
    '''
    state_dict = None
    if os.path.exists(init_model):
        if encoder_type in [EncoderModelType.BERT, EncoderModelType.HMBERT]:
            state_dict = torch.load(init_model)
            config = {}
            for key, val in state_dict['config'].items():
                if key in ['vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', \
                           'hidden_act', 'intermediate_size', \
                           'max_position_embeddings', 'type_vocab_size', \
                           'initializer_range']:
                    config[key] = val
        elif encoder_type == EncoderModelType.ROBERTA:
            model_path = '{}/model.pt'.format(init_model)
            state_dict = torch.load(model_path)
            arch = state_dict['args'].arch
            arch = arch.replace('_', '-')
            # convert model arch
            from data_utils.roberta_utils import update_roberta_keys
            from data_utils.roberta_utils import patch_name_dict
            state = update_roberta_keys(state_dict['model'], nlayer=state_dict['args'].encoder_layers)
            state = patch_name_dict(state)
            literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
            config = config_class.from_pretrained(arch).to_dict()
            state_dict = {'state': state}
    else:
        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
        config = config_class.from_pretrained(init_model).to_dict()

    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p
    config['multi_gpu_on'] = opt["multi_gpu_on"]
    if args.num_hidden_layers != -1:
        config['num_hidden_layers'] = args.num_hidden_layers

    opt.update(config)
    model = MTDNNModel(opt, state_dict=state_dict, num_train_step=num_all_batches)
    if args.resume and args.model_ckpt:
        logger.info('loading model from {}'.format(args.model_ckpt))
        model.load(args.model_ckpt)

    return model, config, opt

def main():
    logger.info('Launching the MT-DNN training')
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    batch_size = args.batch_size

    tasks = {}
    task_def_list = []
    dropout_list = []

    train_collater = Collater(dropout_w=args.dropout_w, encoder_type=encoder_type, soft_label=args.mkd_opt > 0, max_seq_len=args.max_seq_len, do_padding=args.do_padding)
    test_collater = Collater(is_train=False, encoder_type=encoder_type, max_seq_len=args.max_seq_len, do_padding=args.do_padding)

    train_datasets = []
    train_data_list = []
    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks:
            continue
        task_id = len(tasks)
        tasks[prefix] = task_id
        task_def = task_defs.get_task_def(prefix)
        task_def_list.append(task_def)
        train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
        logger.info('Loading {} as task {}'.format(train_path, task_id))
        train_data_set = SingleTaskDataset(train_path, True, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def)
        train_datasets.append(train_data_set)
        train_data = DataLoader(train_data_set, batch_size=args.batch_size_eval, collate_fn=train_collater.collate_fn, pin_memory=args.cuda)
        train_data_list.append(train_data)
    multi_task_train_dataset = MultiTaskDataset(train_datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(train_datasets, args.batch_size, args.mix_opt, args.ratio)
    multi_task_train_data = DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler, collate_fn=train_collater.collate_fn, pin_memory=args.cuda)

    opt['task_def_list'] = task_def_list

    dev_data_list = []
    test_data_list = []
    for dataset in args.test_datasets:
        prefix = dataset.split('_')[0]
        task_def = task_defs.get_task_def(prefix)
        task_id = tasks[prefix]
        task_type = task_def.task_type
        data_type = task_def.data_type
        dev_path = os.path.join(data_dir, '{}_dev.json'.format(dataset))
        dev_data = None
        if os.path.exists(dev_path):
            dev_data_set = SingleTaskDataset(dev_path, False, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def)
            dev_data = DataLoader(dev_data_set, batch_size=args.batch_size_eval, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        dev_data_list.append(dev_data)

        test_path = os.path.join(data_dir, '{}_test.json'.format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data_set = SingleTaskDataset(test_path, False, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def)
            test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        test_data_list.append(test_data)

    logger.info('#' * 20)
    logger.info(opt)
    logger.info('#' * 20)

    # div number of grad accumulation.
    num_all_batches = args.epochs * len(multi_task_train_data) // args.grad_accumulation_step
    logger.info('############# Gradient Accumulation Info #############')
    logger.info('number of step: {}'.format(args.epochs * len(multi_task_train_data)))
    logger.info('number of grad grad_accumulation step: {}'.format(args.grad_accumulation_step))
    logger.info('adjusted number of step: {}'.format(num_all_batches))
    logger.info('############# Gradient Accumulation Info #############')

    if not args.do_train and args.do_mask:
        model, config, opt = load_init_model(args, args.trained_checkpoint, opt, num_all_batches)
    elif args.do_eval:
        model, config, opt = load_init_model(args, args.eval_checkpoint, opt, num_all_batches)
    else:
        model, config, opt = load_init_model(args, args.init_checkpoint, opt, num_all_batches)

    #### model meta str
    headline = '############# Model Arch of MT-DNN #############'
    ### print network
    logger.info('\n{}\n{}\n'.format(headline, model.network))

    # dump config
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as writer:
        writer.write('{}\n'.format(json.dumps(opt)))
        writer.write('\n{}\n{}\n'.format(headline, model.network))

    logger.info("Total number of params: {}".format(model.total_param))

    # tensorboard
    if args.tensorboard:
        args.tensorboard_logdir = os.path.join(args.output_dir, args.tensorboard_logdir)
        tensorboard = SummaryWriter(log_dir=args.tensorboard_logdir)

    if args.encode_mode:
        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            test_data = test_data_list[idx]
            with torch.no_grad():
                encoding = extract_encoding(model, test_data, use_cuda=args.cuda)
            torch.save(encoding, os.path.join(output_dir, '{}_encoding.pt'.format(dataset)))
        return

    if args.do_train:

        logger.info('################ Train & Evaluation ################')

        for epoch in range(0, args.epochs):
            logger.warning('At epoch {}'.format(epoch))
            start = datetime.now()

            for i, (batch_meta, batch_data) in enumerate(multi_task_train_data):

                batch_meta, batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)
                task_id = batch_meta['task_id']
                model.update(batch_meta, batch_data)
                if (model.local_updates) % (args.log_per_updates * args.grad_accumulation_step) == 0 or model.local_updates == 1:
                    ramaining_time = str((datetime.now() - start) / (i + 1) * (len(multi_task_train_data) - i - 1)).split('.')[0]
                    logger.info('Task [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(task_id,
                                                                                                        model.updates,
                                                                                                        model.train_loss.avg,
                                                                                                        ramaining_time))
                    if args.tensorboard:
                        tensorboard.add_scalar('train/loss', model.train_loss.avg, global_step=model.updates)

                if args.save_per_updates_on and ((model.local_updates) % (args.save_per_updates * args.grad_accumulation_step) == 0):
                    model_file = os.path.join(output_dir, 'model_{}_{}.pt'.format(epoch, model.updates))
                    logger.info('Saving mt-dnn model to {}'.format(model_file))
                    model.save(model_file)

            for idx, dataset in enumerate(args.test_datasets):

                prefix = dataset.split('_')[0]
                task_def = task_defs.get_task_def(prefix)
                label_dict = task_def.label_vocab
                dev_data = dev_data_list[idx]
                test_data = test_data_list[idx]

                if dev_data is not None:
                    with torch.no_grad():
                        dev_metrics, dev_predictions, scores, golds, dev_ids= eval_model(model,
                                                                                        dev_data,
                                                                                        metric_meta=task_def.metric_meta,
                                                                                        use_cuda=args.cuda,
                                                                                        label_mapper=label_dict,
                                                                                        task_type=task_def.task_type)
                    for key, val in dev_metrics.items():
                        if args.tensorboard:
                            tensorboard.add_scalar('dev/{}/{}'.format(dataset, key), val, global_step=epoch)
                        if isinstance(val, str):
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                        else:
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}: {3:.3f}'.format(dataset, epoch, key, val))
                    score_file = os.path.join(output_dir, '{}_dev_scores_{}.json'.format(dataset, epoch))
                    results = {'metrics': dev_metrics, 'predictions': dev_predictions, 'uids': dev_ids, 'scores': scores}
                    dump(score_file, results)
                    if args.glue_format_on:
                        from experiments.glue.glue_utils import submit
                        official_score_file = os.path.join(output_dir, '{}_dev_scores_{}.tsv'.format(dataset, epoch))
                        submit(official_score_file, results, label_dict)

                if test_data is not None:
                    with torch.no_grad():
                        test_metrics, test_predictions, test_scores, test_golds, test_ids= eval_model(model,
                                                                                                    test_data,
                                                                                                    metric_meta=task_def.metric_meta,
                                                                                                    use_cuda=args.cuda,
                                                                                                    with_label=False,
                                                                                                    label_mapper=label_dict,
                                                                                                    task_type=task_def.task_type)
                    for key, val in test_metrics.items():
                        if args.tensorboard:
                            tensorboard.add_scalar('test/{}/{}'.format(dataset, key), val, global_step=epoch)
                        if isinstance(val, str):
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                        elif isinstance(val, float):
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}: {3:.3f}'.format(dataset, epoch, key, val))
                        else:
                            test_metrics[key] = str(val)
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                    score_file = os.path.join(output_dir, '{}_test_scores_{}.json'.format(dataset, epoch))
                    results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': test_scores}
                    dump(score_file, results)
                    if args.glue_format_on:
                        from experiments.glue.glue_utils import submit
                        official_score_file = os.path.join(output_dir, '{}_test_scores_{}.tsv'.format(dataset, epoch))
                        submit(official_score_file, results, label_dict)

            model_file = os.path.join(output_dir, 'model_{}.pt'.format(epoch))
            model.save(model_file)

    if args.do_mask:

        logger.info('############### Compute the head and ffn mask for each task ##################')

        n_heads = model.network.bert.config.num_attention_heads
        n_layers = model.network.bert.config.num_hidden_layers

        # Calculate importance scores for each task at each layer
        # head_importance: {task_idx:{layer_idx:[score_list]}}
        head_importance, ffn_importance = calculate_importance(
            args,
            model,
            train_data_list,
            n_heads=n_heads,
            n_layers=n_layers)

        logger.info('Head importance scores:')
        for idx, layer_score in head_importance.items():
            logger.info('Task {}:'.format(idx))
            for layer in range(n_layers):
                layer_scores = head_importance[idx][layer].cpu().data
                logger.info("\t".join(f"{x:.5f}" for x in layer_scores))

        logger.info('FFN importance scores:')
        logger.info('Task {}:'.format(idx))
        for idx, layer_score in ffn_importance.items():
            logger.info("\t".join(f"{x:.5f}" for x in layer_score))

        # Generate head mask json file
        # Determine the sequence of heads to mask
        mask_sequence = determine_masking_sequence(args,
                                                   n_heads=n_heads,
                                                   n_layers=n_layers) # [int, int, ...]
        heads_to_mask = {}
        tot_masked = 0
        for step, n_to_mask in enumerate(mask_sequence):

            # heads_to_mask: {task_idx: {layer_idx:set(head_idx)}
            heads_to_mask = what_to_mask(
                head_importance,
                n_to_mask,
                heads_to_mask,
                args.min_number_attention_heads,
                n_heads=n_heads,
                n_layers=n_layers,
                reverse=args.mask_reverse)

            tot_masked += n_to_mask
            logger.info('Number of heads to be masked: {}'.format(tot_masked))
            for idx, idx_heads_to_mask in heads_to_mask.items():
                logger.info('Task {}:'.format(idx))
                logger.info('{}\n'.format(idx_heads_to_mask))

            head_mask_dir = os.path.join(output_dir, 'mask_{}_heads'.format(tot_masked))
            if not os.path.exists(head_mask_dir):
                os.mkdir(head_mask_dir)
            head_mask_file = os.path.join(head_mask_dir, 'heads_to_mask.json')
            dump(head_mask_file, heads_to_mask)

        # Generate ffn mask json file
        # ffn_to_mask: {task_idx: [ffn_idx]}
        ffn_to_mask = {}
        for idx, layer_score in ffn_importance.items():
            ffn_to_mask[idx] = torch.argsort(torch.abs(layer_score)).cpu().tolist()

        for layer_idx in range(n_layers):
            logger.info('Number of ffn to be masked: {}'.format(layer_idx+1))

            for idx, idx_ffn_to_mask in ffn_to_mask.items():
                logger.info('Task {}:'.format(idx))
                if not args.mask_reverse:
                    logger.info('{}\n'.format(idx_ffn_to_mask[:layer_idx+1]))
                else:
                    logger.info('{}\n'.format(idx_ffn_to_mask[-(layer_idx+1):]))
            ffn_mask_dir = os.path.join(output_dir, 'mask_{}_ffn'.format(layer_idx+1))

            if not os.path.exists(ffn_mask_dir):
                os.mkdir(ffn_mask_dir)
            ffn_mask_file = os.path.join(ffn_mask_dir, 'ffn_to_mask.json')

            if not args.mask_reverse:
                dump(ffn_mask_file, {idx:idx_ffn_to_mask[:layer_idx+1] for idx, idx_ffn_to_mask in ffn_to_mask.items()})
            else:
                dump(ffn_mask_file, {idx:idx_ffn_to_mask[-(layer_idx+1):] for idx, idx_ffn_to_mask in ffn_to_mask.items()})

    if args.do_rewind:

        print('################ Rewind & Evaluation ###################')

        # Re-initialize the model from pre-trained bert
        if args.do_mask:
            model, config, _ = load_init_model(args, args.init_checkpoint, opt, num_all_batches)
        n_heads = model.network.bert.config.num_attention_heads
        n_layers = model.network.bert.config.num_hidden_layers

        assert args.head_mask_file is not None
        head_mask_file = os.path.join(args.head_mask_file, 'heads_to_mask.json')
        heads_to_mask = load_head_mask_file(head_mask_file)

        assert args.ffn_mask_file is not None
        ffn_mask_file = os.path.join(args.ffn_mask_file, 'ffn_to_mask.json')
        ffn_to_mask = load_ffn_mask_file(ffn_mask_file)

        if args.random_rewind:
            heads_to_mask = gen_random_head(heads_to_mask,
                                            n_heads = n_heads,
                                            n_layers = n_layers,
                                            preserve_layer_ratio = args.preserve_layer_ratio)
            ffn_to_mask = gen_random_ffn(ffn_to_mask, n_layers = n_layers)

            logger.info('Random heads to be masked for each task:')
            for idx, idx_heads_to_mask in heads_to_mask.items():
                logger.info('Task {}:'.format(idx))
                logger.info('{}\n'.format(idx_heads_to_mask))
            rand_head_mask_file = os.path.join(output_dir, 'rand_heads_to_mask.json')
            dump(rand_head_mask_file, heads_to_mask, convert_key_type=True)

            logger.info('Random ffn to be masked for each task:')
            for idx, idx_ffn_to_mask in ffn_to_mask.items():
                logger.info('Task {}:'.format(idx))
                logger.info('{}\n'.format(idx_ffn_to_mask))
            rand_fnn_mask_file = os.path.join(output_dir, 'rand_ffn_to_mask.json')
            dump(rand_fnn_mask_file, ffn_to_mask)

        # dump config
        config_file = os.path.join(output_dir, 'rewind_config.json')
        with open(config_file, 'w', encoding='utf-8') as writer:
            writer.write('{}\n'.format(json.dumps(opt)))
            writer.write('\n{}\n{}\n'.format(headline, model.network))
        current_num_param = model.total_param \
                          - 2 * opt['intermediate_size'] * opt['hidden_size'] * int(args.ffn_mask_file.split("/")[-1].split("_")[1]) \
                          - 4 * int(opt['hidden_size'] / opt['num_attention_heads'] * opt['hidden_size']) * int(args.head_mask_file.split("/")[-1].split("_")[1])
        logger.info("Total number of params: {}".format(current_num_param))
        logger.info("Remaining weight percentage: {}".format(current_num_param/model.total_param))

        # Rewind
        for epoch in range(0, args.epochs):

            logger.warning('At epoch {}'.format(epoch))
            start = datetime.now()
            for i, (batch_meta, batch_data) in enumerate(multi_task_train_data):

                batch_meta, batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)
                task_id = batch_meta['task_id']
                if task_id in heads_to_mask:
                    model.mask_heads(heads_to_mask[task_id])
                if task_id in ffn_to_mask:
                    model.mask_ffn(ffn_to_mask[task_id])
                model.update(batch_meta, batch_data)

                if (model.local_updates) % (args.log_per_updates * args.grad_accumulation_step) == 0 or model.local_updates == 1:
                    ramaining_time = str((datetime.now() - start) / (i + 1) * (len(multi_task_train_data) - i - 1)).split('.')[0]
                    logger.info('Task [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(task_id,
                                                                                                        model.updates,
                                                                                                        model.train_loss.avg,
                                                                                                        ramaining_time))
                    if args.tensorboard:
                        tensorboard.add_scalar('rewind_train/loss', model.train_loss.avg, global_step=model.updates)

                if args.save_per_updates_on and ((model.local_updates) % (args.save_per_updates * args.grad_accumulation_step) == 0):
                    model_file = os.path.join(output_dir, 'rewind_model_{}_{}.pt'.format(epoch, model.updates))
                    logger.info('Saving mt-dnn rewind model to {}'.format(model_file))
                    model.save(model_file)

                model.clear_heads()
                model.clear_ffn()

            for idx, dataset in enumerate(args.test_datasets):

                prefix = dataset.split('_')[0]
                task_def = task_defs.get_task_def(prefix)
                task_id = tasks[prefix]

                if task_id in heads_to_mask:
                    model.mask_heads(heads_to_mask[task_id])
                if task_id in ffn_to_mask:
                    model.mask_ffn(ffn_to_mask[task_id])

                label_dict = task_def.label_vocab
                dev_data = dev_data_list[idx]
                test_data = test_data_list[idx]

                if dev_data is not None:
                    with torch.no_grad():
                        dev_metrics, dev_predictions, scores, golds, dev_ids= eval_model(model,
                                                                                        dev_data,
                                                                                        metric_meta=task_def.metric_meta,
                                                                                        use_cuda=args.cuda,
                                                                                        label_mapper=label_dict,
                                                                                        task_type=task_def.task_type)
                    for key, val in dev_metrics.items():
                        if args.tensorboard:
                            tensorboard.add_scalar('rewind_dev/{}/{}'.format(dataset, key), val, global_step=epoch)
                        if isinstance(val, str):
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                        else:
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}: {3:.3f}'.format(dataset, epoch, key, val))
                    score_file = os.path.join(output_dir, '{}_rewind_dev_scores_{}.json'.format(dataset, epoch))
                    results = {'metrics': dev_metrics, 'predictions': dev_predictions, 'uids': dev_ids, 'scores': scores}
                    dump(score_file, results)
                    if args.glue_format_on:
                        from experiments.glue.glue_utils import submit
                        official_score_file = os.path.join(output_dir, '{}_rewind_dev_scores_{}.tsv'.format(dataset, epoch))
                        submit(official_score_file, results, label_dict)

                if test_data is not None:
                    with torch.no_grad():
                        test_metrics, test_predictions, test_scores, test_golds, test_ids= eval_model(model,
                                                                                                    test_data,
                                                                                                    metric_meta=task_def.metric_meta,
                                                                                                    use_cuda=args.cuda,
                                                                                                    with_label=False,
                                                                                                    label_mapper=label_dict,
                                                                                                    task_type=task_def.task_type)
                    for key, val in test_metrics.items():
                        if args.tensorboard:
                            tensorboard.add_scalar('rewind_test/{}/{}'.format(dataset, key), val, global_step=epoch)
                        if isinstance(val, str):
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                        elif isinstance(val, float):
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}: {3:.3f}'.format(dataset, epoch, key, val))
                        else:
                            test_metrics[key] = str(val)
                            logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                    score_file = os.path.join(output_dir, '{}_rewind_test_scores_{}.json'.format(dataset, epoch))
                    results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': test_scores}
                    dump(score_file, results)
                    if args.glue_format_on:
                        from experiments.glue.glue_utils import submit
                        official_score_file = os.path.join(output_dir, '{}_rewind_test_scores_{}.tsv'.format(dataset, epoch))
                        submit(official_score_file, results, label_dict)

                model.clear_heads()
                model.clear_ffn()

            if args.save_rewind_model:
                model_file = os.path.join(output_dir, 'rewind_model_{}.pt'.format(epoch))
                model.save(model_file)

    if args.do_eval:

        logger.info('############### Evaluate the model ##################')

        if args.head_mask_file is not None:
            head_mask_file = os.path.join(args.head_mask_file, 'heads_to_mask.json')
            heads_to_mask = load_head_mask_file(head_mask_file)
        else:
            heads_to_mask = None

        if args.ffn_mask_file is not None:
            ffn_mask_file = os.path.join(args.ffn_mask_file, 'ffn_to_mask.json')
            ffn_to_mask = load_ffn_mask_file(ffn_mask_file)
        else:
            ffn_to_mask = None

        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            task_def = task_defs.get_task_def(prefix)
            if heads_to_mask is not None:
                task_id = tasks[prefix]
                if task_id in heads_to_mask:
                    model.mask_heads(heads_to_mask[task_id])
            if ffn_to_mask is not None:
                task_id = tasks[prefix]
                if task_id in ffn_to_mask:
                    model.mask_ffn(ffn_to_mask[task_id])

            label_dict = task_def.label_vocab
            dev_data = dev_data_list[idx]
            test_data = test_data_list[idx]
            if dev_data is not None:
                with torch.no_grad():
                    dev_metrics, dev_predictions, scores, golds, dev_ids= eval_model(model,
                                                                                    dev_data,
                                                                                    metric_meta=task_def.metric_meta,
                                                                                    use_cuda=args.cuda,
                                                                                    label_mapper=label_dict,
                                                                                    task_type=task_def.task_type)
                for key, val in dev_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar('dev/{}/{}'.format(dataset, key), val, global_step=epoch)
                    if isinstance(val, str):
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                    else:
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}: {3:.3f}'.format(dataset, epoch, key, val))
                score_file = os.path.join(output_dir, '{}_dev_scores_{}.json'.format(dataset, epoch))
                results = {'metrics': dev_metrics, 'predictions': dev_predictions, 'uids': dev_ids, 'scores': scores}
                dump(score_file, results)
                if args.glue_format_on:
                    from experiments.glue.glue_utils import submit
                    official_score_file = os.path.join(output_dir, '{}_dev_scores_{}.tsv'.format(dataset, epoch))
                    submit(official_score_file, results, label_dict)

            if test_data is not None:
                with torch.no_grad():
                    test_metrics, test_predictions, test_scores, test_golds, test_ids= eval_model(model,
                                                                                                test_data,
                                                                                                metric_meta=task_def.metric_meta,
                                                                                                use_cuda=args.cuda,
                                                                                                with_label=False,
                                                                                                label_mapper=label_dict,
                                                                                                task_type=task_def.task_type)
                for key, val in test_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar('test/{}/{}'.format(dataset, key), val, global_step=epoch)
                    if isinstance(val, str):
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                    elif isinstance(val, float):
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}: {3:.3f}'.format(dataset, epoch, key, val))
                    else:
                        test_metrics[key] = str(val)
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                score_file = os.path.join(output_dir, '{}_test_scores_{}.json'.format(dataset, epoch))
                results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': test_scores}
                dump(score_file, results)
                if args.glue_format_on:
                    from experiments.glue.glue_utils import submit
                    official_score_file = os.path.join(output_dir, '{}_test_scores_{}.tsv'.format(dataset, epoch))
                    submit(official_score_file, results, label_dict)

            if heads_to_mask is not None:
                model.clear_heads()
            if ffn_to_mask is not None:
                model.clear_ffn()

    if args.tensorboard:
        tensorboard.close()


if __name__ == '__main__':
    main()
