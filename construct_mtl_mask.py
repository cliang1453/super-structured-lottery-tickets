from train_rewind import dump
import json
import os

task2id= {"mnli": 0,
    "rte": 1,
    "qqp": 2,
    "qnli": 3,
    "mrpc": 4,
    "sst": 5,
    "cola": 6,
    "stsb": 7}

def construct_mtl_mask(single_path, mtl_path, task2num, lr=""):

    heads_to_mask, ffn_to_mask = {}, {}
    head_cnt, ffn_cnt = 0, 0

    for task, num in task2num.items():

        n_head, n_ffn = num
        head_cnt += n_head
        ffn_cnt += n_ffn

        if n_head == 0:
            heads_to_mask[task2id[task]] = {}
        else:
            head_mask_file = single_path + "/mask_" + task + str(lr) + "/mask_" + str(n_head) + "_heads/heads_to_mask.json"
            with open(head_mask_file) as f:
                t = json.load(f)
                heads_to_mask[task2id[task]] = t['0']

        if n_ffn == 0:
            ffn_to_mask[task2id[task]] = []
        else:
            ffn_mask_file = single_path + "/mask_" + task + str(lr) + "/mask_" + str(n_ffn) + "_ffn/ffn_to_mask.json"
            with open(ffn_mask_file) as f:
                t = json.load(f)
                ffn_to_mask[task2id[task]] = t['0']

    avg_head_cnt = head_cnt//len(task2id)
    avg_ffn_cnt = ffn_cnt//len(task2id)

    if not os.path.exists(mtl_path):
        os.mkdir(mtl_path)

    head_mask_dir = os.path.join(mtl_path, 'mask_{}_heads'.format(avg_head_cnt))
    if not os.path.exists(head_mask_dir):
        os.mkdir(head_mask_dir)
    head_mask_file = os.path.join(head_mask_dir, 'heads_to_mask.json')
    dump(head_mask_file, heads_to_mask)

    ffn_mask_dir = os.path.join(mtl_path, 'mask_{}_ffn'.format(avg_ffn_cnt))
    if not os.path.exists(ffn_mask_dir):
        os.mkdir(ffn_mask_dir)
    ffn_mask_file = os.path.join(ffn_mask_dir, 'ffn_to_mask.json')
    dump(ffn_mask_file, ffn_to_mask)

if __name__ == '__main__':

    single_task_seed = 2018

    modelname="base"
    single_task_masks_path="/root/data/mtdnn_ckpt/grad/" + str(single_task_seed) + "/" + modelname
    mtl_head_ffn_mask_path="/root/data/mtdnn_ckpt/mtl_exp/" + str(single_task_seed) + "/" + modelname + "/head_ffn_lr5e-5"
    head_ffn_task2num = {"mnli": (0, 0),
        "rte": (14, 1),
        "qqp": (14, 1),
        "qnli": (14, 1),
        "mrpc": (43, 3),
        "sst": (28, 2),
        "cola": (0, 0), 
        "stsb": (14, 1)}
    construct_mtl_mask(single_task_masks_path, mtl_head_ffn_mask_path, head_ffn_task2num, lr="_lr5e-5")

    # modelname="large"
    # single_task_masks_path="/root/data/mtdnn_ckpt/grad/" + str(single_task_seed) + "/" + modelname
    # mtl_head_ffn_mask_path="/root/data/mtdnn_ckpt/mtl_exp/" + str(single_task_seed) + "/" + modelname + "/head_ffn_lr5e-5"
    # head_ffn_task2num = {"mnli": (38, 2),
    #     "rte": (76, 4),
    #     "qqp": (76, 4),
    #     "qnli": (19, 1),
    #     "mrpc": (115, 6),
    #     "sst": (134, 7),
    #     "cola": (153, 8),
    #     "stsb": (76, 4)}
    # construct_mtl_mask(single_task_masks_path, mtl_head_ffn_mask_path, head_ffn_task2num, lr="_lr5e-5")
