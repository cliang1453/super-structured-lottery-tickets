#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train.sh <gpu>"
  exit 1
fi
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}

train_datasets="mnli,rte,qqp,qnli,mrpc,sst,cola,stsb"
test_datasets="mnli_matched,mnli_mismatched,rte,qqp,qnli,mrpc,sst,cola,stsb"
model_root="/root/data/mtdnn_ckpt"
structure="head_ffn_lr5e-5"
prefix="mtl"

batch_size=32
batch_size_eval=32
answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
encoder_type=5
epochs=5
seed=2018

modelname="base"
bert_path="${model_root}/bert_model_${modelname}_uncased.pt"
data_dir="/root/data/mtdnn/canonical_data/bert_${modelname}_uncased_lower"
head_mask_file="${model_root}/mtl_exp/${seed}/${modelname}/${structure}/mask_15_heads"
ffn_mask_file="${model_root}/mtl_exp/${seed}/${modelname}/${structure}/mask_1_ffn"
model_dir="${model_root}/mtl_exp/${seed}/${modelname}/${structure}/${prefix}"
log_file="${model_dir}/log.log"
python train_rewind.py \
  --data_dir ${data_dir} \
  --batch_size ${batch_size} \
  --batch_size_eval ${batch_size_eval} \
  --output_dir ${model_dir} \
  --log_file ${log_file} \
  --answer_opt ${answer_opt} \
  --optimizer ${optim} \
  --train_datasets ${train_datasets} \
  --test_datasets ${test_datasets} \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --init_checkpoint ${bert_path} \
  --do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file}\
  --epochs ${epochs} \
  --encoder_type ${encoder_type} \
  --multi_gpu_on \
  --tensorboard --seed ${seed}

# modelname="large"
# bert_path="${model_root}/bert_model_${modelname}_uncased.pt"
# data_dir="/root/data/mtdnn/canonical_data/bert_${modelname}_uncased_lower"
# head_mask_file="${model_root}/mtl_exp/${seed}/${modelname}/${structure}/mask_85_heads"
# ffn_mask_file="${model_root}/mtl_exp/${seed}/${modelname}/${structure}/mask_4_ffn"
# model_dir="${model_root}/mtl_exp/${seed}/${modelname}/${structure}/${prefix}"
# log_file="${model_dir}/log.log"
# python train_rewind.py \
#   --data_dir ${data_dir} \
#   --batch_size ${batch_size} \
#   --batch_size_eval ${batch_size_eval} \
#   --output_dir ${model_dir} \
#   --log_file ${log_file} \
#   --answer_opt ${answer_opt} \
#   --optimizer ${optim} \
#   --train_datasets ${train_datasets} \
#   --test_datasets ${test_datasets} \
#   --grad_clipping ${grad_clipping} \
#   --global_grad_clipping ${global_grad_clipping} \
#   --learning_rate ${lr} \
#   --init_checkpoint ${bert_path} \
#   --do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file}\
#   --epochs ${epochs} \
#   --encoder_type ${encoder_type} \
#   --multi_gpu_on \
#   --tensorboard --seed ${seed}
