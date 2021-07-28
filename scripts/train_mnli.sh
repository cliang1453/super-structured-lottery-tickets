#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train.sh <gpu>"
  exit 1
fi
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}

modelname="base"
prefix="train_mnli"
train_datasets="mnli"
test_datasets="mnli_matched,mnli_mismatched"
model_root="/root/data/mtdnn_ckpt"
bert_path="${model_root}/bert_model_${modelname}_uncased.pt"
data_dir="/root/data/mtdnn/canonical_data/bert_${modelname}_uncased_lower"
seed=2018

batch_size=32
batch_size_eval=32
grad_acc_steps=1
answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
encoder_type=5
epochs=3

model_dir="${model_root}/grad/${seed}/${modelname}/${prefix}"
log_file="${model_dir}/log.log"
python train_rewind.py \
  --data_dir ${data_dir} \
  --batch_size ${batch_size} \
  --batch_size_eval ${batch_size_eval} \
  --output_dir ${model_dir} \
  --log_file ${log_file} \
  --answer_opt ${answer_opt} \
  --grad_accumulation_step ${grad_acc_steps} --optimizer ${optim} \
  --train_datasets ${train_datasets} \
  --test_datasets ${test_datasets} \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --init_checkpoint ${bert_path} \
  --do_train \
  --epochs ${epochs} \
  --encoder_type ${encoder_type} \
  --tensorboard --seed ${seed} --multi_gpu_on

prefix="mask_mnli"
trained_path="${model_root}/grad/${seed}/${modelname}/train_mnli/model_2.pt"
mask_percent=`seq 5 5 100`
model_dir="${model_root}/grad/${seed}/${modelname}/${prefix}"
log_file="${model_dir}/log.log"

python train_rewind.py \
  --data_dir ${data_dir} \
  --batch_size ${batch_size} \
  --batch_size_eval ${batch_size_eval} \
  --output_dir ${model_dir} \
  --log_file ${log_file} \
  --answer_opt ${answer_opt} \
  --grad_accumulation_step ${grad_acc_steps} --optimizer ${optim} \
  --train_datasets ${train_datasets} \
  --test_datasets ${test_datasets} \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --init_checkpoint ${bert_path} \
  --do_mask --mask_percent ${mask_percent} --normalize_scores_by_layer --trained_checkpoint ${trained_path} \
  --epochs ${epochs} \
  --encoder_type ${encoder_type} \
  --tensorboard --seed ${seed} --multi_gpu_on

prefix="mask_reverse_mnli"
trained_path="${model_root}/grad/${seed}/${modelname}/train_mnli/model_2.pt"
mask_percent=`seq 5 5 100`
model_dir="${model_root}/grad/${seed}/${modelname}/${prefix}"
log_file="${model_dir}/log.log"

python train_rewind.py \
  --data_dir ${data_dir} \
  --batch_size ${batch_size} \
  --batch_size_eval ${batch_size_eval} \
  --output_dir ${model_dir} \
  --log_file ${log_file} \
  --answer_opt ${answer_opt} \
  --grad_accumulation_step ${grad_acc_steps} --optimizer ${optim} \
  --train_datasets ${train_datasets} \
  --test_datasets ${test_datasets} \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --init_checkpoint ${bert_path} \
  --do_mask --mask_percent ${mask_percent} --mask_reverse --normalize_scores_by_layer --trained_checkpoint ${trained_path} \
  --epochs ${epochs} \
  --encoder_type ${encoder_type} \
  --tensorboard --seed ${seed} --multi_gpu_on
