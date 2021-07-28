#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train.sh <gpu>"
  exit 1
fi
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}

modelname="base"
train_datasets="cola"
test_datasets="cola"
model_root="/root/data/mtdnn_ckpt"
bert_path="${model_root}/bert_model_${modelname}_uncased.pt"
data_dir="/root/data/mtdnn/canonical_data/bert_${modelname}_uncased_lower"
seed=2018
mask_dir="${model_root}/grad/${seed}/${modelname}/mask_cola_lr5e-5"

batch_size=32
batch_size_eval=32
grad_acc_steps=1
answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
encoder_type=5
epochs=6

prefix="rewind_cola_h0.1_f0.1"
head_mask_file="${mask_dir}/mask_14_heads"
ffn_mask_file="${mask_dir}/mask_1_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on

prefix="rewind_cola_h0.2_f0.2"
head_mask_file="${mask_dir}/mask_28_heads"
ffn_mask_file="${mask_dir}/mask_2_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on

prefix="rewind_cola_h0.3_f0.3"
head_mask_file="${mask_dir}/mask_43_heads"
ffn_mask_file="${mask_dir}/mask_3_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on

prefix="rewind_cola_h0.4_f0.4"
head_mask_file="${mask_dir}/mask_57_heads"
ffn_mask_file="${mask_dir}/mask_4_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on

prefix="rewind_cola_h0.5_f0.5"
head_mask_file="${mask_dir}/mask_72_heads"
ffn_mask_file="${mask_dir}/mask_5_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on

prefix="rewind_cola_h0.6_f0.6"
head_mask_file="${mask_dir}/mask_86_heads"
ffn_mask_file="${mask_dir}/mask_6_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on

prefix="rewind_cola_h0.7_f0.7"
head_mask_file="${mask_dir}/mask_100_heads"
ffn_mask_file="${mask_dir}/mask_7_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on

prefix="rewind_cola_h0.8_f0.8"
head_mask_file="${mask_dir}/mask_115_heads"
ffn_mask_file="${mask_dir}/mask_8_ffn"
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
--do_rewind --head_mask_file ${head_mask_file} --ffn_mask_file ${ffn_mask_file} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--tensorboard --seed ${seed} --multi_gpu_on
