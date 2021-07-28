#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train.sh <gpu>"
  exit 1
fi
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}

modelname="base"
model_root="/root/data/mtdnn_ckpt"
bert_path="${model_root}/bert_model_${modelname}_uncased.pt"
data_dir="/root/data/mtdnn/canonical_data/bert_${modelname}_uncased_lower"
seed=2018

batch_size=32
batch_size_eval=32
optim="adamax"
grad_clipping=0
global_grad_clipping=1
encoder_type=5
answer_opt=1
grad_acc_steps=1
lr="5e-5"

###### MNLI ##########
epochs=3
train_datasets="mnli"
test_datasets="mnli_matched,mnli_mismatched"

prefix="train_mnli_lr5e-5"
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

prefix="mask_mnli_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_mnli_lr5e-5/model_2.pt"
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

###### QQP ##########
epochs=6
train_datasets="qqp"
test_datasets="qqp"

prefix="train_qqp_lr5e-5"
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

prefix="mask_qqp_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_qqp_lr5e-5/model_5.pt"
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

###### QNLI ##########
epochs=3
train_datasets="qnli"
test_datasets="qnli"

prefix="train_qnli_lr5e-5"
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

prefix="mask_qnli_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_qnli_lr5e-5/model_2.pt"
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

###### SST ##########
epochs=6
train_datasets="sst"
test_datasets="sst"

prefix="train_sst_lr5e-5"
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

prefix="mask_sst_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_sst_lr5e-5/model_5.pt"
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

###### MRPC ##########
epochs=6
train_datasets="mrpc"
test_datasets="mrpc"

prefix="train_mrpc_lr5e-5"
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

prefix="mask_mrpc_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_mrpc_lr5e-5/model_5.pt"
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

###### COLA ##########
epochs=6
train_datasets="cola"
test_datasets="cola"

prefix="train_cola_lr5e-5"
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

prefix="mask_cola_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_cola_lr5e-5/model_5.pt"
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

###### RTE ##########
epochs=6
train_datasets="rte"
test_datasets="rte"

prefix="train_rte_lr5e-5"
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

prefix="mask_rte_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_rte_lr5e-5/model_3.pt"
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

###### STSB ##########
epochs=6
train_datasets="stsb"
test_datasets="stsb"

prefix="train_stsb_lr5e-5"
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

prefix="mask_stsb_lr5e-5"
trained_path="${model_root}/grad/${seed}/${modelname}/train_stsb_lr5e-5/model_5.pt"
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
