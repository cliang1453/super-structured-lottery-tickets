#!/usr/bin/env bash
##############################################################
# This script is used to download resources for MT-DNN experiments
##############################################################

BERT_DIR=/root/data/checkpoints/mtdnn
if [ ! -d ${BERT_DIR}  ]; then
  echo "Create a folder BERT_DIR"
  mkdir ${BERT_DIR}
fi

## Download bert models
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_base_v2.pt -O "${BERT_DIR}/bert_model_base_uncased.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_large_v2.pt -O "${BERT_DIR}/bert_model_large_uncased.pt"

## Download MT-DNN models
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt -O "${BERT_DIR}/mt_dnn_base_uncased.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt -O "${BERT_DIR}/mt_dnn_large_uncased.pt"

if [ "$1" == "model_only" ]; then
  exit 1
fi

DATA_DIR=/root/data/mtdnn
if [ ! -d ${DATA_DIR}  ]; then
  echo "Create a folder $DATA_DIR"
  mkdir ${DATA_DIR}
fi

## DOWNLOAD GLUE DATA
## Please refer glue-baseline install requirments or other issues.
git clone https://github.com/jsalt18-sentence-repl/jiant.git
cd jiant
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks all

cd ..
rm -rf jiant
