#! /bin/sh
python experiments/glue/split_prepro.py \
  --root_dir /root/data/mtdnn \
  --split_task mnli \
  --split_rate 100
python prepro_std.py \
  --model bert-base-uncased \
  --root_dir /root/data/mtdnn/canonical_data_split100 \
  --task_def experiments/glue/glue_task_def.yml \
  --do_lower_case $1
