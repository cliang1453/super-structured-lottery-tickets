#! /bin/sh
python experiments/glue/glue_prepro.py \
  --root_dir /root/data/mtdnn
python prepro_std.py \
  --model bert-base-uncased \
  --root_dir /root/data/mtdnn/canonical_data \
  --task_def experiments/glue/glue_task_def.yml \
  --do_lower_case $1
