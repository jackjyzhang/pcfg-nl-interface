#!/bin/bash
module load anaconda/2021a; source activate jack

NONCOMP_LABEL_CLASS=$1 # range(LABEL_NUM_CLASSES)
TASK=$2 # concat, concatoracle, alignment
DATASET=$3 # yelp or agnews
LR=$4 # 0.00005, 0.0001
EPOCHS=$5 # 10 if 0.00005, about 5 if 0.0001

if [[ $DATASET == "yelp" ]]; then
    MAX_LENGTH=200
    LABEL_NUM_CLASSES=5
    LENGTH_NUM_CLASSES=5
elif [[ $DATASET == "agnews" ]]; then
    MAX_LENGTH=256
    LABEL_NUM_CLASSES=4
    LENGTH_NUM_CLASSES=3
else
    exit
fi
DATA_DIR="data/${DATASET}-lim${MAX_LENGTH}-3splits"
FILE_STR="${DATASET}_lim${MAX_LENGTH}"
TEMPLATE_STR="templates/${DATASET}.lim${MAX_LENGTH}.template"


SAVE_DIR="ckpt/${TASK}_${FILE_STR}_nc${NONCOMP_LABEL_CLASS}_len_bte_lr${LR}"
echo "SAVE_DIR=$SAVE_DIR"
WANDB_DIR=/home/gridsan/jzhang2/repos/nl-command/wandb \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train_cls_or_baseline.py \
    --task $TASK \
    --data_dir $DATA_DIR \
    --template_file $TEMPLATE_STR \
    --save_dir $SAVE_DIR \
    --epochs $EPOCHS \
    --batch_size 32 \
    --lr $LR \
    --no_dataparallel \
    --max_length $MAX_LENGTH \
    --label_num_classes $LABEL_NUM_CLASSES \
    --length_num_classes $LENGTH_NUM_CLASSES \
    --noncomp_label_class $NONCOMP_LABEL_CLASS \
    --use_length

# python ~/loop.py
