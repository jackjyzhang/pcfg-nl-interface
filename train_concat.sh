#!/bin/bash
module load anaconda/2021a; source activate jack

TASK=$1 # concat, concatfreeze, concatoracle
DATASET=$2 # yelp or agnews
LR=$3 # 0.00005, 0.0001
EPOCHS=$4 # 10 if 0.00005, about 5 if 0.0001
USE_LENGTH=$5 # 0 for False, 1 for True
USE_BASIC_TEMPLATE=$6 # 0, 1
DOUBLE_TEMPLATE=$7 # 0, 1

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
if [[ $DOUBLE_TEMPLATE -eq 1 ]]; then
    DOUBLE_TEMPLATE_STR="x2"
fi
if [[ $USE_BASIC_TEMPLATE -eq 1 ]]; then
    BASIC_TEMPLATE_STR="_basictemplate"
    TEMPLATE_STR="templates/${DATASET}.lim${MAX_LENGTH}.basic${DOUBLE_TEMPLATE_STR}.template"
else
    TEMPLATE_STR="templates/${DATASET}.lim${MAX_LENGTH}.template"
fi

if [[ $USE_LENGTH -eq 1 ]]; then
    USE_LENGTH_FLAG="--use_length"
    USE_LENGTH_STR="_len"
else
    USE_LENGTH_FLAG=""
    USE_LENGTH_STR=""
fi

SAVE_DIR="ckpt/${TASK}_${FILE_STR}${USE_LENGTH_STR}_bte_lr${LR}${BASIC_TEMPLATE_STR}${DOUBLE_TEMPLATE_STR}"
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
    $USE_LENGTH_FLAG

# python ~/loop.py
