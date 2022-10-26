#!/bin/bash
module load anaconda/2021a; source activate jack

TOPK=$1
LMBD=$2 # only used in dir name
NUM=$3
TASK=$4 # concat or concatfreeze or concatoracle
SAVE_DIR=$5 # sth like "ckpt/${TASK}_yelp_lim384_len_bte_xxx"
DATASET=$6 # yelp or agnews
USE_LENGTH=$7 # 0 or 1
USE_TEST_TEMPLATE=$8 # 0 or 1 (if true, need _basictemplate or _regtemplate_test suffix)
SUFFIX=$9 # can leave empty
MIN_LENGTH=20

if [[ "$TASK" == "concatfreeze" ]]; then
    FREEZE_STR="--freeze"
else
    FREEZE_STR=""
fi

if [[ $DATASET == "yelp" ]]; then
    MAX_LENGTH=200
    OUT_STR=""
    LABEL_NUM_CLASSES=5
    LENGTH_NUM_CLASSES=5
elif [[ $DATASET == "agnews" ]]; then
    MAX_LENGTH=256
    OUT_STR="_ag"
    LABEL_NUM_CLASSES=4
    LENGTH_NUM_CLASSES=3
else
    exit
fi
FILE_STR="${DATASET}_lim${MAX_LENGTH}"
if [[ $USE_TEST_TEMPLATE -eq 1 ]]; then
    TEMPLATE_STR="templates/${DATASET}.lim${MAX_LENGTH}.test.template"
else
    TEMPLATE_STR="templates/${DATASET}.lim${MAX_LENGTH}.template"
fi
BACKBONE_STR="backbone/gpt2_${DATASET}_l${MAX_LENGTH}_boseos_ep5_lr0.0001"

if [[ $USE_LENGTH -eq 1 ]]; then
    USE_LENGTH_FLAG=""
    USE_LENGTH_STR=""
else
    USE_LENGTH_FLAG="--exclude_length"
    USE_LENGTH_STR="_nolen"
fi

OUTPUT_DIR="generation${OUT_STR}_topk${TOPK}_lmbd${LMBD}_n${NUM}"
mkdir -p $OUTPUT_DIR
OUTPUT_PATH="${OUTPUT_DIR}/${TASK}_${FILE_STR}${USE_LENGTH_STR}${SUFFIX}.txt"

echo "output_path=${OUTPUT_PATH}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python generate_concat.py \
    --task $TASK \
    --template $TEMPLATE_STR \
    --num $NUM \
    --output $OUTPUT_PATH \
    --ckpt "$SAVE_DIR/model_best.pth.tar" \
    --topk $TOPK \
    --comprehensive \
    --label_num_classes $LABEL_NUM_CLASSES \
    --length_num_classes $LENGTH_NUM_CLASSES \
    --min_length $MIN_LENGTH \
    $FREEZE_STR $USE_LENGTH_FLAG

# python ~/loop.py
