#!/bin/bash
module load anaconda/2021a; source activate jack

TOPK=$1
LMBD=$2
NUM=$3
DATASET=$4 # yelp or agnews
MULTICLASS=$5 # 0 or 1
LABEL_SAVE_DIR=$6 # sth like "ckpt/${TASK}_yelp_lim384_len_bte_xxx"
LENGTH_SAVE_DIR=$7 # can leave empty, then equivalent to no length
MIN_LENGTH=20

if [[ $DATASET == "yelp" ]]; then
    MAX_LENGTH=200
    LABEL_NUM=5
    LENGTH_NUM=5
    OUT_STR=""
elif [[ $DATASET == "agnews" ]]; then
    MAX_LENGTH=256
    LABEL_NUM=4
    LENGTH_NUM=3
    OUT_STR="_ag"
else
    exit
fi
FILE_STR="${DATASET}_lim${MAX_LENGTH}"
BACKBONE_STR="backbone/gpt2_${DATASET}_l${MAX_LENGTH}_boseos_ep5_lr0.0001"

if [[ $MULTICLASS -eq 1 ]]; then
    MULTI_FLAG="--multiclass"
    MULTI_STR="multi"
else
    MULTI_FLAG=""
    MULTI_STR=""
fi

if [[ $LENGTH_SAVE_DIR == "" ]]; then
    USE_LENGTH_STR="_nolen"
    CKPT_LENGTH=""
else
    USE_LENGTH_STR=""
    CKPT_LENGTH="--ckpt_length $LENGTH_SAVE_DIR/model_best.pth.tar"
fi

OUTPUT_DIR="generation${OUT_STR}_topk${TOPK}_lmbd${LMBD}_n${NUM}"
mkdir -p $OUTPUT_DIR
OUTPUT_PATH="${OUTPUT_DIR}/${MULTI_STR}oracle_${FILE_STR}${USE_LENGTH_STR}.txt"

echo "output_path=${OUTPUT_PATH}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python generate_oracle.py \
    --num $NUM \
    --output $OUTPUT_PATH \
    --ckpt_label "$LABEL_SAVE_DIR/model_best.pth.tar" \
    --backbone $BACKBONE_STR \
    --label_num_classes $LABEL_NUM \
    --length_num_classes $LENGTH_NUM \
    --do_sample \
    --condition_lambda_label $LMBD \
    --condition_lambda_length $LMBD \
    --topk $TOPK \
    --length_cutoff $MAX_LENGTH \
    --min_length $MIN_LENGTH \
    $CKPT_LENGTH $MULTI_FLAG

# python ~/loop.py
