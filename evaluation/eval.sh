#!/bin/bash
module load anaconda/2021a; source activate jack

GENERATION_DIR=$1 # put directory name that contain generation
INPUT_FILES=($GENERATION_DIR/*)
DATASET=$2 # yelp or agnews (need to change this depending on the dataset!)
# uncomment to use not finetuned ppl
# ORIG_PPL_FLAG="--orig_ppl"

if [[ $DATASET == "yelp" ]]; then
    MAX_LENGTH=200
    LABEL_NUM=5
    LENGTH_NUM=5
elif [[ $DATASET == "agnews" ]]; then
    MAX_LENGTH=256
    LABEL_NUM=4
    LENGTH_NUM=3
else
    exit
fi
DATASET_STR="${DATASET}-lim${MAX_LENGTH}"
TEMPLATE_STR="templates/${DATASET}.lim${MAX_LENGTH}.template" # only to get length cut-offs, specific content does not matter

echo "generation dir: $GENERATION_DIR"
# echo "files: ${INPUT_FILES[@]}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python eval_alignment.py \
    --dataset $DATASET \
    --template "../${TEMPLATE_STR}" \
    --generation_file ${INPUT_FILES[@]} \
    --reference ../metadata/${DATASET_STR}-3splits-test-100.txt \
    --mauve_reference ../metadata/${DATASET_STR}-3splits-test-3000.txt \
    --sentiment ckpt/roberta_large_$DATASET \
    --comprehensive \
    --label_num_classes $LABEL_NUM \
    --length_num_classes $LENGTH_NUM $ORIG_PPL_FLAG \
    --block_class 0 -1 0 -1 1 -1 1 -1 2 -1 2 -1 3 -1 3 -1 \
    # --block_class 0 -1 1 -1 2 -1 3 -1 \
     
