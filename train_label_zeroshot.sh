#!/bin/bash
module load anaconda/2021a; source activate jack

ZS_CLASS=$1
TASK=$2 # alignment or concat
DATASET=$3 # yelp or agnews
EXTRA_DATA=$4 # 0 for False, 1 for True
LR=$5 # 0.00005, 0.0001
EPOCHS=$6

# uncomment to unblock concat backbone
CONCAT_BB_STR="_bbunb"
CONCAT_BB_FLAG="--concat_model_path backbone/gpt2_agnews_l256_boseos_ep5_lr0.0001"

if [[ $DATASET == "yelp" ]]; then
    MAX_LENGTH=200
elif [[ $DATASET == "agnews" ]]; then
    MAX_LENGTH=256
else
    exit
fi
DATA_DIR="data/${DATASET}-lim${MAX_LENGTH}-3splits"
FILE_STR="${DATASET}_lim${MAX_LENGTH}"
TEMPLATE_STR="templates/${DATASET}.lim${MAX_LENGTH}.template"

if [[ $EXTRA_DATA -eq 1 ]]; then
    if [[ $DATASET == "yelp" ]]; then
        # Yelp extra data
        # amazon review
        AMAZON_DATA_PATH="data/amazon-lim200-zs${ZS_CLASS}"
        AMAZON_TEMPLATE_PATH="templates/amazon.lim200.template"
        # app review
        APP_DATA_PATH="data/app-lim200-zs${ZS_CLASS}"
        APP_TEMPLATE_PATH="templates/app.lim200.template"

        EXTRA_DATA_FLAG="--extra_data $AMAZON_DATA_PATH $AMAZON_TEMPLATE_PATH $APP_DATA_PATH $APP_TEMPLATE_PATH"
    else
        # AG News extra data
        # news popularity
        NEWSPOP_DATA_PATH="data/newspop-lim256"
        NEWSPOP_TEMPLATE_PATH="templates/newspop.lim256.template"
        # news category
        NEWSCAT_DATA_PATH="data/newscat-lim256"
        NEWSCAT_TEMPLATE_PATH="templates/newscat.lim256.template"
        # inshort news
        INSHORT_DATA_PATH="data/inshortnews-lim256"
        INSHORT_TEMPLATE_PATH="templates/inshortnews.lim256.template"

        EXTRA_DATA_FLAG="--extra_data $NEWSPOP_DATA_PATH $NEWSPOP_TEMPLATE_PATH $NEWSCAT_DATA_PATH $NEWSCAT_TEMPLATE_PATH $INSHORT_DATA_PATH $INSHORT_TEMPLATE_PATH"
    fi
    EXTRA_DATA_STR="_ed"
else
    EXTRA_DATA_FLAG=""
    EXTRA_DATA_STR=""
fi

SAVE_DIR="ckpt/${TASK}_${FILE_STR}_zs${ZS_CLASS}${EXTRA_DATA_STR}_lr${LR}${CONCAT_BB_STR}"
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
    --block_label_class ${ZS_CLASS} \
    --block_completely \
    $EXTRA_DATA_FLAG $CONCAT_BB_FLAG

# python ~/loop.py
