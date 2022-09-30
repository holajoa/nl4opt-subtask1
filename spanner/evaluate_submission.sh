#!/bin/bash

# Using conda within a shell script
source ~/miniconda3/etc/profile.d/conda.sh

yes | conda create --name nl4opt-holajoa python=3.9.12
conda activate nl4opt-holajoa


# Ensure the correct interpreter is executed
echo $(which python)
echo $(which pip)


# Install all other dependencies
pip install --no-cache-dir --ignore-installed -r requirements.txt

# Upgrade Pytorch for CUDA 11.5
pip install https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp39-cp39-win_amd64.whl
pip install torchmetrics==0.6.0

# Handle the data
DATA_DIR="./data/nl4opt/"
python ./dataprocess/bio2spannerformat.py --data_dir . --output_dir $DATA_DIR --suffix test

# Evaluate trained model on test set and print results to "results.out"
export PYTHONPATH="$PWD"

PRETRAINED="bert-base-cased"
BERT_DIR=${PRETRAINED}
CHECKPOINT="D:\Develop\nl4opt-subtask1\spanner\trained_model\spanner_bert-base-cased\lightning_logs\version_0\checkpoints\epoch=03-val_loss=0.0335-val_micro@F1=0.9710.ckpt"

mkdir -p vocab/${PRETRAINED}
wget -nc https://huggingface.co/${PRETRAINED}/raw/main/vocab.txt 
mv vocab.txt vocab/${PRETRAINED}/

dataname=nl4opt-task1
n_class=7
BERT_DROPOUT=0.1
MODEL_DROPOUT=0.1
LR=1e-4
MAXLEN=200
MAXNORM=1.0
ADVERSARIAL_PARAM=1e-2
batchSize=16
max_spanLen=6
tokenLen_emb_dim=50
spanLen_emb_dim=100
morph_emb_dim=100

use_prune=True
use_spanLen=True
use_morph=True
use_span_weight=True
neg_span_weight=0.5
gpus="0,"

max_epochs=30

modelName="spanner_"${PRETRAINED}
idtest=${dataname}_${modelName}
param_name=epoch${max_epochs}_batchsize${batchSize}_lr${LR}_maxlen${MAXLEN}

OUTPUT_DIR="./trained_model/${modelName}"
#mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 python evaluate.py \
--pretrained_checkpoint $CHECKPOINT \
--dataname $dataname \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--bert_max_length $MAXLEN \
--batch_size $batchSize \
--gpus=$gpus \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $LR \
--epsilon $ADVERSARIAL_PARAM \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--model_dropout $MODEL_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs $max_epochs \
--n_class $n_class \
--max_spanLen $max_spanLen \
--tokenLen_emb_dim $tokenLen_emb_dim \
--modelName $modelName \
--spanLen_emb_dim $spanLen_emb_dim \
--morph_emb_dim $morph_emb_dim \
--use_prune $use_prune \
--use_spanLen $use_spanLen \
--use_morph $use_morph \
--use_span_weight $use_span_weight \
--neg_span_weight $neg_span_weight \
--param_name $param_name \
--gradient_clip_val $MAXNORM \
--optimizer "adamw"