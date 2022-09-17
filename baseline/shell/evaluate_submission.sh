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

# Evaluate trained model on test set and print results to "results.out"
python evaluate.py --test ./test.txt --out_dir ./trained_model --model_name roberta_squad2_final --gpus 1 --encoder_model deepset/roberta-base-squad2 --batch_size 32 --model ./trained_model/roberta_squad2_lr_2e-5/lightning_logs/version_0 --max_length 128