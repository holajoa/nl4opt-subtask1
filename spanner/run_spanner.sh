export PYTHONPATH="$PWD"

DATA_DIR="./data/nl4opt/"    # "data/conll03"
PRETRAINED="bert-base-uncased"
BERT_DIR=${PRETRAINED}

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
ADVERSARIAL_PARAM=1e-3
batchSize=16
max_spanLen=6
tokenLen_emb_dim=50
spanLen_emb_dim=100
morph_emb_dim=100


use_prune=False
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

CUDA_LAUNCH_BLOCKING=1 python train_model.py \
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
--optimizer "adamw" # >train_logs/${idtest}_epoch${max_epochs}_0606.txt &
