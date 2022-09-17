python predict_tags.py 
    --test ../data/dev/dev.txt \
    --out_dir ./trained_model \
    --model_name roberta_squad2_final_dbg \
    --gpus 1 \
    --encoder_model deepset/roberta-base-squad2 \
    --batch_size 32 \
    --model ./trained_model/roberta_squad2_lr_2e-5/lightning_logs/version_0 \
    --max_length 128