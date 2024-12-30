#!/bin/bash

MODEL=roberta_Novel_punct
BATCH_SIZE=16
EPOCH=20
Learning_rate=3e-5
train_set=data/train_all.json
val_set=data/val_0.json
test_set=data/test.json
for rd in 2 12 14 119 132 401 9999;
do
OUTPUT=${MODEL}_lr${Learning_rate}_BS${BATCH_SIZE}_allTrain_RS${rd}
python modeling_novelty_punct.py \
	--batch_size=$BATCH_SIZE  \
	--epochs=$EPOCH	\
    -lr=$Learning_rate \
	--output_path=$OUTPUT \
    --RD=${rd} \
    --train_set=${train_set} \
    --valid_set=${val_set} \
    --test_set=${test_set} \
    --pretrain=../pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf \
>log.${OUTPUT} 2>err.${OUTPUT} 
done
