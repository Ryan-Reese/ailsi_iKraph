#!/bin/bash

MODEL=roberta_Novel_punct
BATCH_SIZE=16
EPOCH=20
Learning_rate=3e-5
train_set=data/train_all.json
val_set=data/val_0.json
test_set=data/test.json
for rs in 2 12 14 119 132 401 9999;
do 
OUTPUT=${MODEL}_lr${Learning_rate}_BS${BATCH_SIZE}_allTrain_RS${rs}
for checkpoint in `ls -d $OUTPUT/checkpoint-*`;
do
checkpoint_out=${checkpoint##*/}
python pred_novelty_punct.py \
	--batch_size=$BATCH_SIZE  \
	--epochs=$EPOCH	\
    -lr=$Learning_rate \
	--output_path=output/$OUTPUT/${checkpoint_out} \
    --train_set=${train_set} \
    --valid_set=${val_set} \
    --test_set=${test_set} \
    --pretrain=$OUTPUT/${checkpoint_out} \
    --pretrainToken=${OUTPUT}/tokenizer
done
done
