#!/bin/bash -e

pretrain="../pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
export HF_HOME=cache
for rs in 1 2 3 4 6 7 8 9 10 12 14 15 40 42;
do
echo "roberta_BS16_lr3e-5_RS${rs}" 
python -u modeling.py --batch_size=16 \
                  --RD ${rs} \
                  --train_set train.json \
                  --valid_set devel.json \
                  --test_set  test.json \
                  --pretrain ${pretrain} \
	              --epochs 20 \
                  --output_path "roberta_BS16_lr3e-5_RS${rs}" \
2>&1 |tee log.roberta_BS16_lr3e-5_RS${rs} 
done
