export HF_HOME=cache

cd ..
wd=$(pwd)
cd pipeline_step3.5

model_path=$wd/pipeline_step3/pre_trained_models


TRAIN="./split_1/litcoin_train_update_362.json"
DEV="./split_1/litcoin_dev_update_38.json"
CUDA="0"
MODEL="model"

CUDA_VISIBLE_DEVICES="$CUDA" python -u run_ner_roberta.py \
  --model_name_or_path $model_path/RoBERTa-base-PM-M3-Voc-distill-align/RoBERTa-base-PM-M3-Voc-distill-align-hf \
  --task_name ner \
  --train_file "$TRAIN" \
  --validation_file "$DEV" \
  --output_dir "$MODEL" \
  --num_train_epochs=30 \
  --save_steps=1000 \
  --save_total_limit=3 \
  --evaluation_strategy=epoch \
  --save_strategy=epoch \
  --seed 42\
  --metric_for_best_model=f1\
  --learning_rate=3e-05 \
  --label_smoothing_factor=0.05\
  --per_device_train_batch_size=16 \
  --do_train \
  --do_eval \
  --max_seq_length=512 \
  --load_best_model_at_end=True \
  
