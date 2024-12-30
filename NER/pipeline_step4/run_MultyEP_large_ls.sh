export HF_HOME=cache
cd ..
wd=$(pwd)
cd pipeline_step4
model_path=$wd/pipeline_step3/pre_trained_models
data_path=$wd/pipeline_step2/split_3

CUDA="${1}"
TRAIN=$data_path/train.json
DEV=$data_path/dev.json
#MODEL="${4}"
BERT_MODEL=$model_path/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf

LEARNING_RATE=1e-5
echo "CUDA=" $CUDA
MODEL=model/RoBERTa-large-PM-M3-Voc-1r1e-5_bs16_LS0.05_EP

RANDOM_SEED=42
NUM_TRAIN_EPOCHS=30

CUDA_VISIBLE_DEVICES="$CUDA"  python3 -u run_ner_roberta.py \
  --model_name_or_path $BERT_MODEL \
  --task_name ner \
  --train_file "$TRAIN" \
  --validation_file "$DEV" \
  --output_dir "$MODEL$NUM_TRAIN_EPOCHS" \
  --num_train_epochs="$NUM_TRAIN_EPOCHS" \
  --evaluation_strategy=epoch \
  --save_strategy=epoch \
  --save_steps=1000 \
  --learning_rate=$LEARNING_RATE \
  --per_device_train_batch_size=16 \
  --do_train \
  --do_eval \
  --max_seq_length=512 \
  --label_smoothing_factor=0.05 \
  --report_to none \
  --seed $RANDOM_SEED
