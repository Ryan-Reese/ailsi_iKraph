export HF_HOME=cache

INPUT="test_ft.json"
MODEL="model"
CUDA="0"
OUTPUT="output"

CUDA_VISIBLE_DEVICES="$CUDA" python -u run_ner_roberta.py \
  --model_name_or_path "$MODEL" \
  --task_name ner \
  --train_file "split_1/litcoin_train_update_362.json" \
  --validation_file "split_1/litcoin_dev_update_38.json" \
  --test_file "$INPUT" \
  --output_dir "$OUTPUT" \
  --do_predict
  

