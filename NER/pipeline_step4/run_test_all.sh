export HF_HOME=cache
cd ..
wd=$(pwd)
cd pipeline_step4
data_path=$wd/pipeline_step2


CUDA="${1}"
TRAIN=$data_path/split_3/train.json
DEV=$data_path/split_3/dev.json
TEST=$data_path/Litcoin_testset.json
OUTPUT=output/

for x in `cat ensemble_list_runTest.dat`;do
  OUTPUT=output/${x:25}
  MODEL=$wd/pipeline_step4/model/${x:25}
  echo $TRAIN
  echo $DEV
  echo $TEST
  echo $MODEL
  echo $OUTPUT
  echo "------------------------------------------"
  CUDA_VISIBLE_DEVICES="$CUDA" python3 -u run_ner.py \
    --model_name_or_path "$MODEL" \
    --task_name ner \
    --train_file "$TRAIN" \
    --validation_file "$DEV"  \
    --test_file "$TEST"  \
    --output_dir "$OUTPUT" \
    --do_predict
done
