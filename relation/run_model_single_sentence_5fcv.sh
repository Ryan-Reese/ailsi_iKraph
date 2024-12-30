export CUDA_VISIBLE_DEVICES=1
cd model_single_sentence/pmbert_smooth_5epoch_5fcv

# Train the models
bash run_pmbert_smooth_train_all.sh

# After training, run the following script to get the 3 checkpoints of each model setting:
python get_checkpoints.py

# After the 3 checkpoints for each model setting are selected, run the following script to generate config files for prediction of each selected checkpoint:
python generate_pred_configs.py

# Run the following script to produce prodictions of each selected checkpoint:
bash run_pmbert_smooth_predict.sh

# After predictions of each selected checkpoint are produced, run the following script to aggregate all predictions:
python aggregate_checkpoint_preds.py