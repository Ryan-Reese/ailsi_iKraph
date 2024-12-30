conda deactivate
conda create --name Litcoin2 python==3.8.8
conda activate Litcoin2
python -m pip install -r requirements_step3.5.txt
python generate_input.py
./train_roberta_1GPU.sh
./predict_roberta.sh
cp output/predictions.txt output/predictions_ft.txt
mv output/predictions_ft.txt ../pipeline_step5
conda deactivate
conda activate Litcoin
