mkdir pre_trained_models
cd pre_trained_models
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar xf RoBERTa-large-PM-M3-Voc-hf.tar.gz
cd ..

DATA_DIR=$1
mkdir $1
cd data_processing
# poetry install
mkdir nltk_data
python -c 'import nltk; nltk.download("punkt", download_dir="nltk_data"); nltk.download("words", download_dir="nltk_data"); nltk.download("punkt_tab", download_dir="nltk_data")'
# python -m nltk.downloader punkt words -d nltk_data
export NLTK_DATA=nltk_data
python gen_data.py --input_path ../original_data --output_path ../$DATA_DIR
python train_validation_split.py --data_path ../$DATA_DIR
python process_multisentence_data.py --data_path ../$DATA_DIR
python process_annotation_data.py --data_path ../$DATA_DIR
python split_annotation_data.py --data_path ../$DATA_DIR
python multi_sentence_gen_data.py --data_path ../$DATA_DIR --source_path ../original_data
python multi_sentence_split.py --data_path ../$DATA_DIR
python process_no_relations.py --data_path ../$DATA_DIR
python post_processing_make_ids.py
cd ..

cp -r ${DATA_DIR}/annotated_data_split model_single_sentence/pmbert_smooth_5epoch_5fcv/new_train_splits
cp ${DATA_DIR}/processed_data_train_split_annotation_pd.json model_single_sentence/pmbert_smooth_5epoch_5fcv/new_annotated_train_pd.json
cp ${DATA_DIR}/processed_data_test_split_pd.json model_single_sentence/pmbert_smooth_5epoch_5fcv/new_annotated_test_pd.json
cp ${DATA_DIR}/no_rel.csv model_single_sentence/pmbert_smooth_5epoch_5fcv/no_rel.csv

cp -r ${DATA_DIR}/annotated_data_split model_single_sentence/pmbert_smooth_5epoch_alldata/new_train_splits
cp ${DATA_DIR}/processed_data_train_split_annotation_pd.json model_single_sentence/pmbert_smooth_5epoch_alldata/new_annotated_train_pd.json
cp ${DATA_DIR}/processed_data_test_split_pd.json model_single_sentence/pmbert_smooth_5epoch_alldata/new_annotated_test_pd.json
cp ${DATA_DIR}/no_rel.csv model_single_sentence/pmbert_smooth_5epoch_alldata/no_rel.csv

mkdir model_novelty/data
cp ${DATA_DIR}/multi_sentence_split/*.json model_novelty/data
cp ${DATA_DIR}/split/multi_sentence_test.json model_novelty/data/test.json

cp ${DATA_DIR}/split/multi_sentence_train.json model_multi_sentence/train.json
cp ${DATA_DIR}/split/multi_sentence_val.json model_multi_sentence/devel.json
cp ${DATA_DIR}/split/multi_sentence_test.json model_multi_sentence/test.json

mkdir post_processing_and_ensemble/model_ensemble/data
cp ${DATA_DIR}/split/processed_test.json post_processing_and_ensemble/model_ensemble/data/processed_test.json
cp ${DATA_DIR}/split/copy_list_test.csv post_processing_and_ensemble/model_ensemble/data/copy_list_test.csv
