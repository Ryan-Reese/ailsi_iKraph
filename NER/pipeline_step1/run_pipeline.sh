python data_processing.py
python get_litcoin_article_full_text.py  -i litcoin_test_update_100.json -o data_files/fulltext/litcoin_test_fulltext.json
python get_litcoin_article_full_text.py  -i litcoin_train_update_400.json -o data_files/fulltext/litcoin_train_fulltext.json
python data_processing_fulltext.py
rm -f fulltext_train_bio.json
cp fulltext_test_bio.json ../pipeline_step5/test_ft.json
cp fulltext_test_bio.json ../pipeline_step3.5/test_ft.json
cp litcoin_test_update_100.json litcoin_train_update_400.json ../pipeline_step2
