Without going through all the details below, feel free to directly run "run_pipeline.sh".
Alternatively, run_pipeline_2.sh is a version where we improved the tokenization slightly, which may improve the final results. But it was not tested.

./run_pipeline.sh (or ./run_pipeline_2.sh)

The program will output tokenization and labeling errors, which is fine.

Either one of the above two shell scripts completes data preprocessing step. Then feel free to move to "pipeline_step2" by  cd ../pipeline_step2


For detailed explanation of this part of the code, please see following:


Phase 1: Data Processing

Required Python packages:
- NLTK
  - if "Resource punkt not found", please use the NLTK Downloader to obtain the resource:
	>>> import nltk
  	>>> nltk.download('punkt')
- SpaCy==3.0.6 or 3.0.7
- SpaCy model for biomedical data: en_core_sci_sm

Directory Set-up:
- data_files:
  - LitCoin: contains the original LitCoin .csv files: abstracts_train.csv, abstracts_test.csv, entities_train.csv
  - fulltext: contains the fulltext data files that corresonding to the original LitCoin abstracts. The data were downloaded from PubMed database using PubMed IDs. 
- temp: an emplty folder to store intermediate files.

Required programs:
1. You may need to run the following two python code to get the data processed:
- data_processing.py  # The program to process orignal .csv files to ready-to-use training and test set. 
- data_processing_fulltext.py # The program to process fulltext data 
2. You don't need to run the following code directly. These will be used in previous two pieces of code.
- sent_split_post_nltk.py # The program for refining sentence splitting results. 
- sent_split_post_nltk_fulltext.py # The same as before. Used for processing full text.
- NER_labeling.py # Label NER tag and check errors.


#1 Processing the original LitCoin data
Step 1: 
# This part is the default setting, you don't need to check if you run it on Litcoin dataset and skip to step 2.
Make sure the input & output filenames in sent_split_post_nltk.py are the same as following:
    inputfile = "temp/abstr_" + i + "_pmid_sents.txt"
    outputfile = "temp/abstr_" + i + "_pmid_sents_refined.txt"
Make sure the filenames in NER_labeling.py are the same as following:
    tokenFile = "temp/abstr_train_pmid_tokens.json"
    entityFile = "data_files/LitCoin/entities_train.csv"
    outputfile = "temp/abstr_train_bio.txt"
Save after editing.

Step 2:
Run data_processing.py: python data_processing.py

Step 3:
Get the ready-to-use training & testing datasets outputs as:
    litcoin_train_update_400.json  
    litcoin_train_update_100.json  


#2 Processing the fulltext data (Option 1: Original version)

**original version: To reproduce our results at data preprocessing step, please use this version and ignore the error messages.

Step 0:
Aquire full text data by get_litcoin_article_full_text.py:

python get_litcoin_article_full_text.py  -i litcoin_test_update_100.json -o data_files/fulltext/litcoin_test_fulltext.json
python get_litcoin_article_full_text.py  -i litcoin_train_update_400.json -o data_files/fulltext/litcoin_train_fulltext.json


Step 1:
# This part is the default setting, you don't need to check if you run it on Litcoin fulltext and skip to step 2.
Make sure the input & output filenames in sent_split_post_nltk_fulltext.py the same as following:
    inputfile = "temp/full_" + i + "_pmid_sents.txt"
    outputfile = "temp/full_" + i + "_pmid_sents_refined.txt"
Save after editing.

Step 2:
Run data_processing_fulltext.py: python data_processing_fulltext

Step 3:
Get the ready-to-use training & testing datasets outputs as:
    fulltext_train_bio.json
    fulltext_test_bio.json



#2 Processing the fulltext data (Option 2: More precise version)
Step 0:

**More precise version: Comparing to option 1, some tokenization errors were further removed. Please ignore the error messages 

Aquire full text data by get_litcoin_article_full_text.py:

python get_litcoin_article_full_text.py  -i litcoin_test_update_100.json -o data_files/fulltext/litcoin_test_fulltext.json
python get_litcoin_article_full_text.py  -i litcoin_train_update_400.json -o data_files/fulltext/litcoin_train_fulltext.json

Step 1:
# This part is the default setting, you don't need to check if you run it on Litcoin fulltext and skip to step 2.
Make sure the input & output filenames in sent_split_post_nltk_fulltext_2.py the same as following:
    inputfile = "temp/full_" + i + "_pmid_sents.txt"
    outputfile = "temp/full_" + i + "_pmid_sents_refined.txt"
Save after editing.

Step 2:
Run data_processing_fulltext_2.py: python data_processing_fulltext_2

Step 3:
Get the ready-to-use training & testing datasets outputs as:
    fulltext_train_bio.json
    fulltext_test_bio.json







