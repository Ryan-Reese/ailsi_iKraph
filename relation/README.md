# Training Code for Relation Extraction

## Setup

The code was originally developed on python 3.6 and last tested on python 3.10.

To install dependencies, run `pip install -r requirements.txt`, or manually install the following dependencies:
* scikit-learn
* nltk
* pandas
* numpy
* scipy
* transformers
* torch
* accelerate
* datasets

## How to Reproduce

The following shell scripts should be run in order:
1. `sh data_processing.sh litcoin_data`
2. `sh run_model_single_sentence_5fcv.sh`
3. `sh run_model_single_sentence_alldata.sh`
4. `sh run_model_multi_sentence.sh`
5. `sh run_model_novelty.sh`

