#!/bin/bash
input_file=${1}
outout_file=${input_file::-16}_doubleConsist.txt
python consist_postprocess.py \
        -i ${input_file} \
        -t Litcoin_testset.json \
        --pred_ft predictions_ft.txt   \
        --token_ft test_ft.json    \
        -o $outout_file

