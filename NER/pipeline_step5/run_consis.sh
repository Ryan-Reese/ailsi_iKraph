#!/bin/bash
if [ -f new_ensemble_list.txt ];then
rm new_ensemble_list.txt
fi
for x in `ls pp_input/*txt`;
do
output_file=pp_result/${x##*/}
python3 consist_postprocess.py \
        -i ${x} \
        -t Litcoin_testset.json \
        --pred_ft predictions_ft.txt   \
        --token_ft test_ft.json    \
        -o pp.txt
mv pp.txt $output_file
echo $output_file >> new_ensemble_list.txt
done
