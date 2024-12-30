#! /bin/bash

if true; then
ckplist=(`ls ./pred_configs/*.json`)
for ckp in ${ckplist[@]}
do
#ls ${ckp}
python -u ./run_modeling_bert.py ${ckp}
done
fi

