#!/bin/bash
filename=od_oldCode_base_large_LS0.05_30model_fixCC_fixGap1
gap=1
if [ ! -d pp_result/ ];then
mkdir pp_result
fi
if [ ! -d pp_input/ ];then
mkdir pp_input
fi

python3 refine_tag_fixGap.py ensemble_list.dat $gap
echo 'Tag refine finished'
./run_consis.sh >out.cc1
echo 'First step CC finished'
python3 ensemble_noReport.py new_ensemble_list.txt $filename
echo 'Ensemble finished'
./run_consis_second_step.sh ${filename}_sampled_NER.txt >out.cc2
echo 'Second step CC finished'

if [ ! -d cache ];then
mkdir cache
fi
mv out.* phrase* pp_result/ pp_input/ cache
echo "Finish"

