#! /bin/bash
if true; then
sfs=(0.0 0.02)
lrs=(3e-05)
bss=(16 32)
fidxs=(0 1 2 3 4)
for sf in ${sfs[@]}
do
for lr in ${lrs[@]}
do
for bs in ${bss[@]}
do
for fidx in ${fidxs[@]}
do
python -u ./run_modeling_bert.py ./configs/config_pmbert_ls${sf}_split_${fidx}_${bs}_${lr}.json
done
done
done
done
fi

