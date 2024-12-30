./run_training_all.sh
python get_ensemble_list.py
./run_test_all.sh
cp ensemble_list_runTest.dat ../pipeline_step5/ensemble_list.dat
