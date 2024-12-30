import glob
import os

output_path = "model_single_sentence/pmbert_smooth_5epoch_alldata/fine_tuned_models"
files = glob.glob(f"{output_path}/**/predictions.csv", recursive=True)
files2 = glob.glob(f"{output_path}/**/predictions_eval.csv", recursive=True)
for this_file in files + files2:
    print(this_file)
    os.remove(this_file)

output_path = "model_single_sentence/pmbert_smooth_5epoch_5fcv/fine_tuned_models"
files = glob.glob(f"{output_path}/**/predictions.csv", recursive=True)
files2 = glob.glob(f"{output_path}/**/predictions_eval.csv", recursive=True)
for this_file in files + files2:
    print(this_file)
    os.remove(this_file)