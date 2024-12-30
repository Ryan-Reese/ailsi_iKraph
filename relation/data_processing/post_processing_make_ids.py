import json
import os
import argparse

parser =  argparse.ArgumentParser(description="Generate actual training data (multi-sentence) for LitCoin Phase 2.")
parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
parser.add_argument("--abstract_file", action="store", dest="abstract_file", default="all_abstracts.json", help="Path to all LitCoin phase 2 abstract ids, use none to not use this file.")
parser.add_argument("--all_id_file", action="store", dest="all_id_file", default="all_id.dat", help="Path to all LitCoin phase 2 abstract ids, use none to not use this file.")
args = parser.parse_args()

output_dir = "../post_processing_and_ensemble"
splits = json.load(open(args.abstract_file))

def flatten_list(l_of_l):
    ret = []
    for l in l_of_l:
        for elem in l:
            ret.append(elem)
    return ret

all_ids = []
with open(args.all_id_file) as all_ids_fp:
    for line in all_ids_fp:
        all_ids.append(int(line))


all_splits = []
for split_id in range(5):
    all_splits.append(splits[str(split_id)])

splits = [
    [[0, 1, 2, 3], [4]],
    [[1, 2, 3, 4], [0]],
    [[2, 3, 4, 0], [1]],
    [[3, 4, 0, 1], [2]],
    [[4, 0, 1, 2], [3]],
]

for split_id, (split_train, split_val) in enumerate(splits):
    this_train_ids = []
    this_val_ids = []
    all_train_ids = [all_splits[idx] for idx in split_train]
    all_train_ids = flatten_list(all_train_ids)
    all_val_ids = [all_splits[idx] for idx in split_val]
    all_val_ids = flatten_list(all_val_ids)
    for this_id in all_ids:
        if this_id in all_train_ids:
            this_train_ids.append(this_id)
        else:
            this_val_ids.append(this_id)
    with open(os.path.join(output_dir, f"train{split_id}_id.dat"), "w") as out_file:
        for elem in this_train_ids:
            out_file.write(f"{elem}\n")
    with open(os.path.join(output_dir, f"dev{split_id}_id.dat"), "w") as out_file:
        for elem in sorted(this_val_ids):
            out_file.write(f"{elem}\n")