import os
import json
import argparse

parser =  argparse.ArgumentParser(description="Generate actual training data (multi-sentence) for LitCoin Phase 2.")
parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
parser.add_argument("--abstract_file", action="store", dest="abstract_file", default="all_abstracts.json", help="Path to all LitCoin phase 2 abstract ids, use none to not use this file.")
args = parser.parse_args()

train_data_path = os.path.join(args.data_path, "split", "multi_sentence_train.json")
val_data_path = os.path.join(args.data_path, "split", "multi_sentence_val.json")
output_dir = os.path.join(args.data_path, "multi_sentence_split")
os.makedirs(output_dir, exist_ok=True)

all_data = json.load(open(train_data_path)) + json.load(open(val_data_path))
splits = json.load(open(args.abstract_file))

def flatten_list(l_of_l):
    ret = []
    for l in l_of_l:
        for elem in l:
            ret.append(elem)
    return ret

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
    all_train_ids = [all_splits[idx] for idx in split_train]
    all_train_ids = flatten_list(all_train_ids)
    all_val_ids = [all_splits[idx] for idx in split_val]
    all_val_ids = flatten_list(all_val_ids)
    this_train = []
    this_val = []
    for elem in all_data:
        if elem["abstract_id"] in all_train_ids:
            this_train.append(elem)
        else:
            this_val.append(elem)
    json.dump(this_train, open(os.path.join(output_dir, f"train_{split_id}.json"), "w"), indent=4)
    json.dump(this_val, open(os.path.join(output_dir, f"val_{split_id}.json"), "w"), indent=4)
    if split_id == 0:
        json.dump(this_train + this_val, open(os.path.join(output_dir, f"train_all.json"), "w"), indent=4)