import os
import json
import random
import argparse

import pandas


if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description="Split annotation data into 5 folds.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
    parser.add_argument("--abstract_file", action="store", dest="abstract_file", default="all_abstracts.json", help="Path to all LitCoin phase 2 abstract ids, use none to not use this file.")
    args = parser.parse_args()

    processed_train_path = os.path.join(args.data_path, "processed_data_train_split_annotation.json")
    processed_train = json.load(open(processed_train_path))
    train_df = pandas.DataFrame(processed_train)
    pmids = set(list(train_df["abstract_id"]))
    pmids = list(pmids)
    random.shuffle(pmids)
    total_ids = len(pmids)
    num_splits = 5
    per_split_len = total_ids // num_splits
    split_poses = [[idx*per_split_len, idx*per_split_len+per_split_len] for idx in range(0, num_splits-1)] # first n-1 splits
    split_poses.append([per_split_len*(num_splits-1), total_ids])

    pmids_per_split = []
    for split_start, split_end in split_poses:
        pmids_per_split.append(pmids[split_start: split_end])
    if args.abstract_file.lower() != "none":
        all_abstracts = json.load(open(args.abstract_file))
        for split_idx in range(5):
            pmids_per_split[split_idx] = all_abstracts[str(split_idx)]

    train_data = json.load(open(os.path.join(args.data_path, "processed_data_train_split_annotation.json")))
    train_df = pandas.DataFrame(train_data)
    copy_list_df = pandas.read_csv(os.path.join(args.data_path, "copy_list_train.csv"))

    def flatten_list(l_of_l):
        ret = []
        for l in l_of_l:
            for elem in l:
                ret.append(elem)
        return ret

    all_pmids = flatten_list(pmids_per_split)
    is_in_pmids = train_df["abstract_id"].isin(all_pmids)
    assert len(train_df[is_in_pmids]) == len(train_df)   # all in the previous pmids

    import os
    for split_id, this_pmids in enumerate(pmids_per_split):
        os.makedirs(os.path.join(args.data_path, f"annotated_data_split/split_{split_id}"), exist_ok=True)
        this_copy_list = copy_list_df[copy_list_df["abstract_id"].isin(this_pmids)]
        this_data = train_df[train_df["abstract_id"].isin(this_pmids)]

        output_data_fp = os.path.join(args.data_path, f"annotated_data_split/split_{split_id}/data.json")
        this_data.reset_index(drop=True).to_json(open(output_data_fp, "w"), indent=4, orient="table")
        output_copylist_fp = os.path.join(args.data_path, f"annotated_data_split/split_{split_id}/copy_list.csv")
        this_copy_list.reset_index(drop=True).to_csv(open(output_copylist_fp, "w"))