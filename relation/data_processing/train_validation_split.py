import json
import os
import csv
import copy 
import itertools
import argparse
from shutil import copyfile

from sklearn.model_selection import train_test_split

def post_process(copy_list, predictions):
    """
    Make sure all abstract_ids are str, not int!
    """
    copy_list_dict = {}
    for entry in copy_list:
        abstract_id, copy_from, copy_to = entry["abstract_id"], entry["copy_from"], entry["copy_to"]
        if (abstract_id, copy_from) not in copy_list_dict:
            copy_list_dict[(abstract_id, copy_from)] = [copy_to]
        else:
            copy_list_dict[(abstract_id, copy_from)].append(copy_to)

    output_predictions = []
    output_id = 0
    for prediction in predictions:
        if prediction["type"] == "NOT":
            raise ValueError("Prediction Type cannot be NOT. Please remove all these false cases.")
        abstract_id, entity1_id, entity2_id = prediction["abstract_id"], prediction["entity_1_id"], prediction["entity_2_id"]
        entity1_to = copy_list_dict.get((abstract_id, entity1_id), [])
        entity2_to = copy_list_dict.get((abstract_id, entity2_id), [])
        assert entity2_id not in entity1_to
        assert entity1_id not in entity2_to
        entity1_synonyms = entity1_to + [entity1_id]
        entity2_synonyms = entity2_to + [entity2_id]
        for new_entity1_id, new_entity2_id in itertools.product(entity1_synonyms, entity2_synonyms):
            this_entry = copy.deepcopy(prediction)
            this_entry["id"] = output_id
            this_entry["entity_1_id"] = new_entity1_id
            this_entry["entity_2_id"] = new_entity2_id
            output_id += 1
            output_predictions.append(this_entry)
    return output_predictions


def get_article_id(train_data):
    return train_data["abstract_id"]


def has_article_id(train_data, article_ids):
    article_id = get_article_id(train_data)
    if article_id in article_ids:
        return True
    else:
        return False

if __name__ == "__main__":

    parser =  argparse.ArgumentParser(description="Split Litcoin Phase 2 data.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
    parser.add_argument("--abstract_file", action="store", dest="abstract_file", default="all_abstracts.json", help="Path to all LitCoin phase 2 abstract ids, use none to not use this file.")
    args = parser.parse_args()

    with open(os.path.join(args.data_path, "processed_data_train.json"), "r") as json_fp:
        train_data_list = json.load(json_fp)

    train_abstract_list = map(get_article_id, train_data_list)
    train_abstract_uniq = list(set(train_abstract_list))
    if args.abstract_file.lower() != "none":
        train_abstract_uniq = json.load(open(args.abstract_file))
        train_abstract_uniq = train_abstract_uniq["all_pmids"]

    random_seed = 1
    test_size = 0.2
    train_ids, val_ids = train_test_split(train_abstract_uniq, test_size=test_size, random_state=random_seed)

    train_subset = [train_data for train_data in train_data_list if has_article_id(train_data, train_ids)]
    val_subset = [train_data for train_data in train_data_list if has_article_id(train_data, val_ids)]

    if not os.path.isdir(os.path.join(args.data_path, "split")):
        os.mkdir(os.path.join(args.data_path, "split"))

    json.dump(train_subset, open(os.path.join(args.data_path, "split", "train.json"), "w"), indent=4)
    json.dump(val_subset, open(os.path.join(args.data_path, "split", "val.json"), "w"), indent=4)
    copyfile(os.path.join(args.data_path, "processed_data_test.json"), os.path.join(args.data_path, "split", "test.json"))

    train_copy_list = []
    val_copy_list = []
    with open(os.path.join(args.data_path, "copy_list_train.csv"), "r") as input_fp:
        reader = csv.DictReader(input_fp, skipinitialspace=True)
        for line in reader:
            if int(line["abstract_id"]) in train_ids:
                train_copy_list.append(line)
            elif int(line["abstract_id"]) in val_ids:
                val_copy_list.append(line)
            else:
                raise ValueError

    def write_copy_list_csv(file_path, copy_list):
        with open(file_path, "w") as output_fp:
            writer = csv.DictWriter(output_fp, fieldnames=["abstract_id", "copy_from", "copy_to"])
            writer.writeheader()
            for line in copy_list:
                writer.writerow(line)


    write_copy_list_csv(os.path.join(args.data_path, "split", "copy_list_train.csv"), train_copy_list)
    write_copy_list_csv(os.path.join(args.data_path, "split", "copy_list_val.csv"), val_copy_list)
    copyfile(os.path.join(args.data_path, "copy_list_test.csv"), os.path.join(args.data_path, "split", "copy_list_test.csv"))

    current_id = 0
    truths = []
    for entry in val_subset:
        abstract_id = entry["abstract_id"]
        type = entry["type"]
        entity_1_id = entry["entity_a_id"]
        entity_2_id = entry["entity_b_id"]
        novel = entry["novel"]
        if type != "NOT":
            this_truth = {
                "id": current_id,
                "abstract_id": str(abstract_id),
                "type": type,
                "entity_1_id": entity_1_id,
                "entity_2_id": entity_2_id,
                "novel": novel
            }
            current_id += 1
            truths.append(this_truth)

    truths = post_process(val_copy_list, truths)

    with open(os.path.join(args.data_path, "split", "val_truths.csv"), "w") as out_fp:
        writer = csv.DictWriter(out_fp, fieldnames=truths[0].keys(), delimiter="\t")
        writer.writeheader()
        for line in truths:
            writer.writerow(line)