import json
import random
import os
import copy
from tqdm import tqdm

if __name__ == "__main__":

    id_list = []
    data = []
    with open("litcoin_train_update_400.json", "r") as f:
        for line in f:
            data.append(json.loads(line))

    for i in range(2, 3):
        with open("split_" + str(i + 1) + ".json", "r") as f:
            data_list = json.loads(f.read())
        train_list = data_list["train_set"]
        val_list = data_list["validation_set"]
        if not os.path.isdir("split_" + str(i + 1)):
            os.mkdir("split_" + str(i + 1))
        train_file = open("split_" + str(i + 1) + "/train.json", "w")
        val_file = open("split_" + str(i + 1) + "/dev.json", "w")
        count_train = 1
        count_val = 1
        for x in tqdm(data, desc="splitting train dataset"):
            if x["document_id"] in train_list:
                data_dict = copy.deepcopy(x)
                data_dict["id"] = count_train
                if isinstance(data_dict["document_id"], str):
                    data_dict["document_id"] = [data_dict["document_id"]]
                dataString = json.dumps(data_dict)
                train_file.write(dataString + "\n")
                count_train += 1
            elif x["document_id"] in val_list:
                data_dict = copy.deepcopy(x)
                data_dict["id"] = count_val
                if isinstance(data_dict["document_id"], str):
                    data_dict["document_id"] = [data_dict["document_id"]]
                dataString = json.dumps(data_dict)
                val_file.write(dataString + "\n")
                count_val += 1
        train_file.close()
        val_file.close()
