import json
import random
import os
import copy
from tqdm import tqdm

if __name__ == "__main__":

    id_list = []
    data = []
    with open("litcoin_test_update_100.json", "r") as f:
        for line in f:
            data.append(json.loads(line))

    test_file = open("Litcoin_testset.json", "w")
    for x in tqdm(data, desc="generating test dataset"):
        data_dict = copy.deepcopy(x)
        if isinstance(data_dict["document_id"], str):
            data_dict["document_id"] = [data_dict["document_id"]]
        dataString = json.dumps(data_dict)
        test_file.write(dataString + "\n")
    test_file.close()
