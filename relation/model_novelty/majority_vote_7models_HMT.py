import pandas
import sys
import numpy as np
from collections import Counter

def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie, if odd number of models, no need to worry tie
        return False
        #return top_two[0][0], top_two[1][1]
    return top_two[0][0]


PATH = "/data/yuan/8.litcoin_phase2/5.multi_sentence_model/"
TEST_PATHS = ['./output/roberta_Novel_punct_lr3e-5_BS16_allTrain_RS2/',\
              './output/roberta_Novel_punct_lr3e-5_BS16_allTrain_RS12/',\
              './output/roberta_Novel_punct_lr3e-5_BS16_allTrain_RS14/',\
              './output/roberta_Novel_punct_lr3e-5_BS16_allTrain_RS119/',\
              './output/roberta_Novel_punct_lr3e-5_BS16_allTrain_RS132/',\
              './output/roberta_Novel_punct_lr3e-5_BS16_allTrain_RS401/',\
              './output/roberta_Novel_punct_lr3e-5_BS16_allTrain_RS9999/',\
              ]

#test_checkpoints = [[2286, 2540, 2794] for i in range (7)]
test_checkpoints = [[2794, 3810, 4826] for i in range (7)]

predictions = []
test_backbone_df = pandas.read_csv(f"{TEST_PATHS[0]}/checkpoint-{test_checkpoints[0][0]}/model_output.csv")
test_backbone_df = test_backbone_df.drop(columns=["predicted_class"])
for idx, test_checkpoint in enumerate(test_checkpoints):
    for tc in test_checkpoint:
        test_file = f"{TEST_PATHS[idx]}/checkpoint-{tc}/model_output.csv"
        this_prediction_df = pandas.read_csv(test_file)
        this_prediction = this_prediction_df["predicted_class"]
        predictions.append(this_prediction)

predictions = np.array(predictions).T
prediction = []
for L in predictions:
    prediction.append(find_majority(L))
    if len(L) == 2:
        print(L)

test_backbone_df["predicted_class"] = prediction
test_backbone_df.to_csv("majority_voting_7models_HMT.csv")
