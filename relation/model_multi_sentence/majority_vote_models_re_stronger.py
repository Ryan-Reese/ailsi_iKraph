import pandas
import sys
import numpy as np
import random
from scipy.special import softmax

random.seed(42)

def find_majority_withTie(votes):
    label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
    vote = [label_list.index(x) for x in votes] # convert label as index number
    re_cnt = [0 for i in range (9)] # relation count for each type, order as label_list
    for x in vote:
        re_cnt[x] += 1
    has_re = sum(re_cnt) - re_cnt[0]
    if re_cnt[0] > has_re:
        tag = 0
    else:
        m = max(re_cnt[1:])
        max_re = [] #get all the most common members
        for i in range (1,9):
            if re_cnt[i] == m:
                max_re.append(i)
        if len(max_re) == 1:
            tag = max_re[0] # no tie
        elif 2 in max_re and 3 in max_re:   # positive > negative > association > others > not
            tag = 2
        elif 2 in max_re:
            tag = 2
        elif 3 in max_re:
            tag = 3
        elif 1 in max_re:
            tag = 1
        else:
            tag = random.choice(max_re)
    return label_list[tag]      # convert back to label


def read_prob(filename):
    prob = []
    with open(filename, 'r') as f:
        for line in f:
            prob.append(list(map(float, line.split())))
    return prob

def convert_label_stronger(prob):
    prob = softmax(prob)
    label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
    if prob[0] >= 0.5:
        return "NOT"
    else:
        return label_list[np.argmax(prob[1:])+1]


if __name__ == "__main__":
    label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
    rs = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 40, 42]
    file_path = ['roberta_BS16_lr3e-5_RS'+str(i)+'/model_output.csv' for i in rs]
    prob_file_path = ['roberta_BS16_lr3e-5_RS'+str(i)+'/model_output_prob.txt' for i in rs]  

    predictions = []
    test_backbone_df = pandas.read_csv(file_path[0])
    test_backbone_df = test_backbone_df.drop(columns=["predicted_class"])
    #for f in file_path:
    #    this_prediction_df = pandas.read_csv(f)
    #    this_prediction = this_prediction_df["predicted_class"]
    #    predictions.append(this_prediction)
    for f in prob_file_path:
        this_prediction = read_prob(f)
        predictions.append([convert_label_stronger(x) for x in this_prediction])
    
    predictions = np.array(predictions).T
    prediction = []
    for L in predictions:
        prediction.append(find_majority_withTie(L))
    test_backbone_df["predicted_class"] = prediction
    test_backbone_df.to_csv("majority_voting_RE_14models_stronger.csv")



