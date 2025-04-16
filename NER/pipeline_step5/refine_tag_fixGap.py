import numpy as np
from collections import Counter
import json
import pickle
from scipy.special import softmax
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

def read_pred(filename):
    a = []
    with open(filename, 'r') as f:
        for line in f:
            a.append(line.split())
    return a

def read_origin(filename):
    a = []
    with open(filename, 'r') as f:
        for line in f:
            x = json.loads(line)
            a.append(x['ner_tags'])
    return a

def read_prob(filename, ENTITY=[]):     # if ENTITY is given, the result probality will be sorted to match it
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    entity = data['label_list']
    logit_prob = data['probability']
    if ENTITY == []:
        return entity, logit_prob
    elif ENTITY == entity:
        return entity, logit_prob
    else:
        ind = [entity.index(x) for x in ENTITY] # get the index if match the order of entity to ENTITY
        new_entity = [entity[i] for i in ind]
        logit_prob = [np.array(logit_prob[i])[ind] for i in range (len(logit_prob))]
        return new_entity, logit_prob
        


def get_report(true_label, pred_label):
    Report = classification_report(true_label, pred_label,digits=4)
    report = Report.split()
    ini = report.index('support')
    final = report.index('micro')
    f1 = {}
    for i in range (ini+1, final, 5):
        f1[report[i]] = report[i+3]
    f1['overall'] = report[final+4]
    return f1, Report


def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return top_two[0][0], top_two[1][1]
    return top_two[0][0]

def find_candidate_byProb(logit_prob, entity, weight=[]):    # prob = [p1, p2, ...pn], p1 is a probabity vector (13,) of a token from method 1, prob.shape = (n_method, 13)
    prob = softmax(np.array(logit_prob))
    if weight != []:    # weight = [w1, w2, ...wn], w1 is a scale to indicate how important of model 1
        weight = np.array(weight)
        prob = prob * weight[:, None]
    avg_prob = np.mean(prob, axis=0)
    norm_prob = avg_prob/np.sum(avg_prob)
    return entity[np.argmax(norm_prob)]


def prob_vote(sample_prob, full_entity, weight=[]):
    '''
    sample_prob = [sample1_prob, sample2_prob, ...] for one sentence, sample1_prob.shape = (n_token, 13)
    entity = ['B-CellLine', 'B-ChemicalEntity', 'B-DiseaseOrPhenotypicFeature', 'B-GeneOrGeneProduct', 'B-OrganismTaxon', 'B-SequenceVariant', 'I-CellLine', 'I-ChemicalEntity', 'I-DiseaseOrPhenotypicFeature', 'I-GeneOrGeneProduct', 'I-OrganismTaxon', 'I-SequenceVariant', 'O']
    fix: O, I to O, B
         B, B to B, I
         I, B to I, I or B, I
    '''
    entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine'] 
    #entity = list(set([x.replace('B-','').replace('I-','') for x in full_entity]))
    flag = [0 for i in range (len(entity))]        # flag use to decide if asign B or I
    result = []
    for i in range (len(sample_prob[0])):   # go through each token
        prob = [sample_prob[j][i] for j in range (len(sample_prob))]    # prob.shape = (n_method, 13)
        candidate = find_candidate_byProb(prob, full_entity, weight)
        if candidate == 'O':    # the top candidate is "O"
            result.append('O')
            flag = [0 for j in range (len(entity))]
        else:   # top candidate is not "O", has to decide output "B" or "I"
            if flag[entity.index(candidate.split('-')[1])] == 0:    # the previous position "O", assign "B"
                result.append('B-'+candidate.split('-')[1])
                flag = [0 for j in range (len(entity))] # reset flag
                flag[entity.index(candidate.split('-')[1])] = 1
            else:   # the previous position is "B" or "I", assign "I"
                result.append('I-'+candidate.split('-')[1])
    return result


def majority_vote(sample, entity):
    '''
    sample = [sample1, sample2, ...] for one sentence 
    entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine']
    fix: O, I to O, B
         B, B to B, I
         I, B to I, I or B, I

    '''
    flag = [0 for i in range (len(entity))]        # flag use to decide if asign B or I
    sample = np.array(sample)
    result = []
    for i in range (len(sample[0])):
        #candidate = find_majority(sample[:,i])
        candidate = Counter(sample[:,i]).most_common(1)[0][0]
        if candidate == 'O':    # the top candidate is "O"
            result.append('O')
            flag = [0 for j in range (len(entity))]
        else:   # top candidate is not "O", has to decide output "B" or "I"
            if flag[entity.index(candidate.split('-')[1])] == 0:    # the previous position "O", assign "B"
                result.append('B-'+candidate.split('-')[1])
                flag = [0 for j in range (len(entity))] # reset flag
                flag[entity.index(candidate.split('-')[1])] = 1
            else:   # the previous position is "B" or "I", assign "I"
                result.append('I-'+candidate.split('-')[1])
    return result


def single_ensemble(pred, true_label, entity):
    sampled_label = []
    for i in range (len(pred)):
        sampled_label.append(majority_vote([pred[i]], entity))
    f1, report = get_report(true_label, sampled_label)
    return f1, report, sampled_label


def single_ensemble_noEva(pred, entity):
    sampled_label = []
    for i in range (len(pred)):
        sampled_label.append(majority_vote([pred[i]], entity))
    return sampled_label


def refine_tag(input_tag, entity):
    '''
    refine: 'OIB' to 'OBI', 'OBB' to 'OBI', 'OBIB' to 'OBII'. Here B and I are same entity type.
    input: input_tag is a list of tags within 6 entities, e.g. ['I-ChemicalEntity', 'O', 'I-ChemicalEntity','I-ChemicalEntity','O']
    output: refined_tag is a list of refined tags, e.g. ['B-ChemicalEntity', 'O', 'B-ChemicalEntity','I-ChemicalEntity','O']
    '''
    full_refined_tag = []
    for sent in input_tag:
        refined_tag = []
        #entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine']
        flag = [0 for i in range (len(entity))]        # flag use to decide if asign B or I
        for candidate in sent:
            if candidate == 'O':    # the top candidate is "O"
                refined_tag.append('O')
                flag = [0 for j in range (len(entity))]
            else:
                if flag[entity.index(candidate.split('-')[1])] == 0:    # the previous position "O", assign "B"
                    refined_tag.append('B-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
                else:   # the previous position is "B" or "I", assign "I"
                    refined_tag.append('I-'+candidate.split('-')[1])
        full_refined_tag.append(refined_tag)
    return full_refined_tag

def refine_tag_fixGap(input_tag, entity, gap=1):
    '''
    refine: 'OIB' to 'OBI', 'OBB' to 'OBI', 'OBIB' to 'OBII', 'BOI' to 'BII', 'BIOI' to 'BIII'. Here B and I are same entity type.
     input: input_tag is a list of tags within 6 entities, e.g. ['I-ChemicalEntity', 'O', 'I-ChemicalEntity','I-ChemicalEntity','O']
    output: refined_tag is a list of refined tags, e.g. ['B-ChemicalEntity', 'O', 'B-ChemicalEntity','I-ChemicalEntity','O']
    '''
    gap=int(gap)
    full_refined_tag = []
    for sent in input_tag:
        refined_tag = []
        #entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine']
        flag = [0 for i in range (len(entity))]        # flag use to decide if asign B or I
        for candidate in sent:
            if candidate == 'O':    # the top candidate is "O"
                refined_tag.append('O')
                # the flag of any non-zero entity will add 1
                flag =  [x+1 if x!=0 else 0 for x in flag]
            elif candidate.split('-')[0] == 'B':
                if flag[entity.index(candidate.split('-')[1])] != 1: # the previous position "O", assign "B"
                    refined_tag.append('B-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
                else: # the previous position "B" or "I", assign "I"
                    refined_tag.append('I-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
            elif candidate.split('-')[0] == 'I':
                if flag[entity.index(candidate.split('-')[1])] == 1:    # the previous position 'I' or 'B'
                    refined_tag.append('I-'+candidate.split('-')[1])
                elif flag[entity.index(candidate.split('-')[1])] >gap+1:    # the gap larger than gap_cutoff, assign 'B'
                    refined_tag.append('B-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
                elif flag[entity.index(candidate.split('-')[1])] == 0:      # this type happends first time
                    refined_tag.append('B-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
                else:
                    for k in range (1, flag[entity.index(candidate.split('-')[1])]):    # fill the gap
                        refined_tag[-k] = 'I-'+candidate.split('-')[1]
                    refined_tag.append('I-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
        full_refined_tag.append(refined_tag)
    return full_refined_tag


if __name__ == "__main__":
    import sys
    ensemble_list = sys.argv[1]
    gap = sys.argv[2]
    output_folder = 'pp_input/'
    mode = 'majority'  #sys.argv[4]      # majority or prob
    refine_mode = 'input_refine'

    entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine']

    fold = []
    weight = []
    with open(ensemble_list,'r') as f:
        for line in f:
            if mode == 'prob':
                weight.append(float(line.split()[1]))
            fold.append(line.split()[0])

    new_ensemble_list = []
    for i, x in enumerate(fold):
        pred_label = read_pred(x+'/test_predictions.txt')
        RF_pred_label = refine_tag_fixGap(pred_label, entity, gap=gap) # fix I, I to B, I, then calculate F1
        with open(output_folder+str(i+1)+'.txt','w') as f:
            f.writelines('\n'.join([' '.join(sent) for sent in RF_pred_label])+'\n')

