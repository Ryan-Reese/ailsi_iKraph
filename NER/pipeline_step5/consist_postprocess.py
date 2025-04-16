###                     Copyright (C) Feng Pan, fpan@fsu.edu                       ###
### This program is to to post consistency check after predictions from NER models ###
import json
import random
import copy
import argparse
import os.path

ENTITY_LST = ["DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "ChemicalEntity", "OrganismTaxon", "SequenceVariant", "CellLine", "O"]

label = {}
label_true = {}
label_count = {}

# parse arguments
def parse_argument():
    parser = argparse.ArgumentParser(description='Post Processing with Consistency Check')
    parser.add_argument("-i", "--input", default="predictions.txt", help="input prediction file, Default is 'predictions.txt'", type=str)
    parser.add_argument("-t", "--token", default="test.json", help="token info file in json format, Default is 'test.json'", type=str)
    parser.add_argument("--no_fulltext", help="doing consistency check NOT based on fulltext", action="store_true")
    parser.add_argument("--pred_ft", default="predictions_ft.txt", help="prediction file of fulltext, Default is 'predictions_ft.txt'", type=str)
    parser.add_argument("--token_ft", default="test_ft.json", help="token info file of fulltext in json format, Default is 'test_ft.json'", type=str)
    parser.add_argument("-o", "--output", default="predictions_ppConsist.txt", help="output prediction file, Default is 'predictions_ppConsist.txt'", type=str)
    return parser.parse_args()

# check if the token belongs to subphrase
def check_subphrase(tags, i, j):
    if tags[i][0] == 'I':
        return True
    if j<(len(tags)-1) and tags[j+1][0] == 'I':
        return True
    return False

###  Main program ###
args = parse_argument()
if args.no_fulltext:
    use_fulltext = False
else:
    use_fulltext = True

# get predicted tags
if not os.path.exists(args.input):
    print ("ERROR: input prediction file does not exist!")
    exit()
fpred = open(args.input,'r')
pred_label = []
for line in fpred:
    pred_label.append([])
    for tmp in line.split():
        pred_label[-1].append(tmp)
fpred.close()

if use_fulltext:
    if not os.path.exists(args.pred_ft):
        print ("ERROR: input prediction file of fulltext does not exist!")
        exit()
    fpred = open(args.pred_ft,'r')
    pred_label_ft = []
    for line in fpred:
        pred_label_ft.append([])
        for tmp in line.split():
            pred_label_ft[-1].append(tmp)
    fpred.close()

# Open json files with token info
fjson = []
if not os.path.exists(args.token):
    print ("ERROR: input json file does not exist!")
    exit()
fjson.append(open(args.token, 'r'))
if use_fulltext:
    if not os.path.exists(args.token_ft):
        print ("ERROR: input json file of fulltext does not exist!")
        exit()
    fjson.append(open(args.token_ft,'r'))

# Locate tags of phrases in abstract 
ln = 0
for line in fjson[0]:
    ln += 1
    data = json.loads(line)
    data0 = copy.deepcopy(data)

    # update with prediction results
    for i in range(len(data["ner_tags"])):
        data["ner_tags"][i] = pred_label[ln-1][i]

    if type(data["document_id"]) == list:
        pubid = data["document_id"][0]
    else:
        pubid = data["document_id"]
    if pubid not in label.keys():
        label[pubid] = {}
        label_true[pubid] = {}
        label_count[pubid] = {}

    new_phrase = False
    phrase = ''
    for i in range(len(data["tokens"])):
        if data["ner_tags"][i][0] == "B":
            if phrase != '' or new_phrase == True:
                if phrase not in label_count[pubid].keys():
                    label_count[pubid][phrase] = {}
                    label[pubid][phrase] = "NaN"
                    label_true[pubid][phrase] = data0["ner_tags"][i-1].split('-')[-1]
                    for e in ENTITY_LST:
                        label_count[pubid][phrase][e] = 0
                label_count[pubid][phrase][data["ner_tags"][i-1].split('-')[-1]] += 1
            new_phrase = True
            phrase = ''
            phrase = phrase+data["tokens"][i]
            if i == len(data["tokens"]) - 1:
                if phrase not in label_count[pubid].keys():
                    label_count[pubid][phrase] = {}
                    label[pubid][phrase] = "NaN"
                    label_true[pubid][phrase] = data0["ner_tags"][i].split('-')[-1]
                    for e in ENTITY_LST:
                        label_count[pubid][phrase][e] = 0
                label_count[pubid][phrase][data["ner_tags"][i].split('-')[-1]] += 1

        elif data["ner_tags"][i][0] == "I":
            phrase = phrase+' '+data["tokens"][i]
            if i == len(data["tokens"]) - 1:
                if phrase not in label_count[pubid].keys():
                    label_count[pubid][phrase] = {}
                    label[pubid][phrase] = "NaN"
                    label_true[pubid][phrase] = data0["ner_tags"][i].split('-')[-1]
                    for e in ENTITY_LST:
                        label_count[pubid][phrase][e] = 0
                label_count[pubid][phrase][data["ner_tags"][i].split('-')[-1]] += 1
        else:
            if new_phrase == True:
                if phrase not in label_count[pubid].keys():
                    label_count[pubid][phrase] = {}
                    label[pubid][phrase] = "NaN"
                    label_true[pubid][phrase] = data0["ner_tags"][i-1].split('-')[-1]
                    for e in ENTITY_LST:
                        label_count[pubid][phrase][e] = 0
                label_count[pubid][phrase][data["ner_tags"][i-1].split('-')[-1]] += 1
            new_phrase = False
            phrase = ''

# Count tags of phrases in fulltext
ln = 0
if use_fulltext:
    for line in fjson[1]:
        ln += 1
        data = json.loads(line)

        # update with prediction results
        if len(data["ner_tags"]) != len(pred_label_ft[ln-1]):
            print ("Warning: sentence length not equal in fulltext: ", ln)
            continue
        for i in range(len(data["ner_tags"])):
            data["ner_tags"][i] = pred_label_ft[ln-1][i]

        if type(data["document_id"]) == list:
            pubid = data["document_id"][0]
        else:
            pubid = data["document_id"]
        if pubid not in label.keys():
            continue

        new_phrase = False
        phrase = ''
        for i in range(len(data["tokens"])):
            if data["ner_tags"][i][0] == "B":
                if phrase != '' or new_phrase == True:
                    if phrase in label_count[pubid].keys():
                        label_count[pubid][phrase][data["ner_tags"][i-1].split('-')[-1]] += 1
                new_phrase = True
                phrase = ''
                phrase = phrase+data["tokens"][i]
                if i == len(data["tokens"]) - 1:
                    if phrase in label_count[pubid].keys():
                        label_count[pubid][phrase][data["ner_tags"][i].split('-')[-1]] += 1

            elif data["ner_tags"][i][0] == "I":
                phrase = phrase+' '+data["tokens"][i]
                if i == len(data["tokens"]) - 1:
                    if phrase in label_count[pubid].keys():
                        label_count[pubid][phrase][data["ner_tags"][i].split('-')[-1]] += 1
            else:
                if new_phrase == True:
                    if phrase in label_count[pubid].keys():
                        label_count[pubid][phrase][data["ner_tags"][i-1].split('-')[-1]] += 1
                new_phrase = False
                phrase = ''

# double check if any phrases should be 'O', usually should not
fjson[0].seek(0)
ln = 0
for line in fjson[0]:
    data = json.loads(line)
    ln += 1

    # update with prediction results
    for i in range(len(data["ner_tags"])):
        data["ner_tags"][i] = pred_label[ln-1][i]

    if type(data["document_id"]) == list:
        pubid = data["document_id"][0]
    else:
        pubid = data["document_id"]

    for i in range(len(data["tokens"])):
        phrase = data["tokens"][i]
        if data["ner_tags"][i] != 'O': continue
        elif phrase in label[pubid].keys():
            label_count[pubid][phrase]['O'] += 1
        for j in range(i+1, len(data["tokens"])):
            if data["ner_tags"][j] != 'O': break
            phrase = phrase + ' ' + data["tokens"][j]
            if phrase in label[pubid].keys():
                label_count[pubid][phrase]['O'] += 1

if use_fulltext:
    fjson[1].seek(0)
    ln = 0
    for line in fjson[1]:
        data = json.loads(line)
        ln += 1

        # update with prediction results
        if len(data["ner_tags"]) != len(pred_label_ft[ln-1]):
            continue
        for i in range(len(data["ner_tags"])):
            data["ner_tags"][i] = pred_label_ft[ln-1][i]

        if type(data["document_id"]) == list:
            pubid = data["document_id"][0]
        else:
            pubid = data["document_id"]

        if pubid not in label.keys():
            continue

        for i in range(len(data["tokens"])):
            phrase = data["tokens"][i]
            if data["ner_tags"][i] != 'O': continue
            elif phrase in label[pubid].keys():
                label_count[pubid][phrase]['O'] += 1
            for j in range(i+1, len(data["tokens"])):
                if data["ner_tags"][j] != 'O': break
                phrase = phrase + ' ' + data["tokens"][j]
                if phrase in label[pubid].keys():
                    label_count[pubid][phrase]['O'] += 1
    fjson[1].close()

# write label info to file
flabel = open("phrase_label.dat",'w')
for k in label.keys():
    flabel.writelines(k+'\n')
    
    for j in label[k].keys():
        tmpmax = max(label_count[k][j].values())
        tmptags = []
        for e in ENTITY_LST:
            if label_count[k][j][e] == tmpmax: tmptags.append(e)
        if len(tmptags)==1:
            label[k][j] = tmptags[0]
            if label[k][j] == 'O':
                print ("Warning: ",j,"labeled as O")
                flabel.writelines("  "+j+'|'+label[k][j]+'|'+'\n')
            else:
                flabel.writelines("  "+j+'|'+label[k][j]+'|'+'\n')
        else:
            print ("conflicts found in",'|',k,'|',j, '|',label_count[k][j],'|',label_true[k][j])
flabel.close()

print ("Mention labeling finished..")
print ()


###  Fix inconsistencies ###
fjson[0].seek(0)
ln = 0
outtags = []
for line in fjson[0]:
    data = json.loads(line)
    ln += 1

    # update with prediction results
    for i in range(len(data["ner_tags"])):
        data["ner_tags"][i] = pred_label[ln-1][i]

    if type(data["document_id"]) == list:
        pubid = data["document_id"][0]
    else:
        pubid = data["document_id"]

    for i in range(len(data["tokens"])):
        phrase = data["tokens"][i]
        if phrase in label[pubid].keys() and label[pubid][phrase] != 'NaN':
            if not check_subphrase(data["ner_tags"], i, i) and data["ner_tags"][i].split('-')[-1] != label[pubid][phrase]:
                print ("Inconsistency fixed, line ",ln,'|',phrase,':',data["ner_tags"][i],"===>",label[pubid][phrase],'| True(if existing):',label_true[pubid][phrase])
                if label[pubid][phrase] != 'O':
                    data["ner_tags"][i] = 'B-'+label[pubid][phrase]
                else:
                    data["ner_tags"][i] = label[pubid][phrase]
        for j in range(i+1, len(data["tokens"])):
            phrase = phrase + ' ' + data["tokens"][j]
            if phrase in label[pubid].keys() and label[pubid][phrase] != 'NaN':
                if check_subphrase(data["ner_tags"], i, j):
                    continue
                need_fix = False
                for k in range(i,j+1):
                    if data["ner_tags"][k].split('-')[-1] != label[pubid][phrase]:
                        need_fix = True
                        break
                if need_fix:
                    print ("Inconsistency fixed, line ",ln,'|',phrase,':',data["ner_tags"][i:j+1],"===>",label[pubid][phrase],'| True(if existing):',label_true[pubid][phrase])
                    if label[pubid][phrase] != 'O':
                        data["ner_tags"][i] = 'B-'+label[pubid][phrase]
                        for k in range(i+1,j+1):
                            data["ner_tags"][k] = 'I-'+label[pubid][phrase]
                    else:
                        for k in range(i,j+1):
                            data["ner_tags"][k] = label[pubid][phrase]
    outtags.append(data["ner_tags"])
fjson[0].close()

# write fixed tags to output
mod_pred = open(args.output,'w')
for tmptoken in outtags:
    for j in range(len(tmptoken)):
        mod_pred.writelines(tmptoken[j])
        if j != len(tmptoken)-1:
            mod_pred.writelines(' ')
    mod_pred.writelines('\n')
mod_pred.close()

print ("Finished...")

