# NER_labeling.py
# Take a tokenization results and entity information file
# output tokens with the entity labels, ready for training models

import sys, getopt, re
import json

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"ht:e:o:c",["infile=","outfile=",])
except getopt.GetoptError:
    print('usage: TokenParser.py -i inputfile -o outputfile')
    sys.exit()

tokenFile = "temp/abstr_train_pmid_tokens.json"
entityFile = "data_files/LitCoin/entities_train.csv"
outputfile = "temp/abstr_train_bio.txt"
CHECKING = 0

for opt, arg in opts:
    if opt == '-h':
        print('usage: TokenParser.py -i inputfile -o outputfile')
        sys.exit()
    elif opt in ("-t"):
        tokenFile = arg
    elif opt in ("-e"):
        entityFile = arg        
    elif opt in ("-o"):
        outputfile = arg
    elif opt in ("-c"):
        CHECKING = 1
    
TF = open(tokenFile, 'r')
EF = open(entityFile, 'r')
OF = open(outputfile, 'w')

# read the entity file and generate a dictionary
# with key being the abstract ID and value as the list of entities
entityDic = {}
ID = ''
for line in EF:
    if line[:2] == 'id':
        continue
    line = line.strip()
    items = line.split('\t')
    tmpVal = []
    tmpVal.append(int(items[2]))
    tmpVal.append(int(items[3]))
    tmpVal.append(items[4])
    tmpVal.append(items[5])
    if items[1] != ID:
        ID = items[1]
        entityDic[ID] = []
        entityDic[ID].append(tmpVal)
    else:
        entityDic[ID].append(tmpVal)

#print(entityDic)
sentTokens = json.load(TF)
outputList = []
docCnt = {} # keep track which entity is being handled for an abstract
prev_label = ''
cur_cnt = 0
ErrorCnt = 0
ErrorCnt2 = 0
for item in sentTokens:
    tokens = item['tokens']
    spans = item['spans']
    docID = item['document_id']
    labels = []
    if docID not in docCnt:
        docCnt[docID] = 0
        cur_cnt = 0
    else:
        cur_cnt = docCnt[docID]
    for cnt, [s,e] in enumerate(spans):
        #print(s, e)
        #print(cur_cnt, len(entityDic[docID]))
        if cur_cnt >= len(entityDic[docID]) :
            labels.append('O')
            prev_label = 'O'
            continue
        if entityDic[docID][cur_cnt][0] >= e :
            # this is the case that the current word is before the next entity, so label it as O
            labels.append('O')
            prev_label = 'O'
        elif entityDic[docID][cur_cnt][0] == s:
            # this is the case that the current word has the same start span as the next entity, so label it as B-
            labels.append('B-'+entityDic[docID][cur_cnt][2])
            prev_label = 'B-'+entityDic[docID][cur_cnt][2]
            if entityDic[docID][cur_cnt][1] == e:
                # if the current word also has the same end span as the next entity
                # then we finished labeling of this entity and move to the next entity, which is done by the next two lines
                docCnt[docID] = docCnt[docID] + 1
                cur_cnt = docCnt[docID]
            elif entityDic[docID][cur_cnt][1] < e:
                # this is the case that the current entity is only part of this word
                # this is considered as a tokenization error type 1
                print(ErrorCnt, ": Tokenization error 1: ", docID, s, e, tokens[cnt], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                ErrorCnt = ErrorCnt + 1
                docCnt[docID] = docCnt[docID] + 1
                cur_cnt = docCnt[docID]
            #print(prev_label, tokens[cnt], cnt)
        elif entityDic[docID][cur_cnt][0] < s:
            # the start span of current entity is to the left of the start of this word
            if entityDic[docID][cur_cnt][1] > e:
                # the end span of current entity is to the right of the end of this word,
                # so this word is the middle of the current entity, we label it as I-
                # to make sure there is no error, we check whether previous labeling is the same type
                if prev_label == 'B-'+entityDic[docID][cur_cnt][2] or prev_label == 'I-'+entityDic[docID][cur_cnt][2]:
                    labels.append('I-'+entityDic[docID][cur_cnt][2])
                    prev_label = 'I-'+entityDic[docID][cur_cnt][2]
                else:
                    # labeling error occurs. In this case, we label it as B- of the new entity type
                    # this happens because the start of the entity is in the previous word, which is another entity
                    print("Labeling error 1!", docID, s, e, tokens[cnt], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                    labels.append('B-'+entityDic[docID][cur_cnt][2])
                    prev_label = 'B-'+entityDic[docID][cur_cnt][2]
                    #labels.append('O')
                    #prev_label = 'O'
            elif entityDic[docID][cur_cnt][1] == e:
                # the end span of the current entity is the same as the end of this word
                # we finished the labeling of this entity and move to the next entity
                if prev_label == 'B-'+entityDic[docID][cur_cnt][2] or prev_label == 'I-'+entityDic[docID][cur_cnt][2]:
                    labels.append('I-'+entityDic[docID][cur_cnt][2])
                    prev_label = 'I-'+entityDic[docID][cur_cnt][2]
                else:
                    print("Labeling error 2!", docID, s, e, tokens[cnt], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                    # we also change this to B- of another entity
                    labels.append('B-'+entityDic[docID][cur_cnt][2])
                    prev_label = 'B-'+entityDic[docID][cur_cnt][2]
                    #labels.append('O')
                    #prev_label = 'O'
                # again, the following two lines move to the next entity
                docCnt[docID] = docCnt[docID] + 1
                cur_cnt = docCnt[docID]
            else:
                # this means entityDic[docID][cur_cnt][1] < e, the end of current entity is ends before the end of the current word
                # it is a tokenization error
                if entityDic[docID][cur_cnt][1] <= s:
                    # the current entity ends even before the start of the current word
                    # this happens because the current entity and the previous entity are in the same word such as rDEN2Delta30
                    # in such case, rDEN2 is an organism and Delta30 is a sequenceVariant
                    # In this case, we label rDEN2Delta30 as organism and level sequenceVariant entity unlabeled
                    print(ErrorCnt, ": Tokenization error 5: ", docID, s, e, tokens[cnt-1], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                    ErrorCnt = ErrorCnt + 1
                    # we need to move to the next entity first since the current entity cannot be labeled
                    docCnt[docID] = docCnt[docID] + 1
                    cur_cnt = docCnt[docID]
                    # after that we need to check whether the start of the current word is the same as the start of the new entity we just moved to
                    if cur_cnt < len(entityDic[docID]) and entityDic[docID][cur_cnt][0] == s:
                        # if this is the case, then label it as B-
                        labels.append('B-'+entityDic[docID][cur_cnt][2])
                        prev_label = 'B-'+entityDic[docID][cur_cnt][2]
                    elif cur_cnt < len(entityDic[docID]) and entityDic[docID][cur_cnt][0] < e:
                        # if the current entity start in the middle of the current word
                        # Then it is a tokenization error
                        print(ErrorCnt, ": Tokenization error 6: ", docID, s, e, tokens[cnt-1], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                        ErrorCnt = ErrorCnt + 1
                        labels.append('B-'+entityDic[docID][cur_cnt][2])
                        prev_label = 'B-'+entityDic[docID][cur_cnt][2]
                    else:
                        # if the start of the word is not the start of the next entity, and not in the middle of it, we label it as O
                        labels.append('O')
                        prev_label = 'O'
                else:
                    # now entityDic[docID][cur_cnt][1] < e and entityDic[docID][cur_cnt][1] > s. 
                    # The current entity ends in the middle of the current word
                    print(ErrorCnt, ": Tokenization error 4: ", docID, s, e, tokens[cnt], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                    if prev_label == 'B-'+entityDic[docID][cur_cnt][2] or prev_label == 'I-'+entityDic[docID][cur_cnt][2]:
                        labels.append('I-'+entityDic[docID][cur_cnt][2])
                        prev_label = 'I-'+entityDic[docID][cur_cnt][2]
                    else:
                        labels.append('O')
                        prev_label = 'O'
                        print("label consistency error!")
                    ErrorCnt = ErrorCnt + 1
                    # the current entity finished we move to the next
                    docCnt[docID] = docCnt[docID] + 1
                    cur_cnt = docCnt[docID]
        elif entityDic[docID][cur_cnt][0] > s:
            labels.append('B-'+entityDic[docID][cur_cnt][2])
            prev_label = 'B-'+entityDic[docID][cur_cnt][2]
            if entityDic[docID][cur_cnt][1] == e:
                print(ErrorCnt, ": Tokenization error 2: ", docID, s, e, tokens[cnt], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                ErrorCnt = ErrorCnt + 1
                docCnt[docID] = docCnt[docID] + 1
                cur_cnt = docCnt[docID]
            elif entityDic[docID][cur_cnt][1] < e:
                print(ErrorCnt, ": Tokenization error 3: ", docID, s, e, tokens[cnt], entityDic[docID][cur_cnt][0], entityDic[docID][cur_cnt][1])
                ErrorCnt = ErrorCnt + 1
                docCnt[docID] = docCnt[docID] + 1
                cur_cnt = docCnt[docID]
        elif entityDic[docID][cur_cnt][1] < s:
            print("Unknow error!")
            labels.append('O')
            prev_label = 'O'
            
    item['ner_tags'] = labels
    if len(labels) != len(spans):
        print("Error 5! ", docID, cur_cnt, len(labels), len(spans))
        ErrorCnt2 = ErrorCnt2 + 1
    outputList.append(item)

for item in outputList:
    s = json.dumps(item)
    OF.write(s)
    OF.write('\n')    
OF.close()
print("total number of errors is: ", ErrorCnt, ErrorCnt2)