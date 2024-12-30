# This program take the sentence split result from a method based on NLKT and refine it

import sys, getopt, re
import json
import string
import nltk
from nltk.corpus import words
nltk.download('words')

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"hi:o:v",["infile=","outfile=",])
except getopt.GetoptError:
    print('usage: sent_split_post_nltk.py -i inputfile -o outputfile [-v]')
    sys.exit()

for i in ['train', 'test']:
    inputfile = "temp/abstr_" + i + "_pmid_sents.txt"
    outputfile = "temp/abstr_" + i + "_pmid_sents_refined.txt"
    VERBOSE = 0

    for opt, arg in opts:
        if opt == '-h':
            print('usage: sent_split_post_nltk.py -i inputfile -o outputfile [-v]')
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
        elif opt in ('-v'):
            VERBOSE = 1

    IF = open(inputfile, 'r')
    OF = open(outputfile, 'w')

    prev_line = ''
    prev_PMID = ''
    prev_span1 = 0
    prev_span2 = 0
    MERGE = 0
    for line in IF:
        line = line.strip()
        items = line.split('\t')
        if len(items) < 4:
            print('The line does not have four columns!')
            print(line)
            OF.write(line+'\n')
            continue
        PMID = items[0]
        sent = items[1]
        try:
            span1 = int(items[2])
            span2 = int(items[3])
        except:
            print("integer converting error! ")
            print(line)
            OF.write(line+'\n')
            continue
        if MERGE == 1:
            if PMID == prev_PMID:
                OF.write(PMID + '\t' + prev_line + ' '*(span1-prev_span2) + sent + '\t' + str(prev_span1) + '\t' + str(span2) + '\n')
                if VERBOSE:
                    print('Fixed an error by concatenating two sentences below:')
                    print('SENTENCE 1: ', prev_line+'\t'+str(prev_span1)+'\t'+str(prev_span2))
                    print('SENTENCE 2: ', sent+'\t'+str(span1)+'\t'+str(span2))
                    print('MERGED: ', PMID + '\t' + prev_line + ' '*(span1-prev_span2) + sent + '\t' + str(prev_span1) + '\t' + str(span2))
                MERGE = 0
                continue
            else:
                OF.write(prev_PMID + '\t' + prev_line + '\t' + str(prev_span1)+'\t'+str(prev_span2)+'\n')
                #OF.write(PMID + '\t' + sent)
                MERGE = 0

        p = re.compile("[\.\?]\s([A-Z][a-z]*)\s")
        result = p.search(sent)
        if result != None:
            tmpW = result.group(1)
            if tmpW != '' and tmpW.lower() in words.words():
            # we should split this case
                pos = re.search("[\.\?]\s([A-Z][a-z]*)\s", sent)
                pos = pos.span()[0]
                sent1 = sent[0:pos+1]
                sent2 = sent[pos+1:]
                OF.write(PMID + '\t' + sent1 + '\t' + str(span1) + '\t' + str(span1 + pos + 1) + '\n')
                tmp = sent2.lstrip()
                diff = len(sent2) - len(tmp)
                OF.write(PMID + '\t' + tmp + '\t' + str(span1 + pos + 1 + diff) +'\t' + str(span2) + '\n')
                if VERBOSE == 1:
                    print('Fixed an error by splitting a sentence as below:')
                    print(PMID+'\t'+sent+'\t'+str(span1)+'\t'+str(span2))
                    print('Split sentence 1: ', sent1+ '\t' + str(span1) + '\t' + str(span1 + pos))
                    print('Split sentence 2: ', sent2+ '\t' + str(span1 + pos + diff) +'\t' + str(span2))
            else:
                OF.write(PMID + '\t' + sent + '\t' + str(span1) + '\t' + str(span2) + '\n')
        else:
            if sent.endswith(')'):
                MERGE = 1
                # the newline character needs to be removed
                prev_line = sent
                prev_PMID = PMID
                prev_span1 = span1
                prev_span2 = span2
            else:
                OF.write(PMID + '\t' + sent+'\t'+str(span1)+'\t'+str(span2)+'\n')

    OF.close()