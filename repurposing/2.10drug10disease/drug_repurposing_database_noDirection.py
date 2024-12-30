# relation id rule: cell line - chemical - disease - gene - mutations - species


import json
import sys
import argparse
import time
from datetime import datetime
import os
import math
import pandas as pd


def convert_entName(entname):
    entname_ = entname.replace("'"," ").replace('/'," ")
    return '_'.join(entname_.split())

def get_rel_within_time(art_list, time_cut1, time_cut2):
    # art_list = [['pmid', prob, score, 'YYYY-MM-DD'],...]
    art_list = [x[:-1] + [datetime.strptime(x[-1], "%Y-%m-%d")] for x in art_list]
    t1_datetime = datetime.strptime(time_cut1, "%Y-%m-%d")
    t2_datetime = datetime.strptime(time_cut2, "%Y-%m-%d")
    result = [x for x in art_list if t1_datetime <= x[-1] <= t2_datetime]
    return result

def get_prob_fromArtList(rels, MIN_PROB):
    if rels != []:
        cumProb = 1
        sumscore = 0
        for pmid, prob, score, T in rels:
            cumProb = cumProb * (1-prob)
            sumscore += -math.log(1 - prob + 0.01)
        finalProb = round(1 - cumProb, 4)
        if finalProb >= MIN_PROB:
            return finalProb, sumscore
        else:
            return 0, 0
    else:
        return 0, 0

def reverse_corType(cdTypeStr):
    if cdTypeStr == '0':
        return '2'
    elif cdTypeStr == '2':
        return '0'


def repurpose_fromC_database(RELlist, entDic, entSubtype, entName, targetID, MIN_PROB, MIN_PROB_direct, COR_TYPE):
    # get Chemical-Gene, Disease-Gene dict for repurposing; Get Chem-Disease dict for direct relation
    ChemGeneDic = {}
    GeneDisDic = {}
    ChemDisDic = {}
    for rel in RELlist:
        # relation id rule: cell line - chemical - disease - gene - mutations - species
        n1, n2, corType, direction = rel.split('.')
        if n1 not in entDic or n2 not in entDic:
            continue
        if n1 == targetID and entDic[n2] == 'Gene':# and direction == '12':
            # calculate probability
            _, finalProb, sumscore, _ = RELlist[rel]
            if finalProb != 0:
                if n1 not in ChemGeneDic:
                    ChemGeneDic[n1] = {}
                    ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
                else:
                    if n2 not in ChemGeneDic[n1]:
                        ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
                    else:
                        if ChemGeneDic[n1][n2][0] < finalProb:
                            ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
                        elif ChemGeneDic[n1][n2][0] == finalProb:
                            # if the two probabilities are equal, compare score
                            if ChemGeneDic[n1][n2][2] < sumscore:
                                ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
        if n1 == targetID and entDic[n2] == 'Disease':# and direction == '12':
            # calculate probability
            _, finalProb, sumscore, _ = RELlist[rel]
            if finalProb == 0:
                continue
            key = n1+'.'+n2+'.'+corType
            if key not in ChemDisDic:
                ChemDisDic[key] = [finalProb, corType, sumscore]
            elif finalProb > ChemDisDic[key][0]:
                ChemDisDic[key] = [finalProb, corType, sumscore]
        if entDic[n1] == 'Disease' and entDic[n2] == 'Gene':# and direction == '21':
            # calculate probability
            _, finalProb, sumscore, _ = RELlist[rel]
            if finalProb != 0:
                if n2 not in GeneDisDic:
                    GeneDisDic[n2] = {}
                    GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
                else:
                    if n1 not in GeneDisDic[n2]:
                        GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
                    else:
                        if GeneDisDic[n2][n1][0] < finalProb:
                            GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
                        elif GeneDisDic[n2][n1][0] == finalProb:
                            if GeneDisDic[n2][n1][2] < sumscore:
                                GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
    # Finish organize aggregate relation, Begin drug repurposing process.
    relDic = {}
    chem = targetID
    if chem in ChemGeneDic and ChemGeneDic[chem] != {}:
        for gene in ChemGeneDic[chem]:
            prob, corType, sumscore = ChemGeneDic[chem][gene]
            if gene in GeneDisDic:
                for dis in GeneDisDic[gene]:
                    prob2, corType2, sumscore2 = GeneDisDic[gene][dis]
                    finalProb = prob * prob2
                    finalScore = sumscore + sumscore2
                    if finalProb < MIN_PROB:
                        continue
                    cdType = (int(corType)-1) * (int(corType2)-1)
                    if cdType != COR_TYPE:
                        continue
                    cdTypeStr = str(cdType + 1)
                    key = chem+'.'+dis + '.' + cdTypeStr
                    key_op = chem+'.'+dis + '.' + reverse_corType(cdTypeStr)
                    if key not in relDic:
                        tmpDic = {}
                        tmpDic['id'] = key
                        tmpDic['chem'] = chem
                        tmpDic['disease'] = dis
                        tmpDic['genes'] = [gene]
                        tmpDic['type'] = cdType
                        tmpDic['probs'] = [finalProb]
                        tmpDic['score'] = [finalScore]
                        tmpDic['cor_types'] = [[int(corType)-1, int(corType2)-1]]
                        if key in ChemDisDic:
                            dirProb_inT = ChemDisDic[key][0]
                            if key_op in ChemDisDic:
                                dirProb_op_inT = ChemDisDic[key_op][0]
                            else:
                                dirProb_op_inT = 0
                            if dirProb_inT - dirProb_op_inT > 0.1:
                                tmpDic['directProb'] = ChemDisDic[key][0]
                            else:
                                tmpDic['directProb'] = 0
                        else:
                            tmpDic['directProb'] = 0
                        if key in ChemDisDicAll:
                            dirProb_all = ChemDisDicAll[key][0]
                            if key_op in ChemDisDicAll:
                                dirProb_op_all = ChemDisDicAll[key_op][0]
                            else:
                                dirProb_op_all = 0
                            if dirProb_all - dirProb_op_all > 0.1:
                                tmpDic['directProbAll'] = ChemDisDicAll[key][0]
                            else:
                                tmpDic['directProbAll'] = 0
                        else:
                            tmpDic['directProbAll'] = 0
                        relDic[key] = tmpDic
                    else:
                        relDic[key]['genes'].append(gene)
                        relDic[key]['probs'].append(finalProb)
                        relDic[key]['score'].append(finalScore)
                        relDic[key]['cor_types'].append([int(corType)-1, int(corType2)-1])
        for K in relDic:
            sumScore = 0
            cumProb = 1
            for prob in relDic[K]['probs']:
                cumProb = cumProb * (1 - prob)
            for score in relDic[K]['score']:
                sumScore += score
            finalScore = sumScore
            finalProb = 1 - cumProb
            relDic[K]['finalScore'] = finalScore
            relDic[K]['finalProb'] = finalProb
            if entSubtype[relDic[K]['chem']] == 'Drug':
                relDic[K]['Drug'] = 1
            else:
                relDic[K]['Drug'] = 0
            relDic[K]['chem_name'] = entName[relDic[K]['chem']]
            relDic[K]['disease_name'] = entName[relDic[K]['disease']]
            direct_prob = relDic[K]['directProb']
            if direct_prob < MIN_PROB_direct:
                relDic[K]['directRel'] = 0
            else:
                relDic[K]['directRel'] = 1
            direct_prob_all = relDic[K]['directProbAll']
            if direct_prob_all < MIN_PROB_direct:
                relDic[K]['directRelAll'] = 0
            else:
                relDic[K]['directRelAll'] = 1
            relDic[K]['pathCnt'] = len(relDic[K]['genes'])
    # get the direct Chem-Dis in Time slot
    allCD_inT = {}
    for K in ChemDisDic:
        n1, n2, corType = K.split('.')
        if n1 == targetID and int(corType)-1 == COR_TYPE:
            dirProb, _, dirScore = ChemDisDic[K]
            K_op = n1 + '.' + n2 + '.' + reverse_corType(corType)
            if K_op in ChemDisDic:
                dirProb_op = ChemDisDic[K_op][0]
            else:
                dirProb_op = 0
            if dirProb - dirProb_op > 0.1:
                if n1 not in allCD_inT:
                    allCD_inT[n2] = [dirProb, dirScore, entSubtype[n2]]
                elif allCD_inT[n2][0] < dirProb:
                    allCD_inT[n2] = [dirProb, dirScore, entSubtype[n2]]
    return relDic, allCD_inT


def repurpose_fromD_database(RELlist, entDic, entSubtype, entName, targetID, MIN_PROB, MIN_PROB_direct, COR_TYPE):
    # get Chemical-Gene, Disease-Gene dict for repurposing; Get Chem-Disease dict for direct relation
    ChemGeneDic = {}
    GeneDisDic = {}
    ChemDisDic = {}
    for rel in RELlist:
        n1, n2, corType, direction = rel.split('.')
        if n1 not in entDic or n2 not in entDic:
            continue
        if entDic[n1] == 'Chemical' and entDic[n2] == 'Gene':# and direction == '12':
            # calculate probability
            _, finalProb, sumscore, _ = RELlist[rel]
            if finalProb != 0:
                if n1 not in ChemGeneDic:
                    ChemGeneDic[n1] = {}
                    ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
                else:
                    if n2 not in ChemGeneDic[n1]:
                        ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
                    else:
                        if ChemGeneDic[n1][n2][0] < finalProb:
                            ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
                        elif ChemGeneDic[n1][n2][0] == finalProb:
                            # if the two probabilities are equal, compare score
                            if ChemGeneDic[n1][n2][2] < sumscore:
                                ChemGeneDic[n1][n2] = (finalProb, corType, sumscore)
        if entDic[n1] == 'Chemical' and n2 == targetID:# and direction == '12':
            # calculate probability
            _, finalProb, sumscore, _ = RELlist[rel]
            if finalProb == 0:
                continue
            key = n1+'.'+n2+'.'+corType
            if key not in ChemDisDic:
                ChemDisDic[key] = [finalProb, corType, sumscore]
            elif finalProb > ChemDisDic[key][0]:
                ChemDisDic[key] = [finalProb, corType, sumscore]
        if n1 == targetID and entDic[n2] == 'Gene':# and direction == '21':
            # calculate probability
            _, finalProb, sumscore, _ = RELlist[rel]
            if finalProb != 0:
                if n2 not in GeneDisDic:
                    GeneDisDic[n2] = {}
                    GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
                else:
                    if n1 not in GeneDisDic[n2]:
                        GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
                    else:
                        if GeneDisDic[n2][n1][0] < finalProb:
                            GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
                        elif GeneDisDic[n2][n1][0] == finalProb:
                            if GeneDisDic[n2][n1][2] < sumscore:
                                GeneDisDic[n2][n1] = (finalProb, corType, sumscore)
    # Finish organize aggregate relation, Begin drug repurposing process.
    relDic = {}
    dis = targetID
    for chem in ChemGeneDic:
        for gene in ChemGeneDic[chem]:
            prob, corType, sumscore = ChemGeneDic[chem][gene]
            if gene in GeneDisDic:
                if dis in GeneDisDic[gene]:
                    prob2, corType2, sumscore2 = GeneDisDic[gene][dis]
                    finalProb = prob * prob2
                    finalScore = sumscore + sumscore2
                    if finalProb < MIN_PROB:
                        continue
                    cdType = (int(corType)-1) * (int(corType2)-1)
                    if cdType != COR_TYPE:
                        continue
                    cdTypeStr = str(cdType + 1)
                    key = chem+'.'+dis + '.' + cdTypeStr
                    key_op = chem+'.'+dis + '.' + reverse_corType(cdTypeStr)
                    if key not in relDic:
                        tmpDic = {}
                        tmpDic['id'] = key
                        tmpDic['chem'] = chem
                        tmpDic['disease'] = dis
                        tmpDic['genes'] = [gene]
                        tmpDic['type'] = cdType
                        tmpDic['probs'] = [finalProb]
                        tmpDic['score'] = [finalScore]
                        tmpDic['cor_types'] = [[int(corType)-1, int(corType2)-1]]
                        if key in ChemDisDic:
                            dirProb_inT = ChemDisDic[key][0]
                            if key_op in ChemDisDic:
                                dirProb_op_inT = ChemDisDic[key_op][0]
                            else:
                                dirProb_op_inT = 0
                            if dirProb_inT - dirProb_op_inT > 0.1:
                                tmpDic['directProb'] = ChemDisDic[key][0]
                            else:
                                tmpDic['directProb'] = 0
                        else:
                            tmpDic['directProb'] = 0
                        if key in ChemDisDicAll:
                            dirProb_all = ChemDisDicAll[key][0]
                            if key_op in ChemDisDicAll:
                                dirProb_op_all = ChemDisDicAll[key_op][0]
                            else:
                                dirProb_op_all = 0
                            if dirProb_all - dirProb_op_all > 0.1:
                                tmpDic['directProbAll'] = ChemDisDicAll[key][0]
                            else:
                                tmpDic['directProbAll'] = 0
                        else:
                            tmpDic['directProbAll'] = 0
                        relDic[key] = tmpDic
                    else:
                        relDic[key]['genes'].append(gene)
                        relDic[key]['probs'].append(finalProb)
                        relDic[key]['score'].append(finalScore)
                        relDic[key]['cor_types'].append([int(corType)-1, int(corType2)-1])
    for K in relDic:
        sumScore = 0
        cumProb = 1
        for prob in relDic[K]['probs']:
            cumProb = cumProb * (1 - prob)
        for score in relDic[K]['score']:
            sumScore += score
        finalScore = sumScore
        finalProb = 1 - cumProb
        relDic[K]['finalScore'] = finalScore
        relDic[K]['finalProb'] = finalProb
        if entSubtype[relDic[K]['chem']] == 'Drug':
            relDic[K]['Drug'] = 1
        else:
            relDic[K]['Drug'] = 0
        relDic[K]['chem_name'] = entName[relDic[K]['chem']]
        relDic[K]['disease_name'] = entName[relDic[K]['disease']]
        direct_prob = relDic[K]['directProb']
        if direct_prob < MIN_PROB_direct:
            relDic[K]['directRel'] = 0
        else:
            relDic[K]['directRel'] = 1
        direct_prob_all = relDic[K]['directProbAll']
        if direct_prob_all < MIN_PROB_direct:
            relDic[K]['directRelAll'] = 0
        else:
            relDic[K]['directRelAll'] = 1
        relDic[K]['pathCnt'] = len(relDic[K]['genes'])
    # get the direct Chem-Dis in Time slot
    allCD_inT = {}
    for K in ChemDisDic:
        n1, n2, corType = K.split('.')
        if n2 == targetID and int(corType)-1 == COR_TYPE:
            dirProb, _, dirScore = ChemDisDic[K]
            K_op = n1 + '.' + n2 + '.' + reverse_corType(corType)
            if K_op in ChemDisDic:
                dirProb_op = ChemDisDic[K_op][0]
            else:
                dirProb_op = 0
            if dirProb - dirProb_op > 0.1:
                if n1 not in allCD_inT:
                    allCD_inT[n1] = [dirProb, dirScore, entSubtype[n1]]
                elif allCD_inT[n1][0] < dirProb:
                    allCD_inT[n1] = [dirProb, dirScore, entSubtype[n1]]
    return relDic, allCD_inT

def is_ordered(sublist, ordered_list):
    # Get indices of sublist elements in the ordered_list
    indices = [ordered_list.index(item) for item in sublist if item in ordered_list]
    # Check if indices are in ascending order
    return all(earlier <= later for earlier, later in zip(indices, indices[1:]))

def clean_REL_database(REL):
    new_REL = {}
    for rel in REL:
        n1, n2, relType, corType, direction, db = rel['relID'].split('.')
        if rel['node_one_type'] in ['Gene', 'Disease', 'Chemical'] and rel['node_two_type'] in ['Gene', 'Disease', 'Chemical'] and corType in ['0','2']:
            if not is_ordered([rel['node_one_type'], rel['node_two_type']], ['Chemical', 'Disease', 'Gene']):
                n1_ = n2
                n2_ = n1
                if direction == '12':
                    direction_ = '21'
                elif direction == '21':
                    direction_ = '12'
                else:
                    direction_ = direction
            else:
                n1_ = n1
                n2_ = n2
                direction_ = direction
            ID = '.'.join([n1_, n2_, corType, direction_])
            #if ID not in new_REL:
            #    new_REL[ID] = [[db, rel['prob'], rel['score'], 'Time_NA']]
            #else:
            #    new_REL[ID].append([db, rel['prob'], rel['score'], 'Time_NA'])
            new_REL[ID] = [db, rel['prob'], rel['score'], 'Time_NA']
    return new_REL 

if __name__ == "__main__":
    input_list = 'repurpose_ENT.txt'
    output_path = 'DrugRepurposing_noDirection'  

    RELlist_file = '../data/DBRelations.json'
    EntDictFile = '../data/NER_ID_dict_cap_final.json'
    
    # define some parameters
    COR_TYPE = -1
    MIN_PROB = 0.8  # prob cutoff for the overall indirect prob, and for the function of "get_prob_fromArtList" for indirect search
    MIN_PROB_direct = 0.9 # prob cutoff for the prob to decide whether there is a direct validation, default=0.5
    
    output_folder = output_path + f'_cutoff{MIN_PROB}_{MIN_PROB_direct}/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # read entity dict and find the biokde ID for target entities
    print('Load entities.', flush=True)
    time0 = time.time()
    with open(EntDictFile, 'r') as f:
        Edata = json.load(f)
    entDic = {}
    entSubtype = {}
    entName = {}
    for item in Edata:
        bkid = item['biokdeid']
        entType = item['type']
        subType = item['subtype']
        if 'common name' in item:
            commonName = item['common name']
        else:
            commonName = item['official name']
        if bkid not in entDic:
            entDic[bkid] = entType
            entSubtype[bkid] = subType
            entName[bkid] = commonName
        else:
            print('Replicate biokdeid:', item['biokdeid'])
    print(f'Finished reading entities. Using {time.time()-time0} s', flush=True)
    print('IDs for the target drugs:')



    ###################
    # get entity list for repurposing: "Rare Disease"
    targets_D = {}
    targets_C = {}
    with open(input_list, 'r') as f:
        for line in f:
            x = line[:-1].split('\t')
            if len(x)>1 and x[1]=='C':
                targets_C[x[3]] = x[0]
            elif len(x)>1 and x[1]=='D':
                targets_D[x[3]] = x[0]
    ##############

    print(f'Total targets number:{len(targets_C)+len(targets_D)}')
    print('Loading relation list. ', flush=True)
    time0 = time.time()
    with open(RELlist_file, 'r') as f:
        RELlist = json.load(f)
    #RELlist = clean_REL_database(RELlist)  # only keep correlation relaitons with 'chem','disease' and 'gene'.And the order is adjusted to 'chem-disease-gene'
    print(f'Done loading the relation list. Using {time.time()-time0} s.', flush=True)
    
    # the aggregated relation list already remove the cases that corType is not right, And only keep the MP result for time purpose
    # first extract chem-disease collection without time cutoff. Use this collection to tell if the repurposed drug used to treat the disease eventuraly 
    ChemDisDicAll = {}
    for rel in RELlist:
        finalProb = 0
        n1, n2, corType, direction = rel.split('.')
        if n1 not in entDic or n2 not in entDic:
            continue
        if entDic[n1] == 'Chemical' and entDic[n2] == 'Disease':# and direction == '12':
            key = n1+'.'+n2+'.'+corType
            _, finalProb, sumscore, _ = RELlist[rel]
        if finalProb != 0:
            if key not in ChemDisDicAll:
                ChemDisDicAll[key] = [finalProb, corType, sumscore]
            else:
                if finalProb > ChemDisDicAll[key][0]:
                    ChemDisDicAll[key] = [finalProb, corType, sumscore]
                elif finalProb == ChemDisDicAll[key][0]:
                    if sumscore > ChemDisDicAll[key][2]:
                        ChemDisDicAll[key] = [finalProb, corType, sumscore]

    # begin the repurposing for each target
    for target in targets_D:
        targetID = target
        print(f'Begin repurpose for {entName[targetID]} ({targetID})', flush=True)
        if os.path.isfile(output_folder + '.'.join(['_'.join(entName[targetID].replace("'"," ").split()), targetID, 'directTreatment', 'json'])):
            continue
        relDic,ChemDisDicTarget = repurpose_fromD_database(RELlist, entDic, entSubtype, entName, targetID, MIN_PROB, MIN_PROB_direct, COR_TYPE)
        if relDic != {}:
            outputfile = output_folder + '.'.join([convert_entName(entName[targetID]), targetID, 'json'])
            with open(outputfile, 'w') as f:
                json.dump(list(relDic.values()), f, indent=2)
        if ChemDisDicTarget != {}:
            outputfile = output_folder + '.'.join([convert_entName(entName[targetID]), targetID, 'directTreatment', 'json'])
            with open(outputfile, 'w') as f:
                json.dump(ChemDisDicTarget, f, indent=2)
        if relDic == {} and ChemDisDicTarget == {}:
            print(f'No output for {entName[targetID]} ({targetID})', flush=True)
        print(f'Finished repurposing for {entName[targetID]} ({targetID}).', flush=True)
    for target in targets_C:
        targetID = target
        print(f'Begin repurpose for {entName[targetID]} ({targetID})', flush=True)
        if os.path.isfile(output_folder + '.'.join(['_'.join(entName[targetID].replace("'"," ").split()), targetID, 'directTreatment', 'json'])):
            continue
        relDic,ChemDisDicTarget = repurpose_fromC_database(RELlist, entDic, entSubtype, entName, targetID, MIN_PROB, MIN_PROB_direct, COR_TYPE)
        if relDic != {}:
            outputfile = output_folder + '.'.join([convert_entName(entName[targetID]), targetID, 'json'])
            with open(outputfile, 'w') as f:
                json.dump(list(relDic.values()), f, indent=2)
        if ChemDisDicTarget != {}:
            outputfile = output_folder + '.'.join([convert_entName(entName[targetID]), targetID, 'directTreatment', 'json'])
            with open(outputfile, 'w') as f:
                json.dump(ChemDisDicTarget, f, indent=2)
        if relDic == {} and ChemDisDicTarget == {}:
            print(f'No output for {entName[targetID]} ({targetID})', flush=True)
        print(f'Finished repurposing for {entName[targetID]} ({targetID}).', flush=True)
    print('Job finished')

