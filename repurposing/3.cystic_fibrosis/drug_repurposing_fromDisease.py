# relation id rule: cell line - chemical - disease - gene - mutations - species


import json
import sys
import argparse
import time
from datetime import datetime
import os
import math
import pandas as pd

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

if __name__ == "__main__":
    input_list = 'repurpose_ENT_disease.txt'
    output_path = 'DrugRepurposing_time'

    RELlist_time = '../data/Aggregated_PubMedRelList.json'
    EntDictFile = '../data/NER_ID_dict_cap_final.json'
    
    # define some parameters
    COR_TYPE = -1
    MIN_PROB = 0.5  # prob cutoff for the overall indirect prob, and for the function of "get_prob_fromArtList" for indirect search
    MIN_PROB_direct = 0.5 # prob cutoff for the prob to decide whether there is a direct validation, this prob cut is check at the last step, so this value should be always <= MIN_PROB, otherwise the code need to be changed!
    targetType = 'D'
    
    output_folder = output_path + f'_cutoff{MIN_PROB}_{MIN_PROB_direct}/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)


    # read entity for repurposing
    print('Reading the entities for repurposing.')
    if input_list != "":
        with open(input_list,'r') as f:
            Entity_list = [x.split('\n')[0] for x in f if not x.startswith('#')]    #['baricitinib','ibuprofen',...]
        Entity_list = [x.lower() for x in Entity_list if x != '']
    else:
        print('Error! Please give the file containing the target entity list')
    print('Reading entities done.')

    # read entity dict and find the biokde ID for target entities
    print('Load entities.', flush=True)
    time0 = time.time()
    with open(EntDictFile, 'r') as f:
        Edata = json.load(f)
    target_ID = {x:[] for x in Entity_list}  #{entity_name: entID} entity_name='common name' or 'official name'
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
        if item['type'] == 'Disease':
            if commonName.lower() in Entity_list:
                target_ID[commonName.lower()].append(bkid)
            elif item['official name'].lower() in Entity_list:
                target_ID[item['official name'].lower()].append(bkid)
        if bkid not in entDic:
            entDic[bkid] = entType
            entSubtype[bkid] = subType
            entName[bkid] = commonName
        else:
            print('Replicate biokdeid:', item['biokdeid'])
    print(f'Finished reading entities. Using {time.time()-time0} s', flush=True)
    print('IDs for the target drugs:')
    for x in target_ID:
        print(x, target_ID[x])

    print('Loading aggregated relation list with time. ', flush=True)
    time0 = time.time()
    with open(RELlist_time, 'r') as f:
        RELlist = json.load(f)
    print(f'Done loading the aggregated relation list with time. Using {time.time()-time0} s.', flush=True)
    
    # the aggregated relation list already remove the cases that corType is not right, And only keep the MP result for time purpose
    # first extract chem-disease collection without time cutoff. Use this collection to tell if the repurposed drug used to treat the disease eventuraly 
    ChemDisDicAll = {}
    CD_time = {target_ID[target][0]:[] for target in Entity_list}    # {target_id:[target_gene_time, ...], ..}
    for rel in RELlist:
        n1, n2, corType, direction = rel.split('.')
        if n1 not in entDic or n2 not in entDic:
            continue
        if entDic[n1] == 'Chemical' and entDic[n2] == 'Disease' and direction == '12':
            rels = get_rel_within_time(RELlist[rel], '1000-01-01', '2023-01-01')  # [[pmid, prob, score, time_object],...], early time cutoff specially for COVID-19
            finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB_direct)
            if finalProb == 0:
                continue
            key = n1+'.'+n2+'.'+corType
            if key not in ChemDisDicAll:
                ChemDisDicAll[key] = [finalProb, corType, sumscore]
            else:
                if finalProb > ChemDisDicAll[key][0]:
                    ChemDisDicAll[key] = [finalProb, corType, sumscore] 
                elif finalProb == ChemDisDicAll[key][0]:
                    if sumscore > ChemDisDicAll[key][2]:
                        ChemDisDicAll[key] = [finalProb, corType, sumscore]
        if n1 in CD_time and entDic[n2] == 'Gene' and direction == '21':
            CD_time[n1].extend([X[3] for X in RELlist[rel]])

    # begin the repurposing for each target
    for target in Entity_list:
        targetID = target_ID[target][0]
        time1 = '1000-01-01'
        if CD_time[targetID] == []:
            print(f"{target} {targetID} doesn't have any relation with genes", flush=True)
            continue
        # find the earlist time for prediction
        earliest_date = min(datetime.strptime(date, '%Y-%m-%d') for date in CD_time[targetID])
        earliest_year = earliest_date.year
        Time_list = [str(YY)+'-01-01' for YY in range (earliest_year, 2024)]
        allCD_inT = {}  # {time:{disID:[prob,score],..}}
        for time2 in Time_list:
            print(f"Drug repurposint:Disease {target} ({targetID}), Time: {time1}-{time2}", flush=True)
            # get Chemical-Gene, Disease-Gene dict for repurposing; Get Chem-Disease dict for direct relation
            ChemGeneDic = {}
            GeneDisDic = {}
            ChemDisDic = {}
            
            for rel in RELlist:
                # relation id rule: cell line - chemical - disease - gene - mutations - species
                n1, n2, corType, direction = rel.split('.')
                if n1 not in entDic or n2 not in entDic:
                    continue
                if entDic[n1] == 'Chemical' and entDic[n2] == 'Gene' and direction == '12':   
                    # calculate probability
                    rels = get_rel_within_time(RELlist[rel], time1, time2)
                    finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB)
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
                if entDic[n1] == 'Chemical' and n2==targetID and direction == '12':
                    # calculate probability
                    rels = get_rel_within_time(RELlist[rel], time1, time2)  # [[pmid, prob, score, time_object],...]
                    finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB_direct)
                    if finalProb == 0:
                        continue
                    key = n1+'.'+n2+'.'+corType
                    if key not in ChemDisDic:
                        ChemDisDic[key] = [finalProb, corType, sumscore]
                    elif finalProb > ChemDisDic[key][0]:
                        ChemDisDic[key] = [finalProb, corType, sumscore]

                if n1 == targetID and entDic[n2] == 'Gene' and direction == '21':
                    # calculate probability
                    rels = get_rel_within_time(RELlist[rel], time1, time2)
                    finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB)
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
                    
            print("Finish organize aggregate relation.")
            print("Begin drug repurposing process.")
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
                    # adding 0.01 is equavalent to subtract 0.01 from all probabilities, which softens this variable and helps ranking
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
            outputfile = output_folder + '_'.join(target.split()) + '.' + targetID + '.' + time1.replace("-","") +'_'+time2.replace("-","") +'.json'
            if relDic != {}:
                with open(outputfile, 'w') as f:
                    json.dump(list(relDic.values()), f, indent=2)
            
            # Output the direct Chem-Dis in Time slot
            allCD_inT[time2] = {}
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
                        if n1 not in allCD_inT[time2]:
                            allCD_inT[time2][n1] = [dirProb, dirScore]
                        elif allCD_inT[time2][n1][0] < dirProb:
                            allCD_inT[time2][n1] = [dirProb, dirScore]

        # output direct relation
        ChemDisDicTarget = {}
        for key in ChemDisDicAll:
            n1, n2, corType = key.split('.')
            if n2 == targetID and int(corType)-1 == COR_TYPE:
                dirProb, _, dirScore = ChemDisDicAll[key]
                K_op = n1 + '.' + n2 + '.' + reverse_corType(corType)
                if K_op in ChemDisDicAll:
                    dirProb_op = ChemDisDicAll[K_op][0]
                else:
                    dirProb_op = 0
                if dirProb - dirProb_op > 0.1:
                    if n1 not in ChemDisDicTarget:
                        ChemDisDicTarget[n1] = [ChemDisDicAll[key]]
                    else:
                        ChemDisDicTarget[n1].append(ChemDisDicAll[key])
        treatment = {}
        for Chemical in ChemDisDicTarget:
            if len(ChemDisDicTarget[Chemical]) > 1:
                temp = sorted(ChemDisDicTarget[Chemical], key=lambda x:(x[0], x[2]), reverse=True)
                ChemDisDicTarget[Chemical] = temp[0]
            else:
                ChemDisDicTarget[Chemical] = ChemDisDicTarget[Chemical][0]
            if int(ChemDisDicTarget[Chemical][1]) - 1 == COR_TYPE:
                if Chemical not in treatment:
                    treatment[Chemical] = ChemDisDicTarget[Chemical] 
        outputfile = output_folder + '_'.join(target.split()) + '.' + targetID  +'.directTreatment.json'
        with open(outputfile, 'w') as f:
            json.dump(treatment, f, indent=2)

        outputfile = output_folder + '_'.join(target.split()) + '.' + targetID  +'.directTreatment_inT.json'
        with open(outputfile, 'w') as f:
            json.dump(allCD_inT, f, indent=2)
    print('Job finished')
