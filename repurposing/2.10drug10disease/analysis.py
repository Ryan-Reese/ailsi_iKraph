'''
This code is used to calculate the validation score for predicted repurposed drug/indications using
PubMed abstracts and databases relations
'''
import pandas as pd
import json
import sys
import argparse
import time
from datetime import datetime
import os
import math
import pandas as pd
import glob

####################################################################################################
####################################################################################################

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


def combine_twoList(L1, L2):
    if isinstance(L1[0], str):
        L = list(set(L1+L2))
    elif isinstance(L1[0], dict):
        L = L1[:]
        L_id = [x['id'] for x in L]
        for x in L2:
            if x['id'] not in L_id:
                L.append(x)
    return L


def convertTargetName(nameDic):
    name_list = list(nameDic.keys())
    converter = {}
    for x in name_list:
        if '_' in x:
            y = x.split('_')
            converter[x] = '_'.join([z.capitalize() for z in y])
        elif len(x)<=4:
            converter[x] = x.upper()
        else:
            converter[x] = x.capitalize()
    return converter
   

def get_top_candidate(L, top=50, Filter='Drug'):
    if Filter == False:
        L = L
    else:
        L = [x for x in L if x[Filter]==1]
    L = sorted(L, key=lambda x:(x['finalProb'], x['finalScore']), reverse=True)
    return L[:top]

####################################################################################################
####################################################################################################
#TOP = [50, 100, 150, 200, 250, 300, 350, 400]
TOP = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900,1000]

# read entity dict and find the biokde ID for target entities
RELlist_time = '../data/Aggregated_PubMedRelList.json'
EntDictFile = '../data/NER_ID_dict_cap_final.json'
RELlistDB_file = '../data/DBRelations.json'
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

print('Loading aggregated relation list with time. ', flush=True)
time0 = time.time()
with open(RELlist_time, 'r') as f:
    RELlist = json.load(f)
print(f'Done loading the aggregated relation list with time. Using {time.time()-time0} s.', flush=True)

print('Loading DB relation list. ', flush=True)
time0 = time.time()
with open(RELlistDB_file, 'r') as f:
    RELlist_DB = json.load(f)
#RELlist_DB = clean_REL_database(RELlist_DB)  # only keep correlation relaitons with 'chem','disease' and 'gene'.And the order is adjusted to 'chem-disease-gene'
print(f'Done loading the relation list. Using {time.time()-time0} s.', flush=True)

# get the DisChemDic dict which contains all the chem-dis negative correlation from database, chem->disease, DisChemDic_DB = [dis][chem] 
DisChemDic_DB = {}
for rel in RELlist_DB:
    n1, n2, corType, direction = rel.split('.')
    finalProb = 0
    if n1 not in entDic or n2 not in entDic:
        continue
    if entDic[n1] == 'Chemical' and entDic[n2] == 'Disease':# and direction == '12':
        _, finalProb, sumscore, _ = RELlist_DB[rel]
    if finalProb != 0:
        if n2 not in DisChemDic_DB:
            DisChemDic_DB[n2] = {}
            DisChemDic_DB[n2][n1] = [finalProb, sumscore]
        else:
            if n1 not in DisChemDic_DB[n2]:
                DisChemDic_DB[n2][n1] = [finalProb, sumscore]
            else:
                if DisChemDic_DB[n2][n1][0] < finalProb:
                    DisChemDic_DB[n2][n1] = [finalProb, sumscore]

with open('repurpose_ENT.txt','r') as f:
    data = [line[:-1].split('\t') for line in f]

nameDic = {}
for x in data:
    name = x[0]
    common_name = '_'.join(x[2].lower().split())
    name = name[0].upper()+name[1:]
    nameDic[common_name] = {'name':name, 'type':x[1]}
convName = convertTargetName(nameDic)


database_path = 'DrugRepurposing_noDirection_cutoff'
biokde_path = 'DrugRepurposing_result_cutoff'
for folder in ['0.8_0.9']:
    #if folder.lower() == 'exit':
    #    print("Exiting the program.")
    #    break
    if folder.endswith('/'):
        folder = folder[:-1]
    outputname = 'top_10_cutoff'+folder
    MIN_PROB_direct=float(folder.split('_')[1])
    
    # get the DisChemDic dict which contains all the chem-dis negative correlation, chem->disease, DisChemDic = [dis][chem]
    DisChemDic = {}
    for rel in RELlist:
        n1, n2, corType, direction = rel.split('.')
        if n1 not in entDic or n2 not in entDic:
            continue
        if entDic[n1] == 'Chemical' and entDic[n2] == 'Disease' and direction == '12' and corType == '0':
            rels = get_rel_within_time(RELlist[rel], '1000-01-01', '2023-01-01')  # [[pmid, prob, score, time_object],...]
            finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB_direct)
            if finalProb == 0:
                continue
            rel_op = '.'.join([n1, n2, '2', '12'])
            finalProb_op = 0.0
            if rel_op in RELlist:
                rels_op = get_rel_within_time(RELlist[rel_op], '1000-01-01', '2023-01-01') 
                finalProb_op, sumscore_op = get_prob_fromArtList(rels_op, MIN_PROB_direct)
            if finalProb - finalProb_op > 0.1:
                if n2 not in DisChemDic:
                    DisChemDic[n2] = {}
                    DisChemDic[n2][n1] = [finalProb, sumscore]
                else:
                    if n1 not in DisChemDic[n2]:
                        DisChemDic[n2][n1] = [finalProb, sumscore]
                    else:
                        if DisChemDic[n2][n1][0] < finalProb:
                            DisChemDic[n2][n1] = [finalProb, sumscore]
    
    # DisChemDic_DB[n2][n1] = [finalProb, sumscore], use check if finalProb with MIN_PROB_direct to see if the direct relation can be used 
    DisChemDic_DB_sub = {}
    for n2 in DisChemDic_DB:
        for n1 in DisChemDic_DB[n2]:
            finalProb, sumscore = DisChemDic_DB[n2][n1]
            if finalProb >= MIN_PROB_direct:
                if n2 not in DisChemDic_DB_sub:
                    DisChemDic_DB_sub[n2] = {}
                    DisChemDic_DB_sub[n2][n1] = [finalProb, sumscore]
                else:
                    if n1 not in DisChemDic_DB_sub[n2]:
                        DisChemDic_DB_sub[n2][n1] = [finalProb, sumscore]
 
    # process top 10 disease
    topDis = {k:v for k,v in nameDic.items() if v['type']=='D'}
    topDis = {k:topDis[k] for k in sorted(topDis)}
    #result = {'Disease':[], 'Known_Drugs_PM':[], 'Known_Drugs_DB':[], 'Known_Drugs_All':[], \
    #          'Tot_RP_PM':[], 'Tot_RP_DB':[], 'Tot_RP_All':[], \
    #          'RP_byPM_Known_byPM':[], 'RP_byPM_Known_byDB':[], 'RP_byPM_Known_byAll':[], \
    #          'RP_byDB_Known_byPM':[], 'RP_byDB_Known_byDB':[], 'RP_byDB_Known_byAll':[]}
    result = [{'Disease':[], 'Known_Drugs_DB':[], 'Tot_RP_PM':[], 'Tot_RP_DB':[], \
               'RP_byPM_Known_byDB':[], 'RP_byDB_Known_byDB':[], \
               'RP_byPM_Known_byPM':[], 'RP_byDB_Known_byPM':[], \
               'PM_recall_byDB':[], 'DB_recall_byDB':[], 'PM_precision_byDB':[], 'DB_precision_byDB':[], \
               'PM_recall_byPM':[], 'DB_recall_byPM':[], 'PM_precision_byPM':[], 'DB_precision_byPM':[]} for i in range (len(TOP))]

    for d in topDis:
        record_d = {'drug':[], 'RP_byPM':[], 'RP_byDB':[], 'KN_byPM':[], 'KN_byDB':[]}
        print(d)
        FL = glob.glob(biokde_path+folder+'/'+d+'*.json')
        directF = ''
        repF = ''
        for F in FL:
            if 'directTreatment' in F:
                directF = F
            else:
                repF = F
        if directF != '':
            with open(directF, 'r') as f:
                directRe_PM = json.load(f)
        else:
            directRe_PM = []
        if repF != '':
            with open(repF, 'r') as f:
                repRe_PM = json.load(f)
        else:
            repRe_PM = []
        disease = nameDic[F.split('/')[-1].split('.')[0]]['name']
        
        d = convName[d]
        FL = glob.glob(database_path+folder+'/'+d+'*.json')
        directRe_DB = []
        repRe_DB = []
        directF = ''
        repF = ''
        for F in FL:
            if 'directTreatment' in F:
                directF = F
            else:
                repF = F
        if directF != '':
            with open(directF, 'r') as f:
                directRe_DB = json.load(f)
        else:
            directRe_DB = []
        if repF != '':
            with open(repF, 'r') as f:
                repRe_DB = json.load(f)
        else:
            repRe_DB = []
        known_drug_db = [x for x in directRe_DB if entSubtype[x]=='Drug']
        known_drug_pm = [x for x in directRe_PM if entSubtype[x]=='Drug']
        for ind, top in enumerate(TOP):
            tot_rp_pm = get_top_candidate(repRe_PM, top=top, Filter='Drug')
            tot_rp_db = get_top_candidate(repRe_DB, top=top, Filter='Drug')

            rp_byPM_known_byDB = [x['chem'] for x in tot_rp_pm if x['chem'] in known_drug_db]
            rp_byDB_known_byDB = [x['chem'] for x in tot_rp_db if x['chem'] in known_drug_db]
            rp_byPM_known_byPM = [x['chem'] for x in tot_rp_pm if x['chem'] in known_drug_pm]
            rp_byDB_known_byPM = [x['chem'] for x in tot_rp_db if x['chem'] in known_drug_pm]
    
            result[ind]['Disease'].append(disease)
            result[ind]['Known_Drugs_DB'].append(len(known_drug_db))
            result[ind]['Tot_RP_PM'].append(len(tot_rp_pm))
            result[ind]['Tot_RP_DB'].append(len(tot_rp_db))
            result[ind]['RP_byPM_Known_byDB'].append(len(rp_byPM_known_byDB))
            result[ind]['RP_byDB_Known_byDB'].append(len(rp_byDB_known_byDB))
            result[ind]['RP_byPM_Known_byPM'].append(len(rp_byPM_known_byPM))
            result[ind]['RP_byDB_Known_byPM'].append(len(rp_byDB_known_byPM))
            result[ind]['PM_recall_byDB'].append(len(rp_byPM_known_byDB)/len(known_drug_db))
            result[ind]['DB_recall_byDB'].append(len(rp_byDB_known_byDB)/len(known_drug_db))
            result[ind]['PM_recall_byPM'].append(len(rp_byPM_known_byPM)/len(known_drug_pm))
            result[ind]['DB_recall_byPM'].append(len(rp_byDB_known_byPM)/len(known_drug_pm))
            if len(tot_rp_pm) == 0:
                result[ind]['PM_precision_byDB'].append(0)
                result[ind]['PM_precision_byPM'].append(0)
            else:
                result[ind]['PM_precision_byDB'].append(len(rp_byPM_known_byDB)/len(tot_rp_pm))
                result[ind]['PM_precision_byPM'].append(len(rp_byPM_known_byPM)/len(tot_rp_pm))
            if len(tot_rp_db) == 0:
                result[ind]['DB_precision_byDB'].append(0)
                result[ind]['DB_precision_byPM'].append(0)
            else:
                result[ind]['DB_precision_byDB'].append(len(rp_byDB_known_byDB)/len(tot_rp_db))
                result[ind]['DB_precision_byPM'].append(len(rp_byDB_known_byPM)/len(tot_rp_db))


        
    # process top 10 drugs
    topDrugs = {k:v for k,v in nameDic.items() if v['type']=='C'}
    topDrugs = {k:topDrugs[k] for k in sorted(topDrugs)}
    all_dis_withTreatment = set(DisChemDic.keys())
    #result = {'Drug':[], 'Known_Dis_PM':[], 'Known_Dis_DB':[], 'Known_Dis_All':[], \
    #          'Tot_RP_PM':[], 'Tot_RP_DB':[], 'Tot_RP_All':[], \
    #          'RP_byPM_Known_byPM':[], 'RP_byPM_Known_byDB':[], 'RP_byPM_Known_byAll':[], \
    #          'RP_byDB_Known_byPM':[], 'RP_byDB_Known_byDB':[], 'RP_byDB_Known_byAll':[]}
    result_drug = [{'Drug':[], 'Known_Dis_DB':[],'Tot_RP_PM':[], 'Tot_RP_DB':[], \
               'RP_byPM_Known_byDB':[], 'RP_byDB_Known_byDB':[], \
               'RP_byPM_Known_byPM':[], 'RP_byDB_Known_byPM':[], \
               'PM_recall_byDB':[], 'DB_recall_byDB':[], 'PM_precision_byDB':[], 'DB_precision_byDB':[], \
               'PM_recall_byPM':[], 'DB_recall_byPM':[], 'PM_precision_byPM':[], 'DB_precision_byPM':[]} for i in range (len(TOP))]

    for d in topDrugs:
        record_d = {'disease':[], 'RP_byPM':[], 'RP_byDB':[], 'KN_byPM':[], 'KN_byDB':[]}
        FL = glob.glob(biokde_path+folder+'/'+d+'*.json')
        for F in FL:
            if 'directTreatment' in F:
                directF = F
            else:
                repF = F
        with open(directF, 'r') as f:
            directRe_PM = json.load(f)
        with open(repF, 'r') as f:
            repRe_PM = json.load(f)
        drug = nameDic[F.split('/')[-1].split('.')[0]]['name']

        d = convName[d]
        FL = glob.glob(database_path+folder+'/'+d+'*.json')
        directRe_DB = []
        repRe_DB = []
        for F in FL:
            if 'directTreatment' in F:
                directF = F
            else:
                repF = F
        with open(directF, 'r') as f:
            directRe_DB = json.load(f)
        with open(repF, 'r') as f:
            repRe_DB = json.load(f)


        known_dis_db = [x for x in directRe_DB]
        known_dis_pm = [x for x in directRe_PM]
        for ind, top in enumerate(TOP):
            tot_rp_pm = get_top_candidate(repRe_PM, top=top, Filter=False)
            tot_rp_db = get_top_candidate(repRe_DB, top=top, Filter=False)

            rp_byPM_known_byDB = [x['disease'] for x in tot_rp_pm if x['disease'] in known_dis_db]
            rp_byDB_known_byDB = [x['disease'] for x in tot_rp_db if x['disease'] in known_dis_db]
            rp_byPM_known_byPM = [x['disease'] for x in tot_rp_pm if x['disease'] in known_dis_pm]
            rp_byDB_known_byPM = [x['disease'] for x in tot_rp_db if x['disease'] in known_dis_pm]
            

            result_drug[ind]['Drug'].append(drug)
            result_drug[ind]['Known_Dis_DB'].append(len(known_dis_db))
            result_drug[ind]['Tot_RP_PM'].append(len(tot_rp_pm))
            result_drug[ind]['Tot_RP_DB'].append(len(tot_rp_db))
            result_drug[ind]['RP_byPM_Known_byDB'].append(len(rp_byPM_known_byDB))
            result_drug[ind]['RP_byDB_Known_byDB'].append(len(rp_byDB_known_byDB))
            result_drug[ind]['RP_byPM_Known_byPM'].append(len(rp_byPM_known_byPM))
            result_drug[ind]['RP_byDB_Known_byPM'].append(len(rp_byDB_known_byPM))
            result_drug[ind]['PM_recall_byDB'].append(len(rp_byPM_known_byDB)/len(known_dis_db))
            result_drug[ind]['DB_recall_byDB'].append(len(rp_byDB_known_byDB)/len(known_dis_db))
            result_drug[ind]['PM_recall_byPM'].append(len(rp_byPM_known_byPM)/len(known_dis_pm))
            result_drug[ind]['DB_recall_byPM'].append(len(rp_byDB_known_byPM)/len(known_dis_pm))
            if len(tot_rp_pm) == 0:
                result_drug[ind]['PM_precision_byDB'].append(0)
                result_drug[ind]['PM_precision_byPM'].append(0)
            else:
                result_drug[ind]['PM_precision_byDB'].append(len(rp_byPM_known_byDB)/len(tot_rp_pm))
                result_drug[ind]['PM_precision_byPM'].append(len(rp_byPM_known_byPM)/len(tot_rp_pm))
            if len(tot_rp_db) == 0:
                result_drug[ind]['DB_precision_byDB'].append(0)
                result_drug[ind]['DB_precision_byPM'].append(0)
            else:
                result_drug[ind]['DB_precision_byDB'].append(len(rp_byDB_known_byDB)/len(tot_rp_db))
                result_drug[ind]['DB_precision_byPM'].append(len(rp_byDB_known_byPM)/len(tot_rp_db))

    with pd.ExcelWriter(f'10Drug_10Disease_cutoff{folder}.xlsx', engine='xlsxwriter') as writer:
        for ind, top in enumerate(TOP):
            df_dis = pd.DataFrame(result[ind])
            df_drug = pd.DataFrame(result_drug[ind])
            sheet_name1 = 'Disease_Top'+str(top)
            sheet_name2 = 'Drug_Top'+str(top)
            df_dis.to_excel(writer, sheet_name=sheet_name1, index=False)
            df_drug.to_excel(writer, sheet_name=sheet_name2, index=False)
    print(f'Cutoff{folder} finished')
print('Job finished')    
    
    
