import json
import numpy as np
import pandas as pd
import glob
import datetime
import os



def read_json(filename):
    with open(filename,'r') as f:
        data = json.load(f)
    return data

def ana(data, rareDis_list, disDic, geneDic, finalPathGene):
    rareDis_list = set(rareDis_list)
    data_t = {}
    genes = set()
    for x in data:
        genes = set(x['genes'])|genes
    Ngene = len(genes)
    for k in data[0].keys():
        if k in ['genes', 'probs','score', 'cor_types']:
            continue
        data_t[k] = [x[k] for x in data]
    data_t['official_name'] = []
    data_t['rare_disease'] = []
    data_t['genes'] = []
    data_t['final_pathG'] = []
    for x in data:
        data_t['official_name'].append(disDic[x['disease']])
        if x['disease'] in rareDis_list:
            data_t['rare_disease'].append(1)
        else:
            data_t['rare_disease'].append(0)
        gene_names = [geneDic[g] for g in x['genes']]
        data_t['genes'].append(','.join(gene_names))
        data_t['final_pathG'].append(finalPathGene[x['disease']])

    df = pd.DataFrame(data_t)
    N_totDisease = len(df)
    N_dirIntime = len(df.loc[df.directRel==1])
    N_dirAlltime = len(df.loc[df.directRelAll==1])
    N_totRareDis = len(df.loc[df.rare_disease==1])
    N_RareIntime = len(df.loc[(df.rare_disease==1)&(df.directRel==1)])
    N_RareAlltime = len(df.loc[(df.rare_disease==1)&(df.directRelAll==1)])
    N_gene_avg = np.mean(df.pathCnt.tolist())
    df_ss = df.sort_values(by=['rare_disease', 'finalScore', 'finalProb', 'pathCnt'], ascending=False)
    df_ss.reset_index(drop=True, inplace=True)
    return N_totDisease, N_dirIntime, N_dirAlltime, N_totRareDis, N_RareIntime, N_RareAlltime, Ngene, N_gene_avg, df_ss

def ana_fromDisease(data, chemDic, geneDic, finalPathGene):
    data_t = {}
    genes = set()
    for x in data:
        genes = set(x['genes'])|genes
    Ngene = len(genes)
    for k in data[0].keys():
        if k in ['genes', 'probs','score', 'cor_types']:
            continue
        data_t[k] = [x[k] for x in data]
    data_t['chem_official_name'] = []
    data_t['genes'] = []
    data_t['final_pathG'] = []
    for x in data:
        data_t['chem_official_name'].append(chemDic[x['chem']])
        gene_names = [geneDic[g] for g in x['genes']]
        data_t['genes'].append(','.join(gene_names))
        data_t['final_pathG'].append(finalPathGene[x['chem']])
    df = pd.DataFrame(data_t)
    N_totChem = len(df)
    N_dirIntime = len(df.loc[df.directRel==1])
    N_dirAlltime = len(df.loc[df.directRelAll==1])
    N_totDrug = len(df.loc[df.Drug==1])
    N_DrugIntime = len(df.loc[(df.Drug==1)&(df.directRel==1)])
    N_DrugAlltime = len(df.loc[(df.Drug==1)&(df.directRelAll==1)])
    N_gene_avg = np.mean(df.pathCnt.tolist())
    df_ss = df.sort_values(by=['finalScore', 'finalProb', 'pathCnt'], ascending=False)
    df_ss.reset_index(drop=True, inplace=True)
    return N_totChem, N_dirIntime, N_dirAlltime, N_totDrug, N_DrugIntime, N_DrugAlltime, Ngene, N_gene_avg, df_ss


def orginze_df(df):
    df = df.rename(columns={
        'chem': 'chemID',
        'disease': 'diseaseID',
        'directRel': 'reported',
        'directRelAll': 'validatedAllTime'})
    columns_to_keep = ['id', 'chemID', 'diseaseID', 'directProb', 'directProbAll',
                   'finalScore', 'finalProb', 'chem_name', 'disease_name', 'chem_official_name',
                   'reported', 'validatedAllTime', 'pathCnt', 'Drug','validTime', 'genes', 'final_pathG']
    new_df = df[columns_to_keep]
    return new_df

def orginze_df_onlyDrug(df):
    df = df.rename(columns={
        'chem': 'chemID',
        'disease': 'diseaseID',
        'directRel': 'reported',
        'directRelAll': 'validatedAllTime'})
    columns_to_keep = ['id', 'chemID', 'diseaseID', 'directProb', 'directProbAll',
                   'finalScore', 'finalProb', 'chem_name', 'disease_name', 'chem_official_name',
                   'reported', 'validatedAllTime', 'pathCnt', 'Drug','validTime', 'genes', 'final_pathG']
    new_df = df[columns_to_keep] 
    new_df_ = new_df.loc[new_df.Drug==1]
    return new_df_

def organize_sum_df(df):
    df = df.rename(columns={
        'N_valided': 'N_reported', 
        'N_valid_later': 'N_validatedAllTime',
        'N_validedDrug': 'N_reportedDrug',
        'N_validedDrug_later': 'N_validatedAllTimeDrug'})
    return df

def organize_sum_df_onlyDrug(df):
    df = df.rename(columns={
        'N_valided': 'N_reported',
        'N_valid_later': 'N_validatedAllTime',
        'N_validedDrug': 'N_reportedDrug',
        'N_validedDrug_later': 'N_validatedAllTimeDrug'})        
    columns_to_keep = ['time', 'N_totalDrug', 'N_reportedDrug', 'N_validatedAllTimeDrug', 'N_genes', 'N_gene_avg', 'valid_year_Drug']
    new_df = df[columns_to_keep]
    return new_df 

        
if __name__ == "__main__":
    #input_fold = input('Input folder: ')
    input_fold = "DrugRepurposing_time_cutoff0.5_0.5/"
    files = glob.glob(input_fold+'/*.json')
    files = [x for x in files if 'directTreatment' not in x]
    sorted_files = sorted(files, key=lambda x: datetime.datetime.strptime(x.split('.')[-2].split('_')[1], '%Y%m%d'))
    summary = {'time':[], 'N_totalChem':[], 'N_valided':[], 'N_valid_later':[], \
                'N_totalDrug':[], 'N_validedDrug':[], 'N_validedDrug_later':[], 'N_genes':[], 'N_gene_avg':[], 'valid_year':[], 'valid_year_Drug':[]}  
    output_fold = input_fold.split('/')
    if output_fold[-1] == '':
        output_fold = 'summary_'+output_fold[-2] +'_new/'
    else:
        output_fold = 'summary_'+output_fold[-1] +'_new/'
    if not os.path.isdir(output_fold):
        os.mkdir(output_fold)

    with open('../data/GeneEntities.json','r') as f:
        gene_ = json.load(f)
    geneDic = {x['biokdeid']:x['common name'] for x in gene_}
    del gene_

    with open('../data/ChemEntities.json', 'r') as f:
        chem_ = json.load(f)
    chemDic = {}
    drug_list = []
    for x in chem_:
        if x['subtype']=='Drug':
            drug_list.append(x['biokdeid'])
        chemDic[x['biokdeid']] = x['official name']
    drug_list = set(drug_list)
    del chem_

    repurposed_chem = set()
    validate_chem = {}   # {'chemID':{'time', 'drug':0 or 1}}
    final_pathG = {}
    for i, F in enumerate(sorted_files):
        data_ = read_json(F)
        date = F.split('.')[-2].split('_')[1]
        for d in data_:
            final_pathG[d['chem']] = ','.join([geneDic[x] for x in d['genes']])
            if d['directRel'] == 1:
                if d['chem'] in validate_chem:
                    continue    #'Disease already repurposed, and time recorded'
                else:
                    validate_chem[d['chem']] = {'time':date, 'Drug':d['Drug']}
            if i == len(sorted_files)-1 and d['directRel'] == 0 and d['directRelAll'] == 1:
                validate_chem[d['chem']] = {'time':'20230401', 'Drug':d['Drug']}

    for i, F in enumerate(sorted_files):
        data_ = read_json(F)
        date = F.split('.')[-2].split('_')[1]           
        data = [x for x in data_ if x['chem'] not in repurposed_chem] # remove the disease already been repurposed
        disL = [x['chem'] for x in data]
        repurposed_chem = repurposed_chem | set(disL)
        if data == []:
            continue
        N_totChem, N_dirIntime, N_dirAlltime, N_totDrug, N_DrugIntime, N_DrugAlltime, Ngene,N_gene_avg, df_ss = ana_fromDisease(data, chemDic, geneDic, final_pathG)
        df_validTime = [-1 for j in range (len(df_ss))]
        for j,row in df_ss.iterrows():
            if row['chem'] in validate_chem and row['directRel'] == 0:
                df_validTime[j] = int(validate_chem[row['chem']]['time'][:4]) - int(date[:4])
        df_ss['validTime'] = df_validTime
        summary['time'].append(date)
        summary['N_totalChem'].append(N_totChem)
        summary['N_valided'].append(N_dirIntime)
        summary['N_valid_later'].append(N_dirAlltime)
        summary['N_totalDrug'].append(N_totDrug)
        summary['N_validedDrug'].append(N_DrugIntime)
        summary['N_validedDrug_later'].append(N_DrugAlltime)
        summary['N_genes'].append(Ngene)
        summary['N_gene_avg'].append(N_gene_avg)
        df_ss_new = orginze_df_onlyDrug(df_ss)
        df_ss_new.to_csv(output_fold+date+'.csv', index=False)
    

    with open(input_fold+'/cystic_fibrosis.11666.directTreatment_inT.json','r') as f:
        dirT_inT = json.load(f)     # {'time':{'chem':[prob, score], ,,,},,,}
    dirT_inT_dict = {}  # the time for the chem first been validated
    for T in dirT_inT:
        for C in dirT_inT[T]:
            if C not in dirT_inT_dict:
                dirT_inT_dict[C] = T
    
    repurposed_chem = set()
    for F in sorted_files:
        data_ = read_json(F)
        data = [x for x in data_ if x['chem'] not in repurposed_chem] # remove the disease already been repurposed
        if data == []:
            continue
        disL = [x['chem'] for x in data]
        repurposed_chem = repurposed_chem | set(disL)
        T = []
        T_Drug = []
        current_year = int(F.split('.')[-2].split('_')[1][:4])
        for d in data:
            if d['directRelAll'] == 1 and d['directRel'] == 0:
                if d['chem'] not in validate_chem:
                    print('Abnormal prediction:', d['chem'])
                    #validate_year = dirT_inT_dict[d['chem']][:4]
                else:
                    assert int(validate_chem[d['chem']]['time'][:4])>=int(dirT_inT_dict[d['chem']][:4]),print(d['chem'], validate_chem[d['chem']]['time'][:4], dirT_inT_dict[d['chem']][:4]) 
                    #validate_year = validate_chem[d['chem']]['time'][:4]
                validate_year = dirT_inT_dict[d['chem']][:4]
                T.append(int(validate_year)-int(current_year))
                if d['chem'] in drug_list:
                    T_Drug.append(int(validate_year)-int(current_year))
        summary['valid_year'].append(np.mean(T))
        summary['valid_year_Drug'].append(np.mean(T_Drug))
    DF = pd.DataFrame(summary)
    DF = organize_sum_df_onlyDrug(DF)
    DF.to_csv(output_fold+f'summary.csv', index=False)


    # calculate the direct relation and previous predicted ones along time
    direT_prevT = {}
    direT_prevT_drug = {}
    pool = set()
    previous_pred = set()
    for T in dirT_inT:
        T_formatted = T.replace('-','')
        if not os.path.isfile(input_fold+f'/cystic_fibrosis.11666.10000101_{T_formatted}.json'):
            continue
        direT = dirT_inT[T].keys()  
        direT = [x for x in direT if x not in pool] # new direct chem-dis relation in this T year that not reported previously
        direT_drug = [x for x in direT if x in set(drug_list)]
        pool = set(direT) | pool
        overlap = set(direT).intersection(previous_pred)
        overlap_drug = set(direT_drug).intersection(previous_pred)
        data_ = read_json(input_fold+f'/cystic_fibrosis.11666.10000101_{T_formatted}.json')
        disL = [x['chem'] for x in data_ ]
        previous_pred = set(disL)|previous_pred
        if T not in direT_prevT:
            direT_prevT[T] = {'direct_relation':0, 'previous_relation':0, 'direct_chem':[], 'previous_predChem':[]}
            direT_prevT[T]['direct_relation'] = len(direT)
            direT_prevT[T]['previous_relation'] = len(overlap)
            direT_prevT[T]['direct_disease'] = direT
            direT_prevT[T]['previous_predDis'] = list(overlap)
        if T not in direT_prevT_drug:
            direT_prevT_drug[T] = {'direct_relation':0, 'previous_relation':0, 'direct_drug':[], 'previous_predDrug':[]}
            direT_prevT_drug[T]['direct_relation'] = len(direT_drug)
            direT_prevT_drug[T]['previous_relation'] = len(overlap_drug)
            direT_prevT_drug[T]['direct_drug'] = direT_drug
            direT_prevT_drug[T]['previous_predDrug'] = list(overlap_drug)
    with open(output_fold+f'DirectChem_previousPred.json','w') as f:
        json.dump(direT_prevT, f, indent=2)
    with open(output_fold+f'DirectDrug_previousPred.json','w') as f:
        json.dump(direT_prevT_drug, f, indent=2)












