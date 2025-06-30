import json
import time
import numpy as np
import pandas as pd
import glob
import os
import math
from datetime import datetime, timedelta


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def previous_date(date_list):
    if "-" in date_list[0]:
        # Convert date strings to datetime objects and calculate one day before each date
        previous_dates = [
            datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
            for date_str in date_list
        ]
        # Convert datetime objects back to formatted date strings
        previous_date_strings = [
            date_obj.strftime("%Y-%m-%d") for date_obj in previous_dates
        ]
        return previous_date_strings
    elif "/" in date_list[0]:
        # Convert date strings to datetime objects and calculate one day before each date
        previous_dates = [
            datetime.strptime(date_str, "%Y/%m/%d") - timedelta(days=1)
            for date_str in date_list
        ]
        # Convert datetime objects back to formatted date strings
        previous_date_strings = [
            date_obj.strftime("%Y/%m/%d") for date_obj in previous_dates
        ]
        return previous_date_strings
    else:  # "YYYYMMDD"
        # Convert date strings to datetime objects and calculate one day before each date
        previous_dates = [
            datetime.strptime(date_str, "%Y%m%d") - timedelta(days=1)
            for date_str in date_list
        ]
        # Convert datetime objects back to formatted date strings
        previous_date_strings = [
            date_obj.strftime("%Y%m%d") for date_obj in previous_dates
        ]
        return previous_date_strings


def date_range(start_date, end_date):
    date_list = []
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    return date_list


def cleanDrug(data, p, removeNA=0):
    data_t = {}
    genes = set()
    data = [x for x in data if x["Drug"] == 1]
    if removeNA == 1:
        data = [x for x in data if x["ChemDBid"] != "NA"]
    for x in data:
        genes = set(x["genes"]) | genes
    Ngene = len(genes)
    for k in data[0].keys():
        if k in ["genes", "probs", "score", "cor_types"]:
            continue
        data_t[k] = [x[k] for x in data]
    df = pd.DataFrame(data_t)
    df = df.sort_values(by=["finalProb", "finalScore", "pathCnt"], ascending=False)
    df.reset_index(drop=True, inplace=True)
    L = int(len(df) * p)
    df = df.iloc[:L]
    return Ngene, df


def add_chemID(df, ChemDict):
    ChemDBid = []
    ChemCT = []
    for i, x in df.iterrows():
        biokdeid = x["chem"]
        ChemDBid.append(ChemDict[biokdeid]["id"])
        ChemCT.append(max(ChemDict[biokdeid]["synonym_and_cnt"].values()))
    df["ChemDBid"] = ChemDBid
    df["ChemCT"] = ChemCT
    return df


def get_earlist_date(RE_list, p=0):
    RE_list = [x for x in RE_list if x[1] >= p]
    RE_list_sort = sorted(RE_list, key=lambda x: datetime.strptime(x[3], "%Y-%m-%d"))
    return RE_list_sort[0][-1]


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
            cumProb = cumProb * (1 - prob)
            sumscore += -math.log(1 - prob + 0.01)
        finalProb = round(1 - cumProb, 4)
        if finalProb >= MIN_PROB:
            return finalProb, sumscore
        else:
            return 0, 0
    else:
        return 0, 0


def reverse_corType(relID):
    # "2344044.175086.0.12"
    n1, n2, corr, dire = relID.split(".")
    if corr == "0":
        return ".".join([n1, n2, "2", dire])
    elif corr == "2":
        return ".".join([n1, n2, "0", dire])


def fine_ana(df_ss, CT_info):
    N_totDrug, N_drugAlltime, Drug_inClinicTrial, Verified_allSource = 0, 0, 0, 0
    ClinicTrial = []
    verifiedCT = []
    df_ss = df_ss.sort_values(
        by=["directRelAll", "finalProb", "finalScore"], ascending=False
    )
    df_ss.reset_index(drop=True, inplace=True)
    for i, row in df_ss.iterrows():
        if row["chem"] in CT_info.keys():
            ClinicTrial.append(1)
            if not set(CT_info[row["chem"]]["CT_id"]).intersection(set(verifiedCT)):
                if row["directRel"] != 1:
                    N_totDrug += 1
                    Verified_allSource += 1
                    Drug_inClinicTrial += 1
                    if row["directRelAll"] == 1:
                        N_drugAlltime += 1  # 优先使用clinical trial 验证
            verifiedCT.extend(CT_info[row["chem"]]["CT_id"])
        else:
            ClinicTrial.append(0)
            if row["directRel"] != 1:
                N_totDrug += 1
                if row["directRelAll"] == 1:
                    N_drugAlltime += 1
                    Verified_allSource += 1
    df_ss["ClinicTrial"] = ClinicTrial
    df_ss = df_ss.sort_values(
        by=["finalProb", "finalScore", "pathCnt"], ascending=False
    )
    df_ss.reset_index(drop=True, inplace=True)
    df_ss["Verified_allSource"] = [
        (
            1
            if df_ss.ClinicTrial.tolist()[i] == 1 or df_ss.directRelAll.tolist()[i] == 1
            else 0
        )
        for i in range(len(df_ss))
    ]
    return N_totDrug, N_drugAlltime, Drug_inClinicTrial, Verified_allSource, df_ss


def initial_directCD(verify_start_date):  # time1 = '20200401'
    end_date = "20230701"
    date_list = date_range(verify_start_date, end_date)
    directCD = {
        "Time": date_list,
        "verified_repDrug": [0 for i in range(len(date_list))],
        "verified_byPaper": [0 for i in range(len(date_list))],
        "verified_byCT": [0 for i in range(len(date_list))],
    }
    return directCD


def Drug_Verify(df, CT_info, verify_start_date, MIN_PROB):
    # df_ss.column = [id,chem,disease,type,directProb,directProbAll,finalScore,finalProb,Drug,chem_name,disease_name,directRel,directRelAll,pathCnt,ChemDBid,ChemCount]
    # verify_start_date = "YYYYMMDD"
    verify_start_date_obj = datetime.strptime(verify_start_date, "%Y%m%d")
    df_sub = df.loc[
        (df["Drug"] == 1) & (df["directRel"] == 0)
    ]  # get the drugs that doesn't have direct relation before repurpose
    # remove the drugs that verified by clinical trial before the verify_start_date
    remove_byCT = []
    for i, row in df_sub.iterrows():
        if row["chem"] in CT_info:
            if (
                datetime.strptime(CT_info[row["chem"]]["time"].split()[0], "%Y-%m-%d")
                < verify_start_date_obj
            ):
                remove_byCT.append(i)
    df_sub_d = df_sub.drop(remove_byCT)
    df_sub = df_sub_d.copy()
    df_sub = df_sub.sort_values(
        by=["directRelAll", "finalProb", "finalScore"], ascending=False
    )
    df_sub.reset_index(drop=True, inplace=True)
    df_sub["ClinicTrial"] = [
        1 if df_sub.chem.tolist()[i] in CT_info else 0 for i in range(len(df_sub))
    ]
    df_sub["Verified_allSource"] = [
        (
            1
            if df_sub["directRelAll"].tolist()[i] == 1
            or df_sub["ClinicTrial"].tolist()[i] == 1
            else 0
        )
        for i in range(len(df_sub))
    ]
    df_sub = df_sub.sort_values(by=["ClinicTrial", "directRelAll"], ascending=False)
    df_sub.reset_index(drop=True, inplace=True)
    # Now df_sub contains the repurposed drug after verify_start_date, and Verified_allSource based on the information by 20230701 PubMed & Clinical
    directCD = initial_directCD(verify_start_date)
    verifiedCT = []
    remove_index = []
    verified_list = []
    df_output = df_sub.copy()
    for Ti in range(1, len(directCD["Time"])):
        time0 = "2020-01-01"  # the time0 is used for PubMed relation extraction
        time1 = directCD["Time"][Ti - 1]  # the time1 is used for Clinical verification
        time2 = directCD["Time"][
            Ti
        ]  # the time2 is the end date for PubMed and Clinical verification
        for ind, row in df_sub.iterrows():
            if row["Verified_allSource"] != 1:
                # skip the cases that not be verified in 20230701. Some cases can be verified before that but be denied after then, these case
                # should be skipped
                continue
            flag = 0
            relID = row["id"] + ".12"
            relID_op = reverse_corType(relID)
            drugID = relID.split(".")[0]
            if drugID in verified_list:
                continue
            # check in CT
            if drugID in CT_info:
                if set(CT_info[drugID]["CT_id"]).intersection(set(verifiedCT)):
                    # this drug has been verified, should not be count
                    verifiedCT.extend(CT_info[drugID]["CT_id"])
                    remove_index.append(ind)
                    verified_list.append(drugID)
                    continue
                else:
                    time1_t = datetime.strptime(time1, "%Y-%m-%d")
                    time2_t = datetime.strptime(time2, "%Y-%m-%d")
                    timeCT_t = datetime.strptime(
                        CT_info[drugID]["time"].split()[0], "%Y-%m-%d"
                    )
                    if (
                        timeCT_t < verify_start_date_obj
                    ):  # the drug verified by clinical trial before the verify start date should not be count
                        remove_index.append(ind)
                        verified_list.append(drugID)
                        continue
                    if timeCT_t >= time1_t and timeCT_t < time2_t:
                        # this drug can be verified by CT
                        flag = 1
                        verifiedCT.extend(CT_info[drugID]["CT_id"])
                        directCD["verified_repDrug"][Ti] += 1
                        directCD["verified_byCT"][Ti] += 1
                        verified_list.append(drugID)
            if flag == 0:
                if relID in RELlist:
                    rels = get_rel_within_time(RELlist[relID], time0, time2)
                    finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB)
                    if finalProb != 0:
                        if relID_op in RELlist:
                            rels_op = get_rel_within_time(
                                RELlist[relID_op], time0, time2
                            )
                            finalProb_op, sumscore_op = get_prob_fromArtList(
                                rels_op, MIN_PROB
                            )
                        else:
                            finalProb_op = 0
                            sumscore_op = 0
                        if finalProb - finalProb_op > 0.1:
                            directCD["verified_byPaper"][Ti] += 1
                            directCD["verified_repDrug"][Ti] += 1
                            verified_list.append(drugID)
                            if drugID in CT_info:
                                verifiedCT.extend(CT_info[drugID]["CT_id"])
    df_output_drop = df_output.drop(remove_index)
    return directCD, df_output_drop


from tqdm import tqdm


if __name__ == "__main__":
    output_fold = "analysis_results/"
    if not os.path.isdir(output_fold):
        os.mkdir(output_fold)

    # define some parameters
    COR_TYPE = -1
    MIN_PROB_direct = 0.5

    # read covid pool
    with open("../data/covid_drug_with_validID_underClinicTrial.json", "r") as f:
        # dictionary of clinical trials for COVID-19
        CT_info = json.load(f)

    # read ChemEntity
    with open("../data/ChemEntities.json", "r") as f:
        Chem = json.load(f)
    ChemDict = {}
    for C in Chem:
        ChemDict[C["biokdeid"]] = C

    # read Relation
    RELlist_time = "../data/Aggregated_PubMedRelList.json"
    print("Loading aggregated relation list with time.", flush=True)
    time0 = time.time()
    with open(RELlist_time, "r") as f:
        RELlist = json.load(f)
    print(
        f"Done loading the aggregated relation list with time. Using {time.time()-time0} s.",
        flush=True,
    )

    files = glob.glob("results/*.json")
    sorted_files = sorted(
        files, key=lambda x: datetime.strptime(x.split(".")[1].split("_")[1], "%Y%m%d")
    )
    # p is percentage of results?
    for p in [1.00]:
        summary = {
            "time": [],
            "N_totDrug": [],
            "N_inClinicTrial": [],
            "N_inPubMed": [],
            "Drug_verified": [],
            "N_gene": [],
        }
        for F in tqdm(sorted_files[:6], desc="iterating over result files"):
            data = read_json(F)
            if data == []:
                continue
            for x in data:
                x["ChemDBid"] = ChemDict[x["chem"]]["id"]
                # x["ChemCount"] = max(ChemDict[x["chem"]]["synonym_and_cnt"].values())
            Ngene, df_ss = cleanDrug(data, p, removeNA=1)
            verify_start_date = F.split("/")[-1].split("_")[1].split(".")[0]
            directCD, df_ss = Drug_Verify(
                df_ss, CT_info, verify_start_date, MIN_PROB_direct
            )
            summary["time"].append(verify_start_date)

            summary["N_totDrug"].append(len(df_ss))
            summary["N_inClinicTrial"].append(df_ss.ClinicTrial.sum())
            summary["N_inPubMed"].append(df_ss.directRelAll.sum())
            summary["Drug_verified"].append(df_ss.Verified_allSource.sum())
            summary["N_gene"].append(Ngene)
            if df_ss.Verified_allSource.sum() != sum(directCD["verified_repDrug"]):
                print(
                    F, df_ss.Verified_allSource.sum(), sum(directCD["verified_repDrug"])
                )
            if p == 1:
                df_ss.to_csv(output_fold + verify_start_date + ".csv", index=False)
                directCD_df = pd.DataFrame(directCD)
                directCD_df.to_csv(
                    output_fold + verify_start_date + "_direct.csv", index=False
                )
        DF = pd.DataFrame(summary)
        DF.to_csv(output_fold + f"summary_{p}.csv", index=False)
