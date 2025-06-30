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
    # returns list of relations that fall within time interval
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


def reverse_corType(cdTypeStr):
    if cdTypeStr == "0":
        return "2"
    elif cdTypeStr == "2":
        return "0"


from tqdm import tqdm

if __name__ == "__main__":

    input_list = "repurpose_ENT_v3.txt"
    output_path = "results/"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    EntDictFile = "../data/NER_ID_dict_cap_final.json"
    RELlist_time = "../data/Aggregated_PubMedRelList.json"
    # define some parameters
    COR_TYPE = -1
    MIN_PROB = 0.8
    MIN_PROB_direct = 0.5

    # read chemical/disease entity dictionary
    # read entity dictionary
    print("Load entities.", flush=True)
    time0 = time.time()
    with open(EntDictFile, "r") as f:
        Edata = json.load(f)
    entDic = {}
    entSubtype = {}
    entName = {}
    for item in tqdm(Edata, desc="reading entity metadata"):
        bkid = item["biokdeid"]
        entType = item["type"]
        subType = item["subtype"]
        if "common name" in item:
            commonName = item["common name"]
        else:
            commonName = item["official name"]
        if bkid not in entDic:
            # adds to dictionaries if not already present
            entDic[bkid] = entType
            entSubtype[bkid] = subType
            entName[bkid] = commonName
        else:
            print("Replicate biokdeid:", item["biokdeid"])
    print(f"Finished reading entities. Using {time.time()-time0} s", flush=True)

    print("Reading the entities for repurposing.")
    target_list = (
        []
    )  # [[entID, ent_type, cutoff_time1, cutoff_time2], ...] ent_type='C' or 'D', cutoff_time='YYYY-MM-DD'
    target_entName = (
        {}
    )  # {entID:entity_name} entity_name='common name' or 'official name'
    if input_list != "":
        with open(input_list, "r") as f:
            Entity_list = [
                x.split() for x in f if not x.startswith("#")
            ]  # [['Alzheimer's Disease', 'D', 'AD', '3651', '1990-01-01', '2023-08-21'], ...]
        for ent in tqdm(Entity_list, desc="reading input list"):
            temp = [
                ent[3],
                ent[1],
                ent[-2],
                ent[-1],
            ]  # type (D or C), time_cut1, time_cut2
            if ent[1] == "C" and entDic[ent[3]] != "Chemical":
                print(
                    "Chemical ID provided is not in the database:", ent[3], flush=True
                )
            elif ent[1] == "D" and entDic[ent[3]] != "Disease":
                print("Disease ID provided is not in the database:", ent[3], flush=True)
            else:
                target_list.append(temp)
                target_entName[ent[3]] = entName[ent[3]]
    else:
        print("Error! Please give the file containing the target entity list")
    print("Reading entities done.")

    print("Loading aggregated relation list with time. ", flush=True)
    time0 = time.time()
    with open(RELlist_time, "r") as f:
        RELlist = json.load(f)
    print(
        f"Done loading the aggregated relation list with time. Using {time.time()-time0} s.",
        flush=True,
    )

    # the aggregated relation list already remove the cases that corType is not right, And only keep the MP result for time purpose
    # first extract chem-disease collection without time cutoff. Use this collection to tell if the repurposed drug used to treat the disease eventually
    ChemDisDicAll = {}
    # ^ dictionary of all chemical-disease relations
    for rel in tqdm(RELlist, desc="iterating through relations"):
        n1, n2, corType, direction = rel.split(".")
        if n1 not in entDic or n2 not in entDic:
            continue
        if entDic[n1] == "Chemical" and entDic[n2] == "Disease" and direction == "12":
            rels = get_rel_within_time(
                RELlist[rel], "2020-01-01", "2100-01-01"
            )  # [[pmid, prob, score, time_object],...], early time cutoff specially for COVID-19
            finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB_direct)
            if finalProb == 0:
                continue
            key = n1 + "." + n2 + "." + corType
            if key not in ChemDisDicAll:
                ChemDisDicAll[key] = [finalProb, corType, sumscore]
            elif finalProb > ChemDisDicAll[key][0]:
                ChemDisDicAll[key] = [finalProb, corType, sumscore]

    # begin the repurposing for each target
    # [[entID, ent_type, cutoff_time1, cutoff_time2], ...] ent_type='C' or 'D', cutoff_time='YYYY-MM-DD'
    for target in target_list:
        targetID = target[0]
        targetType = target[1]
        time1 = target[2]
        time2 = target[3]
        print(
            f"Drug repurposing:{targetType} {target_entName[targetID]} ({targetID}), Time: {time1}-{time2}",
            flush=True,
        )

        # get Chemical-Gene, Disease-Gene dict for repurposing; Get Chem-Disease dict for direct relation
        ChemGeneDic = {}
        GeneDisDic = {}
        ChemDisDic = {}

        for rel in tqdm(RELlist, desc="iterating through relations"):
            # relation id rule: cell line - chemical - disease - gene - mutations - species
            n1, n2, corType, direction = rel.split(".")
            if n1 not in entDic or n2 not in entDic:
                continue
            if entDic[n1] == "Chemical" and entDic[n2] == "Gene" and direction == "12":
                # calculate probability
                if (
                    targetType == "D"
                ):  # target entity is disease, so there is no cut_time1 for chemical-gene, set cut_time1 as a early date
                    rels = get_rel_within_time(
                        RELlist[rel], "1000-1-1", time2
                    )  # [[pmid, prob, score, time_object],...]
                else:  # target entity is chemical, use the time cutoff from input
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
            if (
                entDic[n1] == "Chemical"
                and entDic[n2] == "Disease"
                and direction == "12"
            ):
                # calculate probability
                rels = get_rel_within_time(
                    RELlist[rel], time1, time2
                )  # [[pmid, prob, score, time_object],...]
                finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB_direct)
                if finalProb == 0:
                    continue
                key = n1 + "." + n2 + "." + corType
                if key not in ChemDisDic:
                    ChemDisDic[key] = [finalProb, corType, sumscore]
                elif finalProb > ChemDisDic[key][0]:
                    ChemDisDic[key] = [finalProb, corType, sumscore]

            if entDic[n1] == "Disease" and entDic[n2] == "Gene" and direction == "21":
                # calculate probability
                if (
                    targetType == "C"
                ):  # target entity is chemical, so there is no cut_time1 for disease-gene, set cut_time1 as a early date
                    rels = get_rel_within_time(
                        RELlist[rel], "1000-1-1", time2
                    )  # [[pmid, prob, score, time_object],...]
                else:  # target entity is disease, use the time cutoff from input
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
        if targetType == "D":  # process for drug repurposing for disease
            print("Searching for relations to disease")
            relDic = {}
            for chem in tqdm(ChemGeneDic, desc="Iterating over chemicals"):
                for gene in ChemGeneDic[chem]:
                    prob, corType, sumscore = ChemGeneDic[chem][gene]
                    if gene in GeneDisDic:
                        if targetID in GeneDisDic[gene]:
                            prob2, corType2, sumscore2 = GeneDisDic[gene][targetID]
                            finalProb = prob * prob2
                            finalScore = sumscore + sumscore2
                            if finalProb < MIN_PROB:
                                continue
                            cdType = (int(corType) - 1) * (int(corType2) - 1)
                            if cdType != COR_TYPE:
                                continue
                            cdTypeStr = str(cdType + 1)
                            key = chem + "." + targetID + "." + cdTypeStr
                            key_op = (
                                chem + "." + targetID + "." + reverse_corType(cdTypeStr)
                            )
                            if key not in relDic:
                                tmpDic = {}
                                tmpDic["id"] = key
                                tmpDic["chem"] = chem
                                tmpDic["disease"] = targetID
                                tmpDic["genes"] = [gene]
                                tmpDic["type"] = cdType
                                tmpDic["probs"] = [finalProb]
                                tmpDic["score"] = [finalScore]
                                tmpDic["cor_types"] = [
                                    [int(corType) - 1, int(corType2) - 1]
                                ]
                                if key in ChemDisDic:
                                    dirProb_inT = ChemDisDic[key][0]
                                    if key_op in ChemDisDic:
                                        dirProb_op_inT = ChemDisDic[key_op][0]
                                    else:
                                        dirProb_op_inT = 0
                                    if dirProb_inT - dirProb_op_inT > 0.1:
                                        tmpDic["directProb"] = ChemDisDic[key][0]
                                    else:
                                        tmpDic["directProb"] = 0
                                else:
                                    tmpDic["directProb"] = 0
                                if key in ChemDisDicAll:
                                    dirProb_all = ChemDisDicAll[key][0]
                                    if key_op in ChemDisDicAll:
                                        dirProb_op_all = ChemDisDicAll[key_op][0]
                                    else:
                                        dirProb_op_all = 0
                                    if dirProb_all - dirProb_op_all > 0.1:
                                        tmpDic["directProbAll"] = ChemDisDicAll[key][0]
                                    else:
                                        tmpDic["directProbAll"] = 0
                                else:
                                    tmpDic["directProbAll"] = 0
                                relDic[key] = tmpDic
                            else:
                                relDic[key]["genes"].append(gene)
                                relDic[key]["probs"].append(finalProb)
                                relDic[key]["score"].append(finalScore)
                                relDic[key]["cor_types"].append(
                                    [int(corType) - 1, int(corType2) - 1]
                                )
            for K in relDic:
                sumScore = 0
                cumProb = 1
                for prob in relDic[K]["probs"]:
                    # adding 0.01 is equavalent to subtract 0.01 from all probabilities, which softens this variable and helps ranking
                    cumProb = cumProb * (1 - prob)
                for score in relDic[K]["score"]:
                    sumScore += score
                finalScore = sumScore
                finalProb = 1 - cumProb
                relDic[K]["finalScore"] = finalScore
                relDic[K]["finalProb"] = finalProb
                if entSubtype[relDic[K]["chem"]] == "Drug":
                    relDic[K]["Drug"] = 1
                else:
                    relDic[K]["Drug"] = 0
                relDic[K]["chem_name"] = entName[relDic[K]["chem"]]
                relDic[K]["disease_name"] = entName[relDic[K]["disease"]]
                direct_prob = relDic[K]["directProb"]
                if direct_prob < MIN_PROB_direct:
                    relDic[K]["directRel"] = 0
                else:
                    relDic[K]["directRel"] = 1
                direct_prob_all = relDic[K]["directProbAll"]
                if direct_prob_all < MIN_PROB_direct:
                    relDic[K]["directRelAll"] = 0
                else:
                    relDic[K]["directRelAll"] = 1
                relDic[K]["pathCnt"] = len(relDic[K]["genes"])

            outputfile = (
                output_path
                + targetID
                + "."
                + time1.replace("-", "")
                + "_"
                + time2.replace("-", "")
                + ".json"
            )
            with open(outputfile, "w") as f:
                json.dump(list(relDic.values()), f, indent=2)

        if targetType == "C":  # process for drug repurposing for chemical
            print("Searching for relations to chemical")
            relDic = {}
            chem = targetID
            for gene in ChemGeneDic[chem]:
                prob, corType, sumscore = ChemGeneDic[chem][gene]
                if gene in GeneDisDic:
                    for dis in GeneDisDic[gene]:
                        prob2, corType2, sumscore2 = GeneDisDic[gene][dis]
                        finalProb = prob * prob2
                        finalScore = sumscore + sumscore2
                        if finalProb < MIN_PROB:
                            continue
                        cdType = (int(corType) - 1) * (int(corType2) - 1)
                        if cdType != COR_TYPE:
                            continue
                        cdTypeStr = str(cdType + 1)
                        key = chem + "." + dis + "." + cdTypeStr
                        key_op = chem + "." + dis + "." + reverse_corType(cdTypeStr)
                        if key not in relDic:
                            tmpDic = {}
                            tmpDic["id"] = key
                            tmpDic["chem"] = chem
                            tmpDic["disease"] = dis
                            tmpDic["genes"] = [gene]
                            tmpDic["type"] = cdType
                            tmpDic["probs"] = [finalProb]
                            tmpDic["score"] = [finalScore]
                            tmpDic["cor_types"] = [
                                [int(corType) - 1, int(corType2) - 1]
                            ]
                            if key in ChemDisDic:
                                dirProb_inT = ChemDisDic[key][0]
                                if key_op in ChemDisDic:
                                    dirProb_op_inT = ChemDisDic[key_op][0]
                                else:
                                    dirProb_op_inT = 0
                                if dirProb_inT - dirProb_op_inT > 0.1:
                                    tmpDic["directProb"] = ChemDisDic[key][0]
                                else:
                                    tmpDic["directProb"] = 0
                            else:
                                tmpDic["directProb"] = 0
                            if key in ChemDisDicAll:
                                dirProb_all = ChemDisDicAll[key][0]
                                if key_op in ChemDisDicAll:
                                    dirProb_op_all = ChemDisDicAll[key_op][0]
                                else:
                                    dirProb_op_all = 0
                                if dirProb_all - dirProb_op_all > 0.1:
                                    tmpDic["directProbAll"] = ChemDisDicAll[key][0]
                                else:
                                    tmpDic["directProbAll"] = 0
                            else:
                                tmpDic["directProbAll"] = 0
                            relDic[key] = tmpDic
                        else:
                            relDic[key]["genes"].append(gene)
                            relDic[key]["probs"].append(finalProb)
                            relDic[key]["score"].append(finalScore)
                            relDic[key]["cor_types"].append(
                                [int(corType) - 1, int(corType2) - 1]
                            )
            for K in relDic:
                sumScore = 0
                cumProb = 1
                for prob in relDic[K]["probs"]:
                    # adding 0.01 is equavalent to subtract 0.01 from all probabilities, which softens this variable and helps ranking
                    cumProb = cumProb * (1 - prob)
                for score in relDic[K]["score"]:
                    sumScore += score
                finalScore = sumScore
                finalProb = 1 - cumProb
                relDic[K]["finalScore"] = finalScore
                relDic[K]["finalProb"] = finalProb
                if entSubtype[relDic[K]["chem"]] == "Drug":
                    relDic[K]["Drug"] = 1
                else:
                    relDic[K]["Drug"] = 0
                relDic[K]["chem_name"] = entName[relDic[K]["chem"]]
                relDic[K]["disease_name"] = entName[relDic[K]["disease"]]
                direct_prob = relDic[K]["directProb"]
                if direct_prob < MIN_PROB_direct:
                    relDic[K]["directRel"] = 0
                else:
                    relDic[K]["directRel"] = 1
                direct_prob_all = relDic[K]["directProbAll"]
                if direct_prob_all < MIN_PROB_direct:
                    relDic[K]["directRelAll"] = 0
                else:
                    relDic[K]["directRelAll"] = 1
                relDic[K]["pathCnt"] = len(relDic[K]["genes"])
            outputfile = (
                output_path
                + targetID
                + "."
                + time1.replace("-", "")
                + "_"
                + time2.replace("-", "")
                + ".json"
            )
            with open(outputfile, "w") as f:
                json.dump(list(relDic.values()), outputfile, indent=2)

    # analysis the direct chem-Covid relation along the time line
    with open(
        output_path
        + target_list[-1][0]
        + "."
        + target_list[-1][2].replace("-", "")
        + "_"
        + target_list[-1][3].replace("-", "")
        + ".json",
        "r",
    ) as f:
        candidate = json.load(f)
    cand_chem = set([x["chem"] for x in candidate])
    directCD = {"Time": [], "allCD": [], "allDrugD": [], "repCD": [], "repDrugD": []}

    # read ChemEntity
    with open("../data/ChemEntities.json", "r") as f:
        Chem = json.load(f)
    ChemDict = {}
    for C in Chem:
        ChemDict[C["biokdeid"]] = C

    for target in target_list:
        targetID = target[0]
        targetType = target[1]
        time1 = target[2]
        time2 = target[3]
        directCD["Time"].append(time2)
        for k in directCD:
            if k != "Time":
                directCD[k].append(0)
        for rel in RELlist:
            # relation id rule: cell line - chemical - disease - gene - mutations - species
            n1, n2, corType, direction = rel.split(".")
            if (
                n2 == targetID
                and entDic[n1] == "Chemical"
                and direction == "12"
                and corType == "0"
            ):
                if ChemDict[n1]["id"] == "NA":
                    continue  # do not count the chemical compond without a valid database ID
                rels = get_rel_within_time(
                    RELlist[rel], time1, time2
                )  # [[pmid, prob, score, time_object],...]
                finalProb, sumscore = get_prob_fromArtList(rels, MIN_PROB_direct)
                if finalProb == 0:
                    continue
                rel_op = ".".join([n1, n2, reverse_corType(corType), direction])
                if rel_op in RELlist:
                    rels_op = get_rel_within_time(RELlist[rel_op], time1, time2)
                    finalProb_op, sumscore_op = get_prob_fromArtList(
                        rels_op, MIN_PROB_direct
                    )
                else:
                    finalProb_op = 0
                    sumscore_op = 0
                if finalProb - finalProb_op > 0.1:
                    directCD["allCD"][-1] += 1
                    if n1 in cand_chem:
                        directCD["repCD"][-1] += 1
                    if entSubtype[n1] == "Drug":
                        directCD["allDrugD"][-1] += 1
                        if n1 in cand_chem:
                            directCD["repDrugD"][-1] += 1
    df = pd.DataFrame(directCD)
    outputfile = output_path + f"dirRel_CD.csv"
    df.to_csv(outputfile, sep="\t", index=False)
    print("Job finished")
