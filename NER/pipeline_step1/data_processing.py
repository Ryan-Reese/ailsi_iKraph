# This program takes in the raw text .csv data file and performs Sentence splitting, Word tokenization and Labeling steps
# Output files are ready for model training
"""
Created on Thu Mar  3 16:08:16 2022

@author: qinghan
"""
import json
import re
import csv
import nltk
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_sci_sm")


abstr_train_pmid = {}
with open("data_files/LitCoin/abstracts_train.csv", "r") as csvfile:
    data = csv.reader(csvfile, delimiter="\t")
    header = next(data)
    for line in tqdm(data, desc="parsing train csv"):
        doc_id = line[0]
        title = line[1]
        abstr = line[2]
        text = title + " " + abstr
        abstr_train_pmid[doc_id] = {"document_id": doc_id, "text": text}

abstr_test_pmid = {}
with open("data_files/LitCoin/abstracts_test.csv", "r") as csvfile:
    data = csv.reader(csvfile, delimiter="\t")
    header = next(data)
    for line in tqdm(data, desc="parsing test csv"):
        doc_id = line[0]
        title = line[1]
        abstr = line[2]
        text = title + " " + abstr
        abstr_test_pmid[doc_id] = {"document_id": doc_id, "text": text}

# json.dump(abstr_train_pmid, open("abstr_train_pmid.json", "w", encoding="utf-8"))
# json.dump(abstr_train_pmid, open("abstr_test_pmid.json", "w", encoding="utf-8"))


### Sentence-splitting by NLTK ###
extra_abbreviations = [
    "e.g",
    "i.e",
    "i.m",
    "a.u",
    "p.o",
    "i.v",
    "i.p",
    "vivo",
    "p.o",
    "i.p",
    "Vmax",
    "i.c.v",
    ")(",
    "E.C",
    "sp",
    "al",
]
sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)

abstr_train_pmid_sents = []
for k, v in tqdm(abstr_train_pmid.items(), desc="tokenising train set"):
    doc_id = v["document_id"]
    text = v["text"]
    sents = sentence_tokenizer.tokenize(text)
    for idx, sent in enumerate(sents):
        record = {}
        record[str(doc_id) + "_" + str(idx)] = {
            "document_id": doc_id,
            "id": idx,
            "sent": sent,
        }
        if idx == 0:
            start = text.find(sent)
        else:
            start = 1 + text[1:].find(sent)
        end = start + len(sent)
        record[str(doc_id) + "_" + str(idx)]["sent_spans"] = [start, end]
        abstr_train_pmid_sents.append(record)

abstr_test_pmid_sents = []
for k, v in tqdm(abstr_test_pmid.items(), desc="tokenising test set"):
    doc_id = v["document_id"]
    text = v["text"]
    sents = sentence_tokenizer.tokenize(text)
    for idx, sent in enumerate(sents):
        record = {}
        record[str(doc_id) + "_" + str(idx)] = {
            "document_id": doc_id,
            "id": idx,
            "sent": sent,
        }
        if idx == 0:
            start = text.find(sent)
        else:
            start = 1 + text[1:].find(sent)
        end = start + len(sent)
        record[str(doc_id) + "_" + str(idx)]["sent_spans"] = [start, end]
        abstr_test_pmid_sents.append(record)

with open("temp/abstr_train_pmid_sents.txt", "w") as output:
    for item in tqdm(abstr_train_pmid_sents, desc="writing train tokenisation"):
        for k, v in item.items():
            output.write(
                v["document_id"]
                + "\t"
                + v["sent"]
                + "\t"
                + str(v["sent_spans"][0])
                + "\t"
                + str(v["sent_spans"][1])
                + "\n"
            )
with open("temp/abstr_test_pmid_sents.txt", "w") as output:
    for item in tqdm(abstr_test_pmid_sents, desc="writing test tokenisation"):
        for k, v in item.items():
            output.write(
                v["document_id"]
                + "\t"
                + v["sent"]
                + "\t"
                + str(v["sent_spans"][0])
                + "\t"
                + str(v["sent_spans"][1])
                + "\n"
            )

### Refine sentence splitting ###
exec(open("sent_split_post_nltk.py").read())

abstr_train_pmid_sents_refined = []
with open("temp/abstr_train_pmid_sents_refined.txt", "r") as f:
    data = f.readlines()[0:]
    for line in tqdm(data, desc="refining sentence splitting for train set"):
        record = {}
        items = line.split("\t")
        doc_id = items[0]
        sent = items[1]
        sent_spans = [int(items[2]), int(items[3])]
        record[doc_id] = {"document_id": doc_id, "sent": sent, "sent_spans": sent_spans}
        abstr_train_pmid_sents_refined.append(record)

abstr_test_pmid_sents_refined = []
with open("temp/abstr_test_pmid_sents_refined.txt", "r") as f:
    data = f.readlines()[0:]
    for line in tqdm(data, desc="refining sentence splitting for test set"):
        record = {}
        items = line.split("\t")
        doc_id = items[0]
        sent = items[1].strip()
        sent_spans = [int(items[2]), int(items[3])]
        record[doc_id] = {"document_id": doc_id, "sent": sent, "sent_spans": sent_spans}
        abstr_test_pmid_sents_refined.append(record)


### Word tokenization ###
def split_punct(text: str, start: int):
    for m in re.finditer(r"""[\w']+|[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", text):
        yield m.group(), m.start() + start, m.end() + start


def get_tok_for_text(text, offset):
    doc = nlp(text)
    token_list = []
    span_list = []
    tokens = []
    spans = []
    for token in doc:
        for tk, start, end in split_punct(token.text, token.idx):
            tokens.append(tk)
            span = [start + offset, end + offset]
            spans.append(span)
    token_list.append(tokens)
    span_list.append(spans)
    return tokens, spans


abstr_train_pmid_tokens = []
i = 0
for item in tqdm(abstr_train_pmid_sents_refined, desc="get tokens for train set"):
    key = list(item.keys())[0]
    docid = item[key]["document_id"]
    sent = item[key]["sent"]
    offset = item[key]["sent_spans"][0]
    tokens = get_tok_for_text(sent, offset)
    record = {"id": i, "document_id": docid, "tokens": tokens[0], "spans": tokens[1]}
    abstr_train_pmid_tokens.append(record)
    i += 1

abstr_test_pmid_tokens = []
j = 0
for item in tqdm(abstr_test_pmid_sents_refined, desc="get tokens for test set"):
    key = list(item.keys())[0]
    docid = item[key]["document_id"]
    sent = item[key]["sent"]
    offset = item[key]["sent_spans"][0]
    tokens = get_tok_for_text(sent, offset)
    record = {"id": j, "document_id": docid, "tokens": tokens[0], "spans": tokens[1]}
    abstr_test_pmid_tokens.append(record)
    j += 1

json.dump(
    abstr_train_pmid_tokens,
    open("temp/abstr_train_pmid_tokens.json", "w", encoding="utf-8"),
)
json.dump(
    abstr_test_pmid_tokens,
    open("temp/abstr_test_pmid_tokens.json", "w", encoding="utf-8"),
)


### Labeling ###
exec(open("NER_labeling.py").read())

# Change the output file to JSON
abstr_train_bio = []
with open("temp/abstr_train_bio.txt", "r") as f:
    data = f.readlines()[0:]
    for line in data:
        abstr_train_bio.append(json.loads(line))
with open("litcoin_train_update_400.json", "w") as output:
    for item in abstr_train_bio:
        output.write(json.dumps(item) + "\n")

# Labeling the test dataset
abstr_test_bio = []
for record in abstr_test_pmid_tokens:
    ner_tags = []
    ner_tags.extend([*(len(record["tokens"]) * ["O"])])
    record["ner_tags"] = ner_tags
    abstr_test_bio.append(record)

with open("litcoin_test_update_100.json", "w") as output:
    for item in abstr_test_bio:
        output.write(json.dumps(item) + "\n")
