# This program takes in the raw full text data file and performs Sentence splitting, Word tokenization and Labeling steps
# Output files are ready for model training

import json
import re
import csv
import nltk
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_sci_sm")


train_fulltext = json.load(
    open("data_files/fulltext/litcoin_train_fulltext.json", "r", encoding="utf-8")
)
test_fulltext = json.load(
    open("data_files/fulltext/litcoin_test_fulltext.json", "r", encoding="utf-8")
)

full_train_pmid = {
    v["pmid"]: {"document_id": v["pmid"], "text": v["fulltext"]}
    for v in train_fulltext.values()
}
full_test_pmid = {
    v["pmid"]: {"document_id": v["pmid"], "text": v["fulltext"]}
    for v in test_fulltext.values()
}

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

full_train_pmid_sents = []
for k, v in tqdm(
    full_train_pmid.items(), desc="sentence splitting full text training data"
):
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
        full_train_pmid_sents.append(record)

full_test_pmid_sents = []
for k, v in tqdm(full_test_pmid.items(), desc="sentence splitting full text test data"):
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
        full_test_pmid_sents.append(record)

with open("temp/full_train_pmid_sents.txt", "w") as output:
    for item in full_train_pmid_sents:
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
with open("temp/full_test_pmid_sents.txt", "w") as output:
    for item in full_test_pmid_sents:
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
exec(open("sent_split_post_nltk_fulltext.py").read())

full_train_pmid_sents_refined = []
with open("temp/full_train_pmid_sents_refined.txt", "r") as f:
    data = f.readlines()[0:]
    for line in tqdm(data, desc="refining sentence splitting full text training data"):
        record = {}
        items = line.split("\t")
        doc_id = items[0]
        sent = items[1]
        sent_spans = [int(items[2]), int(items[3])]
        record[doc_id] = {"document_id": doc_id, "sent": sent, "sent_spans": sent_spans}
        full_train_pmid_sents_refined.append(record)

full_test_pmid_sents_refined = []
with open("temp/full_test_pmid_sents_refined.txt", "r") as f:
    data = f.readlines()[0:]
    for line in tqdm(data, desc="refining sentence splitting full text test data"):
        record = {}
        items = line.split("\t")
        doc_id = items[0]
        sent = items[1].strip()
        sent_spans = [int(items[2]), int(items[3])]
        record[doc_id] = {"document_id": doc_id, "sent": sent, "sent_spans": sent_spans}
        full_test_pmid_sents_refined.append(record)


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


full_train_pmid_tokens = []
i = 0
for item in tqdm(
    full_train_pmid_sents_refined, desc="word tokenization full text training data"
):
    key = list(item.keys())[0]
    docid = item[key]["document_id"]
    sent = item[key]["sent"]
    offset = item[key]["sent_spans"][0]
    tokens = get_tok_for_text(sent, offset)
    record = {"id": i, "document_id": docid, "tokens": tokens[0], "spans": tokens[1]}
    full_train_pmid_tokens.append(record)
    i += 1

full_test_pmid_tokens = []
j = 0
for item in tqdm(
    full_test_pmid_sents_refined, desc="word tokenization full text test data"
):
    key = list(item.keys())[0]
    docid = item[key]["document_id"]
    sent = item[key]["sent"]
    offset = item[key]["sent_spans"][0]
    tokens = get_tok_for_text(sent, offset)
    record = {"id": j, "document_id": docid, "tokens": tokens[0], "spans": tokens[1]}
    full_test_pmid_tokens.append(record)
    j += 1

json.dump(
    full_train_pmid_tokens,
    open("temp/full_train_pmid_tokens.json", "w", encoding="utf-8"),
)
json.dump(
    full_test_pmid_tokens,
    open("temp/full_test_pmid_tokens.json", "w", encoding="utf-8"),
)


### Labeling ###
full_train_bio = []
for record in tqdm(full_train_pmid_tokens, desc="labeling full text training data"):
    ner_tags = []
    ner_tags.extend([*(len(record["tokens"]) * ["O"])])
    record["ner_tags"] = ner_tags
    full_train_bio.append(record)

full_test_bio = []
for record in tqdm(full_test_pmid_tokens, desc="labeling full text test data"):
    ner_tags = []
    ner_tags.extend([*(len(record["tokens"]) * ["O"])])
    record["ner_tags"] = ner_tags
    full_test_bio.append(record)

with open("fulltext_train_bio.json", "w") as output:
    for item in full_train_bio:
        output.write(json.dumps(item) + "\n")
with open("fulltext_test_bio.json", "w") as output:
    for item in full_test_bio:
        output.write(json.dumps(item) + "\n")
