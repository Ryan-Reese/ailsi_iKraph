# to generate training json data for xin's code
import os
import json
import argparse


def load_mentions(entity_fp):
    mentions = {}
    fl = True
    with open(entity_fp) as fent:
        for line in fent:
            if fl:
                fl = False
                continue
            tmp = line[0:-1].split('\t')
            id0 = tmp[0]
            doc_id = tmp[1]
            token = tmp[5]
            entity = tmp[4]
            ent_id = tuple(tmp[6].split(','))

            if doc_id not in mentions.keys():
                mentions[doc_id] = []

            mentions[doc_id].append({})
            mentions[doc_id][-1]["ent_id"] = ent_id
            mentions[doc_id][-1]["tag"] = entity
            mentions[doc_id][-1]["token"] = token
            mentions[doc_id][-1]["id"] = id0
    return mentions

def gen_multi_sentence_data(input_data, mentions):
    output_data = []
    for data in input_data:
        doc_id = str(data["abstract_id"])
        ent1_id = data["entity_a_id"]
        ent2_id = data["entity_b_id"]
        rel_id = data["relation_id"]
        rel = frozenset([ent2_id, ent1_id])
        tag = data["type"]
        novel = data["novel"]
        overlap = data["overlapping"]
        sents = data["sents"]
    
        output_data.append({})
        output_data[-1]["abstract_id"] = int(doc_id)
        output_data[-1]["relation_id"] = rel_id
        output_data[-1]["entity_a_id"] = ent1_id
        output_data[-1]["entity_b_id"] = ent2_id
        output_data[-1]["type"] = tag
        output_data[-1]["novel"] = novel
        output_data[-1]["text"] = ""
        output_data[-1]["entity_a"] = []
        output_data[-1]["entity_b"] = []
        
    
        i = 0
        j = 0
        tmptk1 = []
        tmptag1 = []
        tmptk2 = []
        tmptag2 = []
        for e in mentions[doc_id]:
            if ent1_id in e["ent_id"]:
                tmptk1.append(e["token"])
                tmptag1.append(e["tag"])
            if ent2_id in e["ent_id"]:
                tmptk2.append(e["token"])
                tmptag2.append(e["tag"])
    
        for st in sents:
            tmptxt = st["text"]
            if len(st["entity_a"])>0 or len(st["entity_b"])>0:
                ent_exist = True
            else:
                ent_exist = False
            for e in st["entity_a"]:
                s = e[0]
                f = e[1]
                clen = len(output_data[-1]["text"])
                output_data[-1]["entity_a"].append([clen+s, clen+f, tmptag1[i]])
                i += 1
            for e in st["entity_b"]:
                s = e[0]
                f = e[1]
                clen = len(output_data[-1]["text"])
                output_data[-1]["entity_b"].append([clen+s, clen+f, tmptag2[j]])
                j += 1
            if ent_exist:
                output_data[-1]["text"] = output_data[-1]["text"] + tmptxt + ' '
        
        output_data[-1]["text"] = output_data[-1]["text"].rstrip()
    return output_data


if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description="Generate actual training data (multi-sentence) for LitCoin Phase 2.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
    parser.add_argument("--source_path", action="store", dest="source_path", help="Directory of downloaded original LitCoin phase 2 data.")
    args = parser.parse_args()

    relation_to_id = {'Association': '1', 'Positive_Correlation': '2', 'Negative_Correlation': '3', 'Bind': '4',
                      'Cotreatment': '5', 'Comparison': '6', 'Drug_Interaction': '7', 'Conversion': '8', 'NOT': '0'}

    train_entity_fp = os.path.join(args.source_path, "entities_train.csv")
    test_entity_fp = os.path.join(args.source_path, "entities_test.csv")
    all_train_data = json.load(open(os.path.join(args.data_path, "split", "processed_train.json")))
    all_val_data = json.load(open(os.path.join(args.data_path, "split", "processed_val.json")))
    all_test_data = json.load(open(os.path.join(args.data_path, "split", "processed_test.json")))

    train_entities = load_mentions(train_entity_fp)
    test_entities = load_mentions(test_entity_fp)

    train_outputs = gen_multi_sentence_data(all_train_data, train_entities)
    val_outputs = gen_multi_sentence_data(all_val_data, train_entities)
    test_outputs = gen_multi_sentence_data(all_test_data, test_entities)

    json.dump(train_outputs, open(os.path.join(args.data_path, "split", "multi_sentence_train.json"), "w"), indent=4)
    json.dump(val_outputs, open(os.path.join(args.data_path, "split", "multi_sentence_val.json"), "w"), indent=4)
    json.dump(test_outputs, open(os.path.join(args.data_path, "split", "multi_sentence_test.json"), "w"), indent=4)
