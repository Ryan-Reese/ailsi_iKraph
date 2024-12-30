import os
import itertools
import copy
import json
import argparse

import pandas


def read_training_data(abstract_path, entities_path):
    # Columns in the Abstracts file:
    # abstract_id: PubMed ID of the research paper.
    # title: title of the research paper.
    # abstract: abstract or summary of the research paper.
    abstracts = pandas.read_csv(open(abstract_path, "r"), sep='\t')

    # Columns in the Entities file:
    # id: unique ID of the entity's mention
    # abstract_id: PubMed ID of the research paper where the entity's mention appears.
    # offset_start: position of the character where the entity's mention substring begins in the text (title + abstract).
    # offset_finish: position of the character where the entity's mention substring ends in the text (title + abstract).
    # type: type of entity as one of the 6 possible categories mentioned in the first part of the competition.
    # mention: substring representing the actual entity's mention in the text. Can also be extracted using the offsets and the input text.
    # entity_ids*: comma separated external IDs from a biomedical ontology to specifically identify the entity.
    def convert_entity_ids(entity_id):
        '''
        # entity_ids*: comma separated external IDs from a biomedical ontology to specifically identify the entity.
        Separate this using comma
        '''
        return entity_id.split(",")


    entities = pandas.read_csv(open(entities_path, "r"), sep='\t', converters={"entity_ids": convert_entity_ids})

    return abstracts, entities


def read_training_relations(relations_path):
    # Columns in the relations file:
    # id: unique ID of the relation
    # abstract_id: PubMed ID of the research paper where the relation appears.
    # type: type or predicate connecting the two entities.
    # entity_1_id: external ID of the entity that corresponds to the subject of the relation.
    # entity_2_id: external ID of the entity that corresponds to the object of the relation.
    # novel: whether the relation found corresponds to a novel discovery or not.
    # NOTE that the relation file is only available for the training set.
    relations = pandas.read_csv(open(relations_path, "r"), sep='\t')
    return relations


def is_entity(entity_id, entities):
    ret = [False] * len(entities)
    for idx, (_, entity) in enumerate(entities.iterrows()):
        entity_ids = entity["entity_ids"]
        if entity_id in entity_ids:
            ret[idx] = True
    return ret


# entity_ids with identical appearance
def get_appearance(entity_id, list_of_entity_ids):
    return [entity_id in entity_ids for entity_ids in list_of_entity_ids]


def has_identitcal_appearance(entity_1_id, entity_2_id, list_of_entity_ids):    
    entity_1_appearance = get_appearance(entity_1_id, list_of_entity_ids)
    entity_2_appearance = get_appearance(entity_2_id, list_of_entity_ids)
    return entity_1_appearance == entity_2_appearance


def flatten_list(list_of_list):
    ret = []
    for elem in list_of_list:
        for sub_elem in elem:
            ret.append(sub_elem)
    return ret


def check_overlapping(appearance_a, appearance_b):
    assert len(appearance_a) == len(appearance_b)
    has_a_true_b_false = False
    has_a_false_b_true = False
    has_a_true_b_true = False

    for bool_a, bool_b in zip(appearance_a, appearance_b):
        if bool_a == True and bool_b == False: has_a_true_b_false = True
        if bool_b == True and bool_a == False: has_a_false_b_true = True
        if bool_a == True and bool_b == True: has_a_true_b_true = True

    if not has_a_true_b_true: return "No Overlapping"
    if has_a_true_b_true and has_a_false_b_true and not has_a_true_b_false: return "A subset of B"
    if has_a_true_b_true and has_a_true_b_false and not has_a_false_b_true: return "A superset of B"
    if has_a_true_b_true and has_a_true_b_false and has_a_false_b_true: return "Partial Overlapping"
    raise ValueError


def entity_pd_to_dict(entities, title_text, abstract_text):
    len_title = len(title_text)
    abstract_pos_start = len_title + 1
    ret = []
    for _, entity in entities.iterrows():
        position = "title"
        offset_start, offset_finish = entity["offset_start"], entity["offset_finish"]
        if offset_start >= abstract_pos_start:
            offset_start, offset_finish = offset_start - abstract_pos_start, offset_finish - abstract_pos_start
            assert abstract_text[offset_start:offset_finish] == entity["mention"]
            position = "abstract"
        this_dict = {
            "id": entity["id"],
            "position": position,
            "offset_start": offset_start,
            "offset_finish": offset_finish,
            "type": entity["type"],
            "mention": entity["mention"],
            "entity_ids": entity["entity_ids"]
        }
        ret.append(this_dict)
    return ret


def gen_data(data_type, input_path, output_path, all_entity_ids=None):
    abstract_path = os.path.join(input_path, "abstracts_{}.csv".format(data_type))
    entities_path = os.path.join(input_path, "entities_{}.csv".format(data_type))
    
    if data_type == "train":
        relations_path = os.path.join(input_path, "relations_{}.csv".format(data_type))
        relations = read_training_relations(relations_path)

    abstracts, entities = read_training_data(abstract_path, entities_path)

    false_case_id = 0
    all_cases = []
    copy_list = [] # abstract_id, copy_from, copy_to
    for _, abstract in abstracts.iterrows():
        abstract_id = abstract["abstract_id"]
        title_text, abstract_text = abstract["title"], abstract["abstract"]
        
        related_entities = entities[entities["abstract_id"]==abstract_id]
        if data_type == "train":
            related_relations = relations[relations["abstract_id"]==abstract_id]

        all_entitiy_ids = []
        for entity_ids in list(related_entities["entity_ids"]):
            all_entitiy_ids.append(entity_ids)

        all_entitiy_ids_set = set(flatten_list(all_entitiy_ids))
        if all_entity_ids is not None:
            all_entitiy_ids_set = all_entity_ids[str(abstract_id)]

        filtered_entity_ids_set = copy.deepcopy(all_entitiy_ids_set)

        for entity_a_id, entity_b_id in itertools.combinations(all_entitiy_ids_set, r=2):
            appearance_a = get_appearance(entity_a_id, all_entitiy_ids)
            appearance_b = get_appearance(entity_b_id, all_entitiy_ids)
            if appearance_a == appearance_b and entity_b_id in filtered_entity_ids_set:
                filtered_entity_ids_set.remove(entity_b_id)
                copy_list.append([abstract_id, entity_a_id, entity_b_id])

        for entity_a_id, entity_b_id in itertools.combinations(filtered_entity_ids_set, r=2):
            appearance_a = get_appearance(entity_a_id, all_entitiy_ids)
            appearance_b = get_appearance(entity_b_id, all_entitiy_ids)
            assert appearance_a != appearance_b
            overlapping_flag = check_overlapping(appearance_a, appearance_b)

            entity_a_info = related_entities[appearance_a]
            entity_b_info = related_entities[appearance_b]
            entity_a_dict = entity_pd_to_dict(entity_a_info, title_text, abstract_text)
            entity_b_dict = entity_pd_to_dict(entity_b_info, title_text, abstract_text)
            
            # ep for entity pair
            if data_type == "train":
                ep_related_relations_indicator = ((related_relations["entity_1_id"]==entity_a_id) & (related_relations["entity_2_id"]==entity_b_id)) | ((related_relations["entity_1_id"]==entity_b_id) & (related_relations["entity_2_id"]==entity_a_id))
                ep_related_relations = related_relations[ep_related_relations_indicator]
                assert len(ep_related_relations) <= 1

                if len(ep_related_relations) == 0:
                    relation_type = "NOT"
                    novel = "N/A"
                    relation_id = "False.{}.{}".format(abstract_id, false_case_id)
                    false_case_id += 1

                else:
                    ep_related_relation = ep_related_relations.squeeze()
                    relation_type = ep_related_relation["type"]
                    novel = ep_related_relation["novel"]
                    relation_id = "True.{}.{}".format(abstract_id, ep_related_relation["id"])
            else:
                relation_type = "TBD"
                novel = "TBD"
                relation_id = "Test.{}.{}".format(abstract_id, false_case_id)
                false_case_id += 1

            this_case = {
                "relation_id": relation_id,
                "abstract_id": abstract_id,
                "title": title_text,
                "abstract": abstract_text,
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id,
                "type": relation_type,
                "novel": novel,
                "entity_a_info": entity_a_dict,
                "entity_b_info": entity_b_dict,
                "overlapping": overlapping_flag
            }
            all_cases.append(this_case)

    json.dump(all_cases, open(os.path.join(output_path, "processed_data_{}.json".format(data_type)), "w"), indent=4)

    with open(os.path.join(output_path, "copy_list_{}.csv".format(data_type)), "w") as copy_fp:
        copy_fp.write("abstract_id,copy_from,copy_to\n")
        for abstract_id, copy_from, copy_to in copy_list:
            copy_fp.write("{},{},{}\n".format(abstract_id, copy_from, copy_to))


if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description="Process Litcoin Phase 2 data.")
    parser.add_argument("--input_path", action="store", dest="input_path", help="Directory of downloaded LitCoin phase 2 data.")
    parser.add_argument("--output_path", action="store", dest="output_path", help="Directory of processed LitCoin phase 2 data.")
    parser.add_argument("--entity_id_file", action="store", dest="entity_id_file", default="entity_ids.json", help="Path to the entity id file to ensure same data processing order. Use `none` to not use this file.")
    args = parser.parse_args()
    entity_ids = json.load(open(args.entity_id_file)) if args.entity_id_file.lower() != "none" else None
    gen_data("train", args.input_path, args.output_path, entity_ids)
    gen_data("test", args.input_path, args.output_path, entity_ids)
