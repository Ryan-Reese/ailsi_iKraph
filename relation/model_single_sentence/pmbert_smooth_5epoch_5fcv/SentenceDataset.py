import torch
from torch.utils.data import Dataset
import pandas as pd

class SentenceDataset(Dataset):
    def __init__(self, list_of_dataframes, tokenizer, config):
        """
        config: a dictionary of
            max_len: int, tokenizer maximium length
            is_train: Bool, if it's for training. When training, remove duplicated cases
            transform_method: str in 
                ["entity mask", "entity marker", "entity marker punkt", "typed entity marker", "typed entity marker punct"]
                how sentence is processed
            move_entities_to_start: if True, additionally add entities to start when processing
            model_type: str in ["cls", "triplet"]
                cls: only use the cls token
                triplet: use not only the cls token, but also the positions of entities
            label_column_name: str, the column name in the dataframe that contains the labels
            no_relation_file: str, optional csv file that each row indicates a pair doesn't have relations
            remove_cellline_organismtaxon: bool, remove all cellline and organismtaxon entities
        """
        label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
        self.LABEL_DICT = {idx: val for idx, val in enumerate(label_list)}
        self.ENTITY_LIST = ["CellLine", "ChemicalEntity", "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "OrganismTaxon", "SequenceVariant"]

        og_dataframe = pd.concat(list_of_dataframes, ignore_index=True)
        self.tokenizer = tokenizer
        self.config = config
        self.dataframe = self._process_dataframe(og_dataframe)
        self.no_relation_mat = self._process_no_rel()
        special_tokens = self._get_special_tokens()
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

    def get_processed_dataframe(self):
        return self.dataframe
    
    def get_f1_true_labels(self):
        return list(range(1, len(self.LABEL_DICT)))
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        if self.config["model_type"] == "cls":
            return self._get_item_cls(index)
        elif self.config["model_type"] == "triplet":
            return self._get_item_triplet(index)
        else:
            raise NotImplementedError(f"{self.config['model_type']} is no supported!")

    def _process_dataframe(self, dataframe):
        dataframe = dataframe.copy(deep=True)
        if self.config["is_train"] == True:
            dataframe = dataframe[dataframe["duplicated_flag"]==False]
            dataframe = dataframe.reset_index(drop=True)
            if self.config["remove_cellline_organismtaxon"] is True:
                remove_list = ["CellLine", "OrganismTaxon"]
                keep_flag = [True for _ in range(len(dataframe))]
                for idx, (_, entry) in enumerate(dataframe.iterrows()):
                    ent1, ent2 = entry["entity_a"][2], entry["entity_b"][2]
                    if ent1 in remove_list or ent2 in remove_list:
                        keep_flag[idx] = False
                dataframe = dataframe[keep_flag]
                dataframe = dataframe.reset_index(drop=True)
        
        dataframe["label"] = dataframe[self.config["label_column_name"]]
        dataframe["label"] = dataframe["label"].replace("TBD", 0)
        for key, label in self.LABEL_DICT.items():
            dataframe['label'] = dataframe['label'].replace(label, key)

        texts, new_ent1s, new_ent2s = [], [], []
        for _, entry in dataframe.iterrows():
            new_text, new_ent1, new_ent2 = self._transform_sentence(entry)
            texts.append(new_text)
            new_ent1s.append(new_ent1)
            new_ent2s.append(new_ent2)
        dataframe["processed_text"] = texts
        dataframe["processed_ent1"] = new_ent1s
        dataframe["processed_ent2"] = new_ent2s
        return dataframe
    
    def _transform_sentence(self, entry):
        transform_method = self.config["transform_method"]

        sent = entry["text"]
        ent1_start, ent1_end, ent1_type = entry["entity_a"]
        ent2_start, ent2_end, ent2_type = entry["entity_b"]
        if ent2_start < ent1_start:
            ent1_start, ent1_end, ent1_type = entry["entity_b"]
            ent2_start, ent2_end, ent2_type = entry["entity_a"]
        ent1_mention, ent2_mention = sent[ent1_start:ent1_end], sent[ent2_start:ent2_end]

        if transform_method == "entity_mask":
            pre_ent1 = ''
            ent1 = ent1_type
            post_ent1 = ''
            pre_ent2 = ''
            ent2 = ent2_type
            post_ent2 = ''
        elif transform_method == "entity_marker":
            pre_ent1 = '[E1]'
            ent1 = ent1_mention
            post_ent1 = '[/E1]'
            pre_ent2 = '[E2]'
            ent2 = ent2_mention
            post_ent2 = '[/E2]'
        elif transform_method == "entity_marker_punkt":
            pre_ent1 = '@'
            ent1 = ent1_mention
            post_ent1 = '@'
            pre_ent2 = '#'
            ent2 = ent2_mention
            post_ent2 = '#'
        elif transform_method == "typed_entity_marker":
            pre_ent1 = f'[{ent1_type}]'
            ent1 = ent1_mention
            post_ent1 = f'[/{ent1_type}]'
            pre_ent2 = f'[{ent2_type}]'
            ent2 = ent2_mention
            post_ent2 = f'[/{ent2_type}]'
        elif transform_method == "typed_entity_marker_punct":
            pre_ent1 = f'@ * {ent1_type} *'
            ent1 = ent1_mention
            post_ent1 = '@'
            pre_ent2 = f'# ^ {ent2_type} ^'
            ent2 = ent2_mention
            post_ent2 = '#'
        else:
            raise NotImplementedError(f"{transform_method} is not implemented!")

        tmp_marker_1 = "<##TMP_MARKER_1$$>"
        tmp_marker_2 = "<##TMP_MARKER_2$$>"
        assert tmp_marker_1 not in sent
        assert tmp_marker_2 not in sent
        ent1_tmp = f"{tmp_marker_1}{ent1}{tmp_marker_1}"
        ent2_tmp = f"{tmp_marker_2}{ent2}{tmp_marker_2}"

        sent = ' '.join([sent[0:ent1_start], pre_ent1, ent1_tmp, post_ent1, sent[ent1_end:ent2_start], pre_ent2, ent2_tmp, post_ent2, sent[ent2_end:]]) 
        
        if self.config["move_entities_to_start"] == True:
            sent = ent1_mention + ', ' + ent2_mention + ', ' + sent
        sent = ' '.join(sent.split())  # remove multiple spaces in a row
        sent = sent.replace("LPA1 '3", "LPA1    '3")  # only one special case where it's multiple spaces in the test dataste
        sent = sent.replace("ERK1 '2", "ERK1    '2")
        ent1_start = sent.find(tmp_marker_1)
        ent1_end = sent.find(tmp_marker_1, ent1_start+len(tmp_marker_1))-len(tmp_marker_1)
        sent = sent.replace(tmp_marker_1, '')
        ent2_start = sent.find(tmp_marker_2)
        ent2_end = sent.find(tmp_marker_2, ent2_start+len(tmp_marker_2))-len(tmp_marker_2)
        sent = sent.replace(tmp_marker_2, '')

        assert sent[ent1_start:ent1_end] == ent1
        assert sent[ent2_start:ent2_end] == ent2

        return sent, [ent1_start, ent1_end, ent1_type], [ent2_start, ent2_end, ent2_type]

    def _get_special_tokens(self):
        special_tokens = []
        transform_method = self.config["transform_method"]

        if transform_method == "entity_mask":
            special_tokens += self.ENTITY_LIST
        elif transform_method == "entity_marker":
            special_tokens += ['[E1]', '/[E1]', '[E2]', '[/E2]']
        elif transform_method == "entity_marker_punkt":
            pass  # not adding @ or #
        elif transform_method == "typed_entity_marker":
            special_tokens += [f'[{this_type}]' for this_type in self.ENTITY_LIST]
            special_tokens += [f'[/{this_type}]' for this_type in self.ENTITY_LIST]
        elif transform_method == "typed_entity_marker_punct":
            special_tokens += self.ENTITY_LIST
        else:
            raise NotImplementedError(f"{transform_method} is not implemented!")
        if hasattr(self.tokenizer, "do_lower_case") and self.tokenizer.do_lower_case:
            return sorted([elem.lower() for elem in set(special_tokens)])
        else:
            return []

    def _tokenize_fn(self, sentence, add_special_tokens=True, padding=True):
        sentence = sentence.strip()
        return self.tokenizer(
            sentence,
            add_special_tokens=add_special_tokens,
            padding='max_length' if padding else False,
            truncation=True,
            max_length=self.config["max_len"],
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

    def _get_item_cls(self, index):
        sentence = self.dataframe.loc[index, "processed_text"]
        label = self.dataframe.loc[index, "label"]
        ent_a_type = self.dataframe.loc[index, "entity_a"][2]
        ent_a_idx = self.ENTITY_LIST.index(ent_a_type)
        ent_b_type = self.dataframe.loc[index, "entity_b"][2]
        ent_b_idx = self.ENTITY_LIST.index(ent_b_type)
        encoding = self._tokenize_fn(sentence)

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relation_mask': self.no_relation_mat[ent_a_idx, ent_b_idx, :],
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _get_item_triplet(self, index):
        sentence = self.dataframe.loc[index, "processed_text"]
        ent1_start, ent1_end, ent1_type = self.dataframe.loc[index, "processed_ent1"]
        ent2_start, ent2_end, ent2_type = self.dataframe.loc[index, "processed_ent2"]
        ent_a_idx = self.ENTITY_LIST.index(ent1_type)
        ent_b_idx = self.ENTITY_LIST.index(ent2_type)

        label = self.dataframe.loc[index, "label"]
        
        encoding_1 = self._tokenize_fn(sentence[0: ent1_start], padding=False, add_special_tokens=False)
        encoding_2 = self._tokenize_fn(sentence[0: ent2_start], padding=False, add_special_tokens=False)
        encoding = self._tokenize_fn(sentence)

        input_ids_1, attention_mask_1 = encoding_1['input_ids'].flatten(), encoding_1['attention_mask'].flatten()
        input_ids_2, attention_mask_2 = encoding_2['input_ids'].flatten(), encoding_2['attention_mask'].flatten()

        if len(input_ids_2) >= self.config["max_len"]:
            raise ValueError(f"Tokenization length > {self.config['max_len']} before the appearance of entity2, cannot truncate because entity2 would be discarded!")

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relation_mask': self.no_relation_mat[ent_a_idx, ent_b_idx, :],
            'positions': torch.tensor([0, len(input_ids_1)+1, len(input_ids_2)+1]),
            'label': torch.tensor(label, dtype=torch.long)
        }
    def  _process_no_rel(self):
        ret = torch.zeros((len(self.ENTITY_LIST), len(self.ENTITY_LIST), len(self.LABEL_DICT)))
        if self.config["no_relation_file"] == "":
            return ret
        no_rel_file = pd.read_csv(self.config["no_relation_file"])
        for _, entry in no_rel_file.iterrows():
            type_a, type_b, relation = entry["type_a"], entry["type_b"], entry["relation"]
            idx_a = self.ENTITY_LIST.index(type_a)
            idx_b = self.ENTITY_LIST.index(type_b)
            idx_rel = list(self.LABEL_DICT.values()).index(relation)
            ret[idx_a, idx_b, idx_rel] = -999
            ret[idx_b, idx_a, idx_rel] = -999
        return ret

if __name__ == "__main__":
    from transformers import BertTokenizer, RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # for transform_method in ["entity_mask", "entity_marker", "entity_marker_punkt", "typed_entity_marker", "typed_entity_marker_punct"]:
    for transform_method in ["typed_entity_marker"]:
        config = {
            "model_type": "triplet",
            "max_len": 384,
            "transform_method": transform_method,
            "label_column_name": "annotated_type",
            "move_entities_to_start": False,
            "no_relation_file": "no_rel.csv",
            "is_train": True,
            "remove_cellline_organismtaxon": True
        }
        df = pd.read_json("annotation_data/new_train_splits/split_0/data.json", orient="table")
        train_dataset = SentenceDataset([df], tokenizer, config)
        first_elem = train_dataset[1]
        sent = first_elem["sentence"]
        tokens = train_dataset.tokenizer.convert_ids_to_tokens(first_elem["input_ids"])
        print(sent, first_elem["positions"])
        for idx in first_elem["positions"]:
            print(tokens[idx-1], tokens[idx], tokens[idx+1])
        input()
        print(first_elem)