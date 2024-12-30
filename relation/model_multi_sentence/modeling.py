#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import process
from random import randrange

import pandas as pd
import json
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np

from torch.utils.data import Dataset, DataLoader

import sys
import argparse

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--batch_size', type=int, default=4, help="batch size for training and validation")
parser.add_argument('--epochs', type=int, default=15, help="epoch")
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='learning rate')
parser.add_argument('--RD', '--random_seed', type=int, default=42, help='random seed')
parser.add_argument('--train_set', type=str, default='train.json', help='training data set')
parser.add_argument('--valid_set', type=str, default='valid.json', help='valid data set')
parser.add_argument('--test_set', type=str, default='test.json', help='test data set')
parser.add_argument('--pretrain', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', help='pretrain_model')
parser.add_argument('--output_path', type=str, default='BERT_train', help='checkpoint_path')

hyper_args = parser.parse_args()
for arg in vars(hyper_args):
    print(arg, getattr(hyper_args, arg))

RANDOM_SEED = hyper_args.RD
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
label_dict = {idx: val for idx, val in enumerate(label_list)}

def transform_sentence(entry):
    entity_a = entry["entity_a"]
    entity_b = entry["entity_b"]
    sent = entry["text"]
    if sent == "": return ""
    all_poses = entity_a + entity_b
    all_poses.sort(key=lambda i: i[0], reverse=True)
    for start, end, e_type in all_poses:
        sent = sent[0:start] + e_type + sent[end:]
    sent = entry["text"][entity_a[0][0]:entity_a[0][1]] + ", " + entry["text"][entity_b[0][0]:entity_b[0][1]] + ", " + sent
    return sent

#def process_data(dataframe):
#    dataframe["label"] = dataframe["annotated_type"]
#    for key, label in label_dict.items():
#        dataframe['label'] = dataframe['label'].replace(label, key)
#
#    dataframe["text"] = [transform_sentence(entry) for _, entry in dataframe.iterrows()]
#    return dataframe
#
#MAX_SPLITS = 5
#dataframes = [pd.read_json(open("annotation_data/train_splits/split_{}/data.json".format(split_id)), orient="table") for split_id in range(0, MAX_SPLITS)]
#VAL_SPLIT_INDEX = 3
#TEST_SPLIT_INDEX = 4
#TRAIN_SPLIT_INDEXES = [0, 1, 2]
#df_val = process_data(dataframes[VAL_SPLIT_INDEX])
#df_test = process_data(dataframes[TEST_SPLIT_INDEX])
#df_train = pd.concat([dataframes[idx] for idx in TRAIN_SPLIT_INDEXES], ignore_index=True)
#df_train = process_data(df_train)
def process_data(data_path, mode='train'):
    train_data = json.load(open(data_path, "r"))
    df_train = pd.DataFrame(train_data)
    if mode == 'train':
        df_train["label"] = df_train["type"]
        for key, label in label_dict.items():
            df_train['label'] = df_train['label'].replace(label, key)
    elif mode == 'test':
        df_train["label"] = 0
    df_train["text"] = [transform_sentence(entry) for _, entry in df_train.iterrows()]
    return df_train

df_train = process_data(hyper_args.train_set)
df_val = process_data(hyper_args.valid_set)
df_test = process_data(hyper_args.test_set, mode='test')

# Model Parameters
PRE_TRAINED_MODEL_NAME = hyper_args.pretrain
MAX_LEN = 512
BATCH_SIZE = hyper_args.batch_size

# https://huggingface.co/blog/zero-deepspeed-fairscale
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokens = label_list[1:]
tokenizer.add_tokens(tokens, special_tokens=True)

class SentenceDataset(Dataset):
    
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_data = SentenceDataset(
    sentences=df_train.text.to_numpy(),
    labels=df_train.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
val_data = SentenceDataset(
    sentences=df_val.text.to_numpy(),
    labels=df_val.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_data = SentenceDataset(
    sentences=df_test.text.to_numpy(),
    labels=df_test.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

from transformers import AutoModelForSequenceClassification
n_classes = len(label_dict)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=n_classes)

if model.base_model_prefix == "roberta":
    model.roberta.resize_token_embeddings(len(tokenizer))
elif model.base_model_prefix == "bert":
    model.bert.resize_token_embeddings(len(tokenizer))
else:
    raise NotImplementedError
model = model.to(device)

# Training Parameters
#BATCH_SIZE = 16
#EPOCHS = 5
#LEARNING_RATE = 3e-5
#SAVE_PATH = "annotated_roberta_large"
BATCH_SIZE = hyper_args.batch_size
EPOCHS = hyper_args.epochs
LEARNING_RATE = hyper_args.learning_rate
SAVE_PATH = hyper_args.output_path


from transformers import Trainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=EPOCHS,
    evaluation_strategy='epoch',
    learning_rate=LEARNING_RATE,
    #eval_steps=200,
    save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
    save_strategy='epoch',
    #save_steps=200,
    metric_for_best_model = 'f1',
    warmup_ratio=0.2,
    per_device_train_batch_size=BATCH_SIZE, # 6231 -> 3633MB
    per_device_eval_batch_size=BATCH_SIZE,
    # gradient_accumulation_steps=4,
    # eval_accumulation_steps=4,
    load_best_model_at_end=True,
    # deepspeed="dsconfig.json",
    # gradient_checkpointing=True,
    fp16=False  # 6903 -> 6231
)

# https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
def compute_f1(evalprediction_instance):
    predictions, label_ids = evalprediction_instance.predictions, evalprediction_instance.label_ids
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(label_ids, predictions)
    f1 = f1_score(label_ids, predictions, labels=range(1, len(label_dict)), average='micro')
    return {"accuracy": accuracy, "f1": f1}

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_data, 
    eval_dataset=val_data,
    compute_metrics=compute_f1
)

trainer.train()
print(trainer.evaluate(eval_dataset=test_data))
trainer.save_model("{}/{}".format(SAVE_PATH, "final_model"))
tokenizer.save_pretrained("{}/{}".format(SAVE_PATH, "tokenizer"))

ret = trainer.predict(test_data)

predictions = ret.predictions
with open(SAVE_PATH+'/model_output_prob.txt','w')as f:
    f.writelines('\n'.join(['\t'.join(list(map(str, x))) for x in predictions])+'\n')
predicted_class = np.argmax(predictions, axis=1)
predicted_class = [label_dict[cl] for cl in predicted_class]
expanded_predicted_class = []
for idx in range(len(predicted_class)):
    expanded_predicted_class.append(predicted_class[idx])
    #for _ in range(1, len_relations):
    #    expanded_predicted_class.append("MASKED")
df_test["predicted_class"] = expanded_predicted_class
df_test.to_csv(SAVE_PATH+"/model_output.csv")


