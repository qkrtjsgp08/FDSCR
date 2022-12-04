# -*- coding: utf-8 -*-
"""Untitled19.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dGwKx58bNPtmDNghD5YgPNcZjw7XGN0d
"""

#!unzip /content/TL1_aihub.zip

#!unzip /content/VL1_aihub.zip

#!pip install transformers

#!pip install konlpy

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from transformers import AutoModel, AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import warnings
import os

import json
import codecs
import pprint
from konlpy.tag import Okt
from tqdm import tqdm


def load_json(file_name):
    """
    Load json file
    :param file_name: file name
    :return: loaded data from the file
    """
    with codecs.open(file_name, "r", "utf-8") as json_f:
        return json.load(json_f)

text=[]
emotion=[]
is_immoral=[]
intensity=[]

for file_name in ['/content/talksets-train-1/talksets-train-1_aihub.json','/content/talksets-train-2/talksets-train-2.json','/content/talksets-train-3/talksets-train-3.json','/content/talksets-train-4/talksets-train-4.json']:
  json_data = load_json(file_name)
  len_json=len(json_data)
  for i in range(len_json):
    len_one_json=len(json_data[i]['sentences'])
    for j in range(len_one_json):
      
      text.append(json_data[i]['sentences'][j]['text'].replace("  "," "))
      if ord(json_data[i]['sentences'][j]['types'][0][2])==77:
        emotion.append(0)
      else:
        emotion.append(1)
      is_immoral.append(json_data[i]['sentences'][j]['is_immoral'])
      intensity.append(json_data[i]['sentences'][j]['intensity'])

text_test=[]
emotion_test=[]
is_immoral_test=[]
intensity_test=[]

for file_name in ['/content/talksets-train-5/talksets-train-5.json']:
  json_data = load_json(file_name)
  len_json=len(json_data)
  for i in range(len_json):
    len_one_json=len(json_data[i]['sentences'])
    for j in range(len_one_json):
      text_test.append(json_data[i]['sentences'][j]['text'].replace("  "," "))
      if ord(json_data[i]['sentences'][j]['types'][0][2])==77:
        emotion_test.append(0)
      else:
        emotion_test.append(1)
      
      is_immoral_test.append(json_data[i]['sentences'][j]['is_immoral'])
      intensity_test.append(json_data[i]['sentences'][j]['intensity'])

text_valid=[]
emotion_valid=[]
is_immoral_valid=[]
intensity_valid=[]

for file_name in ['/content/talksets-train-6/talksets-train-6.json']:
  json_data = load_json(file_name)
  len_json=len(json_data)
  for i in range(len_json):
    len_one_json=len(json_data[i]['sentences'])
    for j in range(len_one_json):
      text_valid.append(json_data[i]['sentences'][j]['text'].replace("  "," "))
      if ord(json_data[i]['sentences'][j]['types'][0][2])==77:
        emotion_valid.append(0)
      else:
        emotion_valid.append(1)
     
      is_immoral_valid.append(json_data[i]['sentences'][j]['is_immoral'])
      intensity_valid.append(json_data[i]['sentences'][j]['intensity'])

import json
import codecs
import pprint
from konlpy.tag import Okt
from tqdm import tqdm


def load_json(file_name):
    """
    Load json file
    :param file_name: file name
    :return: loaded data from the file
    """
    with codecs.open(file_name, "r", "utf-8") as json_f:
        return json.load(json_f)

text=[]
emotion=[]
is_immoral=[]
intensity=[]

for file_name in ['/content/talksets-train-1/talksets-train-1_aihub.json','/content/talksets-train-2/talksets-train-2.json','/content/talksets-train-3/talksets-train-3.json','/content/talksets-train-4/talksets-train-4.json']:
  json_data = load_json(file_name)
  len_json=len(json_data)
  for i in range(len_json):
    len_one_json=len(json_data[i]['sentences'])
    for j in range(len_one_json):
      
      text.append(json_data[i]['sentences'][j]['text'].replace("  "," "))
      if ord(json_data[i]['sentences'][j]['types'][0][2])==83:
        emotion.append(0)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==84:
        emotion.append(1)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==78:
        emotion.append(2)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==79:
        emotion.append(3)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==73:
        emotion.append(4)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==88:
        emotion.append(5)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==85:
        emotion.append(6)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==77:
        emotion.append(7)
      is_immoral.append(json_data[i]['sentences'][j]['is_immoral'])
      intensity.append(json_data[i]['sentences'][j]['intensity'])

text_test=[]
emotion_test=[]
is_immoral_test=[]
intensity_test=[]

for file_name in ['/content/talksets-train-5/talksets-train-5.json']:
  json_data = load_json(file_name)
  len_json=len(json_data)
  for i in range(len_json):
    len_one_json=len(json_data[i]['sentences'])
    for j in range(len_one_json):
      text_test.append(json_data[i]['sentences'][j]['text'].replace("  "," "))
      if ord(json_data[i]['sentences'][j]['types'][0][2])==83:
        emotion_test.append(0)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==84:
        emotion_test.append(1)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==78:
        emotion_test.append(2)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==79:
        emotion_test.append(3)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==73:
        emotion_test.append(4)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==88:
        emotion_test.append(5)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==85:
        emotion_test.append(6)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==77:
        emotion_test.append(7)
      is_immoral_test.append(json_data[i]['sentences'][j]['is_immoral'])
      intensity_test.append(json_data[i]['sentences'][j]['intensity'])

text_valid=[]
emotion_valid=[]
is_immoral_valid=[]
intensity_valid=[]

for file_name in ['/content/talksets-train-6/talksets-train-6.json']:
  json_data = load_json(file_name)
  len_json=len(json_data)
  for i in range(len_json):
    len_one_json=len(json_data[i]['sentences'])
    for j in range(len_one_json):
      text_valid.append(json_data[i]['sentences'][j]['text'].replace("  "," "))
      if ord(json_data[i]['sentences'][j]['types'][0][2])==83:
        emotion_valid.append(0)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==84:
        emotion_valid.append(1)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==78:
        emotion_valid.append(2)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==79:
        emotion_valid.append(3)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==73:
        emotion_valid.append(4)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==88:
        emotion_valid.append(5)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==85:
        emotion_valid.append(6)
      elif ord(json_data[i]['sentences'][j]['types'][0][2])==77:
        emotion_valid.append(7)
      is_immoral_valid.append(json_data[i]['sentences'][j]['is_immoral'])
      intensity_valid.append(json_data[i]['sentences'][j]['intensity'])

import csv
def save_csv(file_name, sentences, labels):
    """
    Save the sentences and labels to csv file
    Header is needed to identify the column

    :param file_name: output file name
    :param sentences: human sentences from extract_data function
    :param labels: emotion labels from extract_data function
    :return: None
    """
    fields = ['sentence', 'label']
    with open(file_name,'w',newline='') as f:
      write=csv.writer(f)
      write.writerow(fields)
      for i in range(len(sentences)):
        write.writerow([sentences[i],labels[i]])

##### 감정 종류 분류할 때

save_csv('train_is_immoral.csv', text, emotion)
save_csv('test_is_immoral.csv', text_test, emotion_test)
save_csv('valid_is_immoral.csv', text_valid, emotion_valid)

##### 비윤리 강도 평균 점수 

# save_csv('train.csv', text, intensity)
# save_csv('test.csv', text_test, intensity_test)
# save_csv('valid.csv', text_valid, intensity_valid)

train = pd.read_csv("train_is_immoral.csv")
valid = pd.read_csv("valid_is_immoral.csv")
test = pd.read_csv("test_is_immoral.csv")

train.head()

roberta = AutoModel.from_pretrained("klue/roberta-base")
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

class RoBERTaClassifier(nn.Module):
    def __init__(self, roberta, hidden_size=768, num_classes=2):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = roberta
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 24), nn.ReLU(), nn.Linear(24,2))
        #nn.Linear(hidden_size,num_classes)# Task 2

    def forward(self, input_ids, attention_masks):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_masks)
        return self.classifier(outputs.pooler_output)

device = torch.device("cuda:0")

model = RoBERTaClassifier(roberta=roberta).to(device)

class KERDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset
        document =self.dataset['sentence'].tolist() # Task 3
        inputs = tokenizer(document, padding=True)
        self.input_ids = inputs['input_ids']
        self.attention_masks = inputs['attention_mask']
        self.labels = self.dataset['label'].tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_masks[idx], self.labels[idx])

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return torch.LongTensor(input_ids), torch.LongTensor(attention_masks), torch.FloatTensor(labels)

train_ds = KERDataset(train, tokenizer)
valid_ds = KERDataset(valid, tokenizer)
test_ds = KERDataset(test, tokenizer)

batch_size = 32
warmup_ratio = 0.1
num_epochs = 5
log_interval = 400
learning_rate =  5e-5

train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=5, collate_fn=collate_fn)
valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=5, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=5, collate_fn=collate_fn)

print(len(train_dataloader.dataset[0][0]))

total_steps = len(train_dataloader) * num_epochs
warmup_step = int(total_steps * warmup_ratio)

optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_step,
                                            num_training_steps = total_steps)
loss_fn = nn.MSELoss()

def calc_accuracy(X,Y):
    correct = 0
    X, Y = X.tolist(), Y.tolist()
    for pred, label in zip(X, Y):
        if pred.index(max(pred)) == label.index(max(label)):
            correct += 1
    train_acc = correct/len(X)
    return train_acc

#!pip install torchmetrics

import torch.nn.functional as F
from torchmetrics.classification import BinaryStatScores
metric = BinaryStatScores(multidim_average='samplewise')

for e in range(num_epochs):
    results=torch.zeros(5).cuda()
    train_acc = 0.0
    valid_acc = 0.0
    model.train()
    for batch_id, (input_ids, attention_masks, labels) in enumerate(tqdm_notebook(train_dataloader)):
        if batch_id>8400:
            break
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        out = model(input_ids=input_ids, attention_masks=attention_masks)
        labels = labels.to(torch.int64).to(device)
        labels = F.one_hot(labels)
        labels = labels.float()	   
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate schedule

        train_acc += calc_accuracy(out, labels)

        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    model.eval()
    for batch_id, (input_ids, attention_masks, labels) in enumerate(tqdm_notebook(valid_dataloader)):
        
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(torch.int64).to(device)
        labels = F.one_hot(labels)
        labels = labels.float()
        out = model(input_ids=input_ids, attention_masks=attention_masks)
        valid_acc += calc_accuracy(out, labels)
        for i in range(32):
          if len(out)!=32:
            break
          results+=metric(out,labels)[i]
    print("epoch {} validation acc {}".format(e+1, valid_acc / (batch_id+1)))
    print(results)

test_acc = 0.0
model.eval()
for batch_id, (input_ids, attention_masks, labels) in enumerate(tqdm_notebook(test_dataloader)):
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    labels = torch.FloatTensor([[0, 1] if l == 0 else [1, 0] for l in labels]).to(device)
    out = model(input_ids=input_ids, attention_masks=attention_masks)
    test_acc += calc_accuracy(out, labels)
print("Test acc : {}".format(test_acc / (batch_id+1)))

precision=38174/(38174+7151)
recall = 38174/(38174+7041)
print(2*precision*recall/(precision+recall))

test_acc = 0.0
model.eval()
for batch_id, (input_ids, attention_masks, labels) in enumerate(tqdm_notebook(test_dataloader)):
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    labels = torch.FloatTensor([[0, 1] if l == 0 else [1, 0] for l in labels]).to(device)
    out = model(input_ids=input_ids, attention_masks=attention_masks)
    test_acc += calc_accuracy(out, labels)
print("Test acc : {}".format(test_acc / (batch_id+1)))




