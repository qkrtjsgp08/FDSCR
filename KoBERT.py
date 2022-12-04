#!/usr/bin/env python
# coding: utf-8

# # KoBERT finetuning

# ## Library

# ### Install libraries
get_ipython().system('pip install ipywidgets  # for vscode')
get_ipython().system('pip install git+https://git@github.com/SKTBrain/KoBERT.git@master')


# ### Import libraries
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


# ## Device
## CPU
# device = torch.device("cpu")

## GPU
device = torch.device("cuda:0")


# ## Load KoBERT model
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


# ##Load CSV file
from google.colab import drive
drive.mount('/content/drive')


import pandas as pd

dataset_test = pd.read_csv('/content/drive/MyDrive/valid_is_immoral.csv', sep=',', engine='python', encoding='utf-8') ## valid로 acc하니까 임의로 사용
dataset_train = pd.read_csv('/content/drive/MyDrive/train_is_immoral.csv', sep=',', engine='python', encoding='utf-8')
dataset_valid = pd.read_csv('/content/drive/MyDrive/valid_is_immoral.csv', sep=',', engine='python', encoding='utf-8')
dataset_train.head(15)


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


# ## Load Dataset
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair): 

        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform(i) for i in dataset.iloc[:,sent_idx]]
        self.labels = [data for data in dataset.iloc[:,label_idx]]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# ##Hyperparameter setting
## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 100
learning_rate =  5e-5

print(len(dataset_train))
# dataset_train = dataset_train[:128]
# dataset_valid = dataset_valid[:128]


dataset_train.shape

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_valid = BERTDataset(dataset_valid, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=2)
valid_dataloader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=2)

class EarlyStopping: 
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ## BERTClassifier
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate=0).to(device)


# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)


scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def calc_accuracy(label,output):
    label = label.tolist()
    max_vals, max_indices = torch.max(output, 1)
    guess = max_indices.tolist()

    accuracy = accuracy_score(label, guess)
    recall = recall_score(label, guess)
    precision = precision_score(label, guess)
    f1 = f1_score(label, guess)
    
    return accuracy, recall, precision, f1


# ## Model Training
def model_save():
  ## 학습 모델 저장
  PATH = './drive/MyDrive/KoBERT' # google 드라이브 연동
  torch.save(model, PATH + 'KoBERT32.pt')  # 전체 모델 저장
  torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
  torch.save({
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict()
  }, PATH + 'all.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능

def train(model, batch_size, patience, n_epochs):
  early_stopping = EarlyStopping(patience = patience, verbose = True)
  val_losses = []
  for e in range(num_epochs):
      train_acc, train_recall, train_precision, train_f1 = 0.0, 0.0, 0.0, 0.0
      test_acc, test_recall, test_precision, test_f1 = 0.0, 0.0, 0.0, 0.0
      model.train()
      for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
          optimizer.zero_grad()
          token_ids = token_ids.long().to(device)
          segment_ids = segment_ids.long().to(device)
          valid_length= valid_length
          label = label.long().to(device)
          # print("label:  ", label.tolist())
          out = model(token_ids, valid_length, segment_ids)
          # print("out:  ", out)
          loss = loss_fn(out, label)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          optimizer.step()
          scheduler.step()  # Update learning rate schedule
          accuracy, recall, precision, f1 = calc_accuracy(label, out)
          train_acc += accuracy
          train_recall += recall
          train_precision += precision
          train_f1 += f1
          # if batch_id % log_interval == 0:
            # print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
      print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
      print("epoch {} train recall {}".format(e+1, train_recall / (batch_id+1)))
      print("epoch {} train precision {}".format(e+1, train_precision / (batch_id+1)))
      print("epoch {} train f1 {}".format(e+1, train_f1 / (batch_id+1)))
      model.eval()
      for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
          token_ids = token_ids.long().to(device)
          segment_ids = segment_ids.long().to(device)
          valid_length= valid_length
          label = label.long().to(device)
          out = model(token_ids, valid_length, segment_ids)
          loss = loss_fn(out, label)
          val_losses.append(loss.item())
          accuracy, recall, precision, f1 = calc_accuracy(label, out)
          test_acc += accuracy
          test_recall += recall
          test_precision += precision
          test_f1 += f1
          val_loss = np.average(val_losses)
          # early_stopping(val_loss, model)
      print("epoch {} test accuracy {}".format(e+1, test_acc / (batch_id+1)))
      print("epoch {} test recall {}".format(e+1, test_recall / (batch_id+1)))
      print("epoch {} test precision {}".format(e+1, test_precision / (batch_id+1)))
      print("epoch {} test f1 {}".format(e+1, test_f1 / (batch_id+1)))
      torch.save(model.state_dict(), '/content/data'+str(e)+'.pt')
  model.eval()


def test(model):
  test_acc = 0.00
  with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        test_acc += calc_accuracy(out, label)
  print("valid acc {}".format(test_acc / (batch_id+1)))
  torch.save(model.state_dict(), "./drive/MyDrive/modelstate.pt")
  model.eval()

train(model, batch_size, 7, num_epochs)
test(model)
torch.cuda.empty_cache()

model.load_state_dict(torch.load("./drive/MyDrive/modelstate_100.pt"))
model.eval()