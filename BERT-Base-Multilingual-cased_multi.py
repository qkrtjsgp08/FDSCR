# -*- coding: utf-8 -*-
"""multi_for colab

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CCKeO5Rmg3yfO-pXCPW-TLjuY1k1A5d5
"""

# Hugging Face의 트랜스포머 모델을 설치
!pip install transformers

!pip install sklearn

import tensorflow as tf
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime
from google.colab import drive

drive.mount('/content/drive')

# 판다스로 훈련셋과 테스트셋 데이터 로드
train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/train.csv", sep=',', encoding='cp949')
valid = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/valid.csv", sep=',', encoding='cp949')
test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/test.csv", sep=',', encoding='cp949')

print("train:", train.shape)
print("valid:", valid.shape)
print("test:", test.shape)
train.head(10)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

def preprocess(tokenizer, data):
	sentences=data['sentence']
	
	sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
	
	labels = data['label'].values
	
	tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
	

	# 입력 토큰의 최대 시퀀스 길이
	MAX_LEN = 128

	# 토큰을 숫자 인덱스로 변환
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

	# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움  
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

	
	# 어텐션 마스크 초기화
	attention_masks = []

	# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
	# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
	for seq in input_ids:
		seq_mask = [float(i>0) for i in seq]
		attention_masks.append(seq_mask)	
	
  # 데이터를 파이토치의 텐서로 변환
	inputs = torch.tensor(input_ids)
	labels = torch.tensor(labels)
	masks = torch.tensor(attention_masks)

	return inputs, labels, masks

"""데이터 전처리"""

train_inputs, train_labels, train_masks = preprocess(tokenizer, train)
valid_inputs, valid_labels, valid_masks = preprocess(tokenizer, valid)
test_inputs, test_labels, test_masks = preprocess(tokenizer, test)

print("train:", len(train_inputs), len(train_labels), len(train_masks))
print("valid:", len(valid_inputs), len(valid_labels), len(valid_masks))
print("test:", len(test_inputs), len(test_labels), len(test_masks))

# 배치 사이즈
batch_size = 32

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
validation_sampler = SequentialSampler(validation_data)   #random or qequential 
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

"""모델 생성"""

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# 디바이스 설정
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# 분류를 위한 BERT 모델 생성
# num_labels = 8 로 변경 
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=8)
model.cuda()

# 옵티마이저 설정
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

# 에폭수
epochs = 30

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * epochs
print("len train_dataloader:", len(train_dataloader))
print("toal_steps:", total_steps)
# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

"""모델 학습"""

# 정확도 계산 함수
def flat_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 
def calc_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    accuracy = accuracy_score(labels_flat, pred_flat)
    recall = recall_score(labels_flat, pred_flat)
    precision = precision_score(labels_flat, pred_flat)
    f1 = f1_score(labels_flat, pred_flat)

    return accuracy, recall, precision, f1

# 시간 표시 함수
def format_time(elapsed):

    # 반올림
    elapsed_rounded = int(round((elapsed)))
    
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(epochs):
    
    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()
        
    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # Forward 수행                
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        # 로스 구함
        loss = outputs[0]

        # 총 로스 계산
        total_loss += loss.item()

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)            

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    #시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # 출력 로짓 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # 출력 로짓과 라벨을 비교하여 정확도 계산
        #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("  eval_acc: {0:.4f}".format(eval_acc/nb_eval_steps))
    print("  eval_recall: {0:.4f}".format(eval_recall/nb_eval_steps))
    print("  eval_preci: {0:.4f}".format(eval_preci/nb_eval_steps))
    print("  eval_f1: {0:.4f}".format(eval_f1/nb_eval_steps))

    torch.save(model.state_dict(),"./model"+str(epoch_i)+".pt") 

print("")
print("Training complete!")

#시작 시간 설정
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)
    
    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch
    
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
    # 출력 로짓 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))

"""새로운 문장 테스트 """

# 입력 데이터 변환
def convert_input_data(sentences):

    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks

# 문장 테스트
def test_sentences(sentences):

    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
            
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)

    # 출력 로짓 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return logits

logits = test_sentences(['연기는 별로지만 재미 하나는 끝내줌!'])

print(logits)
print(np.argmax(logits))

logits = test_sentences(['원래 틀딱들은 눈치가 없어서 ㅋㅋㅋ'])

print(logits)
print(np.argmax(logits))

