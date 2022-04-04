import torch
import torch.nn as nn
import time
import math
from data import *
import hparams
from model import Encoder, Decoder, Transformer
#import CreateModel
#import preprocess
#import dill

##################
# model training #
##################
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.en
        trg = batch.de
        optimizer.zero_grad()

        # 입력 start with <sos> & 출력 <eos> 제외
        output, _ = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        # 출력 <sos> 제외
        trg = trg[:, 1:].contiguous().view(-1)

        # model output vs. target sentence
        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # gradient clipping
        optimizer.step()   # parameter update
        epoch_loss += loss.item()   # 전체 loss값

    return epoch_loss / len(iterator)

####################
# model evaluation #
####################
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터 확인
        for i, batch in enumerate(iterator):
            src = batch.en    # src = batch.src
            trg = batch.de    # trg = batch.trg

            # 입력 start with <sos> & 출력 <eos> 제외
            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)

            trg = trg[:,1:].contiguous().view(-1)   # 출력 <eos> 제외
            loss = criterion(output, trg)   # model output vs. target sentence

            # 전체 loss값
            epoch_loss += loss.item()
                # loss shape: tensor(1,)
                # loss.item -> loss의 scalar값
    return epoch_loss / len(iterator)

####################################
# weight, parameter initialization #
####################################
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##################
# epoch별 시간계산 #
##################
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

################
# create model #
################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용
hp = hparams.hparams()

## Encoder, Decoder
enc = Encoder.Encoder(INPUT_DIM, hp.HIDDEN_DIM, hp.ENC_LAYERS, hp.ENC_HEADS, hp.ENC_PF_DIM, hp.ENC_DROPOUT, device)
dec = Decoder.Decoder(OUTPUT_DIM, hp.HIDDEN_DIM, hp.DEC_LAYERS, hp.DEC_HEADS, hp.DEC_PF_DIM, hp.DEC_DROPOUT, device)

## Transformer
model = Transformer.Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

## weight initialization
model.apply(initialize_weights)

## optimizer(Adam)
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

## criterion (뒷 부분의 padding값 무시)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


###############
# start train #
###############
N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time() # 시작 시간 기록
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time() # 종료 시간 기록
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_e2g.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')


model.load_state_dict(torch.load('transformer_e2g.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')