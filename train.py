from tqdm import tqdm
import dill
import time
import math
import os
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Dataset, BucketIterator
# import torchmetrics import BLEUScore
import hparams
from model import Encoder, Decoder, Transformer
from Optim import ScheduledOptim
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



############################################################################################
def patch_trg(trg, pad_idx):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def cal_performance(pred, gold, TRG_PAD_IDX, criterion):
    gold = gold.contiguous().view(-1)
    loss = criterion(pred, gold)
    #loss = F.cross_entropy(pred, gold, ignore_index=TRG_PAD_IDX, reduction='sum')

    pred = pred.max(1)[1]

    non_pad_mask = gold.ne(TRG_PAD_IDX)    # ne is =!, eq is ==
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_correct, n_word    

def epoch_time(start_time, end_time):    # epoch별 시간계산
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
############################################################################################

#############################
# model training (per epoch)# 
#############################
def train(model, iterator, optimizer, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion, data):
    model.train() # 학습 모드
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
        
    for idx,batch in enumerate(tqdm(iterator)):  ##leave=False
        # prepare data
        src_seq = batch.src
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
        
        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)
        

        # backward & update parameters
        loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX, criterion)
        loss.backward()
                
        optimizer.step()
        lr_scheduler.step()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        #if idx % 50 == 0:
        #print(f"{idx} : loss = {total_loss/n_word_total} | acc = {n_word_correct/n_word_total}")
        #trg_tokens = [data['vocab']['trg'].vocab.itos[i] for i in [data['vocab']['trg'].vocab.stoi[data['vocab']['trg'].init_token]]]
        #print(trg_tokens[1:])
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    
    return loss_per_word, accuracy
    
###############################
# model evaluation (per epoch)# 
###############################
def evaluate(model, iterator, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion):
    model.eval() # 평가 모드
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(iterator):  #leave=False
            # prepare data
            src_seq = batch.src.to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
            
            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX, criterion)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()
            

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
 
###############
# start train #
###############
def start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS):
    def print_performances(header, accu, start_time, loss):
        print('  - {header:12}  accuracy: {accu:3.3f} %, loss: {loss:3.3f} '\
                'elapse: {elapse:3.3f} min'.format(
                    header=f"({header})",
                    accu=100*accu, loss=loss, elapse=(time.time()-start_time)/60))
    valid_losses = []
    for epoch in range(N_EPOCHS):
        print(f'[ Epoch | {epoch}:{N_EPOCHS}]')
        
        start_time = time.time() # 시작 시간
        train_loss, train_accu = train(model, train_iterator, optimizer, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion, data)
        print_performances('Training', train_accu, start_time, train_loss)
        
        start_time = time.time()
        valid_loss, valid_accu = evaluate(model, valid_iterator, device, SRC_PAD_IDX, TRG_PAD_IDX, criterion)
        print_performances('Validation', valid_accu, start_time, valid_loss)
        
        valid_losses += [valid_loss]

        if valid_loss <= min(valid_losses):
            torch.save(model.state_dict(), "transformer_e2g.pt")
            print(f'[Info] Model has been updated - epoch: {epoch}')
 
#########################################################################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용
    
#==================== PREPARING ====================

batch_size = 64
data = dill.load(open("data.pickle", 'rb'))

SRC_PAD_IDX = data['vocab']['src'].vocab.stoi["<blank>"]
TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi["<blank>"]
#SRC_PAD_IDX = data['vocab']['src'].vocab.stoi[data['vocab']['src'].pad_token]
#TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi[data['vocab']['trg'].pad_token]

INPUT_DIM = len(data['vocab']['src'].vocab)  # src_vocab_size
OUTPUT_DIM = len(data['vocab']['trg'].vocab)  # trg_vocab_size

fields = {'src': data['vocab']['src'], 'trg' : data['vocab']['trg']}

train_data = Dataset(examples=data['train_data'], fields=fields)
val_data = Dataset(examples=data['valid_data'], fields=fields)
#test_data = Dataset(examples=data['test_data'], fields=fields)

#train_iterator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=True)
#valid_iterator = torch.utils.data.DataLoader(val_data, batch_size=batch_size, drop_last=True)
train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
valid_iterator = BucketIterator(val_data, batch_size=batch_size, device=device)
#test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)

#==================== CREATE MODEL ===================#
## Transformer
model = Transformer.Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX,hparams.HIDDEN_DIM,
                                hparams.ENC_LAYERS, hparams.DEC_LAYERS, hparams.ENC_HEADS, hparams.DEC_HEADS,
                                hparams.ENC_PF_DIM, hparams.DEC_PF_DIM, hparams.ENC_DROPOUT, hparams.DEC_DROPOUT, device).to(device)
#=====================================================#
N_EPOCHS = 50
#=====================================================#  
    
## optimizer(Adam)
LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=8000,
        num_training_steps = N_EPOCHS * len(train_iterator)
    ) 

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

#==================== START TRAIN ====================#
start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS)




'''
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
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f}')


model.load_state_dict(torch.load('transformer_e2g.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')
'''


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Dataset, BucketIterator
from tqdm import tqdm
import dill
import time
import math
import hparams
from model import Encoder, Decoder, Transformer


############################################################################################
def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def cal_performance(pred, gold, TRG_PAD_IDX):
    gold = gold.contiguous().view(-1)
    loss = F.cross_entropy(pred, gold, ignore_index=TRG_PAD_IDX)
    
    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(TRG_PAD_IDX)    # ne is =!, eq is ==
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_correct, n_word    

def epoch_time(start_time, end_time):    # epoch별 시간계산
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

## weight, parameter initialization 
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
############################################################################################

##################
# model training #
##################
def train(model, iterator, optimizer, clip, device, SRC_PAD_IDX, TRG_PAD_IDX): #criterion 제외
    model.train() # 학습 모드
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
    ### epoch_loss = 0
    
    for batch in tqdm(iterator):
        # prepare data
        src_seq = patch_src(batch.src, SRC_PAD_IDX).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
        
        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)
        
        # backward & update parameters
        loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX)
        loss.backward()
        optimizer.step()
           
        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

    '''
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
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
    '''
    
####################
# model evaluation #
####################
def evaluate(model, iterator, device, SRC_PAD_IDX, TRG_PAD_IDX): #criterion 제외
    model.eval() # 평가 모드
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    ### epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            # prepare data
            src_seq = patch_src(batch.src, SRC_PAD_IDX).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, TRG_PAD_IDX))
            
            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(pred, gold, TRG_PAD_IDX)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    '''
    with torch.no_grad():
        # 전체 평가 데이터 확인
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

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
    '''
 
######################
# prepare dataloader #
######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용

batch_size = 128
data = dill.load(open("data.pickle", 'rb'))

SRC_PAD_IDX = data['vocab']['src'].vocab.stoi[data['vocab']['src'].pad_token]
TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi[data['vocab']['trg'].pad_token]

INPUT_DIM = len(data['vocab']['src'].vocab)  # src_vocab_size
OUTPUT_DIM = len(data['vocab']['trg'].vocab)  # trg_vocab_size

fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

train_data = Dataset(examples=data['train_data'], fields=fields)
val_data = Dataset(examples=data['valid_data'], fields=fields)
test_data = Dataset(examples=data['test_data'], fields=fields)

train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True)
valid_iterator = BucketIterator(val_data, batch_size=batch_size, device=device)
test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)


################
# create model #
################
## Encoder, Decoder
enc = Encoder.Encoder(INPUT_DIM, SRC_PAD_IDX, hparams.HIDDEN_DIM, hparams.ENC_LAYERS, hparams.ENC_HEADS, hparams.ENC_PF_DIM, hparams.ENC_DROPOUT, device)
dec = Decoder.Decoder(OUTPUT_DIM, TRG_PAD_IDX, hparams.HIDDEN_DIM, hparams.DEC_LAYERS, hparams.DEC_HEADS, hparams.DEC_PF_DIM, hparams.DEC_DROPOUT, device)
## Transformer
model = Transformer.Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

## weight initialization
model.apply(initialize_weights)

## optimizer(Adam)
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)  ## criterion


###############
# start train #
###############
def start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS, clip):
    def print_performances(header, ppl, accu, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    header=f"({header})", ppl=ppl,
                    accu=100*accu, elapse=(time.time()-start_time)/60))

    valid_losses = []
    for epoch in range(N_EPOCHS):
        start_time = time.time() # 시작 시간
        train_loss, train_accu = train(model, train_iterator, optimizer, clip, device, SRC_PAD_IDX, TRG_PAD_IDX)
        train_ppl = math.exp(min(train_loss, 100))
        print_performances('Training', train_ppl, train_accu, start_time)
        
        start_time = time.time()
        valid_loss, valid_accu = evaluate(model, valid_iterator, device, SRC_PAD_IDX, TRG_PAD_IDX )
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start_time)
        
        valid_losses += [valid_loss]

        if valid_loss <= min(valid_losses):
            torch.save(model.state_dict(), "transformer_e2g.pt")
            print(f'[Info] Model has been updated - epoch: {epoch}')

N_EPOCHS = 10
clip = 1
start_train(model, train_iterator, valid_iterator, optimizer, device, N_EPOCHS, clip)



'''
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
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f}')


model.load_state_dict(torch.load('transformer_e2g.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')
'''
"""