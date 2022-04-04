import spacy
from dataset import CustomDataset
from torchtext.data import TabularDataset, Field, BucketIterator

#############
# tokenizer #
#############
spacy_en = spacy.load('en_core_web_sm')  # en tokenization
spacy_de = spacy.load('de_core_news_sm')  # de tokenization

def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

# (source:영어, target:독일어) _ 데이터전처리(token, 소문자 등)
SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)

#############
# load data #
#############
CustomDataset(phase='train')  # train
CustomDataset(phase='val')  # val
CustomDataset(phase='test')  # test

################
# make dataset #
################
train_data = TabularDataset(path='./wmt16/train.csv', format='csv', fields=[('en', SRC), ('de', TRG)])
valid_data = TabularDataset(path='./wmt16/val.csv', format='csv', fields=[('en', SRC), ('de', TRG)])
test_data = TabularDataset(path='./wmt16/test.csv', format='csv', fields=[('en', SRC), ('de', TRG)])

###############
# build vocab #
###############
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
# [.vocab.stoi] token index ex(없는 단어: 0, padding:1, <sos>:2, <eos>:3)

#################
# make iterator #
#################
batch_size = 128
train_iterator = BucketIterator(train_data, batch_size=batch_size)
valid_iterator = BucketIterator(valid_data, batch_size=batch_size)
test_iterator = BucketIterator(test_data, batch_size=batch_size)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

