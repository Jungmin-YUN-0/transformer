import torch.nn as nn
import torch
from model.EncoderLayer import EncoderLayer
from model.PositionalEncoding import PositionalEncoding
import copy


# encoder

##embedding + encoder layer
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, SRC_PAD_IDX, d_model, n_layers, n_head, ffn_hidden, drop_prob, device, n_position=512):
        super().__init__()
        self.device = device

        # input dimenstion → embedding dimension
        self.tok_embedding = nn.Embedding(enc_voc_size, d_model, padding_idx=SRC_PAD_IDX)
        # positional embedding 학습 (sinusoidal x) =>> self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model, n_position=n_position)
        #self.layers = nn.ModuleList([EncoderLayer.EncoderLayer(d_model, n_head, ffn_hidden, drop_prob, device) for _ in range(n_layers)])
        encoder_layer = EncoderLayer(d_model, n_head, ffn_hidden, drop_prob, device)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])
        self.dropout = nn.Dropout(drop_prob)
        #self.d_model = d_model
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)  ## normalization (가중치에 d_model을 곱함)
        
    def forward(self, src, src_mask):

        x = src
        
        x = self.tok_embedding(x)
        x = x * self.scale
        #src *= self.d_model**0.5
        x = self.dropout(self.pos_encoding(x))
        
        for layer in self.layers :
            x = layer(x, src_mask)     
        return x        
        
        
        
'''

##embedding + encoder layer
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, SRC_PAD_IDX, d_model, n_layers, n_head, ffn_hidden, drop_prob, device, max_len=512, n_position=200):
        super().__init__()
        self.device = device

        # input dimenstion → embedding dimension
        self.tok_embedding = nn.Embedding(enc_voc_size, d_model, padding_idx=SRC_PAD_IDX)
        # positional embedding 학습 (sinusoidal x) =>> self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos_encoding = PositionalEncoding.PositionalEncoding(d_model, n_position=n_position)
        self.layers = nn.ModuleList([EncoderLayer.EncoderLayer(d_model, n_head, ffn_hidden, drop_prob, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(drop_prob)

        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)  ## normalization (가중치에 d_model을 곱함)
        
        #self.emb = TransformerEmbedding(d_model=d_model, max_len=max_len, vocab_size=enc_voc_size, drop_prob=drop_prob, device=device)
        #self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layers)])
        

    def forward(self, src, src_mask):
        batch_size = src.shape[0]  # 문장의 개수
        src_len = src.shape[1]  # 가장 긴 문장의 워드 개수
        #pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  ##
        ## arrange(0~가장 긴 문장의 마지막 단어까지): 해당 크기의 1-d tensor 반환 (src_len+1,)
        ## unsqueeze(0): 첫번째 차원(index 0)에 1인 차원을 추가 (1, src_len+1)
        ## repeat(각각의 문장마다 반복): 해당 tensor를 batch_size*1의 크기로 반복
        # input = embedding + positional encoding
        #src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        src = self.tok_embedding(src) * self.scale
        src = self.dropout(self.pos_encoding(src))
        ## 실제 입력값 : 임베딩 값 + PE 값
        # src : [batch_size, src_len]
        # → src : [batch_size, src_len, hidden_dim]

        # Feed forward through all encoder layer sequentially
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

'''