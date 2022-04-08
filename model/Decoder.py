import torch.nn as nn
import torch
import copy
from model.DecoderLayer import DecoderLayer
from model.PositionalEncoding import PositionalEncoding


# decoder

## embedding + encoder layer + linear transformation
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, TRG_PAD_IDX, d_model, n_layers, n_head, ffn_hidden, drop_prob, device, n_position=512):
        super().__init__()

        self.device = device

        # input dimension → embedding dimension
        self.tok_embedding = nn.Embedding(dec_voc_size, d_model, padding_idx=TRG_PAD_IDX)
        # positional embedding 학습 (sinusoidal x) =>>  self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model, n_position=n_position)
        decoder_layer = DecoderLayer(d_model, n_head, ffn_hidden, drop_prob, device)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])
        #self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, ffn_hidden, drop_prob, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(drop_prob)
        #self.d_model = d_model
        ### self.linear = nn.Linear(d_model, dec_voc_size)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        x = trg
        x = self.tok_embedding(x)
        x = x * self.scale
        #trg *= self.d_model ** 0.5
        x = self.dropout(self.pos_encoding(x))
        
        for layer in self.layers:
            x, attention = layer(x, enc_src, trg_mask, src_mask)
        
        #return trg, attention
        return x
        # linear transformation
        #output = self.linear(trg)
        # return output, attention

'''

## embedding + encoder layer + linear transformation
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, TRG_PAD_IDX, d_model, n_layers, n_head, ffn_hidden, drop_prob, device, max_len=512, n_position=200):
        super().__init__()

        self.device = device

        # input dimenstion → embedding dimension
        self.tok_embedding = nn.Embedding(dec_voc_size, d_model, padding_idx=TRG_PAD_IDX)
        # positional embedding 학습 (sinusoidal x)
        #self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos_embedding = PositionalEncoding.PositionalEncoding(d_model, n_position=n_position)

        self.layers = nn.ModuleList([DecoderLayer.DecoderLayer(d_model, n_head, ffn_hidden, drop_prob, device) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)
        self.dropout = nn.Dropout(drop_prob)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        
        #self.emb = TransformerEmbedding(d_model=d_model, drop_prob=drop_prob, max_len=max_len, vocab_size=dec_voc_size, device=device)
        #self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layers)])
        

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        trg = self.tok_embedding(trg) * self.scale
        trg = self.dropout(self.pos_embedding(trg))
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # linear transformation
        output = self.linear(trg)
        return output, attention
'''