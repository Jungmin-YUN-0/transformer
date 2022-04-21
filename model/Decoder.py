import torch.nn as nn
import torch
import copy
from model.DecoderLayer import DecoderLayer
from model.PositionalEncoding import PositionalEncoding


# decoder

## embedding + encoder layer + linear transformation
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, TRG_PAD_IDX, d_model, n_layers, n_head, ffn_hidden, drop_prob, device, attn_option, n_position=512):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(dec_voc_size, d_model, padding_idx=TRG_PAD_IDX)
        # positional embedding 학습 (sinusoidal x) =>>  self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model, n_position=n_position)
        decoder_layer = DecoderLayer(d_model, n_head, ffn_hidden, drop_prob, device, attn_option, n_position)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])
        self.dropout = nn.Dropout(drop_prob)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        x = trg
        x = self.tok_embedding(x)
        x = x * self.scale
        x = self.dropout(self.pos_encoding(x))
        
        for layer in self.layers:
            x, attention = layer(x, enc_src, trg_mask, src_mask)

        return x


