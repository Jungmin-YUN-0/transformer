import torch.nn as nn
from model import MultiHeadAttention, FeedForward


# decoder layer

##masked_, cross_, feedforward_
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob, device):
        super().__init__()

        self.self_attention = MultiHeadAttention.MultiHeadAttention(d_model, n_head, drop_prob, device)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.enc_dec_attention = MultiHeadAttention.MultiHeadAttention(d_model, n_head, drop_prob, device)
        self.enc_dec_layer_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward.FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # 1-1# masked multi-head attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # 1-2# dropout, residual connection, layer normalization
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # 2-1# cross multi-head attention (encoder-decoder attention)
        _trg, attention = self.enc_dec_attention(trg, enc_src, enc_src, src_mask)
        # 2-2# dropout, residual connection, layer normalization
        trg = self.enc_dec_layer_norm(trg + self.dropout(_trg))
        # 3-1# feed-forward neural network
        _trg = self.ffn(trg)
        # 3-2# dropout, residual connection, layer normalization
        trg = self.ffn_layer_norm(trg + self.dropout(_trg))
        return trg, attention