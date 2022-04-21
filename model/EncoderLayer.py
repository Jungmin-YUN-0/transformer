import torch.nn as nn
from model.MultiHeadAttention import MultiHeadAttention,LinformerSelfAttention
from model.FeedForward import FeedForward



## multihead attention + feed-forward neural network
class EncoderLayer(nn.Module):
    # 입력dim = 출력dim (=> 여러 개 중첩 가능)
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob, device, attn_option, n_position):
        super().__init__()
        
        self.attn_option = attn_option
        if self.attn_option == 'BASE':
            self.attention = MultiHeadAttention(d_model, n_head, drop_prob, device)
        elif self.attn_option == 'LR':
            self.attention = LinformerSelfAttention(d_model, n_head, drop_prob, device, n_position, k=256) 
        elif self.attn_option == 'CT' :
            self.attention = LinformerSelfAttention(d_model, n_head, drop_prob, device, n_position, k=256, pruning=True) 
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        
    def forward(self, src, src_mask):
        # 1-1# self-attention (src가 복제되어 query, key, value로 입력)
        if self.attn_option == 'BASE':
            _src, _ = self.attention(src, src, src, src_mask)
        elif self.attn_option == 'LR':
            _src, _ = self.attention(src, src_mask)
        # 1-2# dropout, residual connection, layer normalization
        src = src + self.attn_layer_norm(self.dropout1(_src)) #!#

        # 2-1# feed forward network
        _src = self.ffn(src)
        # 2-2# dropout, residual connection, layer normalization
        src = src + self.ffn_layer_norm(self.dropout2(_src))
        
        return src
