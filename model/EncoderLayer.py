import torch.nn as nn

from model import MultiHeadAttention, FeedForward


# encoder layer

## multihead attention + feed-forward neural network
class EncoderLayer(nn.Module):
    # 입력dim = 출력dim (=> 여러 개 중첩 가능)
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob, device):
        super().__init__()

        self.attention = MultiHeadAttention.MultiHeadAttention(d_model, n_head, drop_prob, device)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward.FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, src, src_mask):
        # 1-1# self-attention (src가 복제되어 query, key, value로 입력)
        _src, _ = self.attention(src, src, src, src_mask)
        # 1-2# dropout, residual connection, layer normalization
        src = self.attn_layer_norm(src + self.dropout(_src))
        # 2-1# feed forward network
        _src = self.ffn(src)
        # 2-2# dropout, residual connection, layer normalization
        src = self.ffn_layer_norm(src + self.dropout(_src))
        return src