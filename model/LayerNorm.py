import torch.nn as nn
import torch

# layer normalization
## LayerNorm(x + SubLayer(x))
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)  # -1:last dimension
        out = (x - mean) / (std + self.eps)  # zero-mean, unit-variance
        out = self.gamma * out + self.beta  # scale, shift
        return out