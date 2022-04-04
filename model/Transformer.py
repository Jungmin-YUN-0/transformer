import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    # source의 <pad> token -> mask=0 설정
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  ##
        ## unsqueeze -> 차원을 늘릴 때 사용
        ### torch.unsqueeze(input, dim) → Tensor. 지정된 위치에 1차원 크기가 삽입된 새 텐서를 반환
        return src_mask

    # target의 미래시점 단어 -> mask=0 설정
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  ##
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask  # element-wise
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # src_mask : [batch_size, 1, 1, src_len]
        trg_mask = self.make_trg_mask(trg)  # trg_mask : [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention


#model.apply(utils.initialize_weights)