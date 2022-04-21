
import torch
import torch.nn as nn
import math
import tensorflow as tf
from einops import repeat

# multi-head attention 
## input: key, query, value
### d_model : 하나의 단어에 대한 임베딩 차원

#vanilla transformer attention (machine translation task)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, drop_prob, device):
        super().__init__()
        assert d_model % n_head == 0  # 필요조건

        self.d_model = d_model  # 각 word에서의 임베딩 차원
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 각 head에서의 임베딩 차원 

        self.weight_q = nn.Linear(d_model, d_model)  # query weight(FC layer) ## Linear(Q)=Q*W_Q
        self.weight_k = nn.Linear(d_model, d_model)  # key weight(FC layer)
        self.weight_v = nn.Linear(d_model, d_model)  # value weight(FC layer)
        self.fc_concat = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_prob)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # query, key, value -> Q, K, V  
        Q, K, V = self.weight_q(Q), self.weight_k(K), self.weight_v(V)
        
        ## multi-head attention (d_model -> n_head*head_dim 형태로 변환) / (n_head개 마다 각각 head_dim을 갖도록      ##[batch, length, n_head, head_dim] -> [batch, n_head, length, head_dim]
        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        # -1 => 나머지 값들에 의해 결정됨
        
        #=================================================================================================
        ## compute similarity(dot-product)      ##[batch, n_head, query_len, key_len]
        attn_score = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        
        ## option_masking 
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_score = attn_score.masked_fill(mask == 0, -1e10)
            
        attn_dstn = torch.softmax(attn_score, dim=-1)
        # dim=-1 => 현재 input이 tensor(vector였으면 dim 지정 필요 x) last dimension에 대하여 softmax 적용
        # ex. nn.Softmax(dim=-1) torch.randn(2,3) -> 3     
                 
        ## scaled dot-product attention
        out = torch.matmul(self.dropout(attn_dstn), V)
        #=================================================================================================
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        out = self.fc_concat(out)
        #out = self.dropout(out)
        
        return out, attn_score
  
###################################################################################################################
#linformer attention (machine translation task)
class LinformerSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, drop_prob, device, n_position, k):
        super().__init__()
        assert d_model % n_head == 0  # 필요조건

        #self.seq_len = seq_len
        self.k = 256
        self.device = device
        self.n_position = n_position

        self.d_model = d_model  # 각 word에서의 임베딩 차원
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 각 head에서의 임베딩 차원 

        self.weight_q = nn.Linear(d_model, d_model)  # query weight(FC layer) ## Linear(Q)=Q*W_Q
        self.weight_k = nn.Linear(d_model, d_model)  # key weight(FC layer)
        self.weight_v = nn.Linear(d_model, d_model)  # value weight(FC layer)

        self.E = nn.Parameter(torch.randn(self.k, self.n_position), requires_grad=True)
        self.F = nn.Parameter(torch.randn(self.k, self.n_position), requires_grad=True)
     
        self.fc_concat = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_prob)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, x, mask=None):
        batch_size, n, d = x.shape
        assert n == self.n_position
        
        # query, key, value -> Q, K, V  
        Q, K, V = self.weight_q(x), self.weight_k(x), self.weight_v(x)
  
        ##[batch, length, n_head, head_dim] -> [batch, n_head, length, head_dim]
        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        
        K = torch.einsum('kn, bhnd -> bhkd', self.E, K)
        V = torch.einsum('kn, bhnd -> bhkd', self.F, V)
        #=================================================================================================
        attn_score = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            #[batch,1,1,n] -> [batch,1,1,k]로 줄여줘야하므로 (why? linformer에서는 attn_score가 n->k가 됐으므로)
            # 근데 mask는 어차피 fixed된 값들이니까, 그냥 indexing해서 필요한 부분까지만 가져오고자 함
            mask = mask[:, :, :, :self.k]
            attn_score = attn_score.masked_fill(mask == 0, -1e10)
            
        attn_dstn = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(self.dropout(attn_dstn), V)
        #=================================================================================================
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        out = self.fc_concat(out)
        
        return out, attn_score


###################################################################################################################
#vanilla transformer attention (classification task)
class MultiHeadAttention_CF(nn.Module):
    def __init__(self, d_model, n_head, drop_prob, device):
        super().__init__()
        assert d_model % n_head == 0  # 필요조건

        self.d_model = d_model  # 각 word에서의 임베딩 차원
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 각 head에서의 임베딩 차원 

        self.weight_q = nn.Linear(d_model, d_model)  # query weight(FC layer) ## Linear(Q)=Q*W_Q
        self.weight_k = nn.Linear(d_model, d_model)  # key weight(FC layer)
        self.weight_v = nn.Linear(d_model, d_model)  # value weight(FC layer)
        self.fc_concat = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_prob)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        
        Q, K, V = self.weight_q(x), self.weight_k(x), self.weight_v(x)
        
        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        #=================================================================================================
        attn_score = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        attn_dstn = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(self.dropout(attn_dstn), V)
        #=================================================================================================
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        out = self.fc_concat(out)
        
        return out

############################################################################################################
# linformer attention (classification task)
# if pruning=True -> core_token attention까지 mixture
class LinformerSelfAttention_CF(nn.Module):
    def __init__(self, d_model, n_head, drop_prob, device, n_position, k, pruning=False):
        super().__init__()
        assert d_model % n_head == 0  # 필요조건
        
        self.pruning = pruning

        self.k=256
        self.n_position = n_position

        self.d_model = d_model  # 각 word에서의 임베딩 차원
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 각 head에서의 임베딩 차원 

        self.weight_q = nn.Linear(d_model, d_model)  # query weight(FC layer) ## Linear(Q)=Q*W_Q
        self.weight_k = nn.Linear(d_model, d_model)  # key weight(FC layer)
        self.weight_v = nn.Linear(d_model, d_model)  # value weight(FC layer)
        self.E = nn.Parameter(torch.randn(self.k, self.n_position), requires_grad=True)
        self.F = nn.Parameter(torch.randn(self.k, self.n_position), requires_grad=True)

        self.fc_concat = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_prob)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, x, topk_indices=None): #def forward(self, x, mask=None):
        batch_size, n, d = x.shape
        assert n == self.n_position
        Q, K, V = self.weight_q(x), self.weight_k(x), self.weight_v(x)
           
        Q_l = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K_l = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V_l = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        K_l = torch.einsum('kn,bhnd -> bhkd', self.E, K_l)
        V_l = torch.einsum('kn,bhnd -> bhkd', self.F, V_l)

        if self.pruning == True:
            Q_p = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            K_p = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            V_p = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            
            if topk_indices is not None:
                K_p = torch.gather(K_p, -2, repeat(self.topk_indices, 'b h r -> b h r d ', d=K_p.shape[-1]))          
                V_p = torch.gather(V_p, -2, repeat(self.topk_indices, 'b h r -> b h r d ', d=V_p.shape[-1]))   
        #-------------------------------------------------------------------------------------------------
        attn_score_l = torch.matmul(Q_l, K_l.permute(0,1,3,2)) / self.scale
        attn_dstn_l = torch.softmax(attn_score_l, dim=-1)
        out_l = torch.matmul(self.dropout(attn_dstn_l), V_l)
        #-------------------------------------------------------------------------------------------------
        out_l = out_l.permute(0,2,1,3).contiguous()
        out_l = out_l.view(batch_size, -1, self.d_model)
        out_l = self.fc_concat(out_l)
        #=================================================================================================
        if self.pruning == True:
            attn_score_p = torch.matmul(Q_p, K_p.permute(0,1,3,2)) / self.scale
            attn_dstn_p = torch.softmax(attn_score_p, dim=-1)

            importance_score = torch.mean(attn_dstn_l, dim=2) #column mean of attn_dstn

            if topk_indices is not None:
                importance_score = torch.mean(attn_dstn_p, dim=2) #column mean of attn_dstn

            r = round(0.5**(1/6), 1) #round(0.5**(1/n_layer(=6)), 1) #최종적으로 전체 token의 절반 이상은 남기고자 함(token pruning할때 0.5 이하로는 성능이 안좋았다는 논문 결과가 있었음..)
            r = int(n*r)
            
            topk_indices = torch.topk(importance_score, k=r, dim=-1)[1]

            out_p = torch.matmul(self.dropout(attn_dstn_p), V_p)
            out_p = out_p.permute(0,2,1,3).contiguous()
            out_p = out_p.view(batch_size, -1, self.d_model)
            out_p = self.fc_concat(out_p)
                        
            out = out_l+out_p

        if self.pruning == True:
            return out, topk_indices
        else:
            return out_l
