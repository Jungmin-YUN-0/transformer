
import torch
import torch.nn as nn
import math

# multi-head attention
## input: key, query, value
### d_model : 하나의 단어에 대한 임베딩 차원

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
class LinformerSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, drop_prob, device, n_position, k):
        super().__init__()
        assert d_model % n_head == 0  # 필요조건

        #self.seq_len = seq_len
        self.k = 128
        self.device = device
        self.n_position = n_position

        self.d_model = d_model  # 각 word에서의 임베딩 차원
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 각 head에서의 임베딩 차원 

        self.weight_q = nn.Linear(d_model, d_model)  # query weight(FC layer) ## Linear(Q)=Q*W_Q
        self.weight_k = nn.Linear(d_model, d_model)  # key weight(FC layer)
        self.weight_v = nn.Linear(d_model, d_model)  # value weight(FC layer)

        def init_(tensor):
            dim = tensor.shape[-1]
            std = 1 / math.sqrt(dim)
            tensor.uniform_(-std, std)
            return tensor

        #self.proj_k = nn.Parameter(init_(torch.zeros(self.n_position, self.k)))
        #self.proj_v = nn.Parameter(init_(torch.zeros(self.n_position, self.k)))
        self.E = nn.Parameter(torch.randn(self.n_position, self.k), requires_grad=True)

        self.fc_concat = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_prob)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, x, mask=None):
        batch_size, n, d = x.shape
        assert n == self.n_position
        
        # query, key, value -> Q, K, V  
        Q, K, V = self.weight_q(x), self.weight_k(x), self.weight_v(x)
        def project_vk_linformer(V,K,E):
            V = torch.einsum('bhjd,jk -> bhkd', V, E)
            K = torch.einsum('bhjd,jk -> bhkd', K, E)
            return V, K
        
        #proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)
        #kv_projs = (self.proj_k, self.proj_v)
        #K, V = map(proj_seq_len, zip((K,V), kv_projs))

        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        K, V = project_vk_linformer(V,K,self.E)
        #=================================================================================================
        #attn_score = torch.einsum('bhnd,bhkd->bhnk', Q,K) * (d_h ** -0.5)
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

    def forward(self, x): #def forward(self, x, mask=None):
        batch_size = x.shape[0]
         
        Q, K, V = self.weight_q(x), self.weight_k(x), self.weight_v(x)   
        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        
        #=================================================================================================
        attn_score = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        ## option_masking 
        #if mask is not None:
        #    mask = mask.unsqueeze(1)
        #    attn_score = attn_score.masked_fill(mask == 0, -1e10)
        attn_dstn = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(self.dropout(attn_dstn), V)
        #=================================================================================================
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        out = self.fc_concat(out)
        
        return out
