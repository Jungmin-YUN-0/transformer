import torch.nn as nn
import torch
import torch.nn.functional as F
from model.MultiHeadAttention import MultiHeadAttention_CF
from model.PositionalEncoding import PositionalEncoding
       

class Classification_block(nn.Module):
    def __init__(self, HIDDEN_DIM, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device):
        super().__init__()
        heads = ENC_HEADS 
        ffnn_hidden_size = ENC_PF_DIM 
        dmodel = HIDDEN_DIM

        self.attention = MultiHeadAttention_CF(dmodel, heads, ENC_DROPOUT, device)
        self.layer_norm1 = nn.LayerNorm(dmodel)
        self.layer_norm2 = nn.LayerNorm(dmodel)
        
        self.ffnn = nn.Sequential(
                nn.Linear(dmodel, ffnn_hidden_size),
                nn.ReLU(),
                nn.Dropout(ENC_DROPOUT),
                nn.Linear(ffnn_hidden_size, dmodel))

    def forward(self, inputs):
        """Forward propagate through the Transformer block.
        Parameters
        ----------
        inputs: torch.Tensor
            Batch of embeddings.
        Returns
        -------
        torch.Tensor
            Output of the Transformer block (batch_size, seq_length, dmodel)
        """
        # Inputs shape (batch_size, seq_length, embedding_dim = dmodel)
        output = inputs + self.attention(inputs)            
        output = self.layer_norm1(output)            
        output = output + self.ffnn(output)            
        output = self.layer_norm2(output)

        # Output shape (batch_size, seq_length, dmodel)
        return output


class CF_Transformer(nn.Module):
    """Implementation of the Transformer model for classification.
    
    Parameters
    ----------
    vocab_size: int
        The size of the vocabulary.
    dmodel: int
        Dimensionality of the embedding vector.
    max_len: int
        The maximum expected sequence length.
    padding_idx: int, optional (default=0)
        Index of the padding token in the vocabulary and word embedding.
    n_layers: int, optional (default=4)
        Number of the stacked Transformer blocks.    
    ffnn_hidden_size: int, optonal (default=dmodel * 4)
        Position-Wise-Feed-Forward Neural Network hidden size.
    heads: int, optional (default=8)
        Number of the self-attention operations to conduct in parallel.
    pooling: str, optional (default='max')
        Specify the type of pooling to use. Available options: 'max' or 'avg'.
    dropout: float, optional (default=0.2)
        Probability of an element of the tensor to be zeroed.
    """
    
    def __init__(self, enc_voc_size, dec_voc_size, SRC_PAD_IDX, dmodel, n_layers, n_head, ffnn_hidden, drop_prob, device, n_position=256):
        
        super().__init__()
        self.tok_embedding = nn.Embedding(enc_voc_size, dmodel, padding_idx=SRC_PAD_IDX)
        self.pos_encoding = PositionalEncoding(dmodel, n_position=n_position)
        self.tnf_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.tnf_blocks.append(Classification_block(dmodel, n_head, ffnn_hidden, drop_prob, device))
        self.tnf_blocks = nn.Sequential(*self.tnf_blocks)
        self.dropout = nn.Dropout(drop_prob)            
        self.linear = nn.Linear(dmodel, dec_voc_size)
        self.scale = torch.sqrt(torch.FloatTensor([dmodel])).to(device)

    def forward(self, inputs):
        """Forward propagate through the Transformer.
        Parameters ---
        inputs: torch.Tensor
            Batch of input sequences.
        input_lengths: torch.LongTensor
            Batch containing sequences lengths.
            
        Returns ---
        torch.Tensor
            Logarithm of softmaxed class tensor.
        """
        self.batch_size = inputs.size(0)

        output = self.tok_embedding(inputs)
        output = output*self.scale
        output = self.dropout(self.pos_encoding(output))
        output = self.tnf_blocks(output)
        # Output dimensions (batch_size, seq_length, dmodel)
        
        ## opt1 max pooling
        # Permute to the shape (batch_size, dmodel, seq_length)
        # Apply max-pooling, output dimensions (batch_size, dmodel)
        output = F.adaptive_max_pool1d(output.permute(0,2,1), (1,)).view(self.batch_size,-1)
        ## opt2 avg pooling
        # Sum along the batch axis and divide by the corresponding lengths (FloatTensor)
        # Output shape: (batch_size, dmodel)
        #output = torch.sum(output, dim=1) / input_lengths.view(-1,1).type(torch.FloatTensor) 
        
        #seq_logit = self.linear(output)
        #output = seq_logit.view(-1, seq_logit.size(2))

        output = self.linear(output)
        
        return F.log_softmax(output, dim=-1)