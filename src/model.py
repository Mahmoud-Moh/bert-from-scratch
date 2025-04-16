import torch.nn as nn
import torch 
import math
class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(TokenEmbedding, self).__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)


class LayerNorm(nn.Module):
    def __init__(self):
        #I will assume that my input size will be B * max_len * d_model
        super(LayerNorm, self).__init__()
    
    def forward(self, x):
        means = torch.mean(x, dim=2).unsqueeze(2)
        var = torch.var(x, dim=2).unsqueeze(2)
        x = (x - means) / torch.sqrt(var)
        return x 

class ScaledProductAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(ScaledProductAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
    
    def forward(self, q, k, v):
        #size of q, k, v is batch_size * n_heads * max_len * d_h
        # we wanna output something of size batch_size * n_heads * max_len * max_len 
        d_h = self.d_model // self.n_heads
        k_t = k.transpose(2, 3)
        attn = q @ k_t
        attn = attn // math.sqrt(d_h)
        attn = nn.Softmax(attn, dim=-1)
        v = attn @ v 
        return v, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn = ScaledProductAttention(self.d_model, self.n_heads)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_w = nn.Linear(d_model, d_model)

    
    def forward(self, q, k, v):
        #First we must understand that the size of q, k, v is 
        #B * n_heads * max_len * (d_model//n_heads)
        #So q is B * H * L * K 
        # and k is B * H * L * K 
        # and we wanna output something of size B * L * K I mean ( B * max_len * (d_model//n_heads) )
        #so In order to do that we must transpose 
        batch_size, max_len, d_model = q.shape 
        #We wanna reshape it to batch_size, max_len, n_heads, d_h
        d_h = d_model // self.n_heads
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = q.view(batch_size, max_len, self.n_heads, d_h), k.view(batch_size, max_len, self.n_heads, d_h), v.view(batch_size, max_len, self.n_heads, d_h)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        #now q, k, v have the shapes batch_size x n_heads x max_len x d_h
        new_v, attn = self.attn(q, k, v)
        #the returned new_v has the shape batch_size * n_heads * max_len * d_h
        #just stack this stuff contiguous and reshape it 
        v = v.contiguous().view(batch_size, max_len, d_model)
        out = self.w_w(v)
        return  out, attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden):
        super(EncoderLayer, self).__init__()
        self.layer_norm = LayerNorm()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardLayer(d_model=d_model, ffn_hidden=ffn_hidden)
        
    def forward(self, x):
        #we expect x to be of shape B * max_len * d_model 
        _x = x 
        x, attn = self.attn(x, x, x)
        x = self.layer_norm(x + _x)
        _x = x 
        x = self.ffn(x)
        x = self.layer_norm(x + _x)
        return x 


class ClsLinerClassifier(nn.Module):
    def __init__(self, d_model, n_classes):
        super(ClsLinerClassifier, self).__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, x):
        # we expect the input x to be the first vector of the max_len seq returend from encoder layer 
        # more formally the output of encoder layer is of size B * max_len * d_model
        # we don't want all of the sequence (max_len) we want only the first token the cls token 
        # in which all info was accumulated (due to the lack of masking)
        # so we expect size B * d_model 
        out = self.classifier(x)
        out = nn.Softmax(out)
        # we expect out to be of size B * number of classes
        cls = torch.argmax(out, dim=-1)
        return cls, out


class BertEncoder(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, n_layers, max_len, n_classes, device):
        self.tok_emb = TokenEmbedding(max_len, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)
        self.encoder_layers = [EncoderLayer(d_model=d_model, n_heads=n_heads, ffn_hidden=ffn_hidden) for _ in range(n_classes)]
        self.classifier = ClsLinerClassifier(d_model=d_model, n_classes=n_classes)
    
    def forward(self, x):
        # we expect x to be tensor of size B * max_len 
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb
        for layer in self.encoder_layers: 
            x = layer(x)
        #at this stage x size is B * max_len * d_model
        #extract the cls only 
        cls_token = x[:,0,:]
        #shape of cls_token is B * d_model
        cls, out = self.classifier(x)
        return cls, out

#from https://github.com/hyunwoongko/transformer.git
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512] 

class FeedForwardLayer(nn.Module):
    #I will assume that the input to it is of size B * max_len * d_model 
    # and we want to output from it something of size B * max_len * d_model too? 
    def __init__(self, d_model, ffn_hidden):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x