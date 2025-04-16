import torch.nn as nn
import torch
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, device):
        super(TokenEmbedding, self).__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
        self.weight.to(device)

class LayerNorm(nn.Module):
    def __init__(self):
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        d_h = self.d_model // self.n_heads
        k_t = k.transpose(2, 3)
        attn = q @ k_t
        #change here, used to do //, that's wrong cuz it eliminates the fractions
        #attn = attn // math.sqrt(d_h)
        attn = attn / math.sqrt(d_h)
        attn = self.softmax(attn)
        v = attn @ v
        return v, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn = ScaledProductAttention(self.d_model, self.n_heads)
        self.w_q = nn.Linear(d_model, d_model).to(device)
        self.w_k = nn.Linear(d_model, d_model).to(device)
        self.w_v = nn.Linear(d_model, d_model).to(device)
        self.w_w = nn.Linear(d_model, d_model).to(device)

    def forward(self, q, k, v):
        batch_size, max_len, d_model = q.shape
        d_h = d_model // self.n_heads
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = q.view(batch_size, max_len, self.n_heads, d_h), k.view(batch_size, max_len, self.n_heads, d_h), v.view(batch_size, max_len, self.n_heads, d_h)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        new_v, attn = self.attn(q, k, v)
        #Change here, new_ve is returned with shape [BatchSize, n_heads, max_len, d_model]
        #Need to traponse it to [BatchSize, max_len, n_heads, d_model]
        #then do the contiguous thing
        new_v = new_v.transpose(1, 2).contiguous().view(batch_size, max_len, d_model)  # Combine heads
        #v = v.contiguous().view(batch_size, max_len, d_model)
        out = self.w_w(new_v)
        return  out, attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, device):
        super(EncoderLayer, self).__init__()
        self.layer_norm = LayerNorm()
        self.attn = MultiHeadAttention(d_model, n_heads, device)
        self.ffn = FeedForwardLayer(d_model=d_model, ffn_hidden=ffn_hidden, device=device)
        self.device = device

    def forward(self, x):
        x.to(self.device)
        _x = x
        x, attn = self.attn(x, x, x)
        x = self.layer_norm(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.layer_norm(x + _x)
        return x

class ClsLinerClassifier(nn.Module):
    def __init__(self, d_model, n_classes, device):
        super(ClsLinerClassifier, self).__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.classifier = nn.Linear(d_model, n_classes).to(device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.classifier(x)
        cls = torch.argmax(out, dim=-1)
        return cls, out

class BertEncoder(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, n_layers, max_len, vocab_size, n_classes, device):
        super(BertEncoder, self).__init__()
        self.tok_emb = TokenEmbedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)
        self.encoder_layers = [EncoderLayer(d_model=d_model, n_heads=n_heads, ffn_hidden=ffn_hidden, device=device) for _ in range(n_layers)]
        self.classifier = ClsLinerClassifier(d_model=d_model, n_classes=n_classes, device=device)
        self.to(device)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb
        for layer in self.encoder_layers:
            x = layer(x)
        cls_token = x[:,0,:]
        cls, out = self.classifier(cls_token)
        return cls, out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.shape
        return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, device):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden).to(device)
        self.linear2 = nn.Linear(ffn_hidden, d_model).to(device)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
