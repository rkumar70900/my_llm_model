import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_emb(x)        # (B, T, D)
        pos = self.pos_emb[:, :T, :]   # (1, T, D)
        return tok + pos               # (B, T, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.W_q(x)  # (B,T,D)
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,T,hd)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        att = att.masked_fill(mask, float('-inf'))

        att = F.softmax(att, dim=-1)

        out = att @ V  # (B,H,T,hd)
        out = out.transpose(1,2).reshape(B,T,D)  # back to (B,T,D)

        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        ff_dim = 4 * embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=512):
        super().__init__()
        self.embed = Embeddings(vocab_size, embed_dim, max_seq_len)
        self.blocks = nn.Sequential(*[
            DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.out(x)
        return logits