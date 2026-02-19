import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_emb(x)        # (B, T, D)
        pos = self.pos_emb[:, :T, :]   # (1, T, D)
        return self.dropout(tok + pos)               # (B, T, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        Q = self.W_q(x)  # (B,T,D)
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,T,hd)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            att = att.masked_fill(mask, float('-inf'))

        att = F.softmax(att, dim=-1)

        out = att @ V  # (B,H,T,hd)
        out = out.transpose(1,2).reshape(B,T,D)  # back to (B,T,D)

        return self.dropout(self.W_o(out))

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, num_groups * self.head_dim)
        self.W_v = nn.Linear(embed_dim, num_groups * self.head_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False, mask=None):
        B, T, D = x.shape

        Q = self.W_q(x)  # (B,T,D)
        K_new = self.W_k(x)
        V_new = self.W_v(x)

        # reshape into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,T,hd)
        K_new = K_new.view(B, T, self.num_groups, self.head_dim).transpose(1, 2)
        V_new = V_new.view(B, T, self.num_groups, self.head_dim).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            past_len = K_cache.size(2)
            K = torch.cat([K_cache, K_new], dim=2)
            V = torch.cat([V_cache, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        repeats = self.num_heads // self.num_groups
        K = K.repeat_interleave(repeats, dim=1)
        V = V.repeat_interleave(repeats, dim=1)

        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            att = att.masked_fill(mask, float('-inf'))

        att = F.softmax(att, dim=-1)

        out = att @ V  # (B,H,T,hd)
        out = out.transpose(1,2).reshape(B,T,D)  # back to (B,T,D)

        out = self.dropout(self.W_o(out))
        new_kv_cache = (K_new, V_new) if use_cache else None

        return out, new_kv_cache

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, self.head_dim)
        self.W_v = nn.Linear(embed_dim, self.head_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False, mask=None):
        B, T, D = x.shape

        Q = self.W_q(x)  # (B,T,D)
        K_new = self.W_k(x)
        V_new = self.W_v(x)

        # reshape into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,T,hd)
        K_new = K_new.view(B, T, 1, self.head_dim).transpose(1, 2)
        V_new = V_new.view(B, T, 1, self.head_dim).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            past_len = K_cache.size(2)
            K = torch.cat([K_cache, K_new], dim=2)
            V = torch.cat([V_cache, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            att = att.masked_fill(mask, float('-inf'))

        att = F.softmax(att, dim=-1)

        out = att @ V  # (B,H,T,hd)
        out = out.transpose(1,2).reshape(B,T,D)  # back to (B,T,D)

        out = self.dropout(self.W_o(out))
        new_kv_cache = (K, V) if use_cache else None

        return out, new_kv_cache

class MultiHeadAttentionKVCache(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False, mask=None):
        B, T, D = x.shape

        Q = self.W_q(x)  # (B,T,D)
        K_new = self.W_k(x)
        V_new = self.W_v(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        K_new = K_new.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V_new = V_new.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            past_len = K_cache.size(2)
            K = torch.cat([K_cache, K_new], dim=2)
            V = torch.cat([V_cache, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        T_total = K.size(2)
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        q_pos = torch.arange(T, device=x.device).unsqueeze(1) + past_len
        k_pos = torch.arange(T_total, device=x.device).unsqueeze(0)
        mask = k_pos > q_pos
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        att = F.softmax(att, dim=-1)

        out = att @ V  # (B,H,T,hd)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B,T,D)

        out = self.dropout(self.W_o(out))
        new_kv_cache = (K, V) if use_cache else None
        return out, new_kv_cache

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        ff_dim = 4 * embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiQueryAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False, mask=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache, mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        if use_cache:
            return x, new_kv_cache
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()  
        self.embed = Embeddings(vocab_size, embed_dim, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)
        self.out.weight = self.embed.token_emb.weight

    def forward(self, x, kv_caches=None, use_cache=False):
        x = self.embed(x)

        if use_cache and kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        new_kv_caches = [] if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            if use_cache:
                layer_cache = kv_caches[layer_idx]
                x, new_layer_cache = block(x, kv_cache=layer_cache, use_cache=True)
                new_kv_caches.append(new_layer_cache)
            else:
                x = block(x)

        x = self.ln(x)
        logits = self.out(x)
        if use_cache:
            return logits, new_kv_caches
        return logits