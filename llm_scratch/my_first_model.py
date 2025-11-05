import torch
import torch.nn.functional as F
import math

# Hyperparameters
vocab_size = 1000
seq_length = 16
embed_dim = 48
num_heads = 3

# compatability of embedding dimension and number of attention heads
assert embed_dim % num_heads == 0
head_dim = embed_dim // num_heads
ff_dim = 4 * embed_dim

print("Embedding dimension:", embed_dim)
print("Head dimension:", head_dim)

# Simulating tokenizer output
B = 2  # Batch size
T = 10 # Sequence length

# random token IDs generation
torch.manual_seed(1)
input_ids = torch.randint(low=0, high=vocab_size, size=(B, T), dtype=torch.long)
print("Input IDs:", input_ids)

# token embeddings
tok_emb_weight = torch.randn(vocab_size, embed_dim) * 0.02
print("Token Emdeddings: ", tok_emb_weight)
print("Token embeddings weight shape:", tok_emb_weight.shape)

# positional embeddings
pos_emb = torch.zeros(1, seq_length, embed_dim)
torch.nn.init.normal_(pos_emb, mean=0.0, std=0.02)
print("Positional embeddings: ", pos_emb)
print("Positional embeddings shape:", pos_emb.shape)

# Forward pass
# looking up by token id
token_vectors = F.embedding(input_ids, tok_emb_weight)
# print("Token vectors: ", token_vectors)
print("Token vectors shape:", token_vectors.shape)

position_vectors = pos_emb[:, :T, :] # slicing to get only the first T positions (length of the sequences)
position_vectors = position_vectors.expand(B, T, embed_dim)
print("Position vectors shape:", position_vectors.shape)

# adding embeddings with position embeddings
x = token_vectors + position_vectors  # (B, T, D)
print("x shape (B,T,D):", x.shape)
print("example x[0,0,:5] (first token vector first 5 dims):", x[0,0,:5])

# Layer Normalization
def layernorm(x, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var  = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / ((var + eps).sqrt())

# parameters for a decoder block
def init_block_params():
    return {
        "W_q": torch.randn(embed_dim, embed_dim) * 0.02,
        "W_k": torch.randn(embed_dim, embed_dim) * 0.02,
        "W_v": torch.randn(embed_dim, embed_dim) * 0.02,
        "W_o": torch.randn(embed_dim, embed_dim) * 0.02,
        "W_ff1": torch.randn(embed_dim, ff_dim) * 0.02,
        "b_ff1": torch.zeros(ff_dim),
        "W_ff2": torch.randn(ff_dim, embed_dim) * 0.02,
        "b_ff2": torch.zeros(embed_dim)
    }

block1 = init_block_params()
block2 = init_block_params()

# decoder block
mask = torch.triu(torch.ones(T, T), diagonal=1).bool() # causal mask

# attention
x_norm = layernorm(x)

Q = x_norm @ block1["W_q"]
K = x_norm @ block1["W_k"]
V = x_norm @ block1["W_v"]

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

Q = Q.view(B, T, num_heads, head_dim).transpose(1, 2)
K = K.view(B, T, num_heads, head_dim).transpose(1, 2)
V = V.view(B, T, num_heads, head_dim).transpose(1, 2)

print("Q shape (B, H, T, D):", Q.shape)
print("K shape (B, H, T, D):", K.shape)
print("V shape (B, H, T, D):", V.shape)

att = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
print("Attention shape (B, H, T, T):", att.shape)

# applying mask
att = att.masked_fill(mask, -float("inf"))
print("Masked attention shape (B, H, T, T):", att.shape)

att = F.softmax(att, dim=-1)
print("Softmax attention shape (B, H, T, T):", att.shape)

out = att @ V
print("Attention output shape (B, H, T, D):", out.shape)

out = out.transpose(1, 2).reshape(B, T, embed_dim)
print("Attention output shape (B, T, D):", out.shape)

out = out @ block1["W_o"]
print("Output shape (B, T, D):", out.shape)

x = x + out
print("x shape (B, T, D):", x.shape)

print("After Attention:")
x_norm = layernorm(x)
print("x_norm shape (B, T, D):", x_norm.shape)
ff = F.gelu(x_norm @ block1["W_ff1"] + block1["b_ff1"])
ff = ff @ block1["W_ff2"] + block1["b_ff2"]
print("FF shape (B, T, D):", ff.shape)
x = x + ff
print("x shape (B, T, D):", x.shape)

x = layernorm(x)
W_out = torch.randn(embed_dim, vocab_size) * 0.02
print("W_out shape (D, Vocab):", W_out.shape)
logits = x @ W_out
print("Logits shape (B, T, Vocab):", logits.shape)