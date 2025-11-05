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
