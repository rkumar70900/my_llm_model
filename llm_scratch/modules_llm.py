import torch
import torch.nn.functional as F
import math

# Hyperparameters
vocab_size = 1000
seq_length = 16
embed_dim = 48
num_heads = 3

B = 2
T = 10

# compatability of embedding dimension and number of attention heads
assert embed_dim % num_heads == 0
head_dim = embed_dim // num_heads
ff_dim = 4 * embed_dim

print("Embedding dimension:", embed_dim)
print("Head dimension:", head_dim)

def token_embedding(vocab_size, embed_dim):
    return torch.randn(vocab_size, embed_dim, requires_grad=True) * 0.02

def positional_embedding(seq_length, embed_dim):
    pos_emb = torch.zeros(1, seq_length, embed_dim, requires_grad=True)
    torch.nn.init.normal_(pos_emb, mean=0.0, std=0.02)
    return pos_emb

def embeddings(input_ids, tok_emb_weight, pos_emb):
    tok_emb = F.embedding(input_ids, tok_emb_weight)  # [B, T, D]
    # Add batch dimension to pos_emb and slice to match sequence length
    pos_emb = pos_emb[None, :input_ids.size(1), :]  # [1, T, D] -> [1, T, D]
    return tok_emb + pos_emb

def layernorm(x, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var  = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / ((var + eps).sqrt())

def causal_mask(T):
    return torch.triu(torch.ones(T, T), diagonal=1).bool().unsqueeze(0).unsqueeze(0)

def attention(x, block):
    B, T, _ = x.shape  # Get batch size and sequence length from input
    Q = x @ block["W_q"]
    K = x @ block["W_k"]
    V = x @ block["W_v"]
    Q = Q.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, nh, T, hd]
    K = K.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, nh, T, hd]
    V = V.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, nh, T, hd]
    att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    mask = causal_mask(T).to(x.device)
    att = att.masked_fill(mask, -float('inf'))
    att = F.softmax(att, dim=-1)
    out = att @ V  # [B, nh, T, hd]
    out = out.transpose(1, 2).contiguous().view(B, T, embed_dim)  # [B, T, D]
    out = out @ block["W_o"]
    return out

def feed_forward(x, block):
    ff = F.gelu(x @ block["W_ff1"] + block["b_ff1"])  
    ff = ff @ block["W_ff2"] + block["b_ff2"]
    return ff

def init_block_params():
    return torch.nn.ParameterDict({
        "W_q": torch.nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02),
        "W_k": torch.nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02),
        "W_v": torch.nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02),
        "W_o": torch.nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02),
        "W_ff1": torch.nn.Parameter(torch.randn(embed_dim, ff_dim) * 0.02),
        "b_ff1": torch.nn.Parameter(torch.zeros(ff_dim)),
        "W_ff2": torch.nn.Parameter(torch.randn(ff_dim, embed_dim) * 0.02),
        "b_ff2": torch.nn.Parameter(torch.zeros(embed_dim))
    })

def decoder_block(x, block):
    x_norm = layernorm(x)
    att = attention(x_norm, block)
    x = x + att
    x_norm = layernorm(x)
    ff = feed_forward(x_norm, block)
    x = x + ff
    return x

def main():
    B = 2  # Batch size
    T = 10 # Sequence length
    input_ids = torch.randint(low=0, high=vocab_size, size=(B, T), dtype=torch.long)
    tok_emb_weight = token_embedding(vocab_size, embed_dim)
    pos_emb = positional_embedding(seq_length, embed_dim)
    x = embeddings(input_ids, tok_emb_weight, pos_emb)
    block1 = init_block_params()
    x = decoder_block(x, block1)
    x_norm = layernorm(x)
    W_out = torch.randn(embed_dim, vocab_size) * 0.02
    logits = x_norm @ W_out
    return logits

def get_batch():
    ix = torch.randint(len(tokens) - T - 1, (B,))
    x = torch.stack([tokens[i:i+T] for i in ix])
    y = torch.stack([tokens[i+1:i+T+1] for i in ix])  # next char prediction
    return x, y


vocab = list("abcdefghijklmnopqrstuvwxyz \n")
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(vocab)
seq_length = 16
embed_dim = 48
num_heads = 3
assert embed_dim % num_heads == 0
head_dim = embed_dim // num_heads
ff_dim = 4 * embed_dim
# Training params
lr = 3e-3
epochs = 300
# Toy datasete
data = "hello world\nhello ai\nhello gpt\n"
tokens = torch.tensor([stoi[c] for c in data], dtype=torch.long)

# Initialize parameters with nn.Parameter to ensure they are leaf tensors
tok_emb_weight = torch.nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
pos_emb = torch.nn.Parameter(torch.randn(seq_length, embed_dim) * 0.02)
W_out = torch.nn.Parameter(torch.randn(embed_dim, vocab_size) * 0.02)
block1 = init_block_params()


optimizer = torch.optim.Adam([tok_emb_weight, pos_emb, W_out] +
    [param for param in block1.values()], lr=lr)

for step in range(epochs):
    xb, yb = get_batch()

    x = embeddings(xb, tok_emb_weight, pos_emb)
    x = decoder_block(x, block1)
    x = layernorm(x)
    logits = x @ W_out

    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(step, loss.item())

def generate(start, length=20):
    x = torch.tensor([stoi[c] for c in start], dtype=torch.long).unsqueeze(0)
    for _ in range(length):
        inp = embeddings(x[:, -T:], tok_emb_weight, pos_emb)
        out = decoder_block(inp, block1)
        out = layernorm(out)
        logits = out @ W_out
        next_id = torch.argmax(logits[:, -1, :], dim=-1)
        x = torch.cat([x, next_id.unsqueeze(0)], dim=1)
    return "".join(itos[i.item()] for i in x[0])

print(generate("hello "))



if __name__ == "__main__":
    print(generate("hello "))