import torch
from final_model import TinyGPT
import matplotlib.pyplot as plt
import torch.nn.functional as F

# reading dataset
with open("./data/TinyStories-train.txt", "r") as f:  # small text dataset
    text = f.read()

# Character-level tokenizer
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)
tokens = [stoi[c] for c in text]
seq_length = 64
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

print("vocab size: ", vocab_size)
print("number of tokens: ", len(tokens))

# batching
def get_batch(tokens, batch_size, seq_length):
    ix = torch.randint(0, len(tokens) - seq_length - 1, (batch_size,))
    x = torch.stack([torch.tensor(tokens[i:i+seq_length]) for i in ix])
    y = torch.stack([torch.tensor(tokens[i+1:i+seq_length+1]) for i in ix])
    return x.to(device), y.to(device)

# hyperparameters
embed_dim = 64
num_heads = 4
num_layers = 4
max_seq_len = 64
epochs = 300
learning_rate = 0.003

# training loop
model = TinyGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []

for step in range(epochs):
    x, y = get_batch(tokens, batch_size, seq_length)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()s
    
    losses.append(loss.item())

plt.figure()
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Cross-Entropy Loss")
plt.title(f"Training Loss: embed_dim={embed_dim}, heads={num_heads}")
plt.show()