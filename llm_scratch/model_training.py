import torch
from final_model import TinyGPT
import matplotlib.pyplot as plt
import torch.nn.functional as F
import psutil
from pynvml import *
import nvidia_smi
import os
import uuid
from datetime import datetime
import re
import csv

def generate_run_id():
    """Generate a unique run ID with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{unique_id}"

# Generate run ID at the start
run_id = generate_run_id()
print(f"Starting training run: {run_id}")

# Create results directory if it doesn't exist
os.makedirs("training_runs", exist_ok=True)


# reading dataset
try:
    with open("./data/TinyStories-train.txt", "r", encoding='utf-8', errors='replace') as f:  # small text dataset
        text = f.read()
except FileNotFoundError:
    print("Error: The file './data/TinyStories-train.txt' was not found.")
    print("Please ensure the file exists in the 'data' directory.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit(1)


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # Convert to MB
        return allocated, reserved
    return 0, 0

def get_cpu_utilization():
    return psutil.cpu_percent()

# Character-level tokenizer
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)
tokens = [stoi[c] for c in text]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

# print("vocab size: ", vocab_size)
# print("number of tokens: ", len(tokens))

# word level tokenizer
# def simple_tokenize(text):
#     # Split on whitespace and keep only words with length > 1
#     words = re.findall(r'\b\w+\b', text.lower())
#     return words

# # Tokenize the text into words
# words = simple_tokenize(text)
# vocab = sorted(list(set(words)))
# stoi = {word: i for i, word in enumerate(vocab)}
# itos = {i: word for i, word in enumerate(vocab)}
# vocab_size = len(vocab)

# # Convert text to token IDs
# tokens = [stoi[word] for word in words]

def tokenize_with_tiktoken(text, model_name="gpt2"):
    """
    Tokenize text using tiktoken's tokenizer.
    
    Args:
        text (str): Input text to tokenize
        model_name (str): Name of the tokenizer model to use (default: "gpt2")
        
    Returns:
        tuple: (token_ids, tokenizer)
            - token_ids: List of token IDs
            - tokenizer: The tiktoken tokenizer instance
    """
    import tiktoken
    tokenizer = tiktoken.get_encoding(model_name)
    token_ids = tokenizer.encode_ordinary(text)
    return token_ids, tokenizer

# Example usage:
# tokens, tokenizer = tokenize_with_tiktoken(text)
# vocab_size = tokenizer.n_vocab

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

print("vocab size: ", vocab_size)
print("number of tokens: ", len(tokens))

# batching
def get_batch(tokens, batch_size, seq_length):
    ix = torch.randint(0, len(tokens) - seq_length - 1, (batch_size,))
    x = torch.stack([torch.tensor(tokens[i:i+seq_length]) for i in ix])
    y = torch.stack([torch.tensor(tokens[i+1:i+seq_length+1]) for i in ix])
    return x.to(device), y.to(device)

# hyperparameters
seq_length = 256
batch_size = 16
embed_dim = 1024
num_heads = 16
num_layers = 12
max_seq_len = 512
epochs = 500
learning_rate = 0.003

# Create a dictionary to store run information
run_info = {
    'run_id': run_id,
    'tokenizer_type': 'character',  # or 'char' if you switch back
    'embed_dim': embed_dim,
    'num_heads': num_heads,
    'num_layers': num_layers,
    'batch_size': batch_size,
    'seq_length': seq_length,
    'max_seq_len': max_seq_len,
    'learning_rate': learning_rate,
    'epoch': 0,  # Will be updated each epoch
    'loss': None,  # Will be updated each epoch
    'cpu_usage': None,  # Will be updated each epoch
    'gpu_usage': None  # Will be updated each epoch
}


# training loop
model = TinyGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
cpu_usage = []
gpu_memory_percent = []
max_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Convert to MB

# Save to CSV
csv_file = 'training_runs.csv'
file_exists = os.path.isfile(csv_file)

import time

# Training loop with progress tracking
model.train()
start_time = time.time()

for epoch in range(epochs):
    # Get batch
    x, y = get_batch(tokens, batch_size, seq_length)
    
    # Forward pass
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Store loss
    losses.append(loss.item())
    
    # Print training progress
    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
        # Calculate examples per second
        elapsed = time.time() - start_time
        examples_per_sec = (epoch + 1) * batch_size / elapsed if elapsed > 0 else 0
        
        # Print progress
        print(f"Epoch [{epoch + 1:4d}/{epochs}] | "
              f"Loss: {loss.item():.4f} | "
              f"Examples/s: {examples_per_sec:.1f} | "
              f"Time elapsed: {elapsed:.1f}s")
        
        # Print CPU and GPU usage
        cpu_usage.append(get_cpu_utilization())
        allocated, reserved = get_gpu_memory()
        gpu_usage_percent = (allocated / max_gpu_memory) * 100
        gpu_memory_percent.append(gpu_usage_percent)
        print(f"CPU Usage: {cpu_usage[-1]:.2f}%")
        print(f"GPU Memory Usage: {gpu_memory_percent[-1]:.2f}%")
        run_info.update({
            'epoch': epoch + 1,  # 1-based epoch number
            'loss': losses[-1],
            'cpu_usage': cpu_usage[-1],
            'gpu_usage': gpu_memory_percent[-1]
        })
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=run_info.keys())
            if not file_exists:
                writer.writeheader()  # Write header only if file doesn't exist
            writer.writerow(run_info)

# plt.figure()
# plt.plot(losses)
# plt.xlabel("Step")
# plt.ylabel("Cross-Entropy Loss")
# plt.title(f"Training Loss: embed_dim={embed_dim}, heads={num_heads}")
# plt.show()

# plt.figure()
# plt.plot(cpu_usage)
# plt.xlabel("Step")
# plt.ylabel("CPU Usage")
# plt.title(f"CPU Usage: embed_dim={embed_dim}, heads={num_heads}")
# plt.show()

# plt.figure()
# plt.plot(gpu_memory_percent)
# plt.xlabel("Step")
# plt.ylabel("GPU Memory Usage (%)")
# plt.title(f"GPU Memory Usage: embed_dim={embed_dim}, heads={num_heads}")
# plt.show()

# model_save_path = "trained_model.pt"
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'losses': losses,
#     'epoch': epochs,
#     'vocab_size': vocab_size,
#     'embed_dim': embed_dim,
#     'num_heads': num_heads,
#     'num_layers': num_layers,
#     'max_seq_len': max_seq_len,
#     'stoi': stoi,
#     'itos': itos
# }, model_save_path)

# print(f"Model saved to {model_save_path}")