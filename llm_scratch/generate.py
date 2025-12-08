import torch
from final_model import TinyGPT
import torch.nn.functional as F

# Load the saved model
checkpoint = torch.load("trained_model.pt")

# Initialize model with saved parameters
model = TinyGPT(
    vocab_size=checkpoint['vocab_size'],
    embed_dim=checkpoint['embed_dim'],
    num_heads=checkpoint['num_heads'],
    num_layers=checkpoint['num_layers'],
    max_seq_len=checkpoint['max_seq_len']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get the character mappings
stoi = checkpoint['stoi']
itos = checkpoint['itos']

# Function to generate text
def generate_text(prompt, max_length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # Convert prompt to tokens
        tokens = [stoi.get(c, 0) for c in prompt]
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Generate text
        for _ in range(max_length):
            # Get model predictions
            logits = model(input_tensor)[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            tokens.append(next_token.item())
            
            # Prepare for next iteration
            input_tensor = torch.tensor([tokens[-64:]], dtype=torch.long)
            
            # Stop if we hit the end of sequence token (if any)
            if tokens[-1] == 0:  # Assuming 0 is padding/end token
                break
                
    # Convert tokens back to text
    return ''.join([itos[i] for i in tokens])

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt, max_length=200, temperature=0.4)
print(f"Prompt: {prompt}")
print("Generated text:")
print(generated_text)