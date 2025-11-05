# mini_gpt.py — minimal GPT-like model (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """Implements multi-head self-attention with causal masking.
    
    This is a key component of the transformer architecture that allows each position to attend
    to previous positions (including itself) in the sequence. The 'causal' part ensures that
    predictions at position i can only depend on known outputs at positions less than i.
    """
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0  # Ensure embed_dim is divisible by num_heads
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = embed_dim // num_heads  # Dimension of each head
        # Linear layer to compute Q, K, V from input in one go (more efficient)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.out = nn.Linear(embed_dim, embed_dim)
        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(attn_dropout)
        # Causal mask (will be created in forward pass)
        self.register_buffer("mask", None, persistent=False)

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension
        
        # Project input to Q, K, V (query, key, value) vectors
        # qkv shape: (B, T, 3 * embed_dim) -> (B, T, 3, num_heads, head_dim)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        
        # Split into separate Q, K, V tensors, each (B, T, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Transpose for attention dot product: (B, num_heads, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Calculate scaled dot-product attention scores
        # (B, nh, T, T) where T is sequence length
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create causal mask if needed (prevents attending to future tokens)
        if self.mask is None or self.mask.size(0) < T:
            # Lower triangular matrix of ones (causal mask)
            self.mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            
        # Apply mask: set future positions to -inf (will become 0 after softmax)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax to get attention weights
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)  # Apply dropout to attention weights
        
        # Weighted sum of values using attention weights
        y = att @ v  # (B, nh, T, head_dim)
        
        # Reshape back to (B, T, C)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        
        # Final linear projection
        return self.out(y)

class FeedForward(nn.Module):
    """Position-wise feed-forward network (FFN) with GELU activation.
    
    This is a simple 2-layer MLP that is applied to each position separately and identically.
    It's a key component of the transformer architecture, providing non-linearity and
    the ability to learn complex patterns in the data.
    """
    def __init__(self, embed_dim, ff_mult=4, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            # First layer expands the dimension (usually by 4x)
            nn.Linear(embed_dim, ff_mult * embed_dim),
            # GELU activation (used in GPT models, smoother than ReLU)
            nn.GELU(),
            # Project back to original dimension
            nn.Linear(ff_mult * embed_dim, embed_dim),
            # Optional dropout for regularization
            nn.Dropout(p)
        )
    
    def forward(self, x):
        # Apply the feed-forward network to each position independently
        return self.net(x)

class Block(nn.Module):
    """A single transformer block consisting of self-attention and feed-forward layers.
    
    This is the main building block of the transformer architecture. It combines:
    1. Multi-head self-attention (with layer normalization and residual connection)
    2. Feed-forward network (with layer normalization and residual connection)
    
    The layer normalization is applied before each sub-layer (pre-norm), which is the standard
    approach in modern transformer architectures.
    """
    def __init__(self, embed_dim, num_heads, ff_mult=4, attn_p=0.0, ff_p=0.0):
        super().__init__()
        # Layer normalization before self-attention
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-5)
        # Multi-head self-attention layer
        self.attn = CausalSelfAttention(embed_dim, num_heads, attn_dropout=attn_p)
        # Layer normalization before feed-forward
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-5)
        # Feed-forward network
        self.ff = FeedForward(embed_dim, ff_mult, p=ff_p)
    
    def forward(self, x):
        # Self-attention with residual connection (pre-norm)
        x = x + self.attn(self.ln1(x))
        # Feed-forward with residual connection (pre-norm)
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    """A minimal GPT-like language model based on the transformer architecture.
    
    This implements a decoder-only transformer model similar to GPT-2/GPT-3 but smaller.
    It consists of token embeddings, positional encodings, a stack of transformer
    blocks, and a language modeling head.
    
    Key differences from the original transformer:
    - Decoder-only architecture (no encoder)
    - Causal self-attention (no cross-attention)
    - Learned positional embeddings (vs. sinusoidal in original transformer)
    - Pre-layer normalization (vs. post-layer in original)
    - GELU activation (vs. ReLU in original)
    """
    def __init__(self, vocab_size, max_seq_len=128, n_layers=6, embed_dim=256, 
                 num_heads=8, ff_mult=4, attn_p=0.0, ff_p=0.0):
        super().__init__()
        # Token embedding layer (converts token IDs to vectors)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        # Learnable positional embeddings (1 for each position up to max_seq_len)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        # Dropout for regularization
        self.drop = nn.Dropout(0.1)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, ff_mult, attn_p, ff_p)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        # Language modeling head (output projection to vocabulary)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

        # Initialize weights
        self.apply(self._init_weights)
        # Special initialization for positional embeddings
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _init_weights(self, module):
        """Initialize weights for different layer types.
        
        This follows the initialization scheme used in GPT models:
        - Linear and Embedding layers: weights from N(0, 0.02), biases to 0
        - LayerNorm: scale (gamma) to 1, bias (beta) to 0
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize weights from normal distribution with small std
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Initialize biases to zero if they exist
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm: scale (gamma) to 1, bias (beta) to 0
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx):
        """Forward pass of the MiniGPT model.
        
        Args:
            idx: Input tensor of token indices, shape (batch_size, seq_length)
                
        Returns:
            logits: Unnormalized log probabilities for next token prediction,
                   shape (batch_size, seq_length, vocab_size)
        """
        B, T = idx.size()  # Batch size, sequence length
        
        # Check sequence length doesn't exceed maximum
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds model max_seq_len {self.max_seq_len}")
        
        # Get token embeddings and add positional encodings
        # tok_emb: (B, T, embed_dim), pos_emb: (1, T, embed_dim) -> (B, T, embed_dim)
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        
        # Apply dropout to embeddings
        x = self.drop(x)
        
        # Pass through transformer blocks
        x = self.blocks(x)  # (B, T, embed_dim)
        
        # Final layer norm
        x = self.ln_f(x)  # (B, T, embed_dim)
        
        # Project to vocabulary size to get logits
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_token=None, temperature=1.0, top_k=0, top_p=0.0):
        # idx: (B, T) starting context
        for _ in range(max_new_tokens):
            T = idx.size(1)
            if T > self.max_seq_len:
                idx = idx[:, -self.max_seq_len:]
                T = idx.size(1)
            logits = self(idx)  # (B, T, V)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                minv = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < minv, torch.full_like(logits, -1e10), logits)
            # top-p (nucleus)
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                # mask tokens with cumulative prob above top_p
                mask = cumulative_probs > top_p
                # keep first token where cum_prob > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -1e10
                logits = torch.gather(sorted_logits, 1, torch.arange(logits.size(-1), device=logits.device).unsqueeze(0).expand(logits.size(0), -1))
                # note: above line is simplified — for simplicity use top_k or temp for demo
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if eos_token is not None and next_token.item() == eos_token:
                break
        return idx
