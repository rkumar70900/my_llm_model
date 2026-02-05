import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponential (cis = cos + i*sin)
    
    Args:
        dim: Dimension of each attention head (head_dim)
        seq_len: Maximum sequence length
        theta: Base for the geometric progression (default 10000)
    
    Theory:
    - RoPE uses different rotation frequencies for different dimension pairs
    - Formula: θ_i = θ^(-2i/d) for i ∈ [0, d/2)
    - Lower dimensions rotate faster, higher dimensions rotate slower
    - This creates a "multi-scale" positional encoding
    
    Returns:
        Complex tensor of shape (seq_len, dim//2) representing rotation angles
    """
    # Create frequency for each dimension pair
    # Example for dim=8: [1.0, 0.1778, 0.0316, 0.0056]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Shape: (dim//2,)
    
    # Create position indices [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # Shape: (seq_len,)
    
    # Outer product: each position × each frequency
    # This gives us the angle for rotating each dimension at each position
    # Example: position 5 at freq 0.1778 → angle = 5 * 0.1778 = 0.889
    freqs = torch.outer(t, freqs).float()
    # Shape: (seq_len, dim//2)
    
    # Convert to complex exponential: e^(iθ) = cos(θ) + i*sin(θ)
    # This represents the rotation for each dimension pair at each position
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # Shape: (seq_len, dim//2) with complex values
    
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape freqs_cis to broadcast correctly with query/key tensors
    
    Args:
        freqs_cis: (seq_len, head_dim//2) complex tensor
        x: (batch, num_heads, seq_len, head_dim) real tensor
    
    Returns:
        Reshaped freqs_cis: (1, 1, seq_len, head_dim//2) for broadcasting
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # Create shape that broadcasts across batch and num_heads dimensions
    # but matches seq_len dimension
    shape = [d if i == ndim - 2 else 1 for i, d in enumerate(x.shape)]
    # For x.shape = (B, H, T, D), this creates [1, 1, T, 1]
    shape[-1] = freqs_cis.shape[-1]  # Adjust last dim to D//2
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to query and key tensors
    
    Args:
        xq: Query tensor (batch, num_heads, seq_len, head_dim)
        xk: Key tensor (batch, num_heads, seq_len, head_dim)
        freqs_cis: Precomputed rotation angles (seq_len, head_dim//2)
    
    Theory:
    - We treat adjacent dimension pairs as complex numbers (real, imag)
    - Multiply by e^(iθ) to rotate in the complex plane
    - This rotation encodes position information
    - The relative distance between positions is preserved through rotation
    
    Mathematical Operation:
    For dimensions [d₀, d₁] treated as complex number z = d₀ + i·d₁:
    Rotation by angle θ: z' = z × e^(iθ) = z × (cos(θ) + i·sin(θ))
    
    Result: z' = (d₀·cos(θ) - d₁·sin(θ)) + i·(d₀·sin(θ) + d₁·cos(θ))
    Which gives us rotated dimensions [d₀', d₁']
    """
    # Convert real tensors to complex by pairing adjacent dimensions
    # [d0, d1, d2, d3, ...] → [(d0 + i*d1), (d2 + i*d3), ...]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Shape: (batch, num_heads, seq_len, head_dim//2) complex
    
    # Reshape freqs_cis for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # Shape: (1, 1, seq_len, head_dim//2)
    
    # Apply rotation: multiply each complex number by its rotation angle
    # This is the core of RoPE - rotating based on position
    # Complex multiplication rotates the vector in 2D plane
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    # Convert back to real: [(a + i*b) * (cos + i*sin)] → rotated (a', b')
    # Then flatten: [(a', b'), (c', d'), ...] → [a', b', c', d', ...]
    # Shape: (batch, num_heads, seq_len, head_dim)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*80)
print("RoPE Position Encoding: Complete Example Walkthrough")
print("="*80)

# ============================================================================
# Step 1: Tokenization
# ============================================================================

print("\n" + "="*80)
print("STEP 1: TOKENIZATION")
print("="*80)

sentence = "The cat sat on the mat"
print(f"\nOriginal sentence: '{sentence}'")

# Simple word-level tokenization for clarity
words = sentence.lower().split()
print(f"\nTokenized words: {words}")

# Create a simple vocabulary
vocab = {
    '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
    'the': 4, 'cat': 5, 'sat': 6, 'on': 7, 'mat': 8
}
print(f"\nVocabulary: {vocab}")

# Convert words to token IDs
token_ids = [vocab.get(word, vocab['<UNK>']) for word in words]
print(f"\nToken IDs: {token_ids}")
print(f"Token mapping:")
for word, token_id in zip(words, token_ids):
    print(f"  '{word}' -> {token_id}")

# ============================================================================
# Step 2: Token Embeddings (WITHOUT position yet)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: TOKEN EMBEDDINGS (Pure Content, No Position)")
print("="*80)

vocab_size = len(vocab)
embed_dim = 8  # Small for visualization
seq_len = len(token_ids)

# Create embedding layer
token_embedding = nn.Embedding(vocab_size, embed_dim)

# Convert to tensor and embed
input_tensor = torch.tensor([token_ids])  # Shape: (1, seq_len)
token_embeds = token_embedding(input_tensor)  # Shape: (1, seq_len, embed_dim)

print(f"\nEmbedding dimension: {embed_dim}")
print(f"Sequence length: {seq_len}")
print(f"\nToken embeddings shape: {token_embeds.shape}")
print(f"  Batch size: {token_embeds.shape[0]}")
print(f"  Sequence length: {token_embeds.shape[1]}")
print(f"  Embedding dimension: {token_embeds.shape[2]}")

print(f"\nFirst token ('the') embedding:")
print(token_embeds[0, 0].detach().numpy())
print(f"\nSecond token ('cat') embedding:")
print(token_embeds[0, 1].detach().numpy())

# ============================================================================
# Step 3: Create Attention Heads (Q, K projections)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: MULTI-HEAD ATTENTION PROJECTIONS")
print("="*80)

num_heads = 2
head_dim = embed_dim // num_heads  # 8 / 2 = 4
print(f"\nNumber of attention heads: {num_heads}")
print(f"Head dimension: {head_dim}")

# Create projection matrices (simplified - normally these would be in attention module)
W_q = nn.Linear(embed_dim, embed_dim, bias=False)
W_k = nn.Linear(embed_dim, embed_dim, bias=False)

# Project token embeddings to Q and K
Q = W_q(token_embeds)  # Shape: (1, seq_len, embed_dim)
K = W_k(token_embeds)  # Shape: (1, seq_len, embed_dim)

print(f"\nQuery (Q) shape: {Q.shape}")
print(f"Key (K) shape: {K.shape}")

# Reshape into multiple heads
# (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
Q = Q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
K = K.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

print(f"\nAfter reshaping for multi-head:")
print(f"  Q shape: {Q.shape}")
print(f"  K shape: {K.shape}")
print(f"  Format: (batch, num_heads, seq_len, head_dim)")

print(f"\nHead 0 queries for all positions:")
for pos in range(seq_len):
    print(f"  Position {pos} ('{words[pos]}'): {Q[0, 0, pos].detach().numpy()}")

# ============================================================================
# Step 4: RoPE - Precompute Rotation Frequencies
# ============================================================================

print("\n" + "="*80)
print("STEP 4: ROPE - PRECOMPUTE ROTATION FREQUENCIES")
print("="*80)

# For head_dim = 4, we have 2 dimension pairs (0,1) and (2,3)
# Each pair gets its own rotation frequency

theta = 10000.0
freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

print(f"\nHead dimension: {head_dim}")
print(f"Number of dimension pairs: {head_dim // 2}")
print(f"\nRotation frequencies (θᵢ) for each dimension pair:")
for i, freq in enumerate(freqs):
    print(f"  Pair {i} (dimensions {2*i}, {2*i+1}): θ = {freq:.6f}")

print(f"\nInterpretation:")
print(f"  Pair 0 (freq={freqs[0]:.6f}): Fast rotation - captures local positions")
print(f"  Pair 1 (freq={freqs[1]:.6f}): Slow rotation - captures distant positions")

# Create position indices
positions = torch.arange(seq_len)
print(f"\nPositions in our sequence: {positions.numpy()}")

# Compute rotation angles for each position and each frequency
# This is the outer product: positions × frequencies
rotation_angles = torch.outer(positions.float(), freqs)

print(f"\nRotation angles (in radians) for each position and dimension pair:")
print(f"Shape: {rotation_angles.shape} (seq_len={seq_len}, num_pairs={head_dim//2})")
print("\n" + " " * 20 + "Dim Pair 0    Dim Pair 1")
for pos in range(seq_len):
    print(f"  Position {pos} ('{words[pos]:>3s}'): ", end="")
    for pair in range(head_dim // 2):
        angle = rotation_angles[pos, pair].item()
        print(f"{angle:10.4f}    ", end="")
    print()

# Convert to complex exponentials (cos + i*sin)
freqs_cis = torch.polar(torch.ones_like(rotation_angles), rotation_angles)

print(f"\nComplex exponentials (e^(iθ) = cos(θ) + i·sin(θ)):")
print(f"Shape: {freqs_cis.shape} (seq_len={seq_len}, num_pairs={head_dim//2})")
print("\nThese represent the actual rotations we'll apply")

# ============================================================================
# Step 5: Apply RoPE Rotation to Q and K
# ============================================================================

print("\n" + "="*80)
print("STEP 5: APPLY ROPE ROTATION")
print("="*80)

print("\n🎯 THIS IS WHERE POSITION INFORMATION IS INJECTED!")

# Store original Q and K for comparison
Q_original = Q.clone()
K_original = K.clone()

print(f"\nBefore RoPE rotation:")
print(f"  Head 0, Position 0 ('{words[0]}') Query: {Q_original[0, 0, 0].detach().numpy()}")
print(f"  Head 0, Position 2 ('{words[2]}') Query: {Q_original[0, 0, 2].detach().numpy()}")

# Apply RoPE
Q_rotated, K_rotated = apply_rotary_emb(Q, K, freqs_cis)

print(f"\nAfter RoPE rotation:")
print(f"  Head 0, Position 0 ('{words[0]}') Query: {Q_rotated[0, 0, 0].detach().numpy()}")
print(f"  Head 0, Position 2 ('{words[2]}') Query: {Q_rotated[0, 0, 2].detach().numpy()}")

# ============================================================================
# Step 6: Visualize What RoPE Does (2D Rotation)
# ============================================================================

print("\n" + "="*80)
print("STEP 6: VISUALIZING ROPE ROTATION (First Dimension Pair)")
print("="*80)

# Extract first dimension pair (dimensions 0 and 1) from head 0
# This allows us to visualize the rotation in 2D

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Before RoPE
ax1 = axes[0]
for pos in range(seq_len):
    vec = Q_original[0, 0, pos, :2].detach().numpy()  # First 2 dimensions
    ax1.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.05, 
             fc=f'C{pos}', ec=f'C{pos}', linewidth=2, label=f"Pos {pos}: '{words[pos]}'")
    # Add text label
    ax1.text(vec[0]*1.2, vec[1]*1.2, f"{pos}", fontsize=12, fontweight='bold')

ax1.set_xlim(-0.8, 0.8)
ax1.set_ylim(-0.8, 0.8)
ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
ax1.set_xlabel('Dimension 0', fontsize=12)
ax1.set_ylabel('Dimension 1', fontsize=12)
ax1.set_title('Before RoPE: No Position Information\n(Vectors overlap/similar)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: After RoPE
ax2 = axes[1]
for pos in range(seq_len):
    vec = Q_rotated[0, 0, pos, :2].detach().numpy()  # First 2 dimensions
    ax2.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.05, 
             fc=f'C{pos}', ec=f'C{pos}', linewidth=2, label=f"Pos {pos}: '{words[pos]}'")
    # Add text label
    ax2.text(vec[0]*1.2, vec[1]*1.2, f"{pos}", fontsize=12, fontweight='bold')
    
    # Draw rotation arc
    if pos > 0:
        angle_rad = rotation_angles[pos, 0].item()
        angle_deg = np.degrees(angle_rad)
        arc = plt.Circle((0, 0), 0.3, fill=False, color=f'C{pos}', linewidth=1, 
                        linestyle='--', alpha=0.5)
        ax2.add_patch(arc)

ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(-0.8, 0.8)
ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
ax2.set_xlabel('Dimension 0', fontsize=12)
ax2.set_ylabel('Dimension 1', fontsize=12)
ax2.set_title('After RoPE: Position Encoded via Rotation\n(Each position uniquely rotated)', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('./rope_rotation_visualization.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved visualization: rope_rotation_visualization.png")

print("\nWhat you see in the plot:")
print("  • Left: Vectors before RoPE (random directions, no position info)")
print("  • Right: Vectors after RoPE (systematically rotated by position)")
print("  • Each position gets rotated by a unique angle")
print("  • Position 0: minimal rotation, Position 5: maximum rotation")

# ============================================================================
# Step 7: Compute Attention Scores (Where RoPE Magic Happens)
# ============================================================================

print("\n" + "="*80)
print("STEP 7: ATTENTION SCORES - THE ROPE MAGIC")
print("="*80)

# Compute attention scores: Q @ K^T
# Shape: (batch, num_heads, seq_len, seq_len)
attention_scores_original = (Q_original @ K_original.transpose(-2, -1)) / np.sqrt(head_dim)
attention_scores_rope = (Q_rotated @ K_rotated.transpose(-2, -1)) / np.sqrt(head_dim)

print(f"\nAttention scores shape: {attention_scores_rope.shape}")
print(f"  Format: (batch, num_heads, query_positions, key_positions)")

# Focus on head 0
scores_original_h0 = attention_scores_original[0, 0].detach().numpy()
scores_rope_h0 = attention_scores_rope[0, 0].detach().numpy()

print(f"\nHead 0 attention scores (WITHOUT RoPE):")
print(f"Rows = Query positions, Columns = Key positions")
print(" " * 8 + "  ".join([f"{i}:{words[i][:3]}" for i in range(seq_len)]))
for i in range(seq_len):
    print(f"Q{i}:{words[i][:3]} ", end="")
    for j in range(seq_len):
        print(f"{scores_original_h0[i, j]:6.2f} ", end="")
    print()

print(f"\nHead 0 attention scores (WITH RoPE):")
print(f"Rows = Query positions, Columns = Key positions")
print(" " * 8 + "  ".join([f"{i}:{words[i][:3]}" for i in range(seq_len)]))
for i in range(seq_len):
    print(f"Q{i}:{words[i][:3]} ", end="")
    for j in range(seq_len):
        print(f"{scores_rope_h0[i, j]:6.2f} ", end="")
    print()

# Visualize attention patterns
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap 1: Without RoPE
sns.heatmap(scores_original_h0, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=[f"{i}:{words[i]}" for i in range(seq_len)],
            yticklabels=[f"{i}:{words[i]}" for i in range(seq_len)],
            ax=axes[0], cbar_kws={'label': 'Attention Score'})
axes[0].set_title('Attention Scores WITHOUT RoPE\n(No position information)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Key Positions', fontsize=11)
axes[0].set_ylabel('Query Positions', fontsize=11)

# Heatmap 2: With RoPE
sns.heatmap(scores_rope_h0, annot=True, fmt='.2f', cmap='YlGnBu', 
            xticklabels=[f"{i}:{words[i]}" for i in range(seq_len)],
            yticklabels=[f"{i}:{words[i]}" for i in range(seq_len)],
            ax=axes[1], cbar_kws={'label': 'Attention Score'})
axes[1].set_title('Attention Scores WITH RoPE\n(Position encoded in scores)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Key Positions', fontsize=11)
axes[1].set_ylabel('Query Positions', fontsize=11)

plt.tight_layout()
plt.savefig('./rope_attention_scores.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved visualization: rope_attention_scores.png")