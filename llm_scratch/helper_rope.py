import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# RoPE Helper Functions
# ============================================================================

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