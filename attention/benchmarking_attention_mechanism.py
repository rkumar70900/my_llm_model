import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llm_scratch.final_model import MultiHeadAttentionKVCache, MultiQueryAttention, GroupedQueryAttention

# Configuration
BATCH_SIZE = 1
EMBED_DIM = 256
NUM_HEADS = 8
NUM_GROUPS = 4  # For GQA
SEQ_LENS = [64, 128, 256, 512]
NUM_WARMUP = 5
NUM_ITER = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
RESULTS_DIR = "attention_benchmark_results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

def generate_random_inputs(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random input tensors for benchmarking."""
    x = torch.randn(
        BATCH_SIZE, 
        seq_len, 
        EMBED_DIM,
        device=DEVICE,
        dtype=DTYPE
    )
    mask = generate_causal_mask(seq_len).to(DEVICE)
    return x, mask

def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """Generate a causal mask for attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

def measure_latency(model: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Measure the latency of a single forward pass in milliseconds."""
    # Warmup
    for _ in range(NUM_WARMUP):
        _ = model(x, mask=mask)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_ITER):
        _ = model(x, mask=mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return (time.time() - start_time) * 1000 / NUM_ITER  # ms per iteration

def measure_kv_cache_size(model: nn.Module, seq_len: int) -> Dict[str, int]:
    """Measure the KV cache size in bytes."""
    x, _ = generate_random_inputs(seq_len)
    
    # Forward pass to get KV cache
    result = model(x, use_cache=True)
    if isinstance(result, tuple):
        _, kv_cache = result
    else:
        return {"kv_cache_size_bytes": 0, "seq_len": seq_len}
    
    if kv_cache is None:
        return {"kv_cache_size_bytes": 0, "seq_len": seq_len}
        
    # Calculate total memory usage of KV cache
    total_bytes = 0
    if isinstance(kv_cache, tuple):
        for tensor in kv_cache:
            if tensor is not None:
                total_bytes += tensor.element_size() * tensor.nelement()
    elif kv_cache is not None:
        total_bytes = kv_cache.element_size() * kv_cache.nelement()
        
    return {
        "kv_cache_size_bytes": total_bytes,
        "seq_len": seq_len
    }

def compute_entropy(logits: torch.Tensor) -> float:
    """Compute the entropy of the attention weights."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
    return entropy

def compute_loss(logits: torch.Tensor) -> float:
    """Compute the cross-entropy loss for the model's predictions."""
    # Create random target classes (since we don't have real targets)
    # The number of classes is the same as the last dimension of logits
    targets = torch.randint(
        low=0, 
        high=logits.size(-1), 
        size=(logits.size(0), logits.size(1)),
        device=logits.device
    )
    
    # Calculate cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # Reshape to (batch*seq_len, vocab_size)
        targets.view(-1),                  # Reshape to (batch*seq_len)
        reduction='mean'
    )
    return loss.item()

def benchmark_attention_mechanism(model: nn.Module, model_name: str, results: list):
    """Benchmark a single attention mechanism."""
    print(f"\nBenchmarking {model_name}...")
    
    for seq_len in tqdm(SEQ_LENS, desc=f"{model_name} - Sequence Lengths"):
        x, mask = generate_random_inputs(seq_len)
        
        # Measure latency
        latency_ms = measure_latency(model, x, mask)
        
        # Measure KV cache size, entropy, and loss
        with torch.no_grad():
            kv_cache_info = measure_kv_cache_size(model, seq_len)
            
            # Forward pass to get logits
            result = model(x, use_cache=False)
            if isinstance(result, tuple):
                logits = result[0]  # Get logits from (logits, kv_cache) tuple
            else:
                logits = result
                
            # Compute both entropy and loss
            entropy = compute_entropy(logits)
            loss = compute_loss(logits)
        
        # Store results
        result = {
            "model": model_name,
            "seq_len": seq_len,
            "latency_ms": latency_ms,
            "entropy": entropy,
            "loss": loss,
            **kv_cache_info
        }
        results.append(result)
        
        # Print intermediate results
        print(f"{model_name} - SeqLen: {seq_len:4d} | "
              f"Latency: {latency_ms:.3f}ms | "
              f"KV Cache: {kv_cache_info['kv_cache_size_bytes'] / 1024:.1f} KB | "
              f"Entropy: {entropy:.4f} | "
              f"Loss: {loss:.4f}")

def save_results(results: list):
    """Save benchmarking results to CSV and generate plots."""
    # Save raw results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "attention_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate and save plots
    generate_plots(df)

def generate_plots(df: pd.DataFrame):
    """Generate and save plots from benchmarking results."""
    # Define a color map for different models
    model_colors = {
        'MHA': '#1f77b4',      # Blue
        'MQA': '#ff7f0e',      # Orange
        'GQA-4': '#2ca02c',    # Green
        'GQA': '#d62728',      # Red (fallback if group number changes)
    }
    
    # Plot 1: Latency vs Sequence Length
    plt.figure(figsize=(12, 7))
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        color = model_colors.get(model_name, model_colors.get(model_name.split('-')[0], None))
        plt.plot(model_df['seq_len'], model_df['latency_ms'], 'o-', 
                label=model_name, 
                color=color,
                linewidth=2.5,
                markersize=8)
    
    plt.title('Latency vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.legend(fontsize=10, framealpha=1, edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'latency_vs_seqlen.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: KV Cache Size vs Sequence Length
    plt.figure(figsize=(12, 7))
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        color = model_colors.get(model_name, model_colors.get(model_name.split('-')[0], None))
        plt.plot(
            model_df['seq_len'], 
            model_df['kv_cache_size_bytes'] / 1024,  # Convert to KB
            'o-', 
            label=model_name,
            color=color,
            linewidth=2.5,
            markersize=8
        )
    
    plt.title('KV Cache Size vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('KV Cache Size (KB)')
    plt.legend(fontsize=10, framealpha=1, edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'kvcache_vs_seqlen.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Entropy vs Sequence Length
    plt.figure(figsize=(12, 7))
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        color = model_colors.get(model_name, model_colors.get(model_name.split('-')[0], None))
        plt.plot(model_df['seq_len'], model_df['entropy'], 'o-', 
                label=model_name,
                color=color,
                linewidth=2.5,
                markersize=8)
    
    plt.title('Attention Entropy vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Entropy')
    plt.legend(fontsize=10, framealpha=1, edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'entropy_vs_seqlen.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Loss vs Sequence Length
    plt.figure(figsize=(12, 7))
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        color = model_colors.get(model_name, model_colors.get(model_name.split('-')[0], None))
        plt.plot(model_df['seq_len'], model_df['loss'], 'o-', 
                label=model_name,
                color=color,
                linewidth=2.5,
                markersize=8)
    
    plt.title('Cross-Entropy Loss vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Loss')
    plt.legend(fontsize=10, framealpha=1, edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'loss_vs_seqlen.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Initialize attention mechanisms
    attention_models = {
        "MHA": MultiHeadAttentionKVCache(EMBED_DIM, NUM_HEADS).to(DEVICE).to(DTYPE),
        "MQA": MultiQueryAttention(EMBED_DIM, NUM_HEADS).to(DEVICE).to(DTYPE),
        f"GQA-{NUM_GROUPS}": GroupedQueryAttention(
            EMBED_DIM, 
            NUM_HEADS, 
            NUM_GROUPS
        ).to(DEVICE).to(DTYPE)
    }
    
    # Run benchmarks
    results = []
    for name, model in attention_models.items():
        benchmark_attention_mechanism(model, name, results)
    
    # Save results and generate plots
    save_results(results)
    print("\nBenchmarking completed!")

if __name__ == "__main__":
    main()