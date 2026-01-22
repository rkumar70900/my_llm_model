import torch
import torch.nn.functional as F
import time
import psutil
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from llm_scratch.final_model import TinyGPT
import numpy as np

# Configuration
config = {
    'vocab_size': 10000,
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'max_length': 50,
    'output_dir': 'results'
}

# Create output directory
os.makedirs(config['output_dir'], exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TinyGPT(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers']
).to(device)
model.eval()

def calculate_flops(seq_len, embed_dim, num_heads, batch_size=1):
    """Calculate FLOPs for self-attention for a single head."""
    # QK^T: [B, H, T, D/H] @ [B, H, D/H, T] -> [B, H, T, T]
    flops = 2 * batch_size * num_heads * (seq_len * (embed_dim // num_heads) * seq_len)
    # Softmax: ~3 * B * H * T^2
    flops += 3 * batch_size * num_heads * seq_len * seq_len
    # AV: [B, H, T, T] @ [B, H, T, D/H] -> [B, H, T, D/H]
    flops += 2 * batch_size * num_heads * (seq_len * seq_len * (embed_dim // num_heads))
    return flops

def calculate_kv_cache_flops(seq_len, embed_dim, num_heads, batch_size=1):
    head_dim = embed_dim // num_heads

    # QK^T: [B,H,1,D] @ [B,H,D,T] → [B,H,1,T]
    flops = 2 * batch_size * num_heads * head_dim * seq_len

    # Softmax: ~3 * B * H * T
    flops += 3 * batch_size * num_heads * seq_len

    # AV: [B,H,1,T] @ [B,H,T,D] → [B,H,1,D]
    flops += 2 * batch_size * num_heads * seq_len * head_dim

    return flops



def get_kv_cache_size(kv_caches):
    """Calculate total memory used by KV cache in MB"""
    if kv_caches is None:
        return 0.0
    
    total_bytes = 0
    for layer_cache in kv_caches:
        if layer_cache is not None:
            k, v = layer_cache
            if k is not None:
                total_bytes += k.element_size() * k.numel()
            if v is not None:
                total_bytes += v.element_size() * v.numel()
    return total_bytes / (1024 ** 2)  # Convert to MB


# 1. Run with KV Cache
print("=== Running with KV Cache ===")
input_ids = torch.tensor([[0]], device=device)
kv_caches = None
cache_metrics = []
total_flops_saved = 0

with torch.no_grad():
    for step in range(config['max_length']):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        
        
        # Forward pass with cache
        logits, kv_caches = model(
            input_ids[:, -1:],
            kv_caches=kv_caches,
            use_cache=True
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        # Record metrics
        step_time = (time.time() - start_time) * 1000

        kv_cache_mb = get_kv_cache_size(kv_caches)
        
        # Calculate FLOPs saved by KV cache
        if step > 0:  # First step has nothing to compare against
            flops_without_cache = calculate_flops(step + 1, config['embed_dim'], config['num_heads'])
            flops_with_cache = calculate_kv_cache_flops(
                                    seq_len=step + 1,
                                    embed_dim=config['embed_dim'],
                                    num_heads=config['num_heads']
                                )
            flops_saved = flops_without_cache - flops_with_cache
            total_flops_saved += flops_saved
        else:
            flops_saved = 0
        
        # Sample next token
        next_token = logits.argmax(-1)[:, -1:]
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        
        current_seq_len = kv_caches[0][0].size(2) if kv_caches[0][0] is not None else 1
        
        cache_metrics.append({
            'step': step + 1,
            'memory_mb': kv_cache_mb,
            'time_ms': step_time,
            'seq_len': current_seq_len,
            'flops_saved': flops_saved,
            'total_flops_saved': total_flops_saved,
            'tokens_processed': step + 1
        })
        
        print(f"Step {step+1}: SeqLen={current_seq_len}, "
              f"FLOPs saved={flops_saved/1e6:.2f}M, "
              f"Total FLOPs saved={total_flops_saved/1e9:.2f}G")

# 2. Run without KV Cache (for comparison)
print("\n=== Running without KV Cache ===")
input_ids = torch.tensor([[0]], device=device)
no_cache_metrics = []
total_flops = 0

with torch.no_grad():
    for step in range(config['max_length']):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        
        # Forward pass without cache
        logits = model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()
        # Record metrics
        step_time = (time.time() - start_time) * 1000
        
        # Calculate FLOPs for this step
        flops = calculate_flops(step + 1, config['embed_dim'], config['num_heads'])
        total_flops += flops
        
        # Sample next token
        next_token = logits[:, -1:].argmax(-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        
        no_cache_metrics.append({
            'step': step + 1,
            'memory_mb': mem_after - mem_before,
            'time_ms': step_time,
            'flops': flops,
            'total_flops': total_flops,
            'tokens_processed': step + 1
        })

# 3. Save results
def save_metrics(metrics, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

save_metrics(cache_metrics, f"{config['output_dir']}/with_cache_{timestamp}.csv")
save_metrics(no_cache_metrics, f"{config['output_dir']}/without_cache_{timestamp}.csv")

# 4. Plot results
def plot_comparison(cache_metrics, no_cache_metrics):
    steps = [m['step'] for m in cache_metrics]
    base_path = f"{config['output_dir']}/kv_cache_analysis_{timestamp}"
    
    # Create directory for individual plots
    os.makedirs(f"{base_path}_plots", exist_ok=True)
    
    # 1. FLOPs Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, [m['total_flops_saved']/1e9 for m in cache_metrics], 'b-', label='FLOPs saved (KV Cache)')
    plt.plot(steps, [m['total_flops']/1e9 for m in no_cache_metrics], 'r-', label='FLOPs (No Cache)')
    plt.xlabel('Generation Step')
    plt.ylabel('Cumulative FLOPs (G)')
    plt.title('Computational Cost Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_plots/flops_comparison.png")
    plt.close()
    
    # 2. FLOPs Saved per Step Plot
    plt.figure(figsize=(10, 6))
    plt.bar(steps, [m['flops_saved']/1e6 for m in cache_metrics], alpha=0.7, color='green')
    plt.xlabel('Generation Step')
    plt.ylabel('FLOPs Saved (M)')
    plt.title('FLOPs Saved per Step with KV Cache')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_plots/flops_saved_per_step.png")
    plt.close()
    
    # 3. Memory Usage Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, [m['memory_mb'] for m in cache_metrics], 'b-', label='With KV Cache')
    plt.plot(steps, [m['memory_mb'] for m in no_cache_metrics], 'r-', label='Without KV Cache')
    plt.xlabel('Generation Step')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_plots/memory_comparison.png")
    plt.close()
    
    # 4. Inference Time Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, [m['time_ms'] for m in cache_metrics], 'b-', label='With KV Cache')
    plt.plot(steps, [m['time_ms'] for m in no_cache_metrics], 'r-', label='Without KV Cache')
    plt.xlabel('Generation Step')
    plt.ylabel('Time per Step (ms)')
    plt.title('Inference Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f"{base_path}_plots/time_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    
    # 5. Combined plot (for reference)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # FLOPs comparison
    axs[0, 0].plot(steps, [m['total_flops_saved']/1e9 for m in cache_metrics], 'b-', label='FLOPs saved (KV Cache)')
    axs[0, 0].plot(steps, [m['total_flops']/1e9 for m in no_cache_metrics], 'r-', label='FLOPs (No Cache)')
    axs[0, 0].set_xlabel('Generation Step')
    axs[0, 0].set_ylabel('Cumulative FLOPs (G)')
    axs[0, 0].set_title('Computational Cost Comparison')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # FLOPs saved per step
    axs[0, 1].bar(steps, [m['flops_saved']/1e6 for m in cache_metrics], alpha=0.7, color='green')
    axs[0, 1].set_xlabel('Generation Step')
    axs[0, 1].set_ylabel('FLOPs Saved (M)')
    axs[0, 1].set_title('FLOPs Saved per Step with KV Cache')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Memory comparison
    axs[1, 0].plot(steps, [m['memory_mb'] for m in cache_metrics], 'b-', label='With KV Cache')
    axs[1, 0].plot(steps, [m['memory_mb'] for m in no_cache_metrics], 'r-', label='Without KV Cache')
    axs[1, 0].set_xlabel('Generation Step')
    axs[1, 0].set_ylabel('Memory Usage (MB)')
    axs[1, 0].set_title('Memory Usage Comparison')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Time comparison
    axs[1, 1].plot(steps, [m['time_ms'] for m in cache_metrics], 'b-', label='With KV Cache')
    axs[1, 1].plot(steps, [m['time_ms'] for m in no_cache_metrics], 'r-', label='Without KV Cache')
    axs[1, 1].set_xlabel('Generation Step')
    axs[1, 1].set_ylabel('Time per Step (ms)')
    axs[1, 1].set_title('Inference Time Comparison')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}_combined.png")
    plt.close()
    print(f"\nSaved analysis plots to {base_path}_plots/ and {base_path}_combined.png")
    print(f"\nSaved analysis plots to {plot_path}")

# Generate plots
plot_comparison(cache_metrics, no_cache_metrics)

# Print final summary
print("\n=== Final Summary ===")
print(f"Total FLOPs without cache: {no_cache_metrics[-1]['total_flops']/1e9:.2f} GFLOPs")
print(f"Total FLOPs saved with KV cache: {cache_metrics[-1]['total_flops_saved']/1e9:.2f} GFLOPs")
print(f"FLOPs reduction: {cache_metrics[-1]['total_flops_saved']/no_cache_metrics[-1]['total_flops']*100:.1f}%")