import torch
import torch.nn.functional as F
import time
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import both versions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llm_scratch.final_model import TinyGPT as TinyGPT_Absolute  # Original with absolute PE
from llm_scratch.final_model_rope import TinyGPT as TinyGPT_RoPE  # New with RoPE

# Configuration
config = {
    'vocab_size': 10000,
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'train_max_seq_len': 128,  # Train on shorter sequences
    'test_seq_lengths': [128, 256, 384, 512],  # Test on longer sequences
    'num_test_samples': 100,
    'output_dir': 'results/rope_analysis'
}

# Create output directory
os.makedirs(config['output_dir'], exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Helper Functions
# ============================================================================

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def generate_sequence(model, start_token, max_length, use_cache=False):
    """
    Generate a sequence using the model
    
    Returns:
        generated_ids: List of generated token IDs
        time_per_step: List of time taken per generation step (ms)
    """
    model.eval()
    input_ids = torch.tensor([[start_token]], device=device)
    kv_caches = None
    time_per_step = []
    
    with torch.no_grad():
        for step in range(max_length - 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            
            if use_cache:
                logits, kv_caches = model(
                    input_ids[:, -1:],
                    kv_caches=kv_caches,
                    use_cache=True
                )
            else:
                logits = model(input_ids)
                logits = logits[:, -1:, :]
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time = (time.time() - start_time) * 1000
            time_per_step.append(step_time)
            
            # Sample next token (greedy)
            next_token = logits.argmax(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return input_ids[0].tolist(), time_per_step


def compute_perplexity(model, test_data, seq_length):
    """
    Compute perplexity on test data
    
    Args:
        model: The language model
        test_data: List of token IDs
        seq_length: Sequence length to test on
    
    Returns:
        perplexity: Average perplexity across test samples
    """
    model.eval()
    total_loss = 0
    num_samples = min(config['num_test_samples'], len(test_data) - seq_length - 1)
    
    with torch.no_grad():
        for i in range(num_samples):
            start_idx = i * (len(test_data) // num_samples)
            if start_idx + seq_length + 1 > len(test_data):
                break
            
            x = torch.tensor([test_data[start_idx:start_idx + seq_length]], device=device)
            y = torch.tensor([test_data[start_idx + 1:start_idx + seq_length + 1]], device=device)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / num_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


# ============================================================================
# Create Test Data
# ============================================================================

print("Generating synthetic test data...")
# Create synthetic data for testing
torch.manual_seed(42)
test_data = torch.randint(0, config['vocab_size'], (10000,)).tolist()

# ============================================================================
# Initialize Models
# ============================================================================

print("\n" + "="*80)
print("Initializing models...")
print("="*80)

# Model with absolute positional embeddings
model_absolute = TinyGPT_Absolute(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    max_seq_len=config['train_max_seq_len']
).to(device)

# Model with RoPE - initialize with max test sequence length to enable extrapolation
max_test_len = max(config['test_seq_lengths'])
model_rope = TinyGPT_RoPE(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    max_seq_len=max_test_len  # Use max test length for RoPE to enable extrapolation
).to(device)

# Count parameters
params_absolute_total, params_absolute_train = count_parameters(model_absolute)
params_rope_total, params_rope_train = count_parameters(model_rope)

print(f"\nAbsolute PE Model:")
print(f"  Total parameters: {params_absolute_total:,}")
print(f"  Trainable parameters: {params_absolute_train:,}")

print(f"\nRoPE Model:")
print(f"  Total parameters: {params_rope_total:,}")
print(f"  Trainable parameters: {params_rope_train:,}")

print(f"\nParameter savings with RoPE: {params_absolute_total - params_rope_total:,}")
print(f"  (This is max_seq_len × embed_dim = {config['train_max_seq_len']} × {config['embed_dim']} = {config['train_max_seq_len'] * config['embed_dim']:,})")

# ============================================================================
# Test 1: Extrapolation - Perplexity on Different Sequence Lengths
# ============================================================================

print("\n" + "="*80)
print("Test 1: Extrapolation Capability")
print("="*80)
print("Testing perplexity on sequences longer than training length...")
print(f"Training max length: {config['train_max_seq_len']}")
print(f"Test lengths: {config['test_seq_lengths']}")

extrapolation_results = {
    'seq_lengths': config['test_seq_lengths'],
    'absolute_perplexity': [],
    'rope_perplexity': []
}

for seq_len in config['test_seq_lengths']:
    print(f"\nTesting sequence length: {seq_len}")
    
    # Test absolute PE model
    if seq_len <= config['train_max_seq_len']:
        ppl_absolute = compute_perplexity(model_absolute, test_data, seq_len)
        print(f"  Absolute PE perplexity: {ppl_absolute:.2f}")
    else:
        ppl_absolute = float('inf')  # Can't handle longer sequences
        print(f"  Absolute PE perplexity: N/A (exceeds max_seq_len)")
    
    # Test RoPE model
    ppl_rope = compute_perplexity(model_rope, test_data, seq_len)
    print(f"  RoPE perplexity: {ppl_rope:.2f}")
    
    extrapolation_results['absolute_perplexity'].append(ppl_absolute)
    extrapolation_results['rope_perplexity'].append(ppl_rope)

# ============================================================================
# Test 2: Generation Speed Comparison
# ============================================================================

print("\n" + "="*80)
print("Test 2: Generation Speed")
print("="*80)

generation_results = {
    'seq_lengths': [],
    'absolute_avg_time': [],
    'rope_avg_time': []
}

test_gen_lengths = [64, 128, 256]
for gen_len in test_gen_lengths:
    if gen_len > config['train_max_seq_len']:
        print(f"\nSkipping generation test for length {gen_len} (exceeds absolute PE max)")
        continue
    
    print(f"\nGenerating {gen_len} tokens...")
    
    # Absolute PE
    _, times_absolute = generate_sequence(model_absolute, start_token=0, max_length=gen_len, use_cache=True)
    avg_time_absolute = np.mean(times_absolute)
    
    # RoPE
    _, times_rope = generate_sequence(model_rope, start_token=0, max_length=gen_len, use_cache=True)
    avg_time_rope = np.mean(times_rope)
    
    print(f"  Absolute PE avg time/token: {avg_time_absolute:.2f}ms")
    print(f"  RoPE avg time/token: {avg_time_rope:.2f}ms")
    print(f"  Speedup: {avg_time_absolute/avg_time_rope:.2f}x")
    
    generation_results['seq_lengths'].append(gen_len)
    generation_results['absolute_avg_time'].append(avg_time_absolute)
    generation_results['rope_avg_time'].append(avg_time_rope)

# ============================================================================
# Test 3: Memory Efficiency
# ============================================================================

print("\n" + "="*80)
print("Test 3: Memory Efficiency")
print("="*80)

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024**2)

size_absolute = get_model_size_mb(model_absolute)
size_rope = get_model_size_mb(model_rope)

print(f"\nAbsolute PE model size: {size_absolute:.2f} MB")
print(f"RoPE model size: {size_rope:.2f} MB")
print(f"Memory saved: {size_absolute - size_rope:.2f} MB ({(1 - size_rope/size_absolute)*100:.1f}%)")

# ============================================================================
# Test 4: Attention Pattern Analysis (Optional - if you want to visualize)
# ============================================================================

print("\n" + "="*80)
print("Test 4: Analyzing Relative Position Sensitivity")
print("="*80)

# Test how well models encode relative positions
def test_relative_position_sensitivity(model, seq_len=64):
    """
    Test if model is sensitive to relative positions
    by computing attention between different relative distances
    """
    model.eval()
    
    # Create two identical sequences but shifted
    test_seq = torch.randint(0, config['vocab_size'], (1, seq_len), device=device)
    
    with torch.no_grad():
        output1 = model(test_seq)
        # Shift sequence (removing first token, adding random at end)
        test_seq_shifted = torch.cat([test_seq[:, 1:], torch.randint(0, config['vocab_size'], (1, 1), device=device)], dim=1)
        output2 = model(test_seq_shifted)
        
        # Compare predictions at corresponding relative positions
        # If model uses relative positions well, predictions should be similar
        similarity = F.cosine_similarity(output1[:, :-1, :], output2[:, :-1, :], dim=-1).mean().item()
    
    return similarity

print("\nTesting relative position sensitivity...")
print("(Higher similarity means better relative position encoding)")

# Test on sequence length within training range
test_len = 64
sim_absolute = test_relative_position_sensitivity(model_absolute, test_len)
sim_rope = test_relative_position_sensitivity(model_rope, test_len)

print(f"\nSequence length {test_len}:")
print(f"  Absolute PE similarity: {sim_absolute:.4f}")
print(f"  RoPE similarity: {sim_rope:.4f}")

# ============================================================================
# Save Results to CSV
# ============================================================================

print("\n" + "="*80)
print("Saving results...")
print("="*80)

# Save extrapolation results
csv_file = f"{config['output_dir']}/extrapolation_results_{timestamp}.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['seq_length', 'absolute_perplexity', 'rope_perplexity'])
    writer.writeheader()
    for i, seq_len in enumerate(extrapolation_results['seq_lengths']):
        writer.writerow({
            'seq_length': seq_len,
            'absolute_perplexity': extrapolation_results['absolute_perplexity'][i],
            'rope_perplexity': extrapolation_results['rope_perplexity'][i]
        })
print(f"Saved extrapolation results to {csv_file}")

# Save generation speed results
if generation_results['seq_lengths']:
    csv_file = f"{config['output_dir']}/generation_speed_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seq_length', 'absolute_avg_time_ms', 'rope_avg_time_ms'])
        writer.writeheader()
        for i, seq_len in enumerate(generation_results['seq_lengths']):
            writer.writerow({
                'seq_length': seq_len,
                'absolute_avg_time_ms': generation_results['absolute_avg_time'][i],
                'rope_avg_time_ms': generation_results['rope_avg_time'][i]
            })
    print(f"Saved generation speed results to {csv_file}")

# ============================================================================
# Create Visualizations
# ============================================================================

print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

# Create directory for plots
plots_dir = f"{config['output_dir']}/plots_{timestamp}"
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Extrapolation - Perplexity vs Sequence Length
plt.figure(figsize=(10, 6))
seq_lens = extrapolation_results['seq_lengths']

# Handle infinite perplexity for absolute PE
absolute_ppl = [ppl if ppl != float('inf') else None for ppl in extrapolation_results['absolute_perplexity']]
rope_ppl = extrapolation_results['rope_perplexity']

# Plot with markers for valid points
valid_absolute_lens = [seq_lens[i] for i, ppl in enumerate(absolute_ppl) if ppl is not None]
valid_absolute_ppl = [ppl for ppl in absolute_ppl if ppl is not None]

plt.plot(valid_absolute_lens, valid_absolute_ppl, 'o-', label='Absolute PE', linewidth=2, markersize=8, color='#FF6B6B')
plt.plot(seq_lens, rope_ppl, 's-', label='RoPE', linewidth=2, markersize=8, color='#4ECDC4')

# Add vertical line at training max length
plt.axvline(x=config['train_max_seq_len'], color='gray', linestyle='--', alpha=0.5, label=f'Training max ({config["train_max_seq_len"]})')

plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Perplexity (lower is better)', fontsize=12)
plt.title('Extrapolation: Model Performance on Varying Sequence Lengths', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{plots_dir}/extrapolation_perplexity.png", dpi=300)
plt.close()
print(f"Saved plot: {plots_dir}/extrapolation_perplexity.png")

# Plot 2: Generation Speed Comparison
if generation_results['seq_lengths']:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Average time per token
    x = np.arange(len(generation_results['seq_lengths']))
    width = 0.35
    
    ax1.bar(x - width/2, generation_results['absolute_avg_time'], width, label='Absolute PE', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, generation_results['rope_avg_time'], width, label='RoPE', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Avg Time per Token (ms)', fontsize=12)
    ax1.set_title('Generation Speed Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(generation_results['seq_lengths'])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Speedup ratio
    speedup = [generation_results['absolute_avg_time'][i] / generation_results['rope_avg_time'][i] 
               for i in range(len(generation_results['seq_lengths']))]
    
    ax2.plot(generation_results['seq_lengths'], speedup, 'o-', linewidth=2, markersize=10, color='#95E1D3')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Speedup (Absolute PE / RoPE)', fontsize=12)
    ax2.set_title('RoPE Speedup Factor', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/generation_speed.png", dpi=300)
    plt.close()
    print(f"Saved plot: {plots_dir}/generation_speed.png")

# Plot 3: Parameter Efficiency
plt.figure(figsize=(10, 6))
models = ['Absolute PE', 'RoPE']
param_counts = [params_absolute_total, params_rope_total]
colors = ['#FF6B6B', '#4ECDC4']

bars = plt.bar(models, param_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, count in zip(bars, param_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add savings annotation
savings = params_absolute_total - params_rope_total
plt.annotate(f'Saves {savings:,} parameters\n({(savings/params_absolute_total)*100:.1f}%)',
             xy=(1, params_rope_total), xytext=(0.5, params_absolute_total * 0.7),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.ylabel('Number of Parameters', fontsize=12)
plt.title('Model Parameter Count Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{plots_dir}/parameter_comparison.png", dpi=300)
plt.close()
print(f"Saved plot: {plots_dir}/parameter_comparison.png")

# Plot 4: Combined Summary
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Extrapolation
ax1.plot(valid_absolute_lens, valid_absolute_ppl, 'o-', label='Absolute PE', linewidth=2, markersize=8, color='#FF6B6B')
ax1.plot(seq_lens, rope_ppl, 's-', label='RoPE', linewidth=2, markersize=8, color='#4ECDC4')
ax1.axvline(x=config['train_max_seq_len'], color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Sequence Length', fontsize=11)
ax1.set_ylabel('Perplexity', fontsize=11)
ax1.set_title('Extrapolation Capability', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Parameter comparison
ax2.bar(models, param_counts, color=colors, alpha=0.8)
ax2.set_ylabel('Parameters', fontsize=11)
ax2.set_title(f'Parameters (RoPE saves {savings:,})', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Subplot 3: Generation speed
if generation_results['seq_lengths']:
    x = np.arange(len(generation_results['seq_lengths']))
    width = 0.35
    ax3.bar(x - width/2, generation_results['absolute_avg_time'], width, label='Absolute PE', color='#FF6B6B', alpha=0.8)
    ax3.bar(x + width/2, generation_results['rope_avg_time'], width, label='RoPE', color='#4ECDC4', alpha=0.8)
    ax3.set_xlabel('Sequence Length', fontsize=11)
    ax3.set_ylabel('Avg Time/Token (ms)', fontsize=11)
    ax3.set_title('Generation Speed', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(generation_results['seq_lengths'])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Memory efficiency
memory_data = [size_absolute, size_rope]
ax4.bar(models, memory_data, color=colors, alpha=0.8)
ax4.set_ylabel('Model Size (MB)', fontsize=11)
ax4.set_title(f'Memory Usage (RoPE saves {size_absolute - size_rope:.2f} MB)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{plots_dir}/combined_summary.png", dpi=300)
plt.close()
print(f"Saved plot: {plots_dir}/combined_summary.png")

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\n1. PARAMETER EFFICIENCY:")
print(f"   • Absolute PE: {params_absolute_total:,} parameters")
print(f"   • RoPE: {params_rope_total:,} parameters")
print(f"   • Savings: {savings:,} parameters ({(savings/params_absolute_total)*100:.1f}%)")

print("\n2. EXTRAPOLATION CAPABILITY:")
print(f"   • Training max length: {config['train_max_seq_len']}")
for i, seq_len in enumerate(extrapolation_results['seq_lengths']):
    abs_ppl = extrapolation_results['absolute_perplexity'][i]
    rope_ppl = extrapolation_results['rope_perplexity'][i]
    if abs_ppl == float('inf'):
        print(f"   • Length {seq_len}: Absolute PE fails, RoPE achieves {rope_ppl:.2f} perplexity")
    else:
        print(f"   • Length {seq_len}: Absolute PE {abs_ppl:.2f}, RoPE {rope_ppl:.2f} perplexity")

if generation_results['seq_lengths']:
    print("\n3. GENERATION SPEED:")
    for i, seq_len in enumerate(generation_results['seq_lengths']):
        abs_time = generation_results['absolute_avg_time'][i]
        rope_time = generation_results['rope_avg_time'][i]
        speedup = abs_time / rope_time
        print(f"   • Length {seq_len}: {speedup:.2f}x speedup (Absolute: {abs_time:.2f}ms, RoPE: {rope_time:.2f}ms)")

print("\n4. MEMORY EFFICIENCY:")
print(f"   • Absolute PE: {size_absolute:.2f} MB")
print(f"   • RoPE: {size_rope:.2f} MB")
print(f"   • Savings: {size_absolute - size_rope:.2f} MB ({(1 - size_rope/size_absolute)*100:.1f}%)")

print("\n5. RELATIVE POSITION ENCODING:")
print(f"   • Absolute PE similarity score: {sim_absolute:.4f}")
print(f"   • RoPE similarity score: {sim_rope:.4f}")
print(f"   • {'RoPE' if sim_rope > sim_absolute else 'Absolute PE'} shows better relative position encoding")

print("\n" + "="*80)
print(f"All results saved to: {config['output_dir']}/")
print(f"Plots saved to: {plots_dir}/")
print("="*80)