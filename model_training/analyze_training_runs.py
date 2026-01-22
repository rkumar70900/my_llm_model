import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for better-looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [15, 10]

# Path to the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'training_runs.csv')

# Load and clean the data
df = pd.read_csv(csv_path)

# Remove any rows with missing values
df = df.dropna()

# Ensure numeric columns are properly typed
numeric_cols = ['epoch', 'loss', 'cpu_usage', 'gpu_usage']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Print basic statistics
print("\n=== Dataset Overview ===")
print(f"Total data points: {len(df)}")
print(f"Unique tokenizer types: {df['tokenizer_type'].unique().tolist()}")
print(f"Number of unique runs: {df['run_id'].nunique()}")

# Create main output directory
output_dir = 'analysis_plots'
os.makedirs(output_dir, exist_ok=True)

# Get unique tokenizer types
tokenizers = df['tokenizer_type'].unique()
colors = sns.color_palette('husl', len(tokenizers))

# ===== Create tokenizer comparison plots =====
# 1. Loss Plot
plt.figure(figsize=(15, 6))
for i, tokenizer in enumerate(tokenizers):
    tokenizer_data = df[df['tokenizer_type'] == tokenizer]
    sns.lineplot(data=tokenizer_data, x='epoch', y='loss', 
                label=tokenizer, color=colors[i])
plt.title('Training Loss by Tokenizer Type')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(title='Tokenizer Type')
plt.ylim(bottom=0)
plt.tight_layout()
loss_path = os.path.join(output_dir, 'loss_by_tokenizer.png')
plt.savefig(loss_path)
plt.close()

# 2. CPU Usage Plot
plt.figure(figsize=(15, 6))
for i, tokenizer in enumerate(tokenizers):
    tokenizer_data = df[df['tokenizer_type'] == tokenizer]
    sns.lineplot(data=tokenizer_data, x='epoch', y='cpu_usage', 
                label=tokenizer, color=colors[i])
plt.title('CPU Usage by Tokenizer Type')
plt.xlabel('Epoch')
plt.ylabel('CPU Usage (%)')
plt.legend(title='Tokenizer Type')
plt.ylim(0, 100)
plt.tight_layout()
cpu_path = os.path.join(output_dir, 'cpu_usage_by_tokenizer.png')
plt.savefig(cpu_path)
plt.close()

# 3. GPU Usage Plot
plt.figure(figsize=(15, 6))
for i, tokenizer in enumerate(tokenizers):
    tokenizer_data = df[df['tokenizer_type'] == tokenizer]
    sns.lineplot(data=tokenizer_data, x='epoch', y='gpu_usage', 
                label=tokenizer, color=colors[i])
plt.title('GPU Usage by Tokenizer Type')
plt.xlabel('Epoch')
plt.ylabel('GPU Usage (%)')
plt.legend(title='Tokenizer Type')
plt.ylim(0, 100)
plt.tight_layout()
gpu_path = os.path.join(output_dir, 'gpu_usage_by_tokenizer.png')
plt.savefig(gpu_path)
plt.close()

# 4. Combined tokenizer comparison plot
fig, axes = plt.subplots(3, 1, figsize=(15, 18))

# Loss subplot
ax = axes[0]
for i, tokenizer in enumerate(tokenizers):
    tokenizer_data = df[df['tokenizer_type'] == tokenizer]
    sns.lineplot(data=tokenizer_data, x='epoch', y='loss', 
                label=tokenizer, color=colors[i], ax=ax)
ax.set_title('Training Loss by Tokenizer Type')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(title='Tokenizer Type')
ax.set_ylim(bottom=0)

# CPU Usage subplot
ax = axes[1]
for i, tokenizer in enumerate(tokenizers):
    tokenizer_data = df[df['tokenizer_type'] == tokenizer]
    sns.lineplot(data=tokenizer_data, x='epoch', y='cpu_usage', 
                label=tokenizer, color=colors[i], ax=ax)
ax.set_title('CPU Usage by Tokenizer Type')
ax.set_xlabel('Epoch')
ax.set_ylabel('CPU Usage (%)')
ax.legend(title='Tokenizer Type')
ax.set_ylim(0, 100)

# GPU Usage subplot
ax = axes[2]
for i, tokenizer in enumerate(tokenizers):
    tokenizer_data = df[df['tokenizer_type'] == tokenizer]
    sns.lineplot(data=tokenizer_data, x='epoch', y='gpu_usage', 
                label=tokenizer, color=colors[i], ax=ax)
ax.set_title('GPU Usage by Tokenizer Type')
ax.set_xlabel('Epoch')
ax.set_ylabel('GPU Usage (%)')
ax.legend(title='Tokenizer Type')
ax.set_ylim(0, 100)

plt.tight_layout()
combined_path = os.path.join(output_dir, 'all_metrics_combined.png')
plt.savefig(combined_path)
plt.close()

print(f"\nTokenizer comparison plots saved to:")
print(f"- Loss: {loss_path}")
print(f"- CPU:  {cpu_path}")
print(f"- GPU:  {gpu_path}")
print(f"- Combined: {combined_path}")

# ===== Create individual run plots =====
# Create subdirectories for individual run plots
loss_dir = os.path.join(output_dir, 'loss')
cpu_dir = os.path.join(output_dir, 'cpu_usage')
gpu_dir = os.path.join(output_dir, 'gpu_usage')
combined_dir = os.path.join(output_dir, 'combined')

for dir_path in [loss_dir, cpu_dir, gpu_dir, combined_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Group by run_id and tokenizer_type
grouped = df.groupby(['run_id', 'tokenizer_type'])

# Process each run
for (run_id, tokenizer_type), group in grouped:
    safe_run_id = "_".join(str(run_id).split(":"))
    base_filename = f'{safe_run_id}_{tokenizer_type}'
    
    # 1. Individual Loss Plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=group, x='epoch', y='loss')
    plt.title(f'Training Loss\nRun: {run_id}\nTokenizer: {tokenizer_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.tight_layout()
    loss_path = os.path.join(loss_dir, f'{base_filename}_loss.png')
    plt.savefig(loss_path)
    plt.close()
    
    # 2. Individual CPU Usage Plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=group, x='epoch', y='cpu_usage')
    plt.title(f'CPU Usage\nRun: {run_id}\nTokenizer: {tokenizer_type}')
    plt.xlabel('Epoch')
    plt.ylabel('CPU Usage (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    cpu_path = os.path.join(cpu_dir, f'{base_filename}_cpu.png')
    plt.savefig(cpu_path)
    plt.close()
    
    # 3. Individual GPU Usage Plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=group, x='epoch', y='gpu_usage')
    plt.title(f'GPU Usage\nRun: {run_id}\nTokenizer: {tokenizer_type}')
    plt.xlabel('Epoch')
    plt.ylabel('GPU Usage (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    gpu_path = os.path.join(gpu_dir, f'{base_filename}_gpu.png')
    plt.savefig(gpu_path)
    plt.close()
    
    # 4. Combined plot for this run
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    # Loss subplot
    sns.lineplot(data=group, x='epoch', y='loss', ax=axes[0])
    axes[0].set_title(f'Training Loss\nRun: {run_id}\nTokenizer: {tokenizer_type}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(bottom=0)
    
    # CPU Usage subplot
    sns.lineplot(data=group, x='epoch', y='cpu_usage', ax=axes[1])
    axes[1].set_title('CPU Usage')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('CPU Usage (%)')
    axes[1].set_ylim(0, 100)
    
    # GPU Usage subplot
    sns.lineplot(data=group, x='epoch', y='gpu_usage', ax=axes[2])
    axes[2].set_title('GPU Usage')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('GPU Usage (%)')
    axes[2].set_ylim(0, 100)
    
    plt.tight_layout()
    combined_path = os.path.join(combined_dir, f'{base_filename}_combined.png')
    plt.savefig(combined_path)
    plt.close()
    
    print(f"\nSaved plots for {run_id} ({tokenizer_type}) to:")
    print(f"- Loss: {loss_path}")
    print(f"- CPU:  {cpu_path}")
    print(f"- GPU:  {gpu_path}")
    print(f"- Combined: {combined_path}")

print("\nAnalysis complete! Check the 'analysis_plots' directory for all generated plots.")
