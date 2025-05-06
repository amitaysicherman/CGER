from train import load_files
from trie import build_trie, build_mask_from_trie
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os

sns.set(style="whitegrid")
import argparse
parser = argparse.ArgumentParser(description="Analyze Trie Data")
parser.add_argument("--quantize", action="store_true", help="Use quantized tokenizer")
parser.add_argument("--dataset", type=str, default="biosnap", help="Dataset to analyze")
args = parser.parse_args()
quantize = args.quantize
dataset = args.dataset

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)


# Function to analyze data and generate stats
def analyze_data(inputs, tokenizer, max_length=512, is_molecules=True):
    trie = build_trie(inputs, tokenizer)

    length = []
    active_length = []
    active_levels = []
    token_counts = {}  # Dictionary to count token occurrences

    for seq in tqdm(inputs):
        tokens = tokenizer.encode(seq, add_special_tokens=True)
        length.append(len(tokens))

        # Count token occurrences for token distribution analysis
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

        if len(tokens) > max_length:
            if not is_molecules:
                tokens = tokens[:max_length - 1] + [tokenizer.eos_token_id]
            else:
                tokens = tokens[:max_length - 1] + [tokenizer.pad_token_id]

        tensor_tokens = torch.tensor(tokens).unsqueeze(0)
        mask = build_mask_from_trie(trie, tensor_tokens, tokenizer.vocab_size).squeeze(0)
        active_mask = mask.sum(dim=-1) > 1
        active_length.append(active_mask.sum().item())
        levels = mask[active_mask].sum(dim=-1).tolist()
        active_levels.extend(levels)

    # Sort token counts and get the most common tokens
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:50]  # Get top 50 tokens

    return {
        'length': length,
        'active_length': active_length,
        'active_levels': active_levels,
        'top_tokens': top_tokens
    }


# Run analysis for both molecules and proteins
results = {}

# Analyze molecules
molecules = True
print("Analyzing molecules...")
src_train_mol, tgt_train_mol, src_valid_mol, tgt_valid_mol, src_test_mol, tgt_test_mol = load_files(level=dataset,
                                                                                                    gen_mol=molecules,
                                                                                                    quantize=quantize)
inputs_mol = list(set(tgt_train_mol))
tokenizer_mol = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
if quantize:
    from train import QuantizeTokenizer

    tokenizer_mol = QuantizeTokenizer()

results['molecules'] = analyze_data(inputs_mol, tokenizer_mol, is_molecules=True)

# Analyze proteins
molecules = False
print("Analyzing proteins...")
src_train_prot, tgt_train_prot, src_valid_prot, tgt_valid_prot, src_test_prot, tgt_test_prot = load_files(
    level=dataset, gen_mol=molecules, quantize=quantize)
inputs_prot = list(set(tgt_train_prot))
tokenizer_prot = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True)
if quantize:
    from train import QuantizeTokenizer

    tokenizer_prot = QuantizeTokenizer()

results['proteins'] = analyze_data(inputs_prot, tokenizer_prot, is_molecules=False)

# Create and save figures
data_types = ['molecules', 'proteins']

# Figure 1: Sequence Length Distribution
plt.figure(figsize=(10, 4))
for i, data_type in enumerate(data_types):
    plt.subplot(1, 2, i + 1)
    plt.hist(results[data_type]['length'], bins=10, alpha=0.7, label=f"{data_type.capitalize()} Sequences")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Count")
    plt.title(f"{data_type.capitalize()} Sequence Length Distribution")
    # plt.legend()
plt.tight_layout()
plt.savefig(f"figures/{dataset}_{quantize}_sequence_length_distribution.png")
plt.close()

# Figure 2: Active Sequence Length
plt.figure(figsize=(10, 4))
for i, data_type in enumerate(data_types):
    plt.subplot(1, 2, i + 1)
    plt.hist(results[data_type]['active_length'], bins=10, alpha=0.7, label=f"{data_type.capitalize()} Active Length")
    plt.xlabel("Active Sequence Length")
    plt.ylabel("Count")
    plt.title(f"{data_type.capitalize()} Active Sequence Length Distribution")
    # plt.legend()
plt.tight_layout()
plt.savefig(f"figures/{dataset}_{quantize}_active_sequence_length.png")
plt.close()

# Figure 3: Active Tokens Candidate Width
plt.figure(figsize=(10, 4))
for i, data_type in enumerate(data_types):
    plt.subplot(1, 2, i + 1)
    plt.hist(results[data_type]['active_levels'], bins=10, alpha=0.7, label=f"{data_type.capitalize()} Active Width")
    plt.xlabel("Active Tokens Candidate Width")
    plt.ylabel("Count")
    plt.title(f"{data_type.capitalize()} Active Tokens Width Distribution")
    # plt.legend()
plt.tight_layout()
plt.savefig(f"figures/{dataset}_{quantize}_active_tokens_width.png")
plt.close()

# Figure 4: Top Token Distribution
plt.figure(figsize=(10, 4))
for i, data_type in enumerate(data_types):
    plt.subplot(2, 1, i + 1)
    top_tokens = results[data_type]['top_tokens']
    token_ids = [str(t[0]) for t in top_tokens[:20]]  # Convert token IDs to strings for display
    token_counts = [t[1] for t in top_tokens[:20]]

    # Create horizontal bar chart
    bars = plt.barh(range(len(token_ids)), token_counts, align='center', alpha=0.7)
    plt.yticks(range(len(token_ids)), token_ids)
    plt.xlabel('Count')
    plt.ylabel('Token ID')
    plt.title(f"Top 20 Tokens in {data_type.capitalize()} Sequences")

    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{int(width)}',
                 ha='left', va='center')

plt.tight_layout()
plt.savefig(f"figures/{dataset}_{quantize}_top_token_distribution.png")
plt.close()

print("Analysis complete. All figures saved to the 'figures' directory.")
