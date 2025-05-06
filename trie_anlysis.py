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


plt.figure(figsize=(10, 2.5))  # Increased height to make room for labels

# Create subplots with minimal spacing but enough room for labels
gs = plt.GridSpec(1, 4, wspace=0.1, hspace=0.3)

# Find the maximum density value across all histograms to set consistent y-axis limits
max_density = 0
for data_type in data_types:
    # Calculate histogram values for active_length
    counts_length, _ = np.histogram(results[data_type]['active_length'], density=True)
    counts_levels, _ = np.histogram(results[data_type]['active_levels'], density=True)
    max_density = max(max_density, np.max(counts_length), np.max(counts_levels))

# First subplot (active_length - data_type 0)
ax1 = plt.subplot(gs[0])
ax1.hist(results[data_types[0]]['active_length'],
        density=True,
        label=f"{data_types[0].capitalize()}",
        color="tab:blue")
ax1.set_ylabel("Percentage")
ax1.set_title(f"{data_types[0].capitalize()}")
ax1.legend(loc="upper left")
ax1.set_ylim(0, max_density * 1.1)  # Set y-limit with some margin
ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))

# Second subplot (active_length - data_type 1)
ax2 = plt.subplot(gs[1], sharey=ax1)
ax2.hist(results[data_types[1]]['active_length'],
        density=True,
        label=f"{data_types[1].capitalize()}",
        color="tab:blue")
ax2.set_title(f"{data_types[1].capitalize()}")
ax2.legend(loc="upper left")
plt.setp(ax2.get_yticklabels(), visible=False)  # Hide y-ticks for shared y-axis

# Third subplot (active_levels - data_type 0)
ax3 = plt.subplot(gs[2], sharey=ax1)  # Share y-axis with the first plot
ax3.hist(results[data_types[0]]['active_levels'],
        density=True,
        label=f"{data_types[0].capitalize()}",
        color="tab:orange")
# ax3.set_ylabel("Percentage")
ax3.set_title(f"{data_types[0].capitalize()}")
ax3.legend(loc="upper left")
ax3.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
plt.setp(ax3.get_yticklabels(), visible=False)  # Hide y-ticks for shared y-axis

# Fourth subplot (active_levels - data_type 1)
ax4 = plt.subplot(gs[3], sharey=ax1)  # Share y-axis with the first plot
ax4.hist(results[data_types[1]]['active_levels'],
        density=True,
        label=f"{data_types[1].capitalize()}",
        color="tab:orange")
ax4.set_title(f"{data_types[1].capitalize()}")
ax4.legend(loc="upper left")
plt.setp(ax4.get_yticklabels(), visible=False)  # Hide y-ticks for shared y-axis

# Add x-axis labels directly to the subplots instead of using fig.text
ax1.set_xlabel("Active Sequence Length",ha='left')
# ax2.set_xlabel("Active Sequence Length")
ax3.set_xlabel("Mean Branching Factor",ha='left')
# ax4.set_xlabel("Mean Branching Factor")

# set y limit for all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylim(0, 0.4)  # Set y-limit with some margin



# Adjust layout without using tight_layout
plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95, wspace=0.1)
plt.savefig(f"figures/{dataset}_{quantize}_quantitative.png", dpi=300, bbox_inches="tight")
plt.close()

print("Analysis complete. All figures saved to the 'figures' directory.")
