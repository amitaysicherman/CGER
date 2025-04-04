from train import load_files, get_encoder_decoder, EnzymeDecoder, SrcTgtDataset
from trie import build_trie, build_mask_from_trie
from transformers import AutoTokenizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

tab10 = plt.cm.get_cmap('tab10')
colors = [(0, 0, 0, 0), tab10(1), tab10(0)]
custom_cmap = mcolors.ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

sns.set(style="whitegrid")

src_train, tgt_train, src_test, tgt_test = load_files(level="easy")
print(src_train[tgt_train.index("MIEVLLVTICLAVFPYPGSSIILESGNVDDYEVVYPQKLTALPKGAVQPKYEDAMQYEFKVNGEPVVLHLEKNKGLFSEDYSETHYSPDGREITTYPSVEDHCYYHGRIQNDADSTASISACDGLKGYFKLQGETYLIEPLELSDSEAHAVFKYENVEKEDEAPKMCGVTQNWESDESIKKASQLYLTPEQQRFPQRYIELAIVVDHGMYTKYSSNFKKIRKRVHQMVNNINEMYRPLNIAITLSLLDVWSEKDLITMQAVAPTTARLFGDWRETVLLKQKDHDHAQLLTDINFTGNTIGWAYMGGMCNAKNSVGIVKDHSSNVFMVAVTMTHEIGHNLGMEHDDKDKCKCEACIMSAVISDKPSKLFSDCSKDYYQTFLTNSKPQCIINAPLRTDTVSTPVSGNEFLEAGEECDCGSPSNPCCDAATCKLRPGAQCADGLCCDQCRFKKKRTICRRARGDNPDDRCTGQSADCPRNS")])
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
trie = build_trie(list(set(tgt_train + tgt_test)), esm_tokenizer)
all_enzyme_sequences = list(set(tgt_train + tgt_test))

active_seq_len = []
active_width = []
for ii, seq in enumerate(tqdm(all_enzyme_sequences)):
    tokens = esm_tokenizer(
        seq, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )
    mask_in = build_mask_from_trie(trie, tokens["input_ids"], esm_tokenizer.vocab_size)[0]
    active_seq_len.append((mask_in.sum(dim=1) > 1).sum().item())
    last_mask_in = (mask_in.sum(dim=1) > 1)

    if sum(last_mask_in[5:28]) >= 2:
        print(ii, f"seq: {seq}")

        fig, ax = plt.subplots(figsize=(10, 5))
        y_labels = [esm_tokenizer.decode([i]) for i in range(esm_tokenizer.vocab_size)][4:-8]
        mask_to_plot = mask_in.cpu().numpy().T[4:-8, :30]

        for i in range(mask_to_plot.shape[1]):
            if mask_to_plot[:, i].sum() == 1:
                for j in range(mask_to_plot.shape[0]):
                    if mask_to_plot[j, i] == 1:
                        mask_to_plot[j, i] = 2
        sns.heatmap(mask_to_plot, cmap=custom_cmap, norm=norm, cbar=False, ax=ax, yticklabels=y_labels)
        transparent_patch = mpatches.Patch(color='white', label='Restricted Token', alpha=0.3, hatch='/')
        green_patch = mpatches.Patch(color=tab10(1), label='Available Token')
        blue_patch = mpatches.Patch(color=tab10(0), label='Determined Token')

        # Add legend to the plot
        ax.legend(handles=[transparent_patch, green_patch, blue_patch],
                  title="Token Status",
                  loc='upper right',
                  # bbox_to_anchor=(1.05, 1),
                  frameon=True
                  )

        plt.tight_layout()

        plt.savefig(f"figures/mask_{ii}.png")

    mask_in = mask_in[mask_in.sum(dim=1) > 1]
    active_width.extend(mask_in.sum(dim=1).tolist())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(active_seq_len)
ax1.set_xlabel("Active Sequence Length")
ax1.set_ylabel("Count")
ax1.set_title("Active Sequence Length Distribution")
ax2.hist(active_width)
ax2.set_xlabel("Active Width")
ax2.set_ylabel("Count")
ax2.set_title("Active Width Distribution")
plt.tight_layout()
plt.savefig("figures/active_seq_len_width.png")
