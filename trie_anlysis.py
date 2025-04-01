from train import load_files
from trie import build_trie, build_mask_from_trie
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

# set plt backend
# plt.switch_backend('TkAgg')

src_train, tgt_train, src_test, tgt_test = load_files(level="easy")
inputs = list(set(tgt_train + tgt_test))

esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)
trie = build_trie(inputs, esm_tokenizer)

max_length = 512

length = []
active_length = []
active_levels = []
per_input_levels = [[] for _ in range(max_length)]
for seq in tqdm(inputs):
    tokens = esm_tokenizer.encode(seq, add_special_tokens=True)
    length.append(len(tokens))

    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [esm_tokenizer.eos_token_id]
    tensor_tokens = torch.tensor(tokens).unsqueeze(0)
    mask = build_mask_from_trie(trie, tensor_tokens, esm_tokenizer.vocab_size).squeeze(0)
    active_mask = mask.sum(dim=-1) > 1
    active_length.append(active_mask.sum().item())
    levels = mask[active_mask].sum(dim=-1).tolist()
    for i in range(len(levels)):
        per_input_levels[i].append(levels[i])
    active_levels.extend(levels)


fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(15, 5))
ax1.hist(length)
ax2.hist(active_length)
ax3.hist(active_levels)
ax4.plot([np.mean(per_input_levels[i]) for i in range(max_length)])
fig.savefig("figures/trie_analysis.png")

# plt.show()
