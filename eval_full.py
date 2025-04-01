import argparse
from train import load_files, get_encoder_decoder, EnzymeDecoder, SrcTgtDataset
import torch
import numpy as np
from trie import build_trie
import os
from torch.nn import functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--level", type=str, default="easy")
parser.add_argument("--size", type=str, default="m")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--trie", type=int, default=1)
args = parser.parse_args()

src_train, tgt_train, src_test, tgt_test = load_files(level=args.level)
reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=args.size,
                                                                                 dropout=args.dropout)
reaction_model.to(device)
reaction_model.eval()
decoder.to(device)
decoder.eval()
test_dataset = SrcTgtDataset(src_test, tgt_test, reaction_tokenizer, esm_tokenizer, reaction_model)

all_enzyme_tokens = []
all_enzyme_sequences = list(set(tgt_train + tgt_test))
for seq in all_enzyme_sequences:
    tokens = esm_tokenizer.encode(seq, add_special_tokens=True)
    if len(tokens) > 512:
        tokens = tokens[:511] + [esm_tokenizer.eos_token_id]
    all_enzyme_tokens.append(torch.tensor(tokens).unsqueeze(0))

if args.trie:
    trie = build_trie(tgt_train + tgt_test, esm_tokenizer)
else:
    trie = None
model = EnzymeDecoder(decoder, trie=trie)
cp_dir = f"results/{args.level}_{args.size}_{args.dropout}_{args.learning_rate}"
if args.trie == 0:
    cp_dir += "_notrie"
all_cp_dirs = [os.path.join(cp_dir, d) for d in os.listdir(cp_dir) if
               os.path.isdir(os.path.join(cp_dir, d)) and d.startswith("checkpoint")]
all_cp_dirs.sort(key=lambda x: int(x.split("-")[-1]))
last_cp_dir = all_cp_dirs[-1]
model_path = os.path.join(last_cp_dir, "pytorch_model.bin")
model.load_state_dict(torch.load(model_path))
all_res = []
model.eval().to(device)
pbar = tqdm(test_dataset, total=len(test_dataset))
for test_data in pbar:
    test_data = {k: v.unsqueeze(0).to(device) for k, v in test_data.items()}
    all_scores = []
    gt_seq = esm_tokenizer.decode(test_data["input_ids"][0].tolist(), skip_special_tokens=True)
    gt_seq = gt_seq.replace(" ", "")
    gt_index=all_enzyme_sequences.index(gt_seq)
    print(f"Ground truth sequence: {gt_seq} ({gt_index})")
    for enzyme_option in tqdm(all_enzyme_tokens):
        enzyme_option = enzyme_option.to(device)
        with torch.no_grad():
            model_output = model(
                encoder_outputs=test_data["encoder_outputs"],
                encoder_attention_mask=test_data["encoder_attention_mask"],
                input_ids=enzyme_option,
                attention_mask=torch.ones_like(enzyme_option).to(device),
                labels=None
            )
            logits = model_output["logits"][0][:-1]
            mask_out = model_output.trie_mask_out[0]
            enzyme_option = enzyme_option[0][1:]
            enzyme_option = enzyme_option[~mask_out.bool()]
            logits = logits[~mask_out.bool()]
            log_probs = F.log_softmax(logits, dim=-1)
            seq_log_prob = [log_probs[i, enzyme_option[i]].item() for i in range(len(enzyme_option))]
            seq_log_prob = np.mean(seq_log_prob)
            all_scores.append(seq_log_prob)
    all_scores = np.array(all_scores)
    best_idx = np.argmax(all_scores)
    best_score = all_scores[best_idx]
    best_seq = all_enzyme_tokens[best_idx]
    best_seq = esm_tokenizer.decode(best_seq[0].tolist(), skip_special_tokens=True)
    best_seq = best_seq.replace(" ", "")
    gt_seq = esm_tokenizer.decode(test_data["input_ids"][0].tolist(), skip_special_tokens=True)
    gt_seq = gt_seq.replace(" ", "")
    all_res.append(gt_seq == best_seq)
    pbar.set_description(f"Accuracy: {np.mean(all_res):.4f}")
