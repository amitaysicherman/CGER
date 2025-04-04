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
parser.add_argument("--size", type=str, default="l")
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--learning_rate", type=float, default=1e-4)
args = parser.parse_args()

#
# class Args:
#     def __init__(self, level="easy", size="l", dropout=0.3, learning_rate=1e-4, trie=1):
#         self.level = level
#         self.size = size
#         self.dropout = dropout
#         self.learning_rate = learning_rate
#         self.trie = trie
#
#
# args=Args()


src_train, tgt_train, src_test, tgt_test = load_files(level=args.level)
reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=args.size,
                                                                                 dropout=args.dropout)
reaction_model.to(device).eval()
decoder.to(device).eval()

test_dataset = SrcTgtDataset(src_test, tgt_test, reaction_tokenizer, esm_tokenizer, reaction_model)
trie = build_trie(list(set(tgt_train + tgt_test)), esm_tokenizer)

3/0

model = EnzymeDecoder(decoder, trie=trie)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {n_params:,}")

cp_dir = f"results/{args.level}_{args.size}_{args.dropout}_{args.learning_rate}"
all_cp_dirs = [os.path.join(cp_dir, d) for d in os.listdir(cp_dir) if
               os.path.isdir(os.path.join(cp_dir, d)) and d.startswith("checkpoint")]
all_cp_dirs.sort(key=lambda x: int(x.split("-")[-1]))
last_cp_dir = all_cp_dirs[-1]
model_path = os.path.join(last_cp_dir, "pytorch_model.bin")
model.load_state_dict(torch.load(model_path))
model.eval().to(device)
all_enzyme_tokens = []
all_enzyme_sequences = list(set(tgt_train + tgt_test))
for seq in all_enzyme_sequences:
    tokens = esm_tokenizer(
        seq, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )
    all_enzyme_tokens.append(tokens)
all_input_ids = torch.cat([tokens["input_ids"] for tokens in all_enzyme_tokens], dim=0)
all_attention_mask = torch.cat([tokens["attention_mask"] for tokens in all_enzyme_tokens], dim=0)

all_res_pred = []
all_res_scan = []
output_base = "scores"
os.makedirs(output_base, exist_ok=True)
batch_size = 128
pbar = tqdm(test_dataset, total=len(test_dataset))
for line_index, test_data in enumerate(pbar):
    output_file = os.path.join(output_base, f"{line_index}.txt")
    test_data = {k: v.unsqueeze(0).to(device) for k, v in test_data.items()}
    gt_seq = esm_tokenizer.decode(test_data["input_ids"][0].tolist(), skip_special_tokens=True).replace(" ", "")
    with open(output_file, "w") as f:
        f.write(f"GT: {gt_seq}\n")
    with torch.no_grad():
        predict_result = model(**test_data)
    predictions = predict_result.logits[0].argmax(dim=-1).cpu().numpy()
    with open(output_file, "a") as f:
        f.write(f"Pred: {esm_tokenizer.decode(predictions.tolist(), skip_special_tokens=True).replace(' ', '')}\n")
    predictions = predictions[:-1]
    labels = test_data["labels"][0, 1:].cpu().numpy()
    mask = labels != -100
    all_res_pred.append((labels[mask] == predictions[mask]).all())

    with torch.no_grad():
        all_scores = []
        for i in range(0, len(all_input_ids), batch_size):
            batch_input_ids = all_input_ids[i:i + batch_size].to(device)
            batch_attention_mask = all_attention_mask[i:i + batch_size].to(device)
            scan_result = model(
                encoder_outputs=test_data["encoder_outputs"].repeat(len(batch_input_ids), 1, 1),
                encoder_attention_mask=test_data["encoder_attention_mask"].repeat(len(batch_input_ids), 1),
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=None
            )
            all_logits = scan_result["logits"][:, :-1]
            all_mask_out = scan_result.trie_mask_out
            all_enzyme_option = batch_input_ids[:, 1:]
            for enzyme_option, logit, mask_out in zip(all_enzyme_option, all_logits, all_mask_out):
                enzyme_option_text = esm_tokenizer.decode(enzyme_option.tolist(), skip_special_tokens=True).replace(" ",
                                                                                                                    "")
                enzyme_option = enzyme_option[~mask_out.bool()]
                enzyme_option_mask_text = esm_tokenizer.decode(enzyme_option.tolist(),
                                                               skip_special_tokens=True).replace(" ",
                                                                                                 "")
                mask_in_text = "-".join(str(x.item()) for x in (~mask_out.bool()).nonzero(as_tuple=True)[0])
                logit = logit[~mask_out.bool()]
                log_probs = F.log_softmax(logit, dim=-1)
                seq_log_prob = [log_probs[i, enzyme_option[i]].item() for i in range(len(enzyme_option))]
                seq_tokens_rank = []
                # the rank (i.e the index of the scores in sorting) of the token between all the options
                for index_in_seq, token in enumerate(enzyme_option):
                    seq_tokens_rank.append(
                        str((log_probs[index_in_seq].argsort(descending=True) == token).nonzero(as_tuple=True)[
                                0].item()))
                # write enzyme_option_text,enzyme_option_mask_text, mask_in_text, seq_tokens_rank, seq_log_prob
                with open(output_file, "a") as f:
                    f.write(
                        f"Scan: {enzyme_option_text}, Mask: {enzyme_option_mask_text}, Mask_in: {mask_in_text}, "
                        f"Rank: {seq_tokens_rank}, Log_prob: {seq_log_prob}, Score:{np.mean(seq_log_prob)}\n")
                all_scores.append(np.mean(seq_log_prob))
    all_scores = np.array(all_scores)
    best_idx = np.argmax(all_scores)
    best_score = all_scores[best_idx]
    best_seq = all_enzyme_tokens[best_idx]['input_ids']
    best_seq = esm_tokenizer.decode(best_seq[0].tolist(), skip_special_tokens=True)
    best_seq = best_seq.replace(" ", "")
    all_res_scan.append(gt_seq == best_seq)
    pbar.set_description(f"Accuracy Scan: {np.mean(all_res_scan):.4f}, Accuracy Pred: {np.mean(all_res_pred):.4f}")
