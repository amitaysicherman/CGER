import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_train, tgt_train, src_test, tgt_test = load_files(level="drugbank")
reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size="l", dropout=0.2,
                                                                                 drugbank=True)
reaction_model.to(device).eval()
decoder.to(device).eval()

pos_dataset = SrcTgtDataset(src_test, tgt_test, reaction_tokenizer, esm_tokenizer, reaction_model)
trie = build_trie(list(set(tgt_train + tgt_test)), esm_tokenizer)
model = EnzymeDecoder(decoder, trie=trie, encoder_dim=768, bottleneck_dim=128)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {n_params:,}")

model_path = "results_drugbank/drugbank_l_0.2_0.0001_bottleneck_128/checkpoint-7000/pytorch_model.bin"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

with open("data/drugbank/DrugBank.txt", "r") as f:
    lines = f.read().splitlines()
neg_smiles = []
neg_fasta = []
for i, line in enumerate(lines):
    _, __, smiles, fasta, label = line.split(" ")
    label = int(label)
    if label == 1:
        continue
    neg_smiles.append(smiles)
    neg_fasta.append(fasta)

neg_dataset = SrcTgtDataset(neg_smiles, neg_fasta, reaction_tokenizer, esm_tokenizer, reaction_model)


def get_probabilitiy(model, sample):
    with torch.no_grad():
        model.eval()
        sample = {k: v.to(device) for k, v in sample.items()}
        sample = {k: v.unsqueeze(0) for k, v in sample.items()}
        output = model(**sample)
        all_logits = output["logits"][:, :-1]
        all_mask_out = output.trie_mask_out
        input_ids = sample["input_ids"][:, 1:]
        input_ids = input_ids[~all_mask_out]
        logits = all_logits[~all_mask_out]
        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = [log_prob[i, input_ids[i]].item() for i in range(len(input_ids))]
        log_prob_mean = np.mean(log_prob)
        prob = np.exp(log_prob_mean)
        return prob


pos_prob = []
for line_index, test_data in enumerate(pos_dataset):
    prob = get_probabilitiy(model, test_data)
    pos_prob.append(prob)
    print(f"line_index: {line_index}, prob: {prob}")
    if line_index > 100:
        break
pos_prob = np.array(pos_prob)

neg_prob = []
neg_index = random.choices(range(len(neg_dataset)), k=len(pos_prob))
for line_index in neg_index:
    test_data = neg_dataset[line_index]
    prob = get_probabilitiy(model, test_data)
    neg_prob.append(prob)
    print(f"line_index: {line_index}, prob: {prob}")
neg_prob = np.array(neg_prob)

# calculate AUC
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score











def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1


def evaluate_model(pos_prob, neg_prob):
    y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
    y_scores = np.concatenate([pos_prob, neg_prob])

    # Calculate AUC
    auc_score = roc_auc_score(y_true, y_scores)
    print(f"AUC: {auc_score:.4f}")

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred = (y_scores >= threshold).astype(int)
        precision, recall, f1 = calculate_metrics(y_true, y_pred)
        print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


evaluate_model(pos_prob, neg_prob)
