import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
import numpy as np
import random
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_train, tgt_train, src_test, tgt_test = load_files(level="drugbank")
reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size="m", dropout=0.2,
                                                                                 drugbank=True)

all_train_fasta = {x for x in tgt_train}
all_train_smiles = {x for x in src_train}

ignore_indexes = []
for i in range(len(tgt_test)):
    if tgt_test[i] not in all_train_fasta:
        ignore_indexes.append(i)
    if src_test[i] not in all_train_smiles:
        ignore_indexes.append(i)
print(f"len ignore_indexes: {len(ignore_indexes)}")
src_test = [x for i, x in enumerate(src_test) if i not in ignore_indexes]
tgt_test = [x for i, x in enumerate(tgt_test) if i not in ignore_indexes]
print(f"len src_test: {len(src_test)}")
reaction_model.to(device).eval()
decoder.to(device).eval()

pos_dataset = SrcTgtDataset(src_test, tgt_test, reaction_tokenizer, esm_tokenizer, reaction_model, pooling=True)
trie = build_trie(list(set(tgt_train + tgt_test)), esm_tokenizer)
model = EnzymeDecoder(decoder, trie=trie, encoder_dim=768)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {n_params:,}")

model_path = "results_drugbank/drugbank_m_0.2_0.0001_pooling/checkpoint-18000/pytorch_model.bin"
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
    if smiles not in all_train_smiles:
        continue
    if fasta not in all_train_fasta:
        continue
    neg_smiles.append(smiles)
    neg_fasta.append(fasta)

neg_dataset = SrcTgtDataset(neg_smiles, neg_fasta, reaction_tokenizer, esm_tokenizer, reaction_model, pooling=True)


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
for line_index, test_data in tqdm(enumerate(pos_dataset)):
    prob = get_probabilitiy(model, test_data)
    pos_prob.append(prob)
    # print(f"line_index: {line_index}, prob: {prob}")
pos_prob = np.array(pos_prob)

neg_prob = []
neg_index = random.choices(range(len(neg_dataset)), k=len(pos_prob))
for line_index in tqdm(neg_index):
    test_data = neg_dataset[line_index]
    prob = get_probabilitiy(model, test_data)
    neg_prob.append(prob)
    # print(f"line_index: {line_index}, prob: {prob}")
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
    accuracy = (y_true == y_pred).mean()
    return precision, recall, f1, accuracy


def evaluate_model(pos_prob, neg_prob):
    y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
    y_scores = np.concatenate([pos_prob, neg_prob])

    # Calculate AUC
    auc_score = roc_auc_score(y_true, y_scores)
    print(f"AUC: {auc_score:.4f}")
    all_possible_thresholds = np.unique(y_scores)
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_acc = 0
    pbar = tqdm(all_possible_thresholds,total=len(all_possible_thresholds))
    for threshold in pbar:
        y_pred = (y_scores >= threshold).astype(int)
        precision, recall, f1, acc = calculate_metrics(y_true, y_pred)
        print(
            f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Accuracy: {acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if acc > best_acc:
            best_acc = acc
        pbar.set_description(f"Best F1: {best_f1:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best Accuracy: {best_acc:.4f}")
    print(f"Best F1: {best_f1:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best Accuracy: {best_acc:.4f}")

evaluate_model(pos_prob, neg_prob)
