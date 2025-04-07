import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
import numpy as np
import random
from tqdm import tqdm
import os

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import roc_auc_score


def get_data(pooling, reaction_model, reaction_tokenizer, esm_tokenizer):
    src_train, tgt_train, src_test, tgt_test = load_files(level="drugbank")
    all_train_fasta = {x for x in tgt_train}
    all_train_smiles = {x for x in src_train}
    trie = build_trie(list(set(tgt_train + tgt_test)), esm_tokenizer)

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
    pos_dataset = SrcTgtDataset(src_test, tgt_test, reaction_tokenizer, esm_tokenizer, reaction_model, pooling=pooling)
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

    neg_dataset = SrcTgtDataset(neg_smiles, neg_fasta, reaction_tokenizer, esm_tokenizer, reaction_model,
                                pooling=pooling)
    return pos_dataset, neg_dataset, trie


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


def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = (y_true == y_pred).mean()
    return precision, recall, f1, accuracy


def evaluate_model_all(pos_prob, neg_prob):
    y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
    y_scores = np.concatenate([pos_prob, neg_prob])

    all_possible_thresholds = np.unique(y_scores)
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_acc = 0
    pbar = tqdm(all_possible_thresholds, total=len(all_possible_thresholds))
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
        pbar.set_description(
            f"Best F1: {best_f1:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best Accuracy: {best_acc:.4f}")
    print(
        f"Best F1: {best_f1:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best Accuracy: {best_acc:.4f}")


def evaluate_model(pos_dataset, neg_dataset, model, eval_all=False):
    pos_prob = []
    for line_index, test_data in tqdm(enumerate(pos_dataset)):
        prob = get_probabilitiy(model, test_data)
        pos_prob.append(prob)
    pos_prob = np.array(pos_prob)

    neg_prob = []
    neg_index = random.choices(range(len(neg_dataset)), k=len(pos_prob))
    for line_index in tqdm(neg_index):
        test_data = neg_dataset[line_index]
        prob = get_probabilitiy(model, test_data)
        neg_prob.append(prob)
    neg_prob = np.array(neg_prob)
    y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
    y_scores = np.concatenate([pos_prob, neg_prob])
    auc_score = roc_auc_score(y_true, y_scores)

    if eval_all:
        print(f"AUC: {auc_score:.4f}")
        evaluate_model_all(pos_prob, neg_prob)
    return {"auc",auc_score}


def main():
    size = "m"
    dropout = 0.3
    pooling = False
    bottleneck_dim = 0
    learning_rate = 0.0001
    reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=size, dropout=dropout,
                                                                                     drugbank=True)

    pos_dataset, neg_dataset, trie = get_data(pooling, reaction_model, reaction_tokenizer, esm_tokenizer)
    reaction_model.to(device).eval()
    decoder.to(device).eval()

    model = EnzymeDecoder(decoder, trie=trie, encoder_dim=768, bottleneck_dim=bottleneck_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {n_params:,}")

    output_dir = f"drugbank_{size}_{dropout}_{learning_rate}"
    if bottleneck_dim > 0:
        output_dir += f"_bottleneck_{bottleneck_dim}"
    if pooling:
        output_dir += "_pooling"

    model_path = f"results_drugbank/{output_dir}/"
    all_cp_dirs = [os.path.join(model_path, d) for d in os.listdir(model_path) if
                   os.path.isdir(os.path.join(model_path, d)) and d.startswith("checkpoint")]
    all_cp_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    last_cp_dir = all_cp_dirs[-1]
    model_path = f"{last_cp_dir}/pytorch_model.bin"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"Model loaded from {model_path}")

    print(evaluate_model(pos_dataset, neg_dataset, model, eval_all=True))


if __name__ == "__main__":
    main()
