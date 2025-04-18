import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import roc_auc_score, accuracy_score

def load_negative_files(split, mol,cold_smiles=0, cold_fasta=0, random_replace=None,quantize=False):
    basd_path = f"data/drugbank"
    if cold_smiles:
        basd_path += f"_cs"
    if cold_fasta:
        basd_path += f"_cf"
    src_file =f"{basd_path}/{split}_reaction_neg.txt"
    tgt_file = f"{basd_path}/{split}_enzyme_neg.txt"
    if quantize:
        if mol:
            src_file= src_file.replace(".txt", "_q.txt")
        else:
            tgt_file= tgt_file.replace(".txt", "_q.txt")
    with open(src_file, "r") as f:
        src = f.read().splitlines()
    with open(tgt_file, "r") as f:
        tgt = f.read().splitlines()
    if mol:
        src, tgt = tgt, src
    if random_replace is not None:
        tgt = [random_replace.get_random_tokens(x) for x in tgt]
    return src, tgt


def get_data(pooling, src_model, src_tokenizer, tgt_tokenizer, gen_mol, return_files=False,cold_smiles=0, cold_fasta=0,
             random_replace=None,train_encoder=False,quantize=False):
    src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test = load_files(level="drugbank", gen_mol=gen_mol,
                                                                                     cold_smiles=cold_smiles,
                                                                                     cold_fasta=cold_fasta,
                                                                                     quantize=quantize)
    if random_replace is not None:
        tgt_train = [random_replace.get_random_tokens(x) for x in tgt_train]
        tgt_valid = [random_replace.get_random_tokens(x) for x in tgt_valid]
        tgt_test = [random_replace.get_random_tokens(x) for x in tgt_test]

    if train_encoder:
        pos_valid = SrcTgtDataset(src_valid, tgt_valid, src_tokenizer, tgt_tokenizer, None, pooling=pooling,train_encoder=True)
        pos_test = SrcTgtDataset(src_test, tgt_test, src_tokenizer, tgt_tokenizer, None, pooling=pooling,train_encoder=True)
    else:
        pos_valid = SrcTgtDataset(src_valid, tgt_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
        pos_test = SrcTgtDataset(src_test, tgt_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)

    src_neg_valid, tgt_neg_valid = load_negative_files("valid", gen_mol,cold_smiles=cold_smiles, cold_fasta=cold_fasta,random_replace=random_replace,quantize=quantize)

    src_neg_test, tgt_neg_test = load_negative_files("test", gen_mol,cold_smiles=cold_smiles, cold_fasta=cold_fasta,random_replace=random_replace,quantize=quantize)
    if train_encoder:
        neg_valid = SrcTgtDataset(src_neg_valid, tgt_neg_valid, src_tokenizer, tgt_tokenizer, None, pooling=pooling,train_encoder=True)
        neg_test = SrcTgtDataset(src_neg_test, tgt_neg_test, src_tokenizer, tgt_tokenizer, None, pooling=pooling,train_encoder=True)
    else:
        neg_valid = SrcTgtDataset(src_neg_valid, tgt_neg_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
        neg_test = SrcTgtDataset(src_neg_test, tgt_neg_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    if return_files:
        return pos_valid, neg_valid, pos_test, neg_test, tgt_train
    return pos_valid, neg_valid, pos_test, neg_test


def get_batch_probabilities(model, batch):
    with torch.no_grad():
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        all_logits = output["logits"][:, :-1]
        all_mask_out = output.trie_mask_out
        input_ids = batch["input_ids"][:, 1:]

        batch_probs = []
        for idx in range(len(input_ids)):
            sample_logits = all_logits[idx][~all_mask_out[idx]]
            sample_ids = input_ids[idx][~all_mask_out[idx]]

            log_prob = F.log_softmax(sample_logits, dim=-1)
            token_log_probs = [log_prob[i, sample_ids[i]].item() for i in range(len(sample_ids))]
            log_prob_mean = np.mean(token_log_probs)
            prob = np.exp(log_prob_mean)
            batch_probs.append(prob)

        return batch_probs


def find_optimal_threshold(y_true, y_scores, metric='accuracy'):
    """Find the optimal threshold for either accuracy or F1 score"""
    thresholds = np.unique(y_scores)
    best_threshold = 0
    best_score = 0

    for threshold in thresholds:
        y_pred = (np.array(y_scores) >= threshold).astype(int)

        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def evaluate_model(pos_dataset, neg_dataset, model, batch_size=32, return_prob=False, best_acc_threshold=None,
                   best_f1_threshold=None):
    pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=False)
    neg_dataloader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of positive examples: {len(pos_dataset)}")
    print(f"Number of negative examples: {len(neg_dataset)}")


    pos_prob = []

    for batch in tqdm(pos_dataloader):
        batch_probs = get_batch_probabilities(model, batch)
        pos_prob.extend(batch_probs)
    pos_prob = np.array(pos_prob)

    # Process negative examples in batches
    neg_prob = []
    for batch in tqdm(neg_dataloader):
        batch_probs = get_batch_probabilities(model, batch)
        neg_prob.extend(batch_probs)
    neg_prob = np.array(neg_prob)

    # Calculate metrics
    y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
    y_scores = np.concatenate([pos_prob, neg_prob])
    if return_prob:
        return y_true, y_scores

    auc_score = roc_auc_score(y_true, y_scores)
    if best_acc_threshold is None:
        best_acc_threshold, best_acc_score = find_optimal_threshold(y_true, y_scores, metric='accuracy')
    else:
        best_acc_score = accuracy_score(y_true, (y_scores >= best_acc_threshold).astype(int))
    if best_f1_threshold is None:
        best_threshold_f1, best_score_f1 = find_optimal_threshold(y_true, y_scores, metric='f1')
    else:
        best_score_f1 = f1_score(y_true, (y_scores >= best_f1_threshold).astype(int))
    return auc_score, best_acc_threshold, best_acc_score, best_f1_threshold, best_score_f1


def get_best_cp(base_path):
    all_cp_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if
                   os.path.isdir(os.path.join(base_path, d)) and d.startswith("checkpoint")]
    all_cp_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    last_cp_dir = all_cp_dirs[-1]
    import json
    with open(os.path.join(last_cp_dir, "trainer_state.json"), "r") as f:
        trainer_state = json.load(f)
    best_cp_dir = trainer_state["best_model_checkpoint"]
    model_path = f"{best_cp_dir}/pytorch_model.bin"
    return model_path


from dataclasses import dataclass


@dataclass
class Config:
    size: str = "l"
    dropout: float = 0.0
    pooling: bool = True
    bottleneck_dim: int = 128
    learning_rate: float = 0.0001
    mol: bool = True
    cold_smiles: int = 0
    cold_fasta: int = 0

    def to_list(self):
        return [self.size, self.dropout, self.pooling, self.bottleneck_dim, self.learning_rate, self.mol, self.cold_smiles,
                self.cold_fasta]



def get_model(decoder,trie,size, dropout, pooling, bottleneck_dim, learning_rate, mol):
    encoder_dim = 768 if not mol else 1280
    model = EnzymeDecoder(decoder, trie=trie, encoder_dim=encoder_dim, bottleneck_dim=bottleneck_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {n_params:,}")

    output_dir = f"drugbank_{size}_{dropout}_{learning_rate}"
    if bottleneck_dim > 0:
        output_dir += f"_bottleneck_{bottleneck_dim}"
    if pooling:
        output_dir += "_pooling"
    if mol:
        output_dir += "_mol"

    model_path = f"results_db/{output_dir}/"
    model_path = get_best_cp(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"Model loaded from {model_path}")
    return model

def main():
    cong1 = Config("l", 0.0, True, 128, 0.0001, True, 0, 0)
    cong2 = Config("l", 0.0, True, 128, 0.0001, False, 0, 0)

    all_scores = []
    all_true = []

    all_scores_val = []
    all_true_val = []
    for cong in [cong1, cong2]:
        size, dropout, pooling, bottleneck_dim, learning_rate, mol, cold_smiles, cold_fasta = cong.to_list()
        reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=size,
                                                                                         dropout=dropout,
                                                                                         drugbank=True, gen_mol=mol)

        pos_valid, neg_valid, pos_test, neg_test, trie_files = get_data(pooling, reaction_model, reaction_tokenizer,
                                                                        esm_tokenizer,
                                                                        gen_mol=mol,
                                                                        return_files=True,
                                                                        cold_smiles=cold_smiles,
                                                                        cold_fasta=cold_fasta,)

        trie = build_trie(list(set(trie_files)), esm_tokenizer, max_length=512)
        reaction_model.to(device).eval()
        decoder.to(device).eval()
        model = get_model(decoder, trie, size, dropout, pooling, bottleneck_dim, learning_rate, mol)


        y_true, y_scores = evaluate_model(pos_test, neg_test, model, batch_size=32, return_prob=True)
        all_scores.append(y_scores)
        all_true.append(y_true)


        y_true_val, y_scores_val = evaluate_model(pos_valid, neg_valid, model, batch_size=32, return_prob=True)
        all_scores_val.append(y_scores_val)
        all_true_val.append(y_true_val)



    y_true=all_true[0]
    y_scores=all_scores[0]
    y_true_val=all_true_val[0]
    y_scores_val=all_scores_val[0]
    # Calculate metrics
    auc= roc_auc_score(y_true, y_scores)
    best_acc_threshold, best_acc_score = find_optimal_threshold(y_true_val, y_scores_val, metric='accuracy')
    best_f1_threshold, best_f1_score = find_optimal_threshold(y_true_val, y_scores_val, metric='f1')
    print(f"AUC: {auc:.4f}")
    print(f"Best accuracy threshold: {best_acc_threshold:.4f}, Best accuracy score: {best_acc_score:.4f}")
    print(f"Best F1 threshold: {best_f1_threshold:.4f}, Best F1 score: {best_f1_score:.4f}")
    # test acc,f1
    best_acc_score = accuracy_score(y_true, (y_scores >= best_acc_threshold).astype(int))
    best_f1_score = f1_score(y_true, (y_scores >= best_f1_threshold).astype(int))
    print(f"Test accuracy score: {best_acc_score:.4f}")
    print(f"Test F1 score: {best_f1_score:.4f}")


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.hist(y_scores[y_true==1], bins=50, alpha=0.5, label='Positive', color='blue')
    plt.hist(y_scores[y_true==0], bins=50, alpha=0.5, label='Negative', color='red')
    plt.axvline(x=best_acc_threshold, color='green', linestyle='--', label='Best Accuracy Threshold')
    plt.axvline(x=best_f1_threshold, color='orange', linestyle='--', label='Best F1 Threshold')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Probabilities')
    plt.legend()



    assert (all_true[0] == all_true[1]).all()
    all_scores = np.stack(all_scores)
    y_true = all_true[0]
    auc = roc_auc_score(y_true, all_scores.sum(axis=0))
    best_acc_threshold, best_acc_score = find_optimal_threshold(y_true, all_scores.sum(axis=0), metric='accuracy')
    best_f1_threshold, best_f1_score = find_optimal_threshold(y_true, all_scores.sum(axis=0), metric='f1')
    print(f"AUC: {auc:.4f}")
    print(f"Best accuracy threshold: {best_acc_threshold:.4f}, Best accuracy score: {best_acc_score:.4f}")
    print(f"Best F1 threshold: {best_f1_threshold:.4f}, Best F1 score: {best_f1_score:.4f}")
    # Save the results



if __name__ == "__main__":
    main()
