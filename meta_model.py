import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from eval_drug_bank import load_negative_files, get_model


def get_data(pooling, src_model, src_tokenizer, tgt_tokenizer, gen_mol, cold_smiles=0, cold_fasta=0, quantize=False,
             level="drugbank"):
    src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test = load_files(level=level, gen_mol=gen_mol,
                                                                                cold_smiles=cold_smiles,
                                                                                cold_fasta=cold_fasta,
                                                                                quantize=quantize)
    pos_train = SrcTgtDataset(src_train, tgt_train, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    pos_valid = SrcTgtDataset(src_valid, tgt_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    pos_test = SrcTgtDataset(src_test, tgt_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)

    src_neg_train, tgt_neg_train = load_negative_files("train", gen_mol, cold_smiles=cold_smiles, cold_fasta=cold_fasta,
                                                       quantize=quantize, level=level)
    src_neg_valid, tgt_neg_valid = load_negative_files("valid", gen_mol, cold_smiles=cold_smiles, cold_fasta=cold_fasta,
                                                       quantize=quantize, level=level)
    src_neg_test, tgt_neg_test = load_negative_files("test", gen_mol, cold_smiles=cold_smiles, cold_fasta=cold_fasta,
                                                     quantize=quantize, level=level)

    neg_train = SrcTgtDataset(src_neg_train, tgt_neg_train, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    neg_valid = SrcTgtDataset(src_neg_valid, tgt_neg_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    neg_test = SrcTgtDataset(src_neg_test, tgt_neg_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)

    return pos_train, neg_train, pos_valid, neg_valid, pos_test, neg_test, tgt_train


def get_batch_logits(model, batch):
    all_logits = []
    all_labels = []
    for batch in tqdm(batch, desc="Calculating logits", total=len(batch)):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        all_logits.append(output["logits"][:, :-1].detach().cpu().numpy())
        all_labels.append(batch["labels"][:, 1:].detach().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_logits, all_labels


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
    quantize: bool = False

    def to_list(self):
        return [self.size, self.dropout, self.pooling, self.bottleneck_dim, self.learning_rate, self.mol,
                self.cold_smiles,
                self.cold_fasta, self.quantize]


def save_or_load(model, LEVEL):
    datasets = ["train", "valid", "test"]
    splits = ["pos", "neg"]

    logits_dir = "logits"
    os.makedirs(logits_dir, exist_ok=True)
    data = dict()
    for dataset in datasets:
        for split in splits:
            if not os.path.exists(os.path.join(logits_dir, f"{dataset}_{split}_logits_{LEVEL}.npz")):
                data_loader = DataLoader(eval(f"{split}_{dataset}"), batch_size=32, shuffle=False)
                with torch.no_grad():
                    logits, labels = get_batch_logits(model, data_loader)
                np.savez(os.path.join(logits_dir, f"{dataset}_{split}_logits_{LEVEL}.npz"), logits=logits)
                np.savez(os.path.join(logits_dir, f"{dataset}_{split}_labels_{LEVEL}.npz"), labels=labels)
            data[f"{dataset}_{split}_logits"] = \
                np.load(os.path.join(logits_dir, f"{dataset}_{split}_logits_{LEVEL}.npz"))["logits"]
            data[f"{dataset}_{split}_labels"] = \
                np.load(os.path.join(logits_dir, f"{dataset}_{split}_labels_{LEVEL}.npz"))["labels"]
    return data


@dataclass
class Features:
    probs_min: float
    probs_max: float
    probs_mean: float
    probs_std: float
    probs_rank_min: float
    probs_rank_max: float
    probs_rank_mean: float
    probs_rank_std: float
    prob_over_random_min: float
    prob_over_random_max: float
    prob_over_random_mean: float
    prob_over_random_std: float
    probs_to_next_largest_min: float
    probs_to_next_largest_max: float
    probs_to_next_largest_mean: float
    probs_to_next_largest_std: float


def features_list_to_dataframe(features_list):
    """
    Convert a list of Features objects to a pandas DataFrame.
    :param features_list: List of Features objects
    :return: DataFrame
    """
    data = [f.__dict__ for f in features_list]
    df = pd.DataFrame(data)
    return df


def get_sample_features(logits, labels):
    """
    Get the features of the samples from the single sample logits and labels.
    :param logits:  shape (seq_len,vocab_size)
    :param labels:  shape (seq_len) labels is -100 for padding
    """
    probs = []  # log probability of the correct label
    probs_rank = []  # rank of the correct label
    prob_over_random = []  # probability of the correct label over the random uniform distribution
    probs_to_next_largest = []  # probability of the correct label over the next largest label
    for i in range(logits.shape[0]):
        if labels[i] == -100:
            continue
        logit = logits[i]
        log_prob = F.log_softmax(torch.tensor(logit), dim=0)
        prob = log_prob[labels[i]]
        probs.append(prob.item())
        probs_rank.append(torch.argsort(log_prob, descending=True).tolist().index(labels[i]))
        non_zero_log_prob = (~torch.isclose(log_prob, torch.tensor(-1e6))).sum()
        prob_over_random.append(prob.item() - (1 / non_zero_log_prob))
        next_largest = torch.argsort(log_prob, descending=True)[1]
        prob_to_next_largest = prob - log_prob[next_largest]
        probs_to_next_largest.append(prob_to_next_largest.item())
    probs = np.array(probs)
    probs_rank = np.array(probs_rank)
    prob_over_random = np.array(prob_over_random)
    probs_to_next_largest = np.array(probs_to_next_largest)
    features = Features(probs.min(), probs.max(), probs.mean(), probs.std(),
                        probs_rank.min(), probs_rank.max(), probs_rank.mean(), probs_rank.std(),
                        prob_over_random.min(), prob_over_random.max(), prob_over_random.mean(), prob_over_random.std(),
                        probs_to_next_largest.min(), probs_to_next_largest.max(), probs_to_next_largest.mean(),
                        probs_to_next_largest.std())
    return features


def data_split_to_features(logits, labels):
    """
    Get the features of the samples from the logits and labels.
    :param logits:  shape (batch_size,seq_len,vocab_size)
    :param labels:  shape (batch_size,seq_len) labels is -100 for padding
    """
    features = []
    for i in range(logits.shape[0]):
        feature = get_sample_features(logits[i], labels[i])
        features.append(feature)
    return pd.DataFrame(features)


def main():
    cong = Config("l", 0.2, True, 128, 0.0001, False, 0, 0, quantize=True)
    LEVEL = "biosnap"
    size, dropout, pooling, bottleneck_dim, learning_rate, mol, cold_smiles, cold_fasta, quantize = cong.to_list()
    reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=size,
                                                                                     dropout=dropout,
                                                                                     drugbank=True, gen_mol=mol,
                                                                                     quantize=quantize)

    pos_train, neg_train, pos_valid, neg_valid, pos_test, neg_test, tgt_train = get_data(pooling, reaction_model,
                                                                                         reaction_tokenizer,
                                                                                         esm_tokenizer,
                                                                                         gen_mol=mol,
                                                                                         cold_smiles=cold_smiles,
                                                                                         cold_fasta=cold_fasta,
                                                                                         quantize=quantize,
                                                                                         level=LEVEL)

    trie = build_trie(list(set(tgt_train)), esm_tokenizer, max_length=512)
    reaction_model.to(device).eval()
    decoder.to(device).eval()
    model = get_model(decoder, trie, size, dropout, pooling, bottleneck_dim, learning_rate, mol, quantize=quantize,
                      level=LEVEL)
    model.to(device).eval()
    data = save_or_load(model, LEVEL)
    features = {"train": None, "valid": None, "test": None}
    for split in ["train", "valid", "test"]:
        pos_logits = data[f"{split}_pos_logits"]
        pos_labels = data[f"{split}_pos_labels"]
        neg_logits = data[f"{split}_neg_logits"]
        neg_labels = data[f"{split}_neg_labels"]
        pos_features = data_split_to_features(pos_logits, pos_labels)
        neg_features = data_split_to_features(neg_logits, neg_labels)
        pos_features["label"] = 1
        neg_features["label"] = 0
        features[split] = pd.concat([pos_features, neg_features], axis=0)
