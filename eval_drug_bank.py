import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import os

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def remove_stereo_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def load_negative_files(split, mol):
    with open(f"data/drugbank/{split}_reaction_neg.txt", "r") as f:
        src = f.read().splitlines()
    with open(f"data/drugbank/{split}_enzyme_neg.txt", "r") as f:
        tgt = f.read().splitlines()
    if mol:
        src, tgt = tgt, src
    return src, tgt


def get_data(pooling, src_model, src_tokenizer, tgt_tokenizer, gen_mol, return_files=False):
    src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test = load_files(level="drugbank", gen_mol=gen_mol)
    pos_valid = SrcTgtDataset(src_valid, tgt_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    pos_test = SrcTgtDataset(src_test, tgt_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)

    src_neg_valid, tgt_neg_valid = load_negative_files("valid", gen_mol)

    src_neg_test, tgt_neg_test = load_negative_files("test", gen_mol)
    neg_valid = SrcTgtDataset(src_neg_valid, tgt_neg_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    neg_test = SrcTgtDataset(src_neg_test, tgt_neg_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    return pos_valid, neg_valid, pos_test, neg_test



# def get_probabilitiy(model, sample):
#     with torch.no_grad():
#         model.eval()
#         sample = {k: v.to(device) for k, v in sample.items()}
#         sample = {k: v.unsqueeze(0) for k, v in sample.items()}
#         output = model(**sample)
#         all_logits = output["logits"][:, :-1]
#         all_mask_out = output.trie_mask_out
#         input_ids = sample["input_ids"][:, 1:]
#         input_ids = input_ids[~all_mask_out]
#         logits = all_logits[~all_mask_out]
#         log_prob = F.log_softmax(logits, dim=-1)
#         log_prob = [log_prob[i, input_ids[i]].item() for i in range(len(input_ids))]
#         log_prob_mean = np.mean(log_prob)
#         prob = np.exp(log_prob_mean)
#         return prob


# def calculate_metrics(y_true, y_pred):
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     accuracy = (y_true == y_pred).mean()
#     return precision, recall, f1, accuracy
#
#
# def evaluate_model_all(pos_prob, neg_prob):
#     y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
#     y_scores = np.concatenate([pos_prob, neg_prob])
#
#     all_possible_thresholds = np.unique(y_scores)
#     best_precision = 0
#     best_recall = 0
#     best_f1 = 0
#     best_acc = 0
#     pbar = tqdm(all_possible_thresholds, total=len(all_possible_thresholds))
#     for threshold in pbar:
#         y_pred = (y_scores >= threshold).astype(int)
#         precision, recall, f1, acc = calculate_metrics(y_true, y_pred)
#         print(
#             f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Accuracy: {acc:.4f}")
#         if f1 > best_f1:
#             best_f1 = f1
#         if precision > best_precision:
#             best_precision = precision
#         if recall > best_recall:
#             best_recall = recall
#         if acc > best_acc:
#             best_acc = acc
#         pbar.set_description(
#             f"Best F1: {best_f1:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best Accuracy: {best_acc:.4f}")
#     print(
#         f"Best F1: {best_f1:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best Accuracy: {best_acc:.4f}")


# def evaluate_model(pos_dataset, neg_dataset, model, eval_all=False):
#     pos_prob = []
#     for line_index, test_data in tqdm(enumerate(pos_dataset)):
#         prob = get_probabilitiy(model, test_data)
#         pos_prob.append(prob)
#     pos_prob = np.array(pos_prob)
#
#     neg_prob = []
#     neg_index = random.choices(range(len(neg_dataset)), k=len(pos_prob))
#     for line_index in tqdm(neg_index):
#         test_data = neg_dataset[line_index]
#         prob = get_probabilitiy(model, test_data)
#         neg_prob.append(prob)
#     neg_prob = np.array(neg_prob)
#     y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
#     y_scores = np.concatenate([pos_prob, neg_prob])
#     auc_score = roc_auc_score(y_true, y_scores)
#
#     if eval_all:
#         print(f"AUC: {auc_score:.4f}")
#         evaluate_model_all(pos_prob, neg_prob)
#     return {"auc":auc_score}
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


def evaluate_model(pos_dataset, neg_dataset, model, batch_size=32, return_prob=False):
    pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of positive examples: {len(pos_dataset)}")
    print(f"Number of negative examples: {len(neg_dataset)}")
    k = min(len(pos_dataloader), len(neg_dataset))
    neg_indices = random.choices(range(len(neg_dataset)), k=k)
    neg_sampled_dataset = torch.utils.data.Subset(neg_dataset, neg_indices)
    neg_dataloader = DataLoader(neg_sampled_dataset, batch_size=batch_size, shuffle=False)

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

    return auc_score


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

    def to_list(self):
        return [self.size, self.dropout, self.pooling, self.bottleneck_dim, self.learning_rate, self.mol]


def main():
    cong1 = Config("l", 0.0, True, 128, 0.0001, True)
    cong2 = Config("l", 0.0, True, 128, 0.0001, False)

    all_scores = []
    all_true = []
    for cong in [cong1, cong2]:
        size, dropout, pooling, bottleneck_dim, learning_rate, mol = cong.to_list()
        reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=size,
                                                                                         dropout=dropout,
                                                                                         drugbank=True, gen_mol=mol)

        pos_dataset, neg_dataset, trie_files = get_data(pooling, reaction_model, reaction_tokenizer, esm_tokenizer,
                                                        gen_mol=mol,
                                                        return_files=True)
        trie = build_trie(trie_files, esm_tokenizer, max_length=512)
        reaction_model.to(device).eval()
        decoder.to(device).eval()
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

        model_path = f"results_drugbank/{output_dir}/"
        model_path = get_best_cp(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
        print(f"Model loaded from {model_path}")
        y_true, y_scores = evaluate_model(pos_dataset, neg_dataset, model, batch_size=32, return_prob=True)
        all_scores.append(y_scores)
        all_true.append(y_true)
    assert (all_true[0] == all_true[1]).all()
    all_scores = np.stack(all_scores)
    y_true = all_true[0]
    auc = roc_auc_score(y_true, all_scores.sum(axis=0))


if __name__ == "__main__":
    main()
