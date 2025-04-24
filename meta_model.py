import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

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

    return pos_train, neg_train, pos_valid, neg_valid, pos_test, neg_test, src_train


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
    quantize: bool = False

    def to_list(self):
        return [self.size, self.dropout, self.pooling, self.bottleneck_dim, self.learning_rate, self.mol,
                self.cold_smiles,
                self.cold_fasta, self.quantize]


def main():
    cong = Config("l", 0.02, True, 128, 0.0001, True, 0, 0, quantize=True)
    LEVEL = "biosnap"
    size, dropout, pooling, bottleneck_dim, learning_rate, mol, cold_smiles, cold_fasta, quantize = cong.to_list()
    reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=size,
                                                                                     dropout=dropout,
                                                                                     drugbank=True, gen_mol=mol,
                                                                                     quantize=quantize)

    pos_train, neg_train, pos_valid, neg_valid, pos_test, neg_test, src_train = get_data(pooling, reaction_model,
                                                                                         reaction_tokenizer,
                                                                                         esm_tokenizer,
                                                                                         gen_mol=mol,
                                                                                         cold_smiles=cold_smiles,
                                                                                         cold_fasta=cold_fasta,
                                                                                         quantize=quantize,
                                                                                         level=LEVEL)

    trie = build_trie(list(set(src_train)), esm_tokenizer, max_length=512)
    reaction_model.to(device).eval()
    decoder.to(device).eval()
    model = get_model(decoder, trie, size, dropout, pooling, bottleneck_dim, learning_rate, mol, quantize=quantize,
                      level=LEVEL)
    model.to(device).eval()
