import random
from rdkit import Chem
from rdkit import RDLogger
import os
from dataclasses import dataclass

RDLogger.DisableLog('rdApp.*')
import argparse

train_size = 0.85
valid_size = 0.05
test_size = 0.1


class Sample:
    def __init__(self, line):
        _, __, smiles, fasta, label = line.split(" ")
        self.label = int(float(label))
        smiles = remove_stereo_mol(smiles)
        self.smiles = smiles
        self.fasta = fasta

    def is_pos(self):
        return self.label == 1

    def is_neg(self):
        return self.label == 0


random.seed(42)

parser = argparse.ArgumentParser(description="Prepare DrugBank dataset")
parser.add_argument("--cold_smiles", type=int, default=0)
parser.add_argument("--cold_fasta", type=int, default=0)
parser.add_argument("--ds", type=str, default="drugbank")

args = parser.parse_args()
cold_smiles = args.cold_smiles
cold_fasta = args.cold_fasta
ds = args.ds
if cold_fasta > 0 and cold_smiles > 0:
    raise ValueError(
        "You cannot set both cold_smiles and cold_fasta to a value greater than 0. Please set one of them to 0.")


def remove_stereo_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def get_lines(file):
    with open(file, "r") as f:
        lines = f.read().splitlines()
    return lines


def get_out_base(cold_smiles, cold_fasta, ds):
    out_base = f"data/{ds}"
    if cold_smiles:
        out_base += "_cs"
    if cold_fasta:
        out_base += "_cf"
    return out_base


def remove_not_appear_indexes(train_samples, test_val_samples, secondary_attr):
    train_secondary = set([getattr(sample, secondary_attr) for sample in train_samples])
    remove_test_indexes = [
        i for i, sample in enumerate(test_val_samples)
        if getattr(sample, secondary_attr) not in train_secondary
    ]

    for i in remove_test_indexes[::-1]:
        test_val_samples.pop(i)
    return test_val_samples


def process_samples(samples, entities_in_val_tests, primary_attr):
    train_samples = []
    test_val_samples = []

    for sample in samples:
        if getattr(sample, primary_attr) in entities_in_val_tests:
            test_val_samples.append(sample)
        else:
            train_samples.append(sample)
    return train_samples, test_val_samples


def split_val_test(test_val_samples, valid_size, test_size):
    random.shuffle(test_val_samples)
    valid_test_ratio = valid_size / (valid_size + test_size)
    valid_size = int(len(test_val_samples) * valid_test_ratio)
    valid_samples = test_val_samples[:valid_size]
    test_samples = test_val_samples[valid_size:]
    return valid_samples, test_samples


def ds_to_files(ds):
    if ds == "drugbank":
        return "data/drugbank/DrugBank.txt"
    elif ds == "davis":
        return "data/davis/Davis.txt"
    elif ds == "biosnap":
        return "data/biosnap/BIOSNAP.txt"
    elif ds == "kiba":
        return "data/kiba/KIBA.txt"
    elif ds == "bindingdb":
        return "data/bindingdb/BindingDB.txt"
    else:
        raise ValueError("Unknown dataset")


if __name__ == "__main__":

    lines = get_lines(ds_to_files(ds))
    output_base = get_out_base(cold_smiles, cold_fasta, ds)
    os.makedirs(output_base, exist_ok=True)
    random.shuffle(lines)
    samples = [Sample(line) for line in lines]
    neg_samples = [sample for sample in samples if sample.is_neg()]
    samples = [sample for sample in samples if sample.is_pos()]
    pos_smiles = {sample.smiles for sample in samples}
    pos_fasta = {sample.fasta for sample in samples}
    n_neg_samples_before = len(neg_samples)
    neg_samples = [s for s in neg_samples if s.smiles in pos_smiles and s.fasta in pos_fasta]
    random.shuffle(neg_samples)
    n_neg_samples = len(neg_samples)
    print(f"neg samples: {n_neg_samples},before filtering: {n_neg_samples_before}")

    n_samples = len(samples)

    if not cold_fasta and not cold_smiles:
        seen_smiles = set()
        seen_fasta = set()
        train_samples = []
        remaining_samples = []
        for sample in samples:
            new_smiles = sample.smiles not in seen_smiles
            new_fasta = sample.fasta not in seen_fasta
            if sample.is_pos() and (new_smiles or new_fasta):
                seen_smiles.add(sample.smiles)
                seen_fasta.add(sample.fasta)
                train_samples.append(sample)
                continue
            else:
                remaining_samples.append(sample)
        random.shuffle(remaining_samples)
        train_count = int(n_samples * train_size) - len(train_samples)
        train_samples.extend(remaining_samples[:train_count])
        valid_samples = remaining_samples[train_count:train_count + int(n_samples * valid_size)]
        test_samples = remaining_samples[train_count + int(n_samples * valid_size):]

        neg_train_samples = neg_samples[:int(n_neg_samples * train_size)]
        neg_valid_samples = neg_samples[int(n_neg_samples * train_size):int(n_neg_samples * (train_size + valid_size))]
        neg_test_samples = neg_samples[int(n_neg_samples * (train_size + valid_size)):]

        print(f"train size: {len(train_samples)}, valid size: {len(valid_samples)}, test size: {len(test_samples)}")
        print(
            f"neg train size: {len(neg_train_samples)}, neg valid size: {len(neg_valid_samples)}, neg test size: {len(neg_test_samples)}")

    all_smiles_set = set([sample.smiles for sample in samples])
    all_fasta_set = set([sample.fasta for sample in samples])

    if cold_fasta or cold_smiles:
        primary_attr = "fasta" if cold_fasta else "smiles"
        secondary_attr = "smiles" if cold_fasta else "fasta"
        entities_to_choose = all_fasta_set if cold_fasta else all_smiles_set
        entities_in_val_tests = random.sample(entities_to_choose,
                                              int(len(entities_to_choose) * (valid_size + test_size)))

        train_samples, test_val_samples = process_samples(samples, entities_in_val_tests, primary_attr)
        test_val_samples = remove_not_appear_indexes(train_samples, test_val_samples, secondary_attr)
        valid_samples, test_samples = split_val_test(test_val_samples, valid_size, test_size)

        neg_train_samples, neg_test_val_samples = process_samples(neg_samples, entities_in_val_tests, primary_attr)
        neg_test_val_samples = remove_not_appear_indexes(train_samples, neg_test_val_samples, secondary_attr)
        neg_valid_samples, neg_test_samples = split_val_test(neg_test_val_samples, valid_size, test_size)

        print(f"valid size: {len(valid_samples)}, test size: {len(test_samples)}")

    for name, samples, neg_samples in zip(["train", "valid", "test"], [train_samples, valid_samples, test_samples],
                                          [neg_train_samples, neg_valid_samples, neg_test_samples]):
        print(f"Saving {name} samples")
        all_smiles = [sample.smiles for sample in samples]
        all_fasta = [sample.fasta for sample in samples]
        neg_smiles = [sample.smiles for sample in neg_samples]
        neg_fasta = [sample.fasta for sample in neg_samples]
        with open(f"{output_base}/{name}_reaction.txt", "w") as f:
            f.write("\n".join(all_smiles))
        with open(f"{output_base}/{name}_enzyme.txt", "w") as f:
            f.write("\n".join(all_fasta))
        with open(f"{output_base}/{name}_reaction_neg.txt", "w") as f:
            f.write("\n".join(neg_smiles))
        with open(f"{output_base}/{name}_enzyme_neg.txt", "w") as f:
            f.write("\n".join(neg_fasta))
        print(f"{name}. positive: {len(all_smiles)}, {name}. negative: {len(neg_smiles)}")
