import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
import os
from dataclasses import dataclass
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
import argparse

SMILES = "reaction"
Protein = "enzyme"


def remove_stereo_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def prep_csv(dataset, split, cold_fasta=False, cold_smiles=False):
    csv_file = f"data/{dataset}/{split}.csv"
    df = pd.read_csv(csv_file)
    pos_prots = []
    neg_prots = []
    pos_smiles = []
    neg_smiles = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} {dataset}"):
        smiles = row['SMILES']
        smiles = remove_stereo_mol(smiles)
        fasta = row['Protein']
        fasta = fasta.replace(" ", "").replace(".", "")
        label = row["Y"]
        label = int(float(label))
        if label == 1:
            pos_prots.append(fasta)
            pos_smiles.append(smiles)
        else:
            neg_prots.append(fasta)
            neg_smiles.append(smiles)
        with open(f"data/{dataset}/{split}_reaction.txt", "w") as f_src:
            with open(f"data/{dataset}/{split}_enzyme.txt", "w") as f_tgt:
                for i in range(len(pos_prots)):
                    f_src.write(pos_smiles[i] + "\n")
                    f_tgt.write(pos_prots[i] + "\n")
        with open(f"data/{dataset}/{split}_reaction_neg.txt", "w") as f_src:
            with open(f"data/{dataset}/{split}_enzyme_neg.txt", "w") as f_tgt:
                for i in range(len(neg_prots)):
                    f_src.write(neg_smiles[i] + "\n")
                    f_tgt.write(neg_prots[i] + "\n")


def clean_files(src_file, tgt_file, tgt_file_2):
    with open(src_file, "r") as f:
        src_lines = f.read().splitlines()
    with open(tgt_file, "r") as f:
        tgt_lines = f.read().splitlines()
    with open(tgt_file_2, "r") as f:
        tgt_lines_2 = f.read().splitlines()

    src_lines = set(src_lines)
    tgt = []
    tgt2 = []
    remove_count = 0
    for i in range(len(tgt_lines)):
        tgt_line = tgt_lines[i]
        tgt_line_2 = tgt_lines_2[i]
        if tgt_line in src_lines:
            tgt.append(tgt_line)
            tgt2.append(tgt_line_2)
        else:
            remove_count += 1
    print(f"Removed {remove_count} lines from {src_file} and {tgt_file},saved {len(tgt)} lines")
    with open(tgt_file, "w") as f:
        for line in tgt:
            f.write(line + "\n")
    with open(tgt_file_2, "w") as f:
        for line in tgt2:
            f.write(line + "\n")


if __name__ == "__main__":
    splits = ["train", "valid", "test"]
    for split in splits:
        for ds in ["biosnap", "biosnap_cs", "biosnap_cf"]:
            prep_csv(ds, split)

