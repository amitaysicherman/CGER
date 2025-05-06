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


def prep_csv(dataset, split):
    csv_file = f"data/{dataset}/{split}.csv"
    df = pd.read_csv(csv_file)
    pos_prots = []
    neg_prots = []
    pos_smiles = []
    neg_smiles = []

    for i, row in tqdm(df.iterrows(),total=len(df), desc=f"Processing {split} {dataset}"):
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

if __name__ == "__main__":
    splits = ["train", "valid", "test"]
    for split in splits:
        for ds in ["biosnap", "biosnap_cs", "biosnap_cf"]:
            prep_csv(ds, split)
