import random
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def remove_stereo_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


if __name__ == "__main__":
    input_file = "data/drugbank/DrugBank.txt"
    output_base = f"data/drugbank"
    with open(input_file, "r") as f:
        lines = f.read().splitlines()
    all_smiles = []
    all_fasta = []
    for i, line in enumerate(lines):
        _, __, smiles, fasta, label = line.split(" ")
        i = int(i)
        if i == 0:
            continue
        smiles = remove_stereo_mol(smiles)
        all_smiles.append(smiles)
        all_fasta.append(fasta)
    fasta_unique = set(all_fasta)

    train_smiles = []
    train_fasta = []
    test_smiles = []
    test_fasta = []


    for fasta in fasta_unique:
        indexes = [i for i, x in enumerate(all_fasta) if x == fasta]
        if len(indexes) == 1:
            train_indexes = indexes
            test_indexes = []

        test_count = max(int(len(indexes) * 0.1), 1)
        train_count = len(indexes) - test_count
        print(f"train: {train_count}, test: {test_count}")
        train_indexes = random.sample(indexes, train_count)
        test_indexes = list(set(indexes) - set(train_indexes))
        for i in train_indexes:
            train_smiles.append(all_smiles[i])
            train_fasta.append(all_fasta[i])
        for i in test_indexes:
            test_smiles.append(all_smiles[i])
            test_fasta.append(all_fasta[i])

    print(f"train: {len(train_smiles)}")
    print(f"test: {len(test_smiles)}")

    with open(f"{output_base}/test_enzyme.txt", "w") as f:
        f.write("\n".join(test_fasta))
    with open(f"{output_base}/train_enzyme.txt", "w") as f:
        f.write("\n".join(train_fasta))
    with open(f"{output_base}/test_reaction.txt", "w") as f:
        f.write("\n".join(test_smiles))
    with open(f"{output_base}/train_reaction.txt", "w") as f:
        f.write("\n".join(train_smiles))
