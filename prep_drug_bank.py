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
    random.seed(42)
    input_file = "data/drugbank/DrugBank.txt"
    output_base = f"data/drugbank"
    with open(input_file, "r") as f:
        lines = f.read().splitlines()
    all_smiles = []
    all_fasta = []
    neg_smiles = []
    neg_fasta = []
    skip_count = 0
    random.shuffle(lines)
    for i, line in enumerate(lines):
        _, __, smiles, fasta, label = line.split(" ")
        label = int(label)
        smiles = remove_stereo_mol(smiles)

        if label == 0:
            neg_smiles.append(smiles)
            neg_fasta.append(fasta)
            continue
        all_smiles.append(smiles)
        all_fasta.append(fasta)
    print(f"skip count: {skip_count}")
    print(f"total count: {len(all_smiles)}")

    train_count = int(len(all_smiles) * 0.85)
    valid_count = int(len(all_smiles) * 0.05)
    test_count = int(len(all_smiles) * 0.1)
    print(f"train count: {train_count}")
    print(f"valid count: {valid_count}")
    print(f"test count: {test_count}")
    print(f"total count: {len(all_smiles)}")

    indexes = list(range(len(all_smiles)))
    seen_smiles = set()
    seen_fasta = set()
    train_indexes = []
    remaining_indexes = []
    for i in range(len(all_smiles)):
        if all_smiles[i] not in seen_smiles or all_fasta[i] not in seen_fasta:
            seen_smiles.add(all_smiles[i])
            seen_fasta.add(all_fasta[i])
            train_indexes.append(i)
            continue
        else:
            remaining_indexes.append(i)

    random.shuffle(remaining_indexes)
    train_cont_to_add = train_count - len(train_indexes)
    train_indexes += remaining_indexes[:train_cont_to_add]
    valid_indexes = remaining_indexes[train_cont_to_add:train_cont_to_add + valid_count]
    test_indexes = remaining_indexes[train_cont_to_add + valid_count:train_count + valid_count + test_count]

    neg_indexes_filter = []
    for i in range(len(neg_smiles)):
        if neg_smiles[i] in seen_smiles and neg_fasta[i] in seen_fasta:
            neg_indexes_filter.append(i)
    print(f"neg count: {len(neg_smiles)}, neg filter count: {len(neg_indexes_filter)}")
    neg_smiles = [neg_smiles[i] for i in neg_indexes_filter]
    neg_fasta = [neg_fasta[i] for i in neg_indexes_filter]

    neg_indexes = list(range(len(neg_smiles)))
    random.shuffle(neg_indexes)
    train_neg_indexes = neg_indexes[:int(len(neg_indexes) * 0.85)]
    valid_neg_indexes = neg_indexes[int(len(neg_indexes) * 0.85):int(len(neg_indexes) * 0.9)]
    test_neg_indexes = neg_indexes[int(len(neg_indexes) * 0.9):]

    for name, indexes, neg_indexes in zip(["train", "valid", "test"], [train_indexes, valid_indexes, test_indexes],
                                          [train_neg_indexes, valid_neg_indexes, test_neg_indexes]):
        assert all(all_smiles[i] in seen_smiles for i in indexes)
        assert all(all_fasta[i] in seen_fasta for i in indexes)

        smiles = [all_smiles[i] for i in indexes]
        fasta = [all_fasta[i] for i in indexes]

        with open(f"{output_base}/{name}_reaction.txt", "w") as f:
            f.write("\n".join(smiles))
        with open(f"{output_base}/{name}_enzyme.txt", "w") as f:
            f.write("\n".join(fasta))
        assert all(neg_smiles[i] in seen_smiles for i in neg_indexes)
        assert all(neg_fasta[i] in seen_fasta for i in neg_indexes)

        neg_smiles_in = [neg_smiles[i] for i in neg_indexes]
        neg_fasta_in = [neg_fasta[i] for i in neg_indexes]
        with open(f"{output_base}/{name}_reaction_neg.txt", "w") as f:
            f.write("\n".join(neg_smiles_in))
        with open(f"{output_base}/{name}_enzyme_neg.txt", "w") as f:
            f.write("\n".join(neg_fasta_in))

        print(f"Saved {name} , {len(smiles)} positive and {len(neg_smiles_in)} negative samples")
