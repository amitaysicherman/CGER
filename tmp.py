with open("data/drugbank/DrugBank.txt", "r") as f:
    lines = f.read().splitlines()
pos_fasta = set()
neg_fasta = set()
pos_smiles = set()
neg_smiles = set()
for i, line in enumerate(lines):
    _, __, smiles, fasta, label = line.split(" ")
    if label == "1":
        pos_smiles.add(smiles)
        pos_fasta.add(fasta)
    else:
        assert label == "0"
        neg_smiles.add(smiles)
        neg_fasta.add(fasta)


print("Fasta in Positive set: ", len(pos_fasta))
print("Fasta in Negative set: ", len(neg_fasta))
print("Smiles in Positive set: ", len(pos_smiles))
print("Smiles in Negative set: ", len(neg_smiles))

print("Fasta in Positive set and not in Negative set: ", len(pos_fasta - neg_fasta))
print("Fasta in Negative set and not in Positive set: ", len(neg_fasta - pos_fasta))
print("Fasta in Positive set and in Negative set: ", len(pos_fasta & neg_fasta))

print("Smiles in Positive set and not in Negative set: ", len(pos_smiles - neg_smiles))
print("Smiles in Negative set and not in Positive set: ", len(neg_smiles - pos_smiles))
print("Smiles in Positive set and in Negative set: ", len(pos_smiles & neg_smiles))
