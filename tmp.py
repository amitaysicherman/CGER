import pandas as pd

test_data = pd.read_csv("data/bindingdb/full.csv")
print(test_data.columns)
"""
SMILES,Protein,Y,drug_cluster,target_cluster

"""
lines=[]
for index, row in test_data.iterrows():
    lines.append(f'X X {row["SMILES"]} {row["Protein"]} {row["Y"]}')
print(len(lines))

with open("data/bindingdb/BindingDB.txt", "w") as f:
    f.write("\n".join(lines))