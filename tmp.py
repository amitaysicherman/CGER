import pandas as pd
train_file="/home/amitay.s/CGER/MolTrans/dataset/BIOSNAP/full_data/train.csv"
valid_file="/home/amitay.s/CGER/MolTrans/dataset/BIOSNAP/full_data/valid.csv"
test_file="/home/amitay.s/CGER/MolTrans/dataset/BIOSNAP/full_data/test.csv"


def get_src_tgt_for_file(file):
    df = pd.read_csv(file)
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
    # print(df.columns)
    # print(df.head())
    # print(df.columns)
    # print(df.head())
    src = df["SMILES"].tolist()
    tgt = df["Target Sequence"].tolist()
    return src, tgt

train_src, train_tgt = get_src_tgt_for_file(train_file)
train_src=set(train_src)
train_tgt=set(train_tgt)
valid_src, valid_tgt = get_src_tgt_for_file(valid_file)
skip_count = 0
for i in range(len(valid_src)):
    if valid_src[i] not in train_src or valid_tgt[i] not in train_tgt:
        skip_count += 1
print(f"skip count: {skip_count}/{len(valid_src)},({skip_count/len(valid_src):.2%})")
test_src, test_tgt = get_src_tgt_for_file(test_file)
skip_count = 0
for i in range(len(test_src)):
    if test_src[i] not in train_src or test_tgt[i] not in train_tgt:
        skip_count += 1
print(f"skip count: {skip_count}/{len(test_src)},({skip_count/len(test_src):.2%})")
