import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional
import argparse

parser = argparse.ArgumentParser(description="Residual Vector Quantization")
parser.add_argument("--n_layers", type=int, default=10, help="Number of layers for RVQ")
parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for KMeans")
parser.add_argument("--random_state", type=int, default=42, help="Random state for KMeans")
parser.add_argument("--is_molecules", action="store_true", help="Flag to indicate if the input is molecules")
parser.add_argument("--ds", type=str, default="drugbank", help="Dataset name")
args = parser.parse_args()
ds = args.ds
# class Args:
#     def __init__(self, n_layers=10, n_clusters=10, random_state=42, is_molecules=False):
#
#         self.n_layers = n_layers
#         self.n_clusters = n_clusters
#         self.random_state = random_state
#         self.is_molecules = is_molecules
# args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualVectorQuantizer:
    def __init__(
            self,
            n_layers: int = 10,
            n_clusters: int = 20,
            kmeans_kwargs: Optional[dict] = None,
            random_state: Optional[int] = 42,
    ):
        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.kmeans_kwargs = kmeans_kwargs or {}
        self.random_state = random_state
        self.quantizers_ = []
        self.is_fitted_ = False

    def fit(self, X):
        self.quantizers_ = []
        residual = X.copy()
        for i in range(self.n_layers):
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kmeans_kwargs
            )
            kmeans.fit(residual)
            self.quantizers_.append(kmeans)
            centroids = kmeans.cluster_centers_[kmeans.predict(residual)]
            residual = residual - centroids

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise ValueError("RVQ must be fitted before transform can be called")
        n_samples = X.shape[0]
        codes = np.zeros((n_samples, self.n_layers), dtype=int)
        residual = X.copy()
        for i, kmeans in enumerate(self.quantizers_):
            codes[:, i] = kmeans.predict(residual)
            centroids = kmeans.cluster_centers_[codes[:, i]]
            residual = residual - centroids
        return codes


def get_model_tokenizer(is_molecules: bool):
    if is_molecules:
        tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                          trust_remote_code=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True)
        model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    return tokenizer, model


def get_lines(is_molecules: bool, ds: str):
    if is_molecules:
        with open(f"data/{ds}/train_reaction.txt", "r") as f:
            lines = f.read().splitlines()
    else:
        with open(f"data/{ds}/train_enzyme.txt", "r") as f:
            lines = f.read().splitlines()
    return lines


tokenizer, model = get_model_tokenizer(args.is_molecules)
model = model.to(device)
lines = get_lines(args.is_molecules, ds)

lines = list(set(lines))
embeddings = []
for line in tqdm(lines):
    tokens = tokenizer(line, return_tensors="pt", padding=False, truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    embeddings.append(outputs['last_hidden_state'][0].mean(dim=0).detach().cpu().numpy())
embeddings = np.stack(embeddings)

rvq = ResidualVectorQuantizer(
    n_layers=args.n_layers,
    n_clusters=args.n_clusters,
    random_state=args.random_state
)
rvq.fit(embeddings)
codes = rvq.transform(embeddings)
line_to_code = dict()
for i in range(len(lines)):
    line_to_code[lines[i]] = " ".join([str(x) for x in codes[i].tolist()])

assert len(line_to_code) == len(list(set(line_to_code.values()))), "Duplicate codes found"


def convert_files(file_name, line_to_code, output_suffix="_q"):
    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    with open(file_name.replace(".txt", output_suffix + ".txt"), "w") as f:
        for line in lines:
            code = line_to_code[line]
            f.write(code + "\n")


def get_all_file_names(is_molecules: bool, ds: str):
    if is_molecules:
        type_name = "reaction"
        base_dirs = [f"data/{ds}", f"data/{ds}_cf"]
    else:
        type_name = "enzyme"
        base_dirs = [f"data/{ds}", f"data/{ds}_cs"]
    file_names = []
    for base_path in base_dirs:
        for split in ["train", "valid", "test"]:
            file_names.append(f"{base_path}/{split}_{type_name}.txt")
            file_names.append(f"{base_path}/{split}_{type_name}_neg.txt")
    return file_names


file_names = get_all_file_names(args.is_molecules, ds)
for file_name in file_names:
    convert_files(file_name, line_to_code)
