import os

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = model.to(device)
model.eval()

with open("data/drugbank/train_enzyme.txt", "r") as f:
    lines = f.read().splitlines()
lines = list(set(lines))
length = []
embeddings = []
for line in tqdm(lines):
    tokens = tokenizer(line, return_tensors="pt", padding=False, truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    length.append(len(tokens['input_ids'][0]))
    embeddings.append(outputs['last_hidden_state'][0].detach().cpu())

line_to_index = {line: i for i, line in enumerate(lines)}

embeddings = torch.cat(embeddings, dim=0)
print(embeddings.shape)
with open("data/drugbank/enzyme_embeddings.pt", "wb") as f:
    torch.save(embeddings, f)
with open("data/drugbank/enzyme_length.txt", "w") as f:
    for l in length:
        f.write(str(l) + "\n")

kmeans = KMeans(
    n_clusters=20,
    random_state=42
)

kmeans.fit(embeddings.numpy())
labels = kmeans.labels_


def deduplicate(lst):
    final_list = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != final_list[-1]:
            final_list.append(lst[i])
    return final_list


cur_start = 0
all_new_lines = []
for l in length:
    cur_end = cur_start + l
    curr_labels = labels[cur_start:cur_end]
    deduped_labels = deduplicate(curr_labels) + 4
    new_line = tokenizer.decode(deduped_labels)
    all_new_lines.append(new_line)
    cur_start = cur_end
index_to_newline = {i: all_new_lines[i] for i in range(len(all_new_lines))}

os.makedirs("data/drugbank20", exist_ok=True)

# copy all the reactions files to the new directory
import shutil

for split in ["train", "valid", "test"]:
    file_name = f"data/drugbank/{split}_reaction.txt"
    output_file_name = f"data/drugbank20/{split}_reaction.txt"
    shutil.copyfile(file_name, output_file_name)
    file_name = f"data/drugbank/{split}_reaction_neg.txt"
    output_file_name = f"data/drugbank20/{split}_reaction_neg.txt"
    shutil.copyfile(file_name, output_file_name)

for split in ["train", "valid", "test"]:
    for suf in ["", "_neg"]:
        file_name = f"data/drugbank/{split}_enzyme{suf}.txt"
        output_file_name = f"data/drugbank20/{split}_enzyme{suf}.txt"
        with open(file_name, "r") as f:
            lines = f.read().splitlines()
        new_lines = []
        for line in lines:
            new_lines.append(index_to_newline[line_to_index[line]])
        with open(output_file_name, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
