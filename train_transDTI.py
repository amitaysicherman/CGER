import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset as TorchDataset
from os.path import join as pjoin
import numpy as np
import glob
from transformers import AutoModel
from transformers import AutoTokenizer
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransDTI(torch.nn.Module):
    def __init__(self):
        super(TransDTI, self).__init__()

        self.prot_branch = nn.Sequential(
            nn.Linear(1280, 1280),
            # nn.BatchNorm1d(1280)  # batch_normalization: BatchNormalization
        )

        # Branch 2
        self.mol_branch = nn.Sequential(
            nn.Linear(768, 768),
            # nn.BatchNorm1d(768)
        )

        self.post_concat = nn.Sequential(
            nn.Linear(1280 + 768, 1024),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

    def forward(self, prot, mol):
        prot = self.prot_branch(prot)  # Output shape: (None, 1280)
        mol = self.mol_branch(mol)  # Output shape: (None, 768)
        concatenated = torch.cat((prot, mol), dim=-1)
        output = self.post_concat(concatenated)  # Final output shape: (None, 3)
        return output


class SrcTgtDataset(TorchDataset):
    def __init__(self, mol_texts, prot_texts, labels, mol_tokenizer, prot_tokenizer, mol_encoder, prot_encoder,
                 max_length=512):
        self.mol_texts = mol_texts
        self.prot_texts = prot_texts
        self.max_length = max_length
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.mol_encoder = mol_encoder
        self.mol_memory = {}
        self.prot_encoder = prot_encoder
        self.prot_memory = {}

        self.labels = labels

    def __len__(self):
        return len(self.mol_texts)

    def __getitem__(self, idx):
        mol_text = self.mol_texts[idx]
        prot_text = self.prot_texts[idx]
        if mol_text not in self.mol_memory:
            mol_tokens = self.mol_tokenizer(
                mol_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
            )
            mol_tokens = {k: v.to(device) for k, v in mol_tokens.items()}
            mol_encoder_outputs = self.mol_encoder(**mol_tokens)
            mol_encoder_outputs = mol_encoder_outputs.last_hidden_state.mean(dim=1)
            mol_encoder_outputs = mol_encoder_outputs.squeeze(0).detach().cpu()
            self.mol_memory[mol_text] = mol_encoder_outputs
        if prot_text not in self.prot_memory:
            prot_tokens = self.prot_tokenizer(
                prot_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
            )
            prot_tokens = {k: v.to(device) for k, v in prot_tokens.items()}
            prot_encoder_outputs = self.prot_encoder(**prot_tokens)
            prot_encoder_outputs = prot_encoder_outputs.pooler_output.squeeze(0)
            prot_encoder_outputs = prot_encoder_outputs.detach().cpu()
            self.prot_memory[prot_text] = prot_encoder_outputs

        return dict(
            prot=self.prot_memory[prot_text],
            mol=self.mol_memory[mol_text],
            labels=self.labels[idx],
        )


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = predictions.argmax(-1)
    predictions = predictions.flatten()
    labels = labels.flatten()
    acc = (predictions == labels).sum() / len(labels)
    precision = (predictions * labels).sum() / predictions.sum()
    recall = (predictions * labels).sum() / labels.sum()
    f1 = 2 * precision * recall / (precision + recall)
    auc = (predictions == labels).sum() / len(labels)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def get_all_lines(split, base_dir):
    with open(f"data/{base_dir}/{split}_enzyme.txt", "r") as f:
        pos_protein = f.read().splitlines()
    with open(f"data/{base_dir}/{split}_reaction.txt", "r") as f:
        pos_molecule = f.read().splitlines()
    with open(f"data/{base_dir}/{split}_enzyme_neg.txt", "r") as f:
        neg_protein = f.read().splitlines()
    with open(f"data/{base_dir}/{split}_reaction_neg.txt", "r") as f:
        neg_molecule = f.read().splitlines()
    labels = [1] * len(pos_protein) + [0] * len(neg_protein)
    prot_text = pos_protein + neg_protein
    mol_texts = pos_molecule + neg_molecule
    return prot_text, mol_texts, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data", type=str, default="drugbank", help="base data")

    args = parser.parse_args()

    prot_train, mol_train, labels_train = get_all_lines("train", args.base_data)
    prot_valid, mol_valid, labels_valid = get_all_lines("valid", args.base_data)
    prot_test, mol_test, labels_test = get_all_lines("test", args.base_data)

    prot_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    prot_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    prot_model = prot_model.to(device).eval()
    for param in prot_model.parameters():
        param.requires_grad = False
    mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    mol_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True,
                                          deterministic_eval=True)
    mol_model = mol_model.to(device).eval()
    for param in mol_model.parameters():
        param.requires_grad = False

    train_dataset = SrcTgtDataset(mol_train, prot_train, labels_train, mol_tokenizer, prot_tokenizer, mol_model,
                                    prot_model)
    valid_dataset = SrcTgtDataset(mol_valid, prot_valid, labels_valid, mol_tokenizer, prot_tokenizer, mol_model,
                                    prot_model)
    test_dataset = SrcTgtDataset(mol_test, prot_test, labels_test, mol_tokenizer, prot_tokenizer, mol_model,
                                    prot_model)
    train_small_indices = np.random.choice(len(train_dataset), len(test_dataset), replace=False)
    train_small_dataset = torch.utils.data.Subset(train_dataset, train_small_indices)

    eval_dataset = {"test": test_dataset, "train": train_small_dataset, "valid": valid_dataset},
    metric_for_best_model = "eval_valid_auc"

    model = TransDTI()

    output_dir = f"results_db/transDTI_{args.base_data}"
    logs_dir = output_dir.replace("results", "logs")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logs_dir,
        evaluation_strategy="steps",
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        eval_accumulation_steps=30,
        save_total_limit=3,
        num_train_epochs=100,
        logging_steps=100,
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        report_to="tensorboard",
        save_safetensors=False,
        auto_find_batch_size=True,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    # Train model
    print("Training model...")

    trainer.train(resume_from_checkpoint=len(glob.glob(pjoin(output_dir, "checkpoint-*"))) > 0)

    print("Training complete!")
