import random
from collections import defaultdict
import torch
import numpy as np
import logging
from transformers import AutoTokenizer, EsmForMaskedLM
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,

)
from transformers import AutoTokenizer, BertForMaskedLM, BertConfig

from torch.utils.data import Dataset

size = "m"
dropout = 0.0

hidden_size_per_size = {"xs": 64, "s": 128, "m": 256, "l": 512, "xl": 1024}
num_layers_per_size = {"xs": 2, "s": 4, "m": 6, "l": 8, "xl": 12}
num_attention_heads_per_size = {"xs": 2, "s": 4, "m": 4, "l": 8, "xl": 16}
hidden_size = hidden_size_per_size[size]
num_hidden_layers = num_layers_per_size[size]
num_attention_heads = num_attention_heads_per_size[size]


def mask_dict_to_tensor(mask_dict, max_token, max_index):
    """
    Convert a dictionary of masks to a tensor representation.

    Args:
        mask_dict: Dictionary where keys are positions and values are lists of possible token IDs.
        tokenizer: Tokenizer to convert token IDs to tensors.

    Returns:
        Tensor representation of the mask dictionary.
    """
    mask_tensor = torch.zeros((max_index, max_token), dtype=torch.float) - 1e10
    for pos, tokens in mask_dict.items():
        for token in tokens:
            mask_tensor[pos][token] = 0
    return mask_tensor


class CustomMaskedLMDataset(Dataset):
    def __init__(self, base_path, L=10, mask_token=32):
        self.mask_token = mask_token
        input_file = f"{base_path}/train_input.txt"
        mask_file = f"{base_path}/train_mask.txt"
        mask_candidate_file = f"{base_path}/train_mask_cand.txt"
        self.L = L
        with open(input_file) as f:
            lines = f.read().splitlines()

        self.input_sequences = [[int(x) for x in line.strip().split()] for line in lines]
        with open(mask_file) as f:
            lines = f.read().splitlines()

        self.mask_sequences = [[int(x) for x in line.strip().split()] for line in lines]
        self.max_token = max([max(seq) for seq in self.mask_sequences])
        with open(mask_candidate_file) as f:
            lines = f.read().splitlines()
        self.masks_dicts = list()

        for line in lines:
            mask_dict = dict()
            for index_can in line.strip().split(" "):
                index, candidates = index_can.split(":")
                index = int(index)
                candidates = [int(x) for x in candidates.strip().split(",")]
                mask_dict[index] = candidates
            self.masks_dicts.append(mask_dict)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):

        input_ids = self.mask_sequences[idx]
        input_ids = torch.tensor(input_ids)

        labels = self.input_sequences[idx]
        labels = torch.tensor(labels)
        labels[input_ids != self.mask_token] = -100

        mask_candidates = self.masks_dicts[idx]
        possible_tokens = mask_dict_to_tensor(mask_candidates, max_token=self.max_token + 1, max_index=self.L)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "possible_tokens": possible_tokens,
        }
        # Example usage


def compute_metrics(eval_pred):
    """
    Compute the accuracy of the model for the masked tokens

    Args:
        eval_pred: An EvalPrediction object containing predictions and labels
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = np.argmax(predictions, axis=-1)
    labels = labels.flatten()
    predictions = predictions.flatten()
    mask = labels != -100
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy}


class RestrictedMaskedLMTrainer(Trainer):
    """
    Custom Trainer that implements a loss function that only considers
    specified possible tokens for each masked position.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract inputs
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        possible_tokens = inputs.get("possible_tokens", None)

        # Forward pass
        outputs = model(input_ids=input_ids)
        outputs.logits += possible_tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        masked_lm_loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

        return (masked_lm_loss, outputs) if return_outputs else masked_lm_loss


#
#
def main():
    # Configuration
    output_dir = "./finetuned_mlm"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        logging_steps=500,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
        remove_unused_columns=False,
        save_total_limit=2,
        auto_find_batch_size=True,

    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True)
    config = BertConfig(
        vocab_size=len(tokenizer.get_vocab()),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        is_encoder_decoder=False,
        is_decoder=False,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=512,
    )
    model = BertForMaskedLM(config)
    # model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True)

    # model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", trust_remote_code=True)
    train_dataset = CustomMaskedLMDataset("data/drugbank_mlm")
    # eval is random subset of train

    eval_dataset_indices = np.random.choice(len(train_dataset), 50_000, replace=False)
    eval_dataset = torch.utils.data.Subset(train_dataset, eval_dataset_indices)

    trainer = RestrictedMaskedLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    print("Evaluating...")
    print(trainer.evaluate())
    print("Training...")
    trainer.train()


if __name__ == "__main__":
    main()
