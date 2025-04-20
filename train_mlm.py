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
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MaskedSequenceSearch:
    def __init__(self, tokenizer, max_length=10, mask=32):
        """
        Initialize the search structure with a corpus of sequences

        Args:
            sequences: List of sequences (strings)
            max_length: Maximum length of sequences
        """
        self.max_length = max_length
        self.mask = mask
        # Inverted index: position -> character -> set of sequence indices
        with open("data/drugbank/train_enzyme.txt") as f:
            lines = f.read().splitlines()
        seqs = [line.strip() for line in lines]
        seqs = list(set(seqs))
        corpus = []
        for seq in tqdm(seqs):
            if len(seq) < max_length:
                continue
            for index in range(len(seq) - max_length):
                seq_token = tokenizer.encode(seq[index:index + max_length], add_special_tokens=False)
                corpus.append(seq_token)
        self.sequences = corpus

        self.inverted_index = self._build_inverted_index()

    def _build_inverted_index(self):
        """Build the inverted index from the sequence corpus"""
        inverted_index = [{} for _ in range(self.max_length)]

        for seq_idx, sequence in enumerate(self.sequences):
            for pos, char in enumerate(sequence):
                if char not in inverted_index[pos]:
                    inverted_index[pos][char] = set()
                inverted_index[pos][char].add(seq_idx)

        return inverted_index

    def search(self, masked_sequence):
        """
        Search for all sequences that match the masked pattern

        Args:
            masked_sequence: A string with some characters and masks
                             (where masks are represented by '*')

        Returns:
            List of matching sequences
        """
        # Start with all sequences as candidates
        candidate_indices = set(range(len(self.sequences)))

        # Filter candidates by length
        candidate_indices = {i for i in candidate_indices
                             if len(self.sequences[i]) == len(masked_sequence)}

        # For each non-masked position, filter candidates
        for pos, char in enumerate(masked_sequence):
            if pos >= self.max_length:
                break

            if char != self.mask:  # Not a mask
                # Get sequences that have this character at this position
                if char in self.inverted_index[pos]:
                    candidate_indices &= self.inverted_index[pos][char]
                else:
                    # No sequences have this character at this position
                    return []

        # Return the matching sequences
        return [self.sequences[i] for i in candidate_indices]

    def get_mask_candidates(self, masked_sequence):
        candidates = defaultdict(list)
        sequences_candidates = self.search(masked_sequence)

        for pos, char in enumerate(masked_sequence):
            if char != self.mask:
                continue
            for seq in sequences_candidates:
                candidates[pos].append(seq[pos])
        # Remove duplicates
        for pos in candidates:
            candidates[pos] = list(set(candidates[pos]))
        return candidates

    def add_n_masks(self, seq_str, n_can_max=3):
        # add masks to the sequence until the number of returned sequences is less than n_seq_max
        seq = seq_str[:]
        seq_len = len(seq)
        mask_positions = random.sample(range(seq_len), len(seq))
        seq_mask = self.get_mask_candidates(seq)
        for pos in mask_positions:
            seq[pos] = self.mask
            seq_mask = self.get_mask_candidates(seq)
            n_can_mean = np.mean([len(seq_mask[i]) for i in seq_mask if len(seq_mask[i]) > 0])
            if n_can_mean > n_can_max:
                seq[pos] = seq_str[pos]
                return seq, seq_mask
        return seq, seq_mask

    def add_masks(self, seq_str, n):
        seq = seq_str[:]
        seq_len = len(seq)
        mask_positions = random.sample(range(seq_len), n)
        for pos in mask_positions:
            seq[pos] = self.mask
        seq_mask = self.get_mask_candidates(seq)
        return seq, seq_mask


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
    def __init__(self, data_path, tokenizer, L=10):
        with open(data_path) as f:
            seqs = [line.strip() for line in f.readlines()]
        self.tokenizer = tokenizer
        self.L = L
        self.dataset = list(set(seqs))
        self.searcher = MaskedSequenceSearch(tokenizer=self.tokenizer, max_length=L, mask=tokenizer.mask_token_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        rnd_index = random.randint(0, len(item) - self.L)
        seq = item[rnd_index:rnd_index + self.L]
        seq_token = self.tokenizer.encode(seq, add_special_tokens=False)

        src_item = seq_token[:]
        # rnadom normal distribution mean 5 and std 2
        n = int(random.normalvariate(5, 2))
        n= min(n, 7)
        n = max(n, 2)
        n = int(n)
        mask_item, mask_dict = self.searcher.add_masks(src_item, n=n)
        mask_tensor = mask_dict_to_tensor(mask_dict, self.tokenizer.vocab_size, self.L)
        labels = torch.tensor(src_item)
        input_ids = torch.tensor(mask_item)
        labels[input_ids != self.tokenizer.mask_token_id] = -100
        possible_tokens = mask_tensor
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
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=1e-4,
        logging_steps=100,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        remove_unused_columns=False,
        save_total_limit=2,
        auto_find_batch_size=True,

    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", trust_remote_code=True)
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", trust_remote_code=True)

    # model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", trust_remote_code=True)
    train_dataset = CustomMaskedLMDataset("data/drugbank_cf/train_enzyme.txt", tokenizer)
    eval_dataset = CustomMaskedLMDataset("data/drugbank_cf/test_enzyme.txt", tokenizer)

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
    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()
