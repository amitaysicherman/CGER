import torch
from transformers import BertGenerationDecoder, BertGenerationConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments

from rxnfp.main import get_model_and_tokenizer
from trie import build_mask_from_trie, build_trie
from torch.utils.data import Dataset as TorchDataset
from os.path import join as pjoin
import numpy as np
from torch.nn import CrossEntropyLoss
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size_per_size = {"xs": 64, "s": 128, "m": 256, "l": 512, "xl": 1024}
num_layers_per_size = {"xs": 2, "s": 4, "m": 6, "l": 8, "xl": 12}
num_attention_heads_per_size = {"xs": 2, "s": 4, "m": 4, "l": 8, "xl": 16}
ENCODER_DIM = 256


def get_encoder_decoder(decoder_size="m", dropout=0.1):
    # Load the pretrained tokenizers
    reaction_model, reaction_tokenizer = get_model_and_tokenizer()
    reaction_model.eval().to(device)
    for param in reaction_model.parameters():
        param.requires_grad = False
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D", trust_remote_code=True)

    hidden_size = hidden_size_per_size[decoder_size]
    num_hidden_layers = num_layers_per_size[decoder_size]
    num_attention_heads = num_attention_heads_per_size[decoder_size]
    intermediate_size = hidden_size * 4

    # Load the pretrained decoder
    decoder_config = BertGenerationConfig(
        vocab_size=len(esm_tokenizer.get_vocab()),
        eos_token_id=esm_tokenizer.eos_token_id,
        pad_token_id=esm_tokenizer.pad_token_id,
        bos_token_id=esm_tokenizer.bos_token_id,
        decoder_start_token_id=esm_tokenizer.pad_token_id,
        is_encoder_decoder=True,
        is_decoder=True,
        add_cross_attention=True,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=512,

    )
    decoder = BertGenerationDecoder(decoder_config)
    decoder.train().to(device)
    return reaction_model, reaction_tokenizer, decoder, esm_tokenizer


def load_file(file_path):
    """Load text file"""
    with open(file_path) as f:
        texts = f.read().splitlines()
    return texts


def load_files(level="easy"):
    """Load training and testing files"""
    base_dir = f"data/{level}/"
    src_train = load_file(pjoin(base_dir, "train_reaction.txt"))
    tgt_train = load_file(pjoin(base_dir, "train_enzyme.txt"))
    src_test = load_file(pjoin(base_dir, "test_reaction.txt"))
    tgt_test = load_file(pjoin(base_dir, "test_enzyme.txt"))
    print(
        f"src_train: {len(src_train)}, tgt_train: {len(tgt_train)}, src_test: {len(src_test)}, tgt_test: {len(tgt_test)}")
    return src_train, tgt_train, src_test, tgt_test


class SrcTgtDataset(TorchDataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, src_encoder, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_length = max_length
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_encoder = src_encoder

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_tokens = self.src_tokenizer(
            src_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
        src_encoder_outputs = self.src_encoder(**src_tokens)
        tgt_tokens = self.tgt_tokenizer(
            tgt_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = tgt_tokens["input_ids"].clone()
        labels[labels == self.tgt_tokenizer.pad_token_id] = -100
        return dict(
            encoder_outputs=src_encoder_outputs.last_hidden_state.squeeze(0).detach().cpu(),
            encoder_attention_mask=src_tokens["attention_mask"].squeeze(0).detach().cpu(),
            input_ids=tgt_tokens["input_ids"].squeeze(0),
            attention_mask=tgt_tokens["attention_mask"].squeeze(0),
            labels=labels.squeeze(0),
        )


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = predictions.argmax(-1)
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    non_pad_mask = labels != -100
    token_correct = 0
    token_total = 0
    sample_correct = 0
    sample_total = len(labels)

    for i in range(len(labels)):
        # Get mask for this sequence
        seq_mask = non_pad_mask[i]

        # Extract non-padded tokens for this sequence
        seq_true = labels[i][seq_mask]
        seq_pred = predictions[i][seq_mask]

        # Count correct tokens
        token_correct += np.sum(seq_pred == seq_true)
        token_total += len(seq_true)

        # Check if entire sequence is correct (exact match)
        if np.array_equal(seq_pred, seq_true):
            sample_correct += 1

    # Calculate accuracies
    token_accuracy = token_correct / token_total if token_total > 0 else 0
    sample_accuracy = sample_correct / sample_total if sample_total > 0 else 0

    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy
    }


class EnzymeDecoder(torch.nn.Module):
    def __init__(self, decoder, trie=None):
        super(EnzymeDecoder, self).__init__()
        self.decoder = decoder
        self.trie = trie
        if self.decoder.config.hidden_size != ENCODER_DIM:
            self.encoder_project = torch.nn.Linear(
                ENCODER_DIM, self.decoder.config.hidden_size
            )
        else:
            self.encoder_project = torch.nn.Identity()

    def forward(self, input_ids, attention_mask, encoder_outputs, encoder_attention_mask, labels):
        # Encode the input
        encoder_outputs = self.encoder_project(encoder_outputs)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            labels=labels,
        )
        if self.trie is None:
            return decoder_outputs

        trie_mask = build_mask_from_trie(self.trie, input_ids, self.decoder.config.vocab_size)
        trie_mask = trie_mask[:, :-1, :]
        trie_mask_out = trie_mask.sum(dim=-1) <= 1
        decoder_outputs.trie_mask_out = trie_mask_out
        trie_mask = trie_mask.masked_fill(trie_mask == 0, -1e6)
        trie_mask = trie_mask.masked_fill(trie_mask == 1, 0)
        trie_mask = trie_mask.to(decoder_outputs.logits.device)
        decoder_outputs.logits[:, :-1] += trie_mask
        if labels is not None:
            labels[:, 1:][trie_mask_out] = -100
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            decoder_outputs.loss = loss_fct(
                decoder_outputs.logits[:, :-1].reshape(-1, decoder_outputs.logits[:, :-1].size(-1)),
                labels[:, 1:].reshape(-1))
        return decoder_outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--size", type=str, default="m")
    parser.add_argument("--level", type=str, default="easy")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trie", type=int, default=1)
    args = parser.parse_args()

    src_train, tgt_train, src_test, tgt_test = load_files(level=args.level)
    reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=args.size,
                                                                                     dropout=args.dropout)

    # Create datasets and dataloaders
    train_dataset = SrcTgtDataset(src_train, tgt_train, reaction_tokenizer, esm_tokenizer, reaction_model)
    test_dataset = SrcTgtDataset(src_test, tgt_test, reaction_tokenizer, esm_tokenizer, reaction_model)

    train_small_indices = np.random.choice(len(train_dataset), len(test_dataset), replace=False)
    train_small_dataset = torch.utils.data.Subset(train_dataset, train_small_indices)

    if args.trie:
        trie = build_trie(tgt_train + tgt_test, esm_tokenizer)
    else:
        trie = None
    model = EnzymeDecoder(decoder, trie=trie)
    output_dir = f"results/{args.level}_{args.size}_{args.dropout}_{args.learning_rate}"
    if args.trie == 0:
        output_dir += "_notrie"
    logs_dir = output_dir.replace("results", "logs")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logs_dir,
        evaluation_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=30,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        metric_for_best_model="eval_test_token_accuracy",
        report_to=[args.report_to],
        save_safetensors=False,
        auto_find_batch_size=True,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"test": test_dataset, "train": train_small_dataset},
        compute_metrics=compute_metrics
    )
    # Train model
    print("Training model...")

    trainer.train(resume_from_checkpoint=len(glob.glob(pjoin(output_dir, "checkpoint-*"))) > 0)

    print("Training complete!")
