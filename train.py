import torch
from transformers import BertGenerationDecoder, BertGenerationConfig, AutoTokenizer, BertGenerationEncoder
from transformers import Trainer, TrainingArguments
from torch.nn import functional as F
from rxnfp.main import get_model_and_tokenizer
from trie import build_mask_from_trie, build_trie
from torch.utils.data import Dataset as TorchDataset
from os.path import join as pjoin
import numpy as np
from torch.nn import CrossEntropyLoss
import glob
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

hidden_size_per_size = {"xs": 64, "s": 128, "m": 256, "l": 512, "xl": 1024}
num_layers_per_size = {"xs": 2, "s": 4, "m": 6, "l": 8, "xl": 12}
num_attention_heads_per_size = {"xs": 2, "s": 4, "m": 4, "l": 8, "xl": 16}
ENCODER_DIM = 256

import csv
import os
from transformers import TrainerCallback


class EvalLoggingCallback(TrainerCallback):
    """
    Callback to log evaluation metrics to a CSV file during training.
    """

    def __init__(self, output_dir, filename="eval_logs.csv"):
        """
        Initialize the callback.

        Args:
            output_dir: Directory where the CSV file will be saved
            filename: Name of the CSV file
        """
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, filename)
        self.headers = None
        self.csv_file = None
        self.writer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation. Logs metrics to CSV file.
        """
        if metrics is None:
            return

        # Get current step
        step = state.global_step

        # Add step to metrics
        metrics['step'] = step

        # Check if directory exists, create if not
        os.makedirs(self.output_dir, exist_ok=True)

        # If file doesn't exist, create and write headers
        if not os.path.exists(self.csv_path):
            # Create CSV file with headers
            with open(self.csv_path, 'w', newline='') as f:
                # Get all keys from metrics as headers
                self.headers = sorted(metrics.keys())
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerow(metrics)
        else:
            # Append to existing file
            with open(self.csv_path, 'a', newline='') as f:
                # Read existing headers if needed
                if self.headers is None:
                    with open(self.csv_path, 'r', newline='') as read_f:
                        reader = csv.reader(read_f)
                        self.headers = next(reader)  # Get headers from first row

                # Check if we have new metrics that weren't in original headers
                new_headers = [h for h in metrics.keys() if h not in self.headers]

                if new_headers:  # If we have new headers, we need to rewrite the file
                    self.headers.extend(new_headers)
                    # Read existing data
                    with open(self.csv_path, 'r', newline='') as read_f:
                        reader = csv.DictReader(read_f)
                        existing_data = list(reader)

                    # Write all data with updated headers
                    with open(self.csv_path, 'w', newline='') as write_f:
                        writer = csv.DictWriter(write_f, fieldnames=self.headers)
                        writer.writeheader()
                        for row in existing_data:
                            writer.writerow(row)
                        writer.writerow(metrics)
                else:
                    # Just append to existing file with consistent headers
                    writer = csv.DictWriter(f, fieldnames=self.headers)
                    writer.writerow(metrics)

        print(f"Evaluation metrics at step {step} logged to {self.csv_path}")


class QuantizeTokenizer:
    def __init__(self, max_token=15):
        self.eos_token_id = max_token
        self.pad_token_id = max_token + 1
        self.bos_token_id = max_token + 2
        self.vocab_size = max_token + 3

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {i: i for i in range(self.vocab_size)}

    def __call__(self, seq, **kwargs):
        seq = torch.LongTensor([self.bos_token_id] + [int(x) for x in seq.split()] + [self.pad_token_id]).unsqueeze(0)
        mask = torch.ones(seq.shape)
        return {"input_ids": seq, "attention_mask": mask}

    def encode(self, seq, **kwargs):
        return self(seq, **kwargs)["input_ids"][0].tolist()

    def decode(self, seq):
        return " ".join([str(x) for x in seq])


def get_random_tokens(tokenizer, num_tokens=10):
    """Get random tokens from the tokenizer"""

    vocab_size = len(tokenizer)
    random_tokens = np.random.randint(0, vocab_size, num_tokens)
    return random_tokens


class RamdomReplace:
    def __init__(self, tokenizer, num_tokens=512, vocab_size=0):
        self.tokenizer = tokenizer
        if vocab_size == 0:
            vocab_size = len(tokenizer)
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.memory = {}

    def get_random_tokens(self, text):
        if text in self.memory:
            return self.memory[text]
        random_tokens = np.random.randint(0, self.vocab_size, self.num_tokens)
        random_text = self.tokenizer.decode(random_tokens)
        self.memory[text] = random_text
        return random_text


def get_bert_encoder(tokenizer, num_hidden_layers, num_attention_heads, dropout, encoder_dim):
    encoder_config = BertGenerationConfig(
        vocab_size=len(tokenizer.get_vocab()),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        is_encoder_decoder=True,
        is_decoder=False,
        add_cross_attention=False,
        hidden_size=encoder_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=encoder_dim * 4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=512,
    )
    encoder = BertGenerationEncoder(encoder_config)
    return encoder


def get_encoder_decoder(decoder_size="l", dropout=0.2, drugbank=False, gen_mol=False, train_encoder=False,
                        is_text=False, quantize=0, pretrained_encoder=1, encoder_dim=ENCODER_DIM):
    if is_text:
        src_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        src_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        tgt_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t33_650M_UR50D")  # will replace with the quantized tokenizer
    elif gen_mol:
        assert drugbank, "gen_mol can only be used with drugbank"
        src_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        src_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        tgt_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    elif drugbank:
        src_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                              trust_remote_code=True)

        src_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        tgt_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True)
    else:
        src_model, src_tokenizer = get_model_and_tokenizer()
        tgt_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True)
    if quantize:
        tgt_tokenizer = QuantizeTokenizer()

    hidden_size = hidden_size_per_size[decoder_size]
    num_hidden_layers = num_layers_per_size[decoder_size]
    num_attention_heads = num_attention_heads_per_size[decoder_size]
    intermediate_size = hidden_size * 4

    if not pretrained_encoder:
        src_model = get_bert_encoder(src_tokenizer, num_hidden_layers, num_attention_heads, dropout,
                                     encoder_dim=encoder_dim)

    src_model.to(device)
    # Set model state based on train_encoder flag
    if train_encoder:
        src_model.train()
        for param in src_model.parameters():
            param.requires_grad = True
    else:
        src_model.eval()
        for param in src_model.parameters():
            param.requires_grad = False

    # Load the pretrained decoder
    decoder_config = BertGenerationConfig(
        vocab_size=len(tgt_tokenizer.get_vocab()),
        eos_token_id=tgt_tokenizer.eos_token_id,
        pad_token_id=tgt_tokenizer.pad_token_id,
        bos_token_id=tgt_tokenizer.bos_token_id,
        decoder_start_token_id=tgt_tokenizer.pad_token_id,
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

    return src_model, src_tokenizer, decoder, tgt_tokenizer


def load_file(file_path):
    """Load text file"""
    with open(file_path) as f:
        texts = f.read().splitlines()
    return texts


def load_files(level="easy", gen_mol=0, cold_smiles=0, cold_fasta=0, quantize=0):
    """Load training and testing files"""
    base_dir = f"data/{level}"
    if cold_smiles:
        base_dir += "_cs"
    if cold_fasta:
        base_dir += "_cf"

    if gen_mol:
        src_suffix = "_q.txt" if quantize else ".txt"
        tgt_suffix = ".txt"
    else:
        tgt_suffix = "_q.txt" if quantize else ".txt"
        src_suffix = ".txt"

    src_train = load_file(pjoin(base_dir, "train_reaction.txt".replace(".txt", src_suffix)))
    src_valid = load_file(pjoin(base_dir, "valid_reaction.txt".replace(".txt", src_suffix)))
    src_test = load_file(pjoin(base_dir, "test_reaction.txt".replace(".txt", src_suffix)))

    tgt_train = load_file(pjoin(base_dir, "train_enzyme.txt".replace(".txt", tgt_suffix)))
    tgt_valid = load_file(pjoin(base_dir, "valid_enzyme.txt".replace(".txt", tgt_suffix)))
    tgt_test = load_file(pjoin(base_dir, "test_enzyme.txt".replace(".txt", tgt_suffix)))
    print(f"src_train: {len(src_train)}, tgt_train: {len(tgt_train)}")
    print(f"src_valid: {len(src_valid)}, tgt_valid: {len(tgt_valid)}")
    print(f"src_test: {len(src_test)}, tgt_test: {len(tgt_test)}")
    if gen_mol:
        assert level != "easy", "gen_mol can only be used with drugbank"
        src_train, tgt_train = tgt_train, src_train
        src_valid, tgt_valid = tgt_valid, src_valid
        src_test, tgt_test = tgt_test, src_test
    return src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test


class SrcTgtDataset(TorchDataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, src_encoder=None, max_length=512,
                 pooling=False, train_encoder=False):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_length = max_length
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_encoder = src_encoder  # This can be None if we use end-to-end model
        self.pooling = pooling
        self.train_encoder = train_encoder
        self.memory = {}

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Pre-encoded version (used when encoder is frozen)
        if not self.train_encoder and self.src_encoder is not None and src_text in self.memory:
            src_encoder_outputs, src_attention_mask = self.memory[src_text]
        # When training encoder or first time processing
        else:
            src_tokens = self.src_tokenizer(
                src_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
            )

            # If we're not training encoder, we pre-compute encoder outputs
            if not self.train_encoder and self.src_encoder is not None:
                src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
                with torch.no_grad():
                    src_encoder_outputs = self.src_encoder(**src_tokens)
                if self.pooling:
                    if hasattr(self.src_encoder, "pooler_output"):
                        src_encoder_outputs = src_encoder_outputs.pooler_output
                    else:
                        src_encoder_outputs = src_encoder_outputs.last_hidden_state.mean(dim=1)
                    src_attention_mask = torch.ones(1).to(device)
                else:
                    src_encoder_outputs = src_encoder_outputs.last_hidden_state.squeeze(0)
                    src_attention_mask = src_tokens["attention_mask"].squeeze(0)
                src_encoder_outputs = src_encoder_outputs.detach().cpu()
                src_attention_mask = src_attention_mask.detach().cpu()

                # Save to memory if we pre-compute
                self.memory[src_text] = (src_encoder_outputs, src_attention_mask)
            # When training encoder, we just return the input tokens
            else:
                src_encoder_outputs = None
                src_attention_mask = src_tokens["attention_mask"].squeeze(0)
                # We need to keep input_tokens for the model
                src_input_ids = src_tokens["input_ids"].squeeze(0)

        tgt_tokens = self.tgt_tokenizer(
            tgt_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = tgt_tokens["input_ids"].clone()
        labels[labels == self.tgt_tokenizer.pad_token_id] = -100

        # Return format depends on whether we're training the encoder
        if not self.train_encoder and self.src_encoder is not None:
            return dict(
                encoder_outputs=src_encoder_outputs,
                encoder_attention_mask=src_attention_mask,
                input_ids=tgt_tokens["input_ids"].squeeze(0),
                attention_mask=tgt_tokens["attention_mask"].squeeze(0),
                labels=labels.squeeze(0),
            )
        else:
            return dict(
                src_input_ids=src_input_ids,
                src_attention_mask=src_tokens["attention_mask"].squeeze(0),
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


def update_output_with_trie(decoder_outputs, input_ids, trie, vocab_size, labels=None, entropy_normalize=False,
                            path_weights_normalize=False):
    trie_mask, path_weights = build_mask_from_trie(trie, input_ids, vocab_size, return_path_weights=True)

    trie_mask = trie_mask[:, :-1, :]
    trie_mask_out = trie_mask.sum(dim=-1) <= 1
    decoder_outputs.trie_mask_out = trie_mask_out
    valid_token_count = trie_mask.sum(dim=-1)

    trie_mask = trie_mask.masked_fill(trie_mask == 0, -1e6)
    trie_mask = trie_mask.masked_fill(trie_mask == 1, 0)
    trie_mask = trie_mask.to(decoder_outputs.logits.device)
    decoder_outputs.logits[:, :-1] += trie_mask
    if labels is not None:
        labels[:, 1:][trie_mask_out] = -100
        if entropy_normalize:
            information_weights = torch.log(valid_token_count + 1 + 1e-6)  # add 2 to avoid log(1)=0
            info_weights_expanded = information_weights.unsqueeze(-1)
            info_weights_expanded = info_weights_expanded.to(decoder_outputs.logits.device)
            normalized_logits = decoder_outputs.logits[:, :-1] / info_weights_expanded
            decoder_outputs.logits[:, :-1] = normalized_logits

        if path_weights_normalize:
            path_weights = path_weights[:, 1:]
            path_weights = (trie.total_paths * path_weights + 1).log()
            path_weights = path_weights / path_weights.sum(dim=-1, keepdim=True)
            path_weights = 1 - path_weights

            path_weights = path_weights.to(decoder_outputs.logits.device)
            path_weights = path_weights.view(-1)

            # Compute weighted cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            per_token_loss = loss_fct(
                decoder_outputs.logits[:, :-1].reshape(-1, decoder_outputs.logits[:, :-1].size(-1)),
                labels[:, 1:].reshape(-1))  # shape: (batch_size * seq_len)
            per_token_loss = per_token_loss * path_weights
            per_token_loss = per_token_loss.sum() / path_weights.sum()
            decoder_outputs.loss = per_token_loss

        else:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            decoder_outputs.loss = loss_fct(
                decoder_outputs.logits[:, :-1].reshape(-1, decoder_outputs.logits[:, :-1].size(-1)),
                labels[:, 1:].reshape(-1))
    return decoder_outputs


class EndToEndModel(torch.nn.Module):
    def __init__(self, encoder, decoder, trie=None, encoder_dim=ENCODER_DIM, bottleneck_dim=0, pooling=False,
                 entropy_normalize=False, path_weights_normalize=False):
        super(EndToEndModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.entropy_normalize = entropy_normalize
        self.path_weights_normalize = path_weights_normalize
        self.trie = trie
        self.pooling = pooling

        if bottleneck_dim > 0:
            self.encoder_project = torch.nn.Sequential(
                torch.nn.Linear(encoder_dim, bottleneck_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(bottleneck_dim, self.decoder.config.hidden_size)
            )
        else:
            self.encoder_project = torch.nn.Linear(
                encoder_dim, self.decoder.config.hidden_size
            )

    def forward(self, src_input_ids, src_attention_mask, input_ids, attention_mask, labels=None):
        # Run through encoder
        encoder_outputs = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask)

        if self.pooling:
            if hasattr(encoder_outputs, "pooler_output") and encoder_outputs.pooler_output is not None:
                encoder_hidden_states = encoder_outputs.pooler_output.unsqueeze(1)
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state.mean(dim=1)
            if encoder_hidden_states.ndim == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            encoder_attention_mask = torch.ones(src_attention_mask.shape[0], 1).to(src_attention_mask.device)
        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = src_attention_mask

        # Project encoder outputs
        projected_encoder_hidden_states = self.encoder_project(encoder_hidden_states)

        # Run through decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=projected_encoder_hidden_states,
            labels=labels,
        )

        # Apply trie constraints if needed
        if self.trie is None:
            return decoder_outputs
        decoder_outputs = update_output_with_trie(decoder_outputs, input_ids, self.trie, self.decoder.config.vocab_size,
                                                  labels, entropy_normalize=self.entropy_normalize,
                                                  path_weights_normalize=self.path_weights_normalize)
        return decoder_outputs


class EnzymeDecoder(torch.nn.Module):
    def __init__(self, decoder, trie=None, encoder_dim=ENCODER_DIM, bottleneck_dim=0, entropy_normalize=False,
                 path_weights_normalize=False):
        super(EnzymeDecoder, self).__init__()
        self.decoder = decoder
        self.trie = trie
        self.entropy_normalize = entropy_normalize
        self.path_weights_normalize = path_weights_normalize
        if bottleneck_dim > 0:
            self.encoder_project = torch.nn.Sequential(
                torch.nn.Linear(encoder_dim, bottleneck_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(bottleneck_dim, self.decoder.config.hidden_size)
            )
        else:
            self.encoder_project = torch.nn.Linear(
                encoder_dim, self.decoder.config.hidden_size
            )

    def forward(self, input_ids, attention_mask, encoder_outputs, encoder_attention_mask, labels):
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

        decoder_outputs = update_output_with_trie(decoder_outputs, input_ids, self.trie, self.decoder.config.vocab_size,
                                                  labels, entropy_normalize=self.entropy_normalize,
                                                  path_weights_normalize=self.path_weights_normalize)
        return decoder_outputs


def get_auc_valid_test(pos_valid, neg_valid, pos_test, neg_test, model, batch_size=64, auc_only=False):

    from eval_dti import evaluate_model
    return evaluate_model(pos_valid, neg_valid, pos_test, neg_test, model,
                          batch_size=batch_size)

    # auc_score, ap_score, best_acc_threshold, best_acc_score, best_f1_threshold, best_score_f1, fmax = evaluate_model(
    #     pos_valid,
    #     neg_valid, model,
    #     batch_size=batch_size,
    #     auc_only=auc_only,
    #     use_f1_max=auc_only)
    # test_auc_score, test_ap_score, _, test_best_acc_score, _, test_best_score_f1, test_fmax = evaluate_model(pos_test,
    #                                                                                                          neg_test,
    #                                                                                                          model,
    #                                                                                                          best_acc_threshold=best_acc_threshold,
    #                                                                                                          best_f1_threshold=best_f1_threshold,
    #                                                                                                          batch_size=batch_size,
    #                                                                                                          auc_only=auc_only,
    #                                                                                                          use_f1_max=auc_only)
    # return {"auc": auc_score, "auc_test": test_auc_score, "acc": best_acc_score, "f1": best_score_f1,
    #         "acc_test": test_best_acc_score, "f1_test": test_best_score_f1, "ap": ap_score, "ap_test": test_ap_score,
    #         "fmax": fmax, "fmax_test": test_fmax}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--size", type=str, default="l")
    parser.add_argument("--level", type=str, default="biosnap")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--trie", type=int, default=1)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--pooling", type=int, default=1)
    parser.add_argument("--gen_mol", type=int, default=0)
    parser.add_argument("--cold_smiles", type=int, default=0)
    parser.add_argument("--cold_fasta", type=int, default=0)
    parser.add_argument("--random_tgt", type=int, default=0)
    parser.add_argument("--train_encoder", type=int, default=0,
                        help="Whether to train the encoder (1) or freeze it (0)")
    parser.add_argument("--pretrained_encoder", type=int, default=1)
    parser.add_argument("--quantize", type=int, default=1)
    parser.add_argument("--entropy_normalize", type=int, default=0)
    parser.add_argument("--path_weights_normalize", type=int, default=0)
    parser.add_argument("--auto_pretrained", type=int, default=0)

    args = parser.parse_args()

    src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test = load_files(level=args.level, gen_mol=args.gen_mol,
                                                                                cold_smiles=args.cold_smiles,
                                                                                cold_fasta=args.cold_fasta,
                                                                                quantize=args.quantize)
    encoder_dim = ENCODER_DIM
    if args.level == "mf":
        encoder_dim = 1280
    elif args.level != "easy":
        encoder_dim = 768
    if args.gen_mol:
        encoder_dim = 1280
    if args.auto_pretrained:
        encoder_dim = 512
    src_model, src_tokenizer, decoder, tgt_tokenizer = get_encoder_decoder(decoder_size=args.size,
                                                                           dropout=args.dropout,
                                                                           drugbank=args.level != "easy",
                                                                           gen_mol=args.gen_mol,
                                                                           train_encoder=args.train_encoder,
                                                                           is_text=args.level == "mf",
                                                                           quantize=args.quantize,
                                                                           pretrained_encoder=args.pretrained_encoder,
                                                                           encoder_dim=encoder_dim)
    if args.auto_pretrained:
        from train_auto import get_auto_prep
        src_model = get_auto_prep(is_mol=args.gen_mol==0)
        src_model.to(device)
        src_model.eval()
        for param in src_model.parameters():
            param.requires_grad = False

    random_replace = None
    if args.random_tgt:
        vocab_siz = len(tgt_tokenizer) if args.random_tgt == 1 else args.random_tgt
        random_replace = RamdomReplace(tgt_tokenizer, vocab_size=vocab_siz)
        tgt_train = [random_replace.get_random_tokens(x) for x in tgt_train]
        tgt_valid = [random_replace.get_random_tokens(x) for x in tgt_valid]
        tgt_test = [random_replace.get_random_tokens(x) for x in tgt_test]

    # Configure dataset based on whether we're training the encoder
    if args.train_encoder:
        # Don't pass the encoder to dataset when training end-to-end
        train_dataset = SrcTgtDataset(src_train, tgt_train, src_tokenizer, tgt_tokenizer,
                                      max_length=512, pooling=args.pooling, train_encoder=True)
        valid_dataset = SrcTgtDataset(src_valid, tgt_valid, src_tokenizer, tgt_tokenizer,
                                      max_length=512, pooling=args.pooling, train_encoder=True)
        test_dataset = SrcTgtDataset(src_test, tgt_test, src_tokenizer, tgt_tokenizer,
                                     max_length=512, pooling=args.pooling, train_encoder=True)
    else:
        # Use the original approach with pre-computed encoder outputs
        train_dataset = SrcTgtDataset(src_train, tgt_train, src_tokenizer, tgt_tokenizer, src_model,
                                      pooling=args.pooling)
        valid_dataset = SrcTgtDataset(src_valid, tgt_valid, src_tokenizer, tgt_tokenizer, src_model,
                                      pooling=args.pooling)
        test_dataset = SrcTgtDataset(src_test, tgt_test, src_tokenizer, tgt_tokenizer, src_model,
                                     pooling=args.pooling)

    train_small_indices = np.random.choice(len(train_dataset), len(test_dataset), replace=False)
    train_small_dataset = torch.utils.data.Subset(train_dataset, train_small_indices)

    if args.trie:
        trie = build_trie(list(set(tgt_train + tgt_test)), tgt_tokenizer)
    else:
        trie = None

    # Choose the appropriate model based on whether we're training the encoder
    if args.train_encoder:
        model = EndToEndModel(
            encoder=src_model,
            decoder=decoder,
            trie=trie,
            encoder_dim=encoder_dim,
            bottleneck_dim=args.bottleneck_dim,
            pooling=args.pooling,
            entropy_normalize=args.entropy_normalize,
            path_weights_normalize=args.path_weights_normalize
        )
    else:
        model = EnzymeDecoder(
            decoder,
            trie=trie,
            encoder_dim=encoder_dim,
            bottleneck_dim=args.bottleneck_dim,
            entropy_normalize=args.entropy_normalize,
            path_weights_normalize=args.path_weights_normalize
        )

    if args.level != "easy":
        from eval_drug_bank import evaluate_model, get_data

        pos_valid, neg_valid, pos_test, neg_test = get_data(args.pooling, src_model, src_tokenizer, tgt_tokenizer,
                                                            gen_mol=args.gen_mol, cold_smiles=args.cold_smiles,
                                                            cold_fasta=args.cold_fasta, random_replace=random_replace,
                                                            train_encoder=args.train_encoder, quantize=args.quantize,
                                                            level=args.level)

        compute_metrics_func = lambda x: get_auc_valid_test(pos_valid, neg_valid, pos_test, neg_test, model,
                                                            batch_size=args.batch_size, auc_only=args.level == "mf")

        # subset with size 1, run the script
        test_dataset_dummy = torch.utils.data.Subset(test_dataset, [0])
        eval_dataset = {"valid": test_dataset_dummy}
        metric_for_best_model = "eval_valid_auc"

    else:
        compute_metrics_func = lambda x: compute_metrics(x)
        eval_dataset = {"test": test_dataset, "train": train_small_dataset, "valid": valid_dataset}
        metric_for_best_model = "eval_test_token_accuracy"

    print(f"Training with {args.level} level")
    print("Src model:")
    print(src_model)
    print("Number of parameters:", sum(p.numel() for p in src_model.parameters() if p.requires_grad))
    print("model:")
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    output_dir = f"results/{args.level}_{args.size}_{args.dropout}_{args.learning_rate}"
    if args.trie == 0:
        output_dir += "_notrie"
    if args.bottleneck_dim > 0:
        output_dir += f"_bottleneck_{args.bottleneck_dim}"
    if args.pooling:
        output_dir += "_pooling"
    if args.gen_mol:
        output_dir += "_mol"
    if args.cold_smiles:
        output_dir += "_cs"
    if args.cold_fasta:
        output_dir += "_cf"
    if args.random_tgt:
        output_dir += f"_rnd{args.random_tgt}"
    if args.train_encoder:
        output_dir += f"_trainenc"
    if args.quantize:
        output_dir += "_quantize"
    if args.entropy_normalize:
        output_dir += "_entropy"
    if args.path_weights_normalize:
        output_dir += "_pathinv"
    if not args.pretrained_encoder:
        output_dir += "_noenc"
    if args.auto_pretrained:
        output_dir += "_autoenc"
    output_dir = output_dir.replace("results", f"results_{args.level}")
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
        max_steps=args.steps,
        fp16=args.fp16,
        logging_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        report_to=[args.report_to],
        save_safetensors=False,
        auto_find_batch_size=True,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_func
    )
    eval_logging_callback = EvalLoggingCallback(output_dir=output_dir)
    trainer.add_callback(eval_logging_callback)

    print(trainer.evaluate(eval_dataset["valid"]))
    # Train model
    print("Training model...")

    # trainer.train(resume_from_checkpoint=len(glob.glob(pjoin(output_dir, "checkpoint-*"))) > 0)
    trainer.train()
    print("Training complete!")
