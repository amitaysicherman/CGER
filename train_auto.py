from transformers import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder
from transformers import AutoTokenizer
import torch
from transformers import TrainingArguments, Trainer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoDataset(torch.utils.data.Dataset):
    def __init__(self, input_file="data/biosnap/train_enzyme.txt", tokenizer="facebook/esm2_t33_650M_UR50D"):
        with open(input_file, "r") as f:
            self.sequences = list(set(f.read().splitlines()))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(device),
            "labels": labels.squeeze(0).to(device),
        }


def get_encoder_decoder_decoder(tokenizer, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size,
                                dropout):
    encoder_config = BertGenerationConfig(
        vocab_size=len(tokenizer.get_vocab()),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        is_encoder_decoder=True,
        is_decoder=False,
        add_cross_attention=False,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=512,
    )
    encoder = BertGenerationEncoder(encoder_config)

    decoder_config = BertGenerationConfig(
        vocab_size=len(tokenizer.get_vocab()),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
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
    return encoder, decoder


from train import update_output_with_trie


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, trie):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trie = trie

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = encoder_outputs.last_hidden_state.mean(dim=1).unsqueeze(1)
        mask = encoder_outputs.new_ones(encoder_outputs.size()[:-1], dtype=torch.bool)
        decoder_outputs = self.decoder(encoder_hidden_states=encoder_outputs, attention_mask=mask,
                                       labels=labels, input_ids=input_ids)
        decoder_outputs = update_output_with_trie(decoder_outputs, input_ids, self.trie,
                                                  vocab_size=self.decoder.config.vocab_size, labels=labels)
        return decoder_outputs


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model with a trie.")
    parser.add_argument("--input_file", type=str, default="data/biosnap/train_enzyme.txt",
                        help="Path to the input file containing sequences.")
    parser.add_argument("--output_dir", type=str, default="./finetuned_mlm",
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden size of the model.")
    parser.add_argument("--num_hidden_layers", type=int, default=4,
                        help="Number of hidden layers in the model.")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                        help="Number of attention heads in the model.")
    parser.add_argument("--intermediate_size", type=int, default=1024,
                        help="Intermediate size of the model.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for the model.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                        help="Learning rate scheduler type.")
    parser.add_argument("--logging_steps", type=int, default=250,
                        help="Number of steps between logging.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Number of steps between saving the model.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Number of steps between evaluations.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        help="Evaluation strategy.")
    parser.add_argument("--metric_for_best_model", type=str, default="sample_accuracy",
                        help="Metric for selecting the best model.")
    parser.add_argument("--remove_unused_columns", type=bool, default=False,
                        help="Whether to remove unused columns.")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Total limit of checkpoints to save.")
    parser.add_argument("--auto_find_batch_size", type=bool, default=True,
                        help="Whether to automatically find the batch size.")
    args = parser.parse_args()
    output_dir = args.output_dir
    input_file = args.input_file
    hidden_size = args.hidden_size
    num_hidden_layers = args.num_hidden_layers
    num_attention_heads = args.num_attention_heads
    intermediate_size = args.intermediate_size
    dropout = args.dropout
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    lr_scheduler_type = args.lr_scheduler_type
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    evaluation_strategy = args.evaluation_strategy
    metric_for_best_model = args.metric_for_best_model
    remove_unused_columns = args.remove_unused_columns
    save_total_limit = args.save_total_limit
    auto_find_batch_size = args.auto_find_batch_size
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")





    tokenizer_file = "facebook/esm2_t33_650M_UR50D"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file)
    dataset = AutoDataset(input_file=input_file, tokenizer=tokenizer_file)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset,
                                                                [int(0.9 * len(dataset)), int(0.1 * len(dataset))])
    from trie import build_trie

    trie = build_trie(dataset.sequences, tokenizer)
    encoder, decoder = get_encoder_decoder_decoder(tokenizer, hidden_size, num_hidden_layers, num_attention_heads,
                                                   intermediate_size, dropout)
    model = EncoderDecoder(encoder, decoder, trie).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy=evaluation_strategy,
        metric_for_best_model=metric_for_best_model,
        remove_unused_columns=remove_unused_columns,
        save_total_limit=save_total_limit,
        auto_find_batch_size=auto_find_batch_size,
        save_safetensors=False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
