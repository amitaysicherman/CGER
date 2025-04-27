import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from eval_drug_bank import load_negative_files, get_model


def get_data(pooling, src_model, src_tokenizer, tgt_tokenizer, gen_mol, cold_smiles=0, cold_fasta=0, quantize=False,
             level="drugbank"):
    src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test = load_files(level=level, gen_mol=gen_mol,
                                                                                cold_smiles=cold_smiles,
                                                                                cold_fasta=cold_fasta,
                                                                                quantize=quantize)
    pos_train = SrcTgtDataset(src_train, tgt_train, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    pos_valid = SrcTgtDataset(src_valid, tgt_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    pos_test = SrcTgtDataset(src_test, tgt_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)

    src_neg_train, tgt_neg_train = load_negative_files("train", gen_mol, cold_smiles=cold_smiles, cold_fasta=cold_fasta,
                                                       quantize=quantize, level=level)
    src_neg_valid, tgt_neg_valid = load_negative_files("valid", gen_mol, cold_smiles=cold_smiles, cold_fasta=cold_fasta,
                                                       quantize=quantize, level=level)
    src_neg_test, tgt_neg_test = load_negative_files("test", gen_mol, cold_smiles=cold_smiles, cold_fasta=cold_fasta,
                                                     quantize=quantize, level=level)

    neg_train = SrcTgtDataset(src_neg_train, tgt_neg_train, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    neg_valid = SrcTgtDataset(src_neg_valid, tgt_neg_valid, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)
    neg_test = SrcTgtDataset(src_neg_test, tgt_neg_test, src_tokenizer, tgt_tokenizer, src_model, pooling=pooling)

    return pos_train, neg_train, pos_valid, neg_valid, pos_test, neg_test, tgt_train


def get_batch_logits(model, batch):
    all_logits = []
    all_labels = []
    for batch in tqdm(batch, desc="Calculating logits", total=len(batch)):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        all_logits.append(output["logits"][:, :-1].detach().cpu().numpy())
        all_labels.append(batch["labels"][:, 1:].detach().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_logits, all_labels


@dataclass
class Config:
    size: str = "l"
    dropout: float = 0.0
    pooling: bool = True
    bottleneck_dim: int = 128
    learning_rate: float = 0.0001
    mol: bool = True
    cold_smiles: int = 0
    cold_fasta: int = 0
    quantize: bool = False

    def to_list(self):
        return [self.size, self.dropout, self.pooling, self.bottleneck_dim, self.learning_rate, self.mol,
                self.cold_smiles,
                self.cold_fasta, self.quantize]


def save_or_load(model, LEVEL):
    datasets = ["valid", "test"]
    splits = ["pos", "neg"]

    logits_dir = "logits"
    os.makedirs(logits_dir, exist_ok=True)
    data = dict()
    for dataset in datasets:
        for split in splits:
            if not os.path.exists(os.path.join(logits_dir, f"{dataset}_{split}_logits_{LEVEL}.npz")):
                data_loader = DataLoader(eval(f"{split}_{dataset}"), batch_size=32, shuffle=False)
                with torch.no_grad():
                    logits, labels = get_batch_logits(model, data_loader)
                np.savez(os.path.join(logits_dir, f"{dataset}_{split}_logits_{LEVEL}.npz"), logits=logits)
                np.savez(os.path.join(logits_dir, f"{dataset}_{split}_labels_{LEVEL}.npz"), labels=labels)
            data[f"{dataset}_{split}_logits"] = \
                np.load(os.path.join(logits_dir, f"{dataset}_{split}_logits_{LEVEL}.npz"))["logits"]
            data[f"{dataset}_{split}_labels"] = \
                np.load(os.path.join(logits_dir, f"{dataset}_{split}_labels_{LEVEL}.npz"))["labels"]
    return data


@dataclass
class Features:
    log_prob: float
    rank: float
    prob_over_random: float
    prob_to_next_largest: float
    entropy: float
    opt_count: float
    index: int


def features_list_to_dataframe(features_list):
    """
    Convert a list of Features objects to a pandas DataFrame.
    :param features_list: List of Features objects
    :return: DataFrame
    """
    data = [f.__dict__ for f in features_list]
    df = pd.DataFrame(data)
    return df


def get_sample_features(logits, labels):
    """
    Get the features of the samples from the single sample logits and labels.
    :param logits:  shape (seq_len,vocab_size)
    :param labels:  shape (seq_len) labels is -100 for padding
    """
    features_list = []
    for i in range(logits.shape[0]):
        if labels[i] == -100:
            continue
        logit = logits[i]
        log_probs = F.log_softmax(torch.tensor(logit), dim=0)
        probs = F.softmax(torch.tensor(logit), dim=0)
        non_zero_count = (~torch.isclose(log_probs, torch.tensor(-1e6))).sum()
        features = Features(
            log_prob=log_probs[labels[i]].item(),
            rank=torch.argsort(log_probs, descending=True).tolist().index(labels[i]),
            prob_over_random=log_probs[labels[i]].item() - (1 / non_zero_count),
            prob_to_next_largest=log_probs[labels[i]].item() - log_probs[
                torch.argsort(log_probs, descending=True)[1]].item(),
            entropy=-torch.sum(probs * log_probs).item(),
            opt_count=non_zero_count.item(),
            index=i
        )
        features_list.append([features.log_prob, features.rank, features.prob_over_random,
                              features.prob_to_next_largest, features.entropy, features.opt_count, features.index])
    return features_list


def data_split_to_features(logits, labels):
    """
    Get the features of the samples from the logits and labels.
    :param logits:  shape (batch_size,seq_len,vocab_size)
    :param labels:  shape (batch_size,seq_len) labels is -100 for padding
    """
    features = []
    lens = []
    for i in range(logits.shape[0]):
        feature = get_sample_features(logits[i], labels[i])
        features.extend(feature)
        lens.append(len(feature))
    return np.array(features), np.array(lens)


def main():
    cong = Config("l", 0.2, True, 128, 0.0001, False, 0, 0, quantize=True)
    LEVEL = "biosnap"
    size, dropout, pooling, bottleneck_dim, learning_rate, mol, cold_smiles, cold_fasta, quantize = cong.to_list()
    reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size=size,
                                                                                     dropout=dropout,
                                                                                     drugbank=True, gen_mol=mol,
                                                                                     quantize=quantize)

    pos_train, neg_train, pos_valid, neg_valid, pos_test, neg_test, tgt_train = get_data(pooling, reaction_model,
                                                                                         reaction_tokenizer,
                                                                                         esm_tokenizer,
                                                                                         gen_mol=mol,
                                                                                         cold_smiles=cold_smiles,
                                                                                         cold_fasta=cold_fasta,
                                                                                         quantize=quantize,
                                                                                         level=LEVEL)

    trie = build_trie(list(set(tgt_train)), esm_tokenizer, max_length=512)
    reaction_model.to(device).eval()
    decoder.to(device).eval()
    model = get_model(decoder, trie, size, dropout, pooling, bottleneck_dim, learning_rate, mol, quantize=quantize,
                      level=LEVEL, cold_fasta=cold_fasta, cold_smiles=cold_smiles)
    model.to(device).eval()
    data = save_or_load(model, LEVEL)
    features = {"valid": None, "test": None}
    for split in [ "valid", "test"]:
        pos_logits = data[f"{split}_pos_logits"]
        pos_labels = data[f"{split}_pos_labels"]
        neg_logits = data[f"{split}_neg_logits"]
        neg_labels = data[f"{split}_neg_labels"]
        pos_features,pos_len = data_split_to_features(pos_logits, pos_labels)
        neg_features,nen_len = data_split_to_features(neg_logits, neg_labels)
        all_features = np.concatenate([pos_features, neg_features], axis=0)
        all_len = np.concatenate([pos_len, nen_len], axis=0)
        labels = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))], axis=0)
        features[split] = (all_features, all_len, labels)


print("Preparing training and evaluation data...")
valid_features, valid_lengths, valid_labels = features["valid"]
test_features, test_lengths, test_labels = features["test"]

# Create sequence level data with padding
max_seq_length = max(np.max(valid_lengths), np.max(test_lengths))


def prepare_sequence_data(features, lengths, labels):
    """
    Prepare sequence data by grouping features by their sequence and padding to max_seq_length
    """
    seq_features = []
    seq_labels = []

    # Group features by their sequence
    start_idx = 0
    for length in lengths:
        if length == 0:
            continue
        # Extract features for this sequence
        seq_feats = features[start_idx:start_idx + length]
        # Pad to max_seq_length
        padded_feats = np.zeros((max_seq_length, seq_feats.shape[1]))
        padded_feats[:length] = seq_feats
        seq_features.append(padded_feats)
        start_idx += length

    return np.array(seq_features), labels[:len(seq_features)]


# Prepare sequence data for training and evaluation
X_valid, y_valid = prepare_sequence_data(valid_features, valid_lengths, valid_labels)
X_test, y_test = prepare_sequence_data(test_features, test_lengths, test_labels)

print(f"Prepared data shapes - Valid: {X_valid.shape}, Test: {X_test.shape}")

# Configure Transformer model with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from datetime import datetime
import csv
import random

# Set random seed base for reproducibility
base_seed = 42
np.random.seed(base_seed)
random.seed(base_seed)
torch.manual_seed(base_seed)


# Define Transformer Encoder model
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding - learnable
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sequence mask for padding (1 for tokens to attend to, 0 for padded positions)
        self.register_buffer('sequence_mask', torch.ones(max_seq_length))

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        # x shape: [batch_size, seq_len, feature_dim]

        # Create attention mask based on sequence lengths if provided
        if lengths is not None:
            mask = torch.zeros(x.size(0), x.size(1), device=x.device)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1.0
            mask = mask.bool()
        else:
            mask = None

        # Project input features to model dimension
        x = self.input_projection(x)

        # Add positional encodings
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Apply transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=~mask if mask is not None else None)

        # Global average pooling over non-padded sequence elements
        if mask is not None:
            # Apply mask for accurate mean calculation
            encoded = encoded * mask.unsqueeze(-1)
            pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = encoded.mean(dim=1)

        # Apply classifier
        output = self.classifier(pooled)

        return output.squeeze(-1)


# Create data loaders
def create_data_loader(features, labels, batch_size, shuffle=True):
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels)
    dataset = TensorDataset(features_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Training and evaluation functions
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, auc


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)  # Average precision
    return epoch_loss, auc, ap, all_preds, all_labels


# Define a function for a single training run with specific hyperparameters
def run_training(config, run_id, X_valid, y_valid, X_test, y_test):
    # Set seed for reproducibility
    run_seed = base_seed + run_id
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    # Extract hyperparameters from config
    d_model = config['d_model']
    nhead = config['nhead']
    num_layers = config['num_layers']
    dropout = config['dropout']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    patience = config['patience']

    print(f"Run {run_id} - Config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
          f"dropout={dropout:.2f}, lr={learning_rate:.2e}, weight_decay={weight_decay:.2e}")

    # Create data loaders
    valid_loader = create_data_loader(X_valid, y_valid, batch_size, shuffle=True)
    test_loader = create_data_loader(X_test, y_test, batch_size, shuffle=False)

    # Initialize model
    input_dim = X_valid.shape[2]  # Feature dimension
    model = TransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=num_epochs)
    # Train model
    best_valid_auc = 0
    best_model_state = None
    no_improve = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # Train
        train_loss, train_auc = train_epoch(model, valid_loader, criterion, optimizer)

        # Evaluate on test set (using as validation)
        valid_loss, valid_auc, valid_ap, _, _ = evaluate(model, test_loader, criterion)

        # Update learning rate based on validation AUC
        scheduler.step(valid_auc)

        # Save best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_model_state = model.state_dict().copy()
            no_improve = 0
            best_epoch = epoch
        else:
            no_improve += 1

        # Print progress
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                  f"Valid Loss: {valid_loss:.4f}, AUC: {valid_auc:.4f}")

        # Early stopping
        if no_improve >= patience:
            print(f"  Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_loss, test_auc, test_ap, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"  Run {run_id} Results | Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}, Best Epoch: {best_epoch + 1}")

    # Save model

    # Return results
    return {
        'run_id': run_id,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'best_epoch': best_epoch + 1,  # 1-indexed for human readability
        'test_loss': test_loss,
        'test_auc': test_auc,
        'test_ap': test_ap,
    }


# Define hyperparameter grid for random search
param_grid = {
    'd_model': [64, 128, 256, 512],
    'nhead': [2, 4, 8],
    'num_layers': [1, 2, 3, 4],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'batch_size': [16, 32, 64],
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    'weight_decay': [0, 1e-6, 1e-5, 1e-4, 1e-3],
    'num_epochs': [100],  # Fixed for all runs
    'patience': [15]  # Fixed for all runs
}

# Set number of random configurations to try
num_random_configs = 10_000


# Generate random hyperparameter configurations
def generate_random_configs(param_grid, num_configs):
    configs = []

    for i in range(num_configs):
        config = {}
        for param, values in param_grid.items():
            config[param] = random.choice(values)

        # Ensure nhead divides d_model
        while config['d_model'] % config['nhead'] != 0:
            config['nhead'] = random.choice([n for n in param_grid['nhead'] if config['d_model'] % n == 0])

        configs.append(config)

    return configs


# Generate random configurations
random_configs = generate_random_configs(param_grid, num_random_configs)

# Create directory for results
os.makedirs('results_meta', exist_ok=True)

# Prepare CSV file for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'results/hyperparam_search_{timestamp}.csv'

# Define CSV header
csv_header = [
    'run_id', 'd_model', 'nhead', 'num_layers', 'dropout',
    'batch_size', 'learning_rate', 'weight_decay',
    'best_epoch', 'test_loss', 'test_auc', 'test_ap'
]

# Create CSV file and write header
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
    writer.writeheader()

# Run random grid search
all_results = []

print(f"\nStarting Random Grid Search with {num_random_configs} configurations")
print(f"Results will be saved to: {csv_filename}")

for run_id, config in enumerate(random_configs):
    print(f"\n=== Starting Run {run_id}/{num_random_configs} ===")
    result = run_training(config, run_id, X_valid, y_valid, X_test, y_test)
    all_results.append(result)

    # Append result to CSV file
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writerow(result)

# Sort results by test AUC
all_results.sort(key=lambda x: x['test_auc'], reverse=True)

# Print top configurations
print("\n=== Top 5 Configurations ===")
for i in range(min(5, len(all_results))):
    result = all_results[i]
    print(f"Rank {i + 1} (Run {result['run_id']}) | "
          f"Test AUC: {result['test_auc']:.4f}, AP: {result['test_ap']:.4f}")
    print(f"  d_model={result['d_model']}, nhead={result['nhead']}, "
          f"num_layers={result['num_layers']}, dropout={result['dropout']:.2f}, "
          f"batch_size={result['batch_size']}, lr={result['learning_rate']:.2e}, "
          f"weight_decay={result['weight_decay']:.2e}")

# Save top configuration to a separate file
best_result = all_results[0]
with open(f'results/best_config_{timestamp}.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
    writer.writeheader()
    writer.writerow(best_result)

print(f"\nRandom Grid Search complete. Results saved to {csv_filename}")
print(f"Best configuration saved to: results/best_config_{timestamp}.csv")