# --- Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import math
from transformers import AutoTokenizer, AutoModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- Configuration ---
config = {
    "vector_dim": 768,
    "context_vector_dim": 1280,
    "hidden_dim": 256,
    "num_layers": 4,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_train_epochs": 100,
    "num_inference_steps": 50,
    "num_train_timesteps": 1000,
    "output_dir": "simple_1d_diffusion_model",
    "seed": 42
}


# --- Dummy Dataset ---
class SimpleVectorDataset(Dataset):
    def __init__(self, split="train", max_len=512):
        self.src_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.src_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        self.tgt_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                                   trust_remote_code=True)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.src_model.to(device)
        self.src_model.eval()

        self.tgt_model.to(device)
        self.tgt_model.eval()

        self.src_mem = dict()
        self.tgt_mem = dict()

        with open(f"data/biosnap/{split}_enzyme.txt") as f:
            self.src_lines = f.read().splitlines()
        with open(f"data/biosnap/{split}_reaction.txt") as f:
            self.tgt_lines = f.read().splitlines()
        assert len(self.src_lines) == len(self.tgt_lines)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx]
        tgt_line = self.tgt_lines[idx]
        if src_line not in self.src_mem:
            src_tokens = self.src_tokenizer(src_line, truncation=True, padding="max_length", max_length=512,
                                            return_tensors="pt")
            with torch.no_grad():
                src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
                src_embedding = self.src_model(**src_tokens)
                src_embedding = src_embedding.last_hidden_state.mean(dim=1).detach()
            self.src_mem[src_line] = src_embedding
        else:
            src_embedding = self.src_mem[src_line]
        if tgt_line not in self.tgt_mem:
            tgt_tokens = self.tgt_tokenizer(tgt_line, truncation=True, padding="max_length", max_length=512,
                                            return_tensors="pt")

            with torch.no_grad():
                tgt_tokens = {k: v.to(device) for k, v in tgt_tokens.items()}
                tgt_embedding = self.tgt_model(**tgt_tokens)
                tgt_embedding = tgt_embedding.pooler_output.detach()

            self.tgt_mem[tgt_line] = tgt_embedding
        else:
            tgt_embedding = self.tgt_mem[tgt_line]
        return {
            "target_vector": tgt_embedding.squeeze(0),
            "context_vector": src_embedding.squeeze(0)
        }


# --- Simple 1D Transformer Model ---
class Simple1DTransformer(nn.Module):
    def __init__(self, vector_dim, context_dim, hidden_dim, num_layers):
        super().__init__()

        # Project input vector to hidden dim
        self.input_proj = nn.Linear(vector_dim, hidden_dim)

        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Context projection
        self.context_proj = nn.Linear(context_dim, hidden_dim)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vector_dim)

    def forward(self, x, timestep, context):
        """
        x: (batch_size, vector_dim)
        timestep: (batch_size,)
        context: (batch_size, context_dim)
        """
        # Reshape to sequence format (batch_size, 1, dim)
        x = x.unsqueeze(1)

        # Project input
        x = self.input_proj(x)

        # Add timestep embedding
        t_emb = self.timestep_embed(timestep.unsqueeze(-1).float())
        x = x + t_emb.unsqueeze(1)

        # Prepare context
        context = self.context_proj(context).unsqueeze(1)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context)

        # Project back to original dimension
        output = self.output_proj(x)

        return output.squeeze(1)


# --- Simple Noise Scheduler ---
class SimpleNoiseScheduler:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def add_noise(self, x_0, noise, timesteps):
        """Add noise to data according to timesteps"""
        alpha_prod = self.alphas_cumprod[timesteps].to(device)
        alpha_prod = alpha_prod.view(-1, 1)  # (batch, 1)

        sqrt_alpha_prod = torch.sqrt(alpha_prod)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_prod)

        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

    def step(self, model_output, timestep, sample):
        """Denoise one step"""
        t = timestep.item()
        alpha = self.alphas[t].to(sample.device)
        alpha_prod = self.alphas_cumprod[t].to(sample.device)
        beta = self.betas[t].to(sample.device)

        if t > 0:
            prev_alpha_prod = self.alphas_cumprod[t - 1].to(sample.device)
        else:
            prev_alpha_prod = torch.tensor(1.0).to(sample.device)

        # Predict original sample
        pred_original = (sample - torch.sqrt(1 - alpha_prod) * model_output) / torch.sqrt(alpha_prod)

        # Compute mean of previous timestep
        pred_prev_sample = torch.sqrt(prev_alpha_prod) * pred_original + torch.sqrt(1 - prev_alpha_prod) * model_output

        # Add noise for all steps except the last one
        if t > 0:
            noise = torch.randn_like(sample)
            variance = ((1 - prev_alpha_prod) / (1 - alpha_prod)) * beta
            sigma = torch.sqrt(variance)
            pred_prev_sample = pred_prev_sample + sigma * noise

        return pred_prev_sample


# --- Training Function ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = Simple1DTransformer(
        vector_dim=config["vector_dim"],
        context_dim=config["context_vector_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"]
    ).to(device)

    # Create dataset and dataloader
    dataset = SimpleVectorDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    noise_scheduler = SimpleNoiseScheduler(num_timesteps=config["num_train_timesteps"])
    all_target_vectors = []
    for i in range(len(dataset)):
        all_target_vectors.append(dataset[i]["target_vector"])
    all_target_vectors = torch.stack(all_target_vectors).to(device)

    # Training loop
    for epoch in range(config["num_train_epochs"]):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['num_train_epochs']}")

        for batch in progress_bar:
            clean_vectors = batch["target_vector"].to(device)
            context = batch["context_vector"].to(device)

            # Sample noise and timesteps
            noise = torch.randn_like(clean_vectors)
            timesteps = torch.randint(0, config["num_train_timesteps"], (clean_vectors.shape[0],), device=device)

            # Add noise to vectors
            noisy_vectors = noise_scheduler.add_noise(clean_vectors, noise, timesteps)

            # Predict noise
            predicted_noise = model(noisy_vectors, timesteps, context)

            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch + 1} average loss: {epoch_loss / len(dataloader):.4f}")
        avg_rank, mrr, hits_at_k = evaluate_epoch(model, dataloader, noise_scheduler,
                                                  all_target_vectors, config["num_inference_steps"], device)

        print(f"  Average Rank: {avg_rank:.2f}")
        print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
        for i, k in enumerate([1, 5, 10]):
            print(f"  Hits@{k}: {hits_at_k[i]:.4f}")
        print("-" * 50)
    return model, noise_scheduler


# --- Metric Calculation Function ---
def calculate_similarity_rank(generated_vectors, gt_vectors, ks=[1, 5, 10]):
    """
    Calculate the rank of ground truth vector in the similarity to generated vector

    Args:
        generated_vectors: (batch_size, vector_dim) - generated vectors
        gt_vectors: (total_samples, vector_dim) - all ground truth vectors
        k: top-k rank to calculate

    Returns:
        avg_rank: average rank of GT vector among similarities
        avg_reciprocal_rank: average reciprocal rank (MRR)
        hits_at_k: percentage of GT vectors within top-k
    """
    batch_size = generated_vectors.shape[0]
    ranks = []
    reciprocal_ranks = []
    hits_at_k_count = [0] * len(ks)

    for i in range(batch_size):
        # Compute cosine similarity between generated vector and all GT vectors
        similarities = F.cosine_similarity(generated_vectors[i].unsqueeze(0).to(device), gt_vectors.to(device), dim=1)

        # Get the rank of the GT vector (corresponding to batch index i)
        _, sorted_indices = similarities.sort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank

        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        for k in ks:
            if rank <= k:
                hits_at_k_count[ks.index(k)] += 1

    avg_rank = np.mean(ranks)
    avg_reciprocal_rank = np.mean(reciprocal_ranks)
    hits_at_k = [count / batch_size for count in hits_at_k_count]

    return avg_rank, avg_reciprocal_rank, hits_at_k


# --- Evaluation Function ---
@torch.no_grad()
def evaluate_epoch(model, dataloader, scheduler, all_target_vectors, num_inference_steps, device):
    """Evaluate the model by generating samples and computing metrics"""
    model.eval()

    all_generated = []
    all_gt_indices = []

    for batch_idx, batch in enumerate(dataloader):
        context_vectors = batch["context_vector"].to(device)
        target_vectors = batch["target_vector"].to(device)
        all_target_vectors = batch["target_vector"].to(device)
        batch_size = context_vectors.shape[0]

        # Generate vectors for this batch
        generated_vectors = generate_samples_batch(model, scheduler, context_vectors, target_vectors,
                                                   num_inference_steps, device)

        all_generated.append(generated_vectors.cpu())
        all_gt_indices.extend(range(batch_idx * dataloader.batch_size,
                                    min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))))

    # Concatenate all generated vectors
    all_generated = torch.cat(all_generated, dim=0)

    # Calculate metrics
    avg_rank, mrr, hits_at_k = calculate_similarity_rank(all_generated, all_target_vectors)

    return avg_rank, mrr, hits_at_k


# --- Inference Function ---
@torch.no_grad()
def generate_samples_batch(model, scheduler, context_vectors, target_vectors, num_inference_steps, device):
    """Generate vectors conditioned on context for a batch"""
    model.eval()
    batch_size = context_vectors.shape[0]

    # Start with random noise
    x = torch.randn(batch_size, config["vector_dim"], device=device)

    # Denoise step by step
    timesteps = torch.linspace(config["num_train_timesteps"] - 1, 0, num_inference_steps, device=device).long()
    pbar= tqdm(timesteps, desc="Generating samples", unit="step")
    for t in pbar:
        # Predict noise
        predicted_noise = model(x, torch.full((batch_size,), t, device=device), context_vectors)

        # Denoise
        x = scheduler.step(predicted_noise, t, x)
        loss_in_step = F.mse_loss(x, target_vectors)
        pbar.set_postfix({"loss_in_step": loss_in_step.item()})

    return x


# --- Main ---
if __name__ == "__main__":
    print("Training model...")
    model, scheduler = train()

    print("\nGenerating samples...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate 4 samples
    # test_context = torch.randn(4, config["context_vector_dim"]).to(device)
    # generated_vectors = generate_samples_batch(model, scheduler, test_context, config["num_inference_steps"], device)

    # print(f"\nGenerated vectors shape: {generated_vectors.shape}")
    # print(f"First generated vector:\n{generated_vectors[0]}")
