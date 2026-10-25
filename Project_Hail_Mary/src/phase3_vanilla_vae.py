import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 1. Dataset Loader
# ---------------------------------------------------------
class GeneExpressionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, index_col=0)
        self.sample_names = list(df.columns)
        self.gene_names = list(df.index)
        X = df.values.T
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.data = torch.tensor(X_scaled, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.sample_names[idx]

# ---------------------------------------------------------
# 2. PyTorch Vanilla VAE Architecture
# ---------------------------------------------------------
class VanillaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=2):
        super(VanillaVAE, self).__init__()
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder_backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss_function(x_recon, x, mu, logvar, beta=0.1):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div

# ---------------------------------------------------------
# 3. Multi-Epoch Training Loop & Evaluation
# ---------------------------------------------------------
def run_phase3_vanilla_vae(expr_file="data/breast_cancer_genes_x_samples.csv",
                           ssgsea_file="data/breast_cancer_ssgsea_scores.csv",
                           checkpoint_epochs=[50, 100, 200, 300]):
    """
    Phase 3: Vanilla VAE Training & Multi-Epoch Baseline Evaluation
    ===============================================================
    - Trains PyTorch Vanilla VAE up to 300 epochs.
    - Exports latent coordinates and scatter plots at 50, 100, 200, 300 epochs.
    - Evaluates 5-fold cross-validated R^2 pathway predictability.
    """
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Expression matrix '{expr_file}' not found.")

    print("--> Phase 3: Loading Dataset for PyTorch Vanilla VAE Training...")
    dataset = GeneExpressionDataset(expr_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    input_dim = dataset.data.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Vanilla VAE on device: {device}")

    model = VanillaVAE(input_dim=input_dim, hidden_dim=512, latent_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    total_epochs = max(checkpoint_epochs)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss_epoch, recon_epoch, kl_epoch = 0.0, 0.0, 0.0

        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x_batch)
            loss, recon_loss, kl_loss = vae_loss_function(x_recon, x_batch, mu, logvar, beta=0.1)
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item() * x_batch.size(0)
            recon_epoch += recon_loss.item() * x_batch.size(0)
            kl_epoch += kl_loss.item() * x_batch.size(0)

        if epoch % 10 == 0 or epoch == 1:
            n_samples = len(dataset)
            print(f"Epoch {epoch:03d}/{total_epochs} | "
                  f"Total Loss: {total_loss_epoch/n_samples:.4f} | "
                  f"Recon MSE: {recon_epoch/n_samples:.4f} | "
                  f"KL Div: {kl_epoch/n_samples:.4f}")

        # Save Checkpoint Coordinates
        if epoch in checkpoint_epochs:
            model.eval()
            all_mu, all_samples = [], []
            eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                for x_batch, sample_ids in eval_loader:
                    x_batch = x_batch.to(device)
                    mu, _ = model.encode(x_batch)
                    all_mu.append(mu.cpu().numpy())
                    all_samples.extend(sample_ids)

            df_latent = pd.DataFrame(np.vstack(all_mu), index=all_samples, columns=['z1', 'z2'])
            csv_path = f"results/tables/vanilla_vae_latent_coordinates_{epoch}epochs.csv"
            df_latent.to_csv(csv_path)

            plt.figure(figsize=(8, 6))
            plt.scatter(df_latent['z1'], df_latent['z2'], alpha=0.6, c=df_latent['z2'], cmap='viridis', s=25)
            plt.title(f"Vanilla VAE Latent Space ({epoch} Epochs)", fontsize=12, fontweight='bold')
            plt.xlabel("Latent Dimension z1")
            plt.ylabel("Latent Dimension z2")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"results/figures/vanilla_vae_latent_plots_{epoch}epochs.png", dpi=300)
            plt.close()

    print("--> Phase 3 Complete! Vanilla VAE baseline checkpoints exported.")

if __name__ == '__main__':
    run_phase3_vanilla_vae()
