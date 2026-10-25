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
# 2. Position-Dependent Metric Tensor Layer G(z)
# ---------------------------------------------------------
class MetricTensor(nn.Module):
    def __init__(self, latent_dim=2, num_centroids=15, temperature=0.8, lambda_reg=1e-2):
        super(MetricTensor, self).__init__()
        self.latent_dim = latent_dim
        self.num_centroids = num_centroids
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        
        self.centroids = nn.Parameter(torch.randn(num_centroids, latent_dim))
        self.L_params = nn.Parameter(torch.randn(num_centroids, latent_dim, latent_dim))

    def forward(self, z):
        batch_size = z.size(0)
        dists = torch.cdist(z, self.centroids, p=2) ** 2
        weights = torch.exp(-dists / (self.temperature ** 2))
        
        L = torch.tril(self.L_params)
        M_i = torch.matmul(L, L.transpose(-1, -2))
        M_z = torch.einsum('bc,cij->bij', weights, M_i)
        
        identity = torch.eye(self.latent_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        M_z = M_z + self.lambda_reg * identity
        G_z = torch.linalg.inv(M_z)
        return G_z, M_z

# ---------------------------------------------------------
# 3. RHVAE Architecture
# ---------------------------------------------------------
class RHVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=2, num_centroids=15):
        super(RHVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.metric = MetricTensor(latent_dim=latent_dim, num_centroids=num_centroids)
        
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
        h = self.encoder(x)
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
        G_z, M_z = self.metric(z)
        return x_recon, mu, logvar, z, G_z, M_z

def rhvae_loss_function(x_recon, x, mu, logvar, z, G_z, M_z, beta=0.1):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    log_det_G = torch.logdet(G_z)
    metric_reg = -0.5 * torch.mean(log_det_G)
    total_loss = recon_loss + beta * (kl_div + 0.1 * metric_reg)
    return total_loss, recon_loss, kl_div, metric_reg

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
def run_phase4_rhvae(expr_file="data/breast_cancer_genes_x_samples.csv",
                     checkpoint_epochs=[50, 100, 200, 300],
                     num_centroids=15):
    """
    Phase 4: RHVAE Model Training & Metric Tensor Learning
    ======================================================
    - Trains Riemannian Hamiltonian VAE with dynamic metric tensor G(z).
    - Saves PyTorch weights, latent coordinates, and centroid scatter plots.
    """
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Expression matrix '{expr_file}' not found.")

    print("--> Phase 4: Loading Dataset for PyTorch RHVAE Training...")
    dataset = GeneExpressionDataset(expr_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = dataset.data.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training RHVAE on device: {device} with C={num_centroids} centroids.")

    model = RHVAE(input_dim=input_dim, hidden_dim=512, latent_dim=2, num_centroids=num_centroids).to(device)
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
            x_recon, mu, logvar, z, G_z, M_z = model(x_batch)
            loss, recon_loss, kl_loss, metric_reg = rhvae_loss_function(x_recon, x_batch, mu, logvar, z, G_z, M_z, beta=0.1)
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
            df_latent.to_csv(f"results/tables/rhvae_latent_coordinates_{epoch}epochs.csv")

            centroids_np = model.metric.centroids.detach().cpu().numpy()
            plt.figure(figsize=(9, 7))
            plt.scatter(df_latent['z1'], df_latent['z2'], alpha=0.6, c=df_latent['z2'], cmap='plasma', s=25)
            plt.scatter(centroids_np[:, 0], centroids_np[:, 1], color='black', marker='X', s=120, label='Metric Centroids $c_i$')
            plt.title(f"RHVAE Latent Space & Centroids $c_i$ ({epoch} Epochs)", fontsize=12, fontweight='bold')
            plt.xlabel("Latent Dimension z1")
            plt.ylabel("Latent Dimension z2")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/figures/rhvae_latent_plots_{epoch}epochs.png", dpi=300)
            plt.close()

    torch.save(model.state_dict(), "results/tables/rhvae_model.pt")
    print("--> Phase 4 Complete! RHVAE model and checkpoint artifacts exported.")

if __name__ == '__main__':
    run_phase4_rhvae()
