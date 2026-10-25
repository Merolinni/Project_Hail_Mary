import os
import pandas as pd
import numpy as np

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
# Dataset Loader
# ---------------------------------------------------------
class GeneExpressionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, index_col=0)
        self.sample_names = list(df.columns)
        X = df.values.T
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.data = torch.tensor(X_scaled, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.sample_names[idx]

# ---------------------------------------------------------
# Dynamic Vanilla VAE & RHVAE for Sweeps
# ---------------------------------------------------------
class DynamicVanillaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=2):
        super(DynamicVanillaVAE, self).__init__()
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
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar

class DynamicRHVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=2, num_centroids=15):
        super(DynamicRHVAE, self).__init__()
        self.latent_dim = latent_dim
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
        
        self.centroids = nn.Parameter(torch.randn(num_centroids, latent_dim))
        self.L_params = nn.Parameter(torch.randn(num_centroids, latent_dim, latent_dim))
        self.temperature = 0.8
        self.lambda_reg = 1e-2

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def get_metric(self, z):
        batch_size = z.size(0)
        dists = torch.cdist(z, self.centroids, p=2) ** 2
        weights = torch.exp(-dists / (self.temperature ** 2))
        L = torch.tril(self.L_params)
        M_i = torch.matmul(L, L.transpose(-1, -2))
        M_z = torch.einsum('bc,cij->bij', weights, M_i)
        M_z = M_z + self.lambda_reg * torch.eye(self.latent_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.linalg.inv(M_z), M_z

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        G_z, M_z = self.get_metric(z)
        return self.decoder(z), mu, logvar, z, G_z, M_z

# ---------------------------------------------------------
# Sweeps Implementation
# ---------------------------------------------------------
def run_hyperparameter_sweeps(expr_file="data/breast_cancer_genes_x_samples.csv",
                               ssgsea_file="data/breast_cancer_ssgsea_scores.csv"):
    """
    Phase 4 Sweeps: Latent Dimension (d in [2, 5, 10, 20, 50])
                   Metric Centroids (C in [5, 10, 15, 20, 30, 50])
    """
    if not os.path.exists(expr_file) or not os.path.exists(ssgsea_file):
        raise FileNotFoundError("Expression matrix or ssGSEA file missing.")

    dataset = GeneExpressionDataset(expr_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    df_ssgsea = pd.read_csv(ssgsea_file, index_col=0)
    input_dim = dataset.data.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    target_keywords = ['E2F', 'G2M', 'HYPOXIA', 'EPITHELIAL', 'DNA_REPAIR', 'GLYCOLYSIS']
    matched_pathways = [t for t in df_ssgsea.index if any(kw in t.upper().replace(' ', '_') for kw in target_keywords)]

    os.makedirs("results/tables", exist_ok=True)

    # 1. Latent Dimension Sweep (d in [2, 5, 10, 20, 50])
    print("\n--> Running Latent Dimension Capacity Sweep (d in [2, 5, 10, 20, 50])...")
    dim_results = []
    for d in [2, 5, 10, 20, 50]:
        model = DynamicVanillaVAE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(100):
            model.train()
            for x_batch, _ in dataloader:
                x_batch = x_batch.to(device)
                optimizer.zero_grad()
                x_recon, mu, logvar = model(x_batch)
                loss = nn.functional.mse_loss(x_recon, x_batch) + 0.1 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                loss.backward()
                optimizer.step()

        model.eval()
        eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_mu, all_recons, all_samples = [], [], []
        with torch.no_grad():
            for x_batch, sample_ids in eval_loader:
                x_batch_dev = x_batch.to(device)
                mu = model.fc_mu(model.encoder(x_batch_dev))
                x_recon = model.decoder(mu)
                all_mu.append(mu.cpu().numpy())
                all_recons.append(x_recon.cpu().numpy())
                all_samples.extend(sample_ids)

        X_latent = np.vstack(all_mu)
        X_recon_full = np.vstack(all_recons)
        mse_val = mean_squared_error(dataset.data.numpy(), X_recon_full)

        df_latent = pd.DataFrame(X_latent, index=all_samples)
        common_samples = df_latent.index.intersection(df_ssgsea.columns)
        df_latent_sub, df_ssgsea_sub = df_latent.loc[common_samples], df_ssgsea[common_samples]

        r2_list = [r2_score(df_ssgsea_sub.loc[p].values, cross_val_predict(Ridge(alpha=1.0), df_latent_sub.values, df_ssgsea_sub.loc[p].values, cv=kf)) for p in matched_pathways]
        mean_r2 = np.mean(r2_list)
        print(f"  * d={d:2d} | Recon MSE: {mse_val:.4f} | Mean Pathway R^2: {mean_r2:.4f}")
        dim_results.append({'Dimension (d)': d, 'Reconstruction MSE': mse_val, 'Mean Pathway R2': mean_r2})

    df_dim = pd.DataFrame(dim_results)
    df_dim.to_csv("results/tables/latent_dimension_sweep_results.csv", index=False)

    # 2. Metric Centroid Sweep (C in [5, 10, 15, 20, 30, 50])
    print("\n--> Running RHVAE Metric Centroid Sweep (C in [5, 10, 15, 20, 30, 50])...")
    centroid_results = []
    for C in [5, 10, 15, 20, 30, 50]:
        model = DynamicRHVAE(input_dim=input_dim, hidden_dim=512, latent_dim=2, num_centroids=C).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(100):
            model.train()
            for x_batch, _ in dataloader:
                x_batch = x_batch.to(device)
                optimizer.zero_grad()
                x_recon, mu, logvar, z, G_z, M_z = model(x_batch)
                recon_loss = nn.functional.mse_loss(x_recon, x_batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                metric_reg = -0.5 * torch.mean(torch.logdet(G_z))
                loss = recon_loss + 0.1 * (kl_loss + 0.1 * metric_reg)
                loss.backward()
                optimizer.step()

        model.eval()
        eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_mu, all_recons, all_samples = [], [], []
        with torch.no_grad():
            for x_batch, sample_ids in eval_loader:
                x_batch_dev = x_batch.to(device)
                mu = model.fc_mu(model.encoder(x_batch_dev))
                x_recon = model.decoder(mu)
                all_mu.append(mu.cpu().numpy())
                all_recons.append(x_recon.cpu().numpy())
                all_samples.extend(sample_ids)

        X_latent = np.vstack(all_mu)
        X_recon_full = np.vstack(all_recons)
        mse_val = mean_squared_error(dataset.data.numpy(), X_recon_full)

        df_latent = pd.DataFrame(X_latent, index=all_samples)
        common_samples = df_latent.index.intersection(df_ssgsea.columns)
        df_latent_sub, df_ssgsea_sub = df_latent.loc[common_samples], df_ssgsea[common_samples]

        r2_list = [r2_score(df_ssgsea_sub.loc[p].values, cross_val_predict(Ridge(alpha=1.0), df_latent_sub.values, df_ssgsea_sub.loc[p].values, cv=kf)) for p in matched_pathways]
        mean_r2 = np.mean(r2_list)
        print(f"  * C={C:2d} | Recon MSE: {mse_val:.4f} | Mean Pathway R^2: {mean_r2:.4f}")
        centroid_results.append({'Centroids (C)': C, 'Reconstruction MSE': mse_val, 'Mean Pathway R2': mean_r2})

    df_cen = pd.DataFrame(centroid_results)
    df_cen.to_csv("results/tables/rhvae_centroid_sweep_results.csv", index=False)
    print("\n--> Phase 4 Sweeps Complete! Saved sweep matrices to 'results/tables/'.")

if __name__ == '__main__':
    run_hyperparameter_sweeps()
