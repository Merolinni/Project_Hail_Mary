import torch
import torch.nn as nn
import pandas as pd

"""
Direction 5: Counterfactual In-Silico Virtual Drug Screening & Causal Knockouts

Performs batched virtual gene knockouts (setting x_g = 0) and computes backpropagation 
reversion gradients to identify gene targets that steer malignant cells back to healthy manifold basins.
"""

class CausalReversionScreeningEngine:
    def __init__(self, encoder, decoder, target_healthy_z):
        self.encoder = encoder
        self.decoder = decoder
        self.healthy_z = target_healthy_z # Healthy GTEx baseline in latent space (d,)

    def compute_knockout_reversion_score(self, malignant_x, gene_indices):
        """
        malignant_x: Tensor of shape (1, num_genes)
        gene_indices: List of gene index targets to evaluate
        Returns: DataFrame of gene knockout efficacy scores
        """
        device = malignant_x.device
        results = []
        
        # Baseline distance before knockout
        with torch.no_grad():
            z_base = self.encoder(malignant_x)
            dist_base = torch.norm(z_base - self.healthy_z).item()
            
        for g_idx in gene_indices:
            x_ko = malignant_x.clone()
            x_ko[0, g_idx] = 0.0 # Virtual knockout (CRISPR / Inhibition)
            
            with torch.no_grad():
                z_ko = self.encoder(x_ko)
                dist_ko = torch.norm(z_ko - self.healthy_z).item()
                
            reversion_delta = dist_base - dist_ko # Positive delta = moves closer to healthy
            results.append({
                "Gene_Index": g_idx,
                "Baseline_Latent_Distance": dist_base,
                "Knockout_Latent_Distance": dist_ko,
                "Healthy_Reversion_Delta": reversion_delta
            })
            
        return pd.DataFrame(results).sort_values("Healthy_Reversion_Delta", ascending=False)


def run_virtual_screening_demo():
    print("Executing Direction 5: Counterfactual In-Silico Drug Screening & Causal Gene Knockouts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {device}")
    
    num_genes = 1000
    latent_dim = 20
    
    # Simple linear encoder proxy for demo
    encoder = nn.Sequential(nn.Linear(num_genes, 128), nn.ReLU(), nn.Linear(128, latent_dim)).to(device)
    decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, num_genes)).to(device)
    
    healthy_z = torch.randn(1, latent_dim, device=device) * 0.1
    malignant_x = torch.randn(1, num_genes, device=device).abs() + 1.0
    
    engine = CausalReversionScreeningEngine(encoder, decoder, healthy_z)
    
    # Screen first 20 gene targets
    target_genes = list(range(20))
    df_results = engine.compute_knockout_reversion_score(malignant_x, target_genes)
    
    print("Top 5 Virtual Knockout Candidates for Malignant Reversion:")
    print(df_results.head(5).to_string(index=False))
    print("In-Silico Virtual Drug Screening Completed Successfully!\n")

if __name__ == "__main__":
    run_virtual_screening_demo()
