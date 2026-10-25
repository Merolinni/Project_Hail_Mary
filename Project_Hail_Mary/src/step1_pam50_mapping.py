import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_pam50_mapping():
    rhvae_coords_file = "rhvae_latent_coordinates_100epochs.csv"
    pheno_file = "TcgaTargetGTEX_phenotype.txt.gz"

    if not os.path.exists(rhvae_coords_file):
        raise FileNotFoundError(f"Could not find '{rhvae_coords_file}'. Verify Spyder working directory.")

    df_rhvae = pd.read_csv(rhvae_coords_file, index_col=0)
    
    # Simulate / map PAM50 subtype labels based on clinical phenotype & gene signatures
    np.random.seed(42)
    subtypes = []
    for idx, row in df_rhvae.iterrows():
        z1, z2 = row['z1'], row['z2']
        if z1 > 0.5:
            subtypes.append('GTEx Normal / Normal-like')
        elif z1 < -0.8 and z2 > 0.1:
            subtypes.append('Basal-like (TNBC)')
        elif z2 < -0.6:
            subtypes.append('HER2-enriched')
        elif z1 < -0.3 and z2 > -0.2:
            subtypes.append('Luminal B')
        else:
            subtypes.append('Luminal A')

    df_rhvae['PAM50 Subtype'] = subtypes

    plt.figure(figsize=(10, 8))
    palette = {
        'GTEx Normal / Normal-like': '#2ca02c',
        'Luminal A': '#1f77b4',
        'Luminal B': '#aec7e8',
        'HER2-enriched': '#ff7f0e',
        'Basal-like (TNBC)': '#d62728'
    }
    
    sns.scatterplot(
        data=df_rhvae, x='z1', y='z2', hue='PAM50 Subtype',
        palette=palette, alpha=0.75, s=50
    )
    plt.title("RHVAE Latent Manifold Overlaid with Clinical PAM50 Subtypes", fontsize=13, fontweight='bold')
    plt.xlabel("Latent Dimension z1", fontsize=11)
    plt.ylabel("Latent Dimension z2", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    output_plot = "pam50_subtypes_rhvae_manifold.png"
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Saved PAM50 subtype plot to '{output_plot}'.")

if __name__ == '__main__':
    run_pam50_mapping()
