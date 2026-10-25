import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_phase1_pca(expr_file="data/breast_cancer_genes_x_samples.csv",
                   ssgsea_file="data/breast_cancer_ssgsea_scores.csv",
                   pheno_file="data/TcgaTargetGTEX_phenotype.txt.gz",
                   output_plot="results/figures/pca_continuous_pathway_gradients.png"):
    """
    Phase 1: Preprocessing & Continuous PCA Gradient Visualization
    ==============================================================
    - Standardizes gene expression (58,581 genes x 1,384 samples).
    - Computes 2D PCA projection (PC1 vs PC2).
    - Plots 4-panel grid overlaying discrete tissue labels and continuous 
      ssGSEA pathway enrichment scores (E2F Targets, Hypoxia, EMT).
    """
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Expression matrix '{expr_file}' not found. Please verify dataset location.")

    print("--> Phase 1: Loading Gene Expression Matrix and Computing PCA...")
    df_expr = pd.read_csv(expr_file, index_col=0)  # Shape: (Genes x Samples)
    X = df_expr.values.T                           # Shape: (Samples x Genes)

    # Standardization across genes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2D PCA Projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var1, var2 = pca.explained_variance_ratio_ * 100

    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1]}, index=df_expr.columns)

    # Map Discrete Tissue Annotations
    if os.path.exists(pheno_file):
        df_pheno = pd.read_csv(pheno_file, sep='\t', compression='gzip')
        sample_type_map = {}
        for s in df_pca.index:
            match = df_pheno[df_pheno['sample'] == s]
            if not match.empty:
                study = match['_study'].values[0] if len(match['_study'].values) > 0 else ''
                stype = match['_sample_type'].values[0] if len(match['_sample_type'].values) > 0 else ''
                if study == 'GTEX': sample_type_map[s] = 'GTEx Normal'
                elif stype == 'Solid Tissue Normal': sample_type_map[s] = 'TCGA Adjacent Normal'
                elif stype == 'Primary Tumor': sample_type_map[s] = 'TCGA Primary Tumor'
        df_pca['Tissue Type'] = [sample_type_map.get(s, 'Unknown') for s in df_pca.index]
    else:
        df_pca['Tissue Type'] = 'Sample'

    # Map Continuous ssGSEA Pathway Targets if available
    if os.path.exists(ssgsea_file):
        df_ssgsea = pd.read_csv(ssgsea_file, index_col=0)
        pathway_map = {
            'E2F Proliferation': [t for t in df_ssgsea.index if 'E2F' in t.upper()],
            'Hypoxia': [t for t in df_ssgsea.index if 'HYPOXIA' in t.upper()],
            'EMT (Invasion)': [t for t in df_ssgsea.index if 'EPITHELIAL' in t.upper()]
        }
        for label, terms in pathway_map.items():
            if terms:
                term = terms[0]
                df_pca[label] = [df_ssgsea.loc[term, s] if s in df_ssgsea.columns else np.nan for s in df_pca.index]

    # Generate 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # Panel A: Discrete Tissue Categories
    sns.scatterplot(
        data=df_pca, x='PC1', y='PC2', hue='Tissue Type', ax=axes[0, 0],
        palette={'GTEx Normal': '#2ca02c', 'TCGA Adjacent Normal': '#ff7f0e', 'TCGA Primary Tumor': '#d62728', 'Sample': '#1f77b4'},
        alpha=0.7, s=40
    )
    axes[0, 0].set_title(f"A. PCA - Discrete Tissue Categories\n(PC1: {var1:.1f}% | PC2: {var2:.1f}%)", fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Panels B-D: Continuous Pathway Scores
    panels = [
        (axes[0, 1], 'E2F Proliferation', 'B. PCA - E2F Proliferation Score (NES)', 'magma'),
        (axes[1, 0], 'Hypoxia', 'C. PCA - Hypoxia Stress Score (NES)', 'viridis'),
        (axes[1, 1], 'EMT (Invasion)', 'D. PCA - EMT Invasion Score (NES)', 'plasma')
    ]

    for ax, col, title, cmap in panels:
        if col in df_pca.columns:
            sc = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca[col], cmap=cmap, alpha=0.7, s=35)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(f"{col} NES", fontsize=9)
        else:
            ax.scatter(df_pca['PC1'], df_pca['PC2'], color='gray', alpha=0.5, s=20)
        ax.set_title(f"{title}\n(PC1: {var1:.1f}% | PC2: {var2:.1f}%)", fontsize=11, fontweight='bold')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"--> Phase 1 Complete! Plot saved to '{output_plot}'.")

if __name__ == '__main__':
    run_phase1_pca()
