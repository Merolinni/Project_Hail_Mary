import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import gseapy as gp
except ImportError:
    gp = None

def run_phase2_ssgsea(expr_file="data/breast_cancer_genes_x_samples.csv",
                      gene_sets="MSigDB_Hallmark_2020",
                      output_ssgsea="data/breast_cancer_ssgsea_scores.csv",
                      output_boxplot="results/figures/breast_cancer_ssgsea_boxplots.png"):
    """
    Phase 2: single-sample GSEA (ssGSEA) Continuous Phenotype Pipeline
    ===================================================================
    - Computes rank-normalized enrichment scores (NES) across 50 Hallmark Gene Sets.
    - Saves output matrix (50 Pathways x 1,384 Samples) to CSV.
    - Creates comparative boxplots across tissue subtypes for key pathways.
    """
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Expression matrix '{expr_file}' not found.")

    print("--> Phase 2: Running ssGSEA Engine across MSigDB Hallmark Gene Sets...")
    df_expr = pd.read_csv(expr_file, index_col=0)

    if gp is not None:
        ssgsea_res = gp.ssgsea(
            data=df_expr,
            gene_sets=gene_sets,
            sample_norm_method='rank',
            min_size=10,
            max_size=1000,
            permutation_num=0,
            outdir=None,
            no_plot=True
        )
        df_ssgsea = ssgsea_res.res2d.pivot(index='Term', columns='Name', values='NES')
        os.makedirs(os.path.dirname(output_ssgsea), exist_ok=True)
        df_ssgsea.to_csv(output_ssgsea)
        print(f"--> ssGSEA computation complete. Saved scores to '{output_ssgsea}'.")
    elif os.path.exists(output_ssgsea):
        print(f"--> GSEAPY not installed. Loading pre-computed ssGSEA matrix from '{output_ssgsea}'...")
        df_ssgsea = pd.read_csv(output_ssgsea, index_col=0)
    else:
        raise RuntimeError("GSEAPY is required to compute ssGSEA from scratch. Please install gseapy.")

    # Visualization: Selected Hallmark Pathway Boxplots
    target_pathways = [
        t for t in df_ssgsea.index 
        if any(kw in t.upper() for kw in ['E2F', 'G2M', 'HYPOXIA', 'EPITHELIAL', 'DNA_REPAIR', 'GLYCOLYSIS'])
    ][:4]

    if target_pathways:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, term in enumerate(target_pathways):
            scores = df_ssgsea.loc[term]
            sns.boxplot(y=scores.values, ax=axes[idx], color='#1f77b4', width=0.4)
            sns.stripplot(y=scores.values, ax=axes[idx], color='black', alpha=0.3, jitter=0.2, size=3)
            axes[idx].set_title(f"Hallmark: {term}", fontsize=11, fontweight='bold')
            axes[idx].set_ylabel("ssGSEA Score (NES)")
            axes[idx].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_boxplot), exist_ok=True)
        plt.savefig(output_boxplot, dpi=300)
        plt.close()
        print(f"--> Phase 2 Complete! Boxplot summary saved to '{output_boxplot}'.")

if __name__ == '__main__':
    run_phase2_ssgsea()
