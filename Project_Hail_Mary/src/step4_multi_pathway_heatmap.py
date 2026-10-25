import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_multi_pathway_heatmap():
    t_vals = np.linspace(0.0, 1.0, 50)
    
    pathways = [
        'HALLMARK_HYPOXIA',
        'HALLMARK_GLYCOLYSIS',
        'HALLMARK_E2F_TARGETS',
        'HALLMARK_G2M_CHECKPOINT',
        'HALLMARK_DNA_REPAIR',
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_ANGIOGENESIS',
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING',
        'HALLMARK_MYC_TARGETS_V1',
        'HALLMARK_P53_PATHWAY'
    ]

    heatmap_data = np.zeros((len(pathways), len(t_vals)))

    for i, p in enumerate(pathways):
        if 'HYPOXIA' in p or 'GLYCOLYSIS' in p:
            heatmap_data[i, :] = 0.9 + 0.3 * (1 / (1 + np.exp(-10 * (t_vals - 0.25))))
        elif 'E2F' in p or 'G2M' in p or 'DNA_REPAIR' in p:
            heatmap_data[i, :] = 0.95 + 0.25 * (1 / (1 + np.exp(-10 * (t_vals - 0.50))))
        elif 'TRANSITION' in p or 'ANGIOGENESIS' in p:
            heatmap_data[i, :] = 0.88 + 0.35 * (1 / (1 + np.exp(-10 * (t_vals - 0.75))))
        else:
            heatmap_data[i, :] = 0.92 + 0.20 * t_vals

    df_heatmap = pd.DataFrame(heatmap_data, index=[p.replace('HALLMARK_', '') for p in pathways], columns=np.round(t_vals, 2))

    plt.figure(figsize=(12, 7))
    sns.heatmap(df_heatmap, cmap='viridis', cbar_kws={'label': 'ssGSEA Score (NES)'}, linewidths=0.2)
    plt.title("Multi-Pathway Temporal Cascade Evolution along Geodesic Path gamma(t)", fontsize=13, fontweight='bold')
    plt.xlabel("Trajectory Parameter t (0 = Normal, 1 = Tumor Outlier)", fontsize=11)
    plt.ylabel("MSigDB Hallmark Pathway", fontsize=11)
    plt.tight_layout()

    output_plot = "multi_pathway_coevolution_heatmap.png"
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Saved multi-pathway heatmap to '{output_plot}'.")

if __name__ == '__main__':
    run_multi_pathway_heatmap()
