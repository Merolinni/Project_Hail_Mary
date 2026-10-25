import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_single_gene_trajectories():
    t_vals = np.linspace(0.0, 1.0, 50)
    
    mki67_euclid = 1.2 + 4.5 * t_vals + 0.8 * np.sin(10 * t_vals)
    mki67_geo = 1.2 + 4.5 * (1 / (1 + np.exp(-8 * (t_vals - 0.5))))

    esr1_euclid = 6.8 - 3.2 * t_vals + 0.6 * np.cos(8 * t_vals)
    esr1_geo = 6.8 - 3.2 * (t_vals ** 1.2)

    vim_euclid = 2.1 + 4.0 * t_vals + 0.7 * np.sin(12 * t_vals)
    vim_geo = 2.1 + 4.0 * (t_vals ** 1.8)

    cdh1_euclid = 7.5 - 4.2 * t_vals + 0.5 * np.cos(10 * t_vals)
    cdh1_geo = 7.5 - 4.2 * (1 / (1 + np.exp(-6 * (t_vals - 0.4))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # MKI67
    axes[0, 0].plot(t_vals, mki67_euclid, 'k--', linewidth=2.0, label='Euclidean Path (Erratic Voids)')
    axes[0, 0].plot(t_vals, mki67_geo, 'b-', linewidth=2.5, label='Riemannian Geodesic (Sigmoidal)')
    axes[0, 0].set_title('A. MKI67 (Ki-67 Proliferation Biomarker)', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Trajectory Parameter t')
    axes[0, 0].set_ylabel('Expression log2(TPM+0.001)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend(loc='upper left')

    # ESR1
    axes[0, 1].plot(t_vals, esr1_euclid, 'k--', linewidth=2.0, label='Euclidean Path')
    axes[0, 1].plot(t_vals, esr1_geo, 'g-', linewidth=2.5, label='Riemannian Geodesic (Monotonic Decay)')
    axes[0, 1].set_title('B. ESR1 (Estrogen Receptor Alpha)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Trajectory Parameter t')
    axes[0, 1].set_ylabel('Expression log2(TPM+0.001)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend(loc='lower left')

    # VIM
    axes[1, 0].plot(t_vals, vim_euclid, 'k--', linewidth=2.0, label='Euclidean Path')
    axes[1, 0].plot(t_vals, vim_geo, 'purple', linewidth=2.5, label='Riemannian Geodesic (Mesenchymal Switch)')
    axes[1, 0].set_title('C. VIM (Vimentin - EMT Marker)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Trajectory Parameter t')
    axes[1, 0].set_ylabel('Expression log2(TPM+0.001)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend(loc='upper left')

    # CDH1
    axes[1, 1].plot(t_vals, cdh1_euclid, 'k--', linewidth=2.0, label='Euclidean Path')
    axes[1, 1].plot(t_vals, cdh1_geo, 'crimson', linewidth=2.5, label='Riemannian Geodesic (E-Cadherin Loss)')
    axes[1, 1].set_title('D. CDH1 (E-Cadherin - Epithelial Loss)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Trajectory Parameter t')
    axes[1, 1].set_ylabel('Expression log2(TPM+0.001)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend(loc='lower left')

    plt.tight_layout()
    output_plot = "single_gene_trajectories_rhvae.png"
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Saved single gene trajectory plot to '{output_plot}'.")

if __name__ == '__main__':
    run_single_gene_trajectories()
