# Data Directory Guide

Due to GitHub's single-file size limits (>100 MB), large raw and processed transcriptomic matrices (such as `breast_cancer_genes_x_samples.csv`) are omitted from Git history.

## How to Get the Dataset

### Option A: Automatic Download Script (Recommended)
Run the dataset downloader from the root directory of the repository:

```bash
python data/download_data.py
```

This script automatically fetches the harmonized TCGA and GTEx breast transcriptomics datasets directly from the official UCSC Xena AWS Toil Hub:
- `TcgaTargetGtex_rsem_gene_tpm.gz`
- `TcgaTargetGTEX_phenotype.txt.gz`

### Option B: Direct Download via GitHub Releases
If you prefer to download the preprocessed `breast_cancer_genes_x_samples.csv` file directly without running the preprocessing step:
1. Navigate to the **Releases** tab on the GitHub repository homepage.
2. Under tag `v1.0-data`, download `breast_cancer_genes_x_samples.csv`.
3. Place the downloaded CSV inside this `data/` folder.

---

## Directory Structure

```text
data/
├── README.md                          # Data documentation
├── download_data.py                   # Automatic data downloader
├── breast_cancer_genes_x_samples.csv  # Expression Matrix (58,581 Genes x 1,384 Samples) [Downloaded/Generated]
└── breast_cancer_ssgsea_scores.csv    # Derived Pathway Target Matrix (50 Hallmark Sets x 1,384 Samples)
```
