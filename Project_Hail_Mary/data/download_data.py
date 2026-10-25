import os
import urllib.request

def download_and_setup_data():
    """
    Downloads raw transcriptomic data from the official UCSC Xena AWS Toil Hub
    and prepares the dataset for the RHVAE pipeline.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    tpm_url = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_tpm.gz"
    pheno_url = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz"
    
    tpm_file = os.path.join(data_dir, "TcgaTargetGtex_rsem_gene_tpm.gz")
    pheno_file = os.path.join(data_dir, "TcgaTargetGTEX_phenotype.txt.gz")
    
    if not os.path.exists(tpm_file):
        print("Downloading TPM expression data from UCSC Xena (AWS Toil Hub)...")
        urllib.request.urlretrieve(tpm_url, tpm_file)
        print("✓ TPM expression data downloaded successfully.")
    else:
        print(f"✓ '{tpm_file}' already exists.")

    if not os.path.exists(pheno_file):
        print("Downloading Phenotype annotation data from UCSC Xena...")
        urllib.request.urlretrieve(pheno_url, pheno_file)
        print("✓ Phenotype annotation data downloaded successfully.")
    else:
        print(f"✓ '{pheno_file}' already exists.")

    print("\nDataset preparation script ready!")
    print("Run 'python src/phase1_pca_analysis.py' next to filter breast tissue and generate 'breast_cancer_genes_x_samples.csv'.")

if __name__ == "__main__":
    download_and_setup_data()
