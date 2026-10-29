import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

"""
Direction 1: Foundation Model Feature Extraction & Tokenization
(Geneformer / scGPT Transformer Embedding Adapter)

This module implements rank-based tokenization and transformer embedding
extraction for high-dimensional transcriptomic profiles, adapting raw gene expression 
(58,581 genes) into context-aware 512-dimensional foundation embeddings for the RHVAE encoder.
"""

class TranscriptomicRankTokenizer:
    def __init__(self, max_seq_len=2048):
        self.max_seq_len = max_seq_len

    def tokenize_batch(self, expression_matrix):
        """
        Sorts non-zero genes per sample by expression rank and converts to token IDs.
        expression_matrix: Tensor of shape (batch_size, num_genes)
        Returns: token_ids (batch_size, max_seq_len), attention_mask (batch_size, max_seq_len)
        """
        batch_size, num_genes = expression_matrix.shape
        # Sort indices in descending order of expression value
        sorted_indices = torch.argsort(expression_matrix, dim=1, descending=True)
        
        # Truncate or pad to max_seq_len
        token_ids = sorted_indices[:, :self.max_seq_len] + 1  # 1-indexed (0 reserved for padding)
        
        # Create attention mask (1 for valid token, 0 for padding)
        non_zero_counts = (expression_matrix > 0).sum(dim=1)
        attention_mask = torch.zeros_like(token_ids, dtype=torch.float32)
        
        for i in range(batch_size):
            valid_len = min(non_zero_counts[i].item(), self.max_seq_len)
            attention_mask[i, :valid_len] = 1.0
            
        return token_ids, attention_mask


class FoundationEmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size=60000, embed_dim=512, n_heads=8, n_layers=4):
        super().__init__()
        self.gene_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(2048, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=2048, 
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids, attention_mask):
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        
        # Gene rank + position embeddings
        x = self.gene_embeddings(token_ids) + self.position_embeddings(positions)
        
        # Apply transformer attention
        padding_mask = (attention_mask == 0)
        hidden_states = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Mean-pooled representation across valid tokens
        mask_expanded = attention_mask.unsqueeze(-1)
        pooled_embedding = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        return self.pooling_layer(pooled_embedding)


def run_foundation_extraction_demo():
    print("Executing Direction 1: Transcriptomic Foundation Embedding Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {device}")
    
    # Simulate batch of 16 samples x 58,581 genes
    mock_expression = torch.randn(16, 58581).abs()
    
    tokenizer = TranscriptomicRankTokenizer(max_seq_len=2048)
    token_ids, attention_mask = tokenizer.tokenize_batch(mock_expression)
    
    model = FoundationEmbeddingEncoder(vocab_size=60000, embed_dim=512).to(device)
    token_ids, attention_mask = token_ids.to(device), attention_mask.to(device)
    
    with torch.no_grad():
        embeddings = model(token_ids, attention_mask)
        
    print(f"Input Matrix Shape: {mock_expression.shape}")
    print(f"Tokenized Sequence Shape: {token_ids.shape}")
    print(f"Extracted Foundation Embeddings Shape: {embeddings.shape}")
    print("Foundation Embedding Extraction Completed Successfully!\n")

if __name__ == "__main__":
    run_foundation_extraction_demo()
