import pandas as pd
import numpy as np
import os
import torch
from .config import FILES


def load_data():
    print(">>> Loading PAN-CANCER Data (High Accuracy Mode)...")
    dfs = {}
    required_keys = ['exp', 'cnv', 'mut', 'dep', 'model']
    for key in required_keys:
        if os.path.exists(FILES[key]):
            print(f"   -> Reading {key}...")
            if key == 'model':
                dfs[key] = pd.read_csv(FILES[key])
            else:
                dfs[key] = pd.read_csv(FILES[key], index_col=0)
        else:
            print(f"❌ Missing {FILES[key]}")
            return None, None, None, None

    meta = dfs['model']
    meta['ModelID'] = meta['ModelID'].astype(str).str.strip()
    all_ids = meta['ModelID'].tolist()

    common_samples = set(all_ids)
    for key in ['exp', 'cnv', 'mut', 'dep']:
        if key in dfs:
            dfs[key].index = dfs[key].index.astype(str).str.strip()
          
            common_samples = common_samples.intersection(set(dfs[key].index))

    common_samples = sorted(list(common_samples))
    print(f"   -> Total Samples: {len(common_samples)}")

    available_genes = set(dfs['exp'].columns)
    for key in ['cnv', 'mut', 'dep']:
        available_genes = available_genes.intersection(set(dfs[key].columns))

    final_genes = sorted(list(available_genes))
    print(f"   -> Final Gene Set: {len(final_genes)} Genes (Full Coverage)")

    try:
        def get_aligned(key, fill_val=np.nan):
            df = dfs[key].loc[common_samples]
            return df.reindex(columns=final_genes, fill_value=fill_val).values.astype(np.float32)

        exp = get_aligned('exp', fill_val=np.nan)
        cnv = get_aligned('cnv', fill_val=np.nan)
        mut = get_aligned('mut', fill_val=np.nan)
        X_np = np.stack([exp, cnv, mut], axis=-1)
        Y_np = get_aligned('dep', fill_val=np.nan)

        return X_np, Y_np, final_genes, common_samples

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None, None


def build_adjacency_matrix(genes, ppi_path='data/raw/ppi.csv'):
    print("   -> Constructing PPI Adjacency Matrix...")
    num_genes = len(genes)

    clean_genes = [str(g).split(' (')[0].strip() if ' (' in str(g) else str(g).strip() for g in genes]
    gene_to_idx = {gene: i for i, gene in enumerate(clean_genes)}

    adj_matrix = np.zeros((num_genes, num_genes), dtype=np.float32)
    np.fill_diagonal(adj_matrix, 1.0)

    if os.path.exists(ppi_path):
        ppi_df = pd.read_csv(ppi_path)
        edges_added = 0
        for _, row in ppi_df.iterrows():
            g1, g2 = str(row['Gene1']).strip(), str(row['Gene2']).strip()
            if g1 in gene_to_idx and g2 in gene_to_idx:
                i, j = gene_to_idx[g1], gene_to_idx[g2]
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
                edges_added += 1
        print(f"   -> Mapped {edges_added} physical protein interactions into the matrix.")
    else:
        print(f"   ⚠️ Warning: PPI file not found at {ppi_path}. Using identity matrix.")

    return torch.FloatTensor(adj_matrix)
