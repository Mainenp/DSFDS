import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys
import random
import copy
import json
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dataset import load_data, build_adjacency_matrix
from src.data_utils import DataSplitter, OmicsPreprocessor
from src.model import PPI_MOGAT
from src.config import FILES


def generate_or_load_kge(gene_ids, kg_raw_file="data/raw/raw_kg.csv",
                         kge_save_path="results/models/synlethdb_kge_256d.csv", kge_dim=256):
    kge_dict = {}
    if os.path.exists(kge_save_path):
        print(f"\n>>> [KGE 模块] 加载预训练图谱向量: {kge_save_path}")
        kge_df = pd.read_csv(kge_save_path, index_col=0)
        for gid in gene_ids:
            kge_dict[gid] = kge_df.loc[gid].values.astype(np.float32) if gid in kge_df.index else np.zeros(kge_dim,
                                                                                                           dtype=np.float32)
        return kge_dict, kge_dim

    print(f"\n>>> [KGE 模块] 提取 256维 语义特征...")
    if not os.path.exists(kg_raw_file): return {gid: np.zeros(kge_dim, dtype=np.float32) for gid in gene_ids}, kge_dim

    cols = ['x_id', 'y_id', 'relation', 'display_relation']
    edges_df = pd.read_csv(kg_raw_file, usecols=cols).dropna(subset=['x_id', 'y_id'])

    # 剔除所有正向(SL/SR)和负向(NONSL)答案
    edges_df['relation'] = edges_df['relation'].astype(str).str.upper()
    edges_df['display_relation'] = edges_df['display_relation'].astype(str).str.upper()

    leakage_relations = ['SL', 'SR', 'NONSL']
    leakage_keywords = ['SL', 'SR', 'NONSL', 'SYNTHETIC LETHAL', 'SYNTHETIC RESCUE', 'NON-SYNTHETIC LETHAL']

    edges_df = edges_df[~edges_df['relation'].isin(leakage_relations)]
    mask = edges_df['display_relation'].apply(lambda x: any(kw in x for kw in leakage_keywords))
    edges_df = edges_df[~mask]

    adj_list = defaultdict(list)
    x_vals = edges_df['x_id'].astype(str).str.strip().values
    y_vals = edges_df['y_id'].astype(str).str.strip().values
    for u, v in zip(x_vals, y_vals):
        adj_list[u].append(v)
        adj_list[v].append(u)

    nodes = list(adj_list.keys())
    walks = []
    for _ in range(15):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            curr_node = node
            for _ in range(20):
                neighbors = adj_list.get(curr_node)
                if not neighbors: break
                curr_node = random.choice(neighbors)
                walk.append(curr_node)
            walks.append(walk)

    cpu_cores = os.cpu_count() or 4
    n2v_model = Word2Vec(walks, vector_size=kge_dim, window=5, min_count=1, sg=1, workers=cpu_cores)

    extracted_data = {}
    for gid in gene_ids:
        kge_dict[gid] = n2v_model.wv[gid] if gid in n2v_model.wv else np.zeros(kge_dim, dtype=np.float32)
        extracted_data[gid] = kge_dict[gid]

    os.makedirs(os.path.dirname(kge_save_path), exist_ok=True)
    pd.DataFrame.from_dict(extracted_data, orient='index').to_csv(kge_save_path)
    return kge_dict, kge_dim



class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        probas = torch.sigmoid(inputs)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - targets)
        loss = focal_weight * alpha_weight * bce_loss
        return loss.mean()



class TransformerCrossFusionNet(nn.Module):
    def __init__(self, gat_dim, kge_dim, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.3):
        super(TransformerCrossFusionNet, self).__init__()
        self.gat_proj = nn.Linear(gat_dim, hidden_dim)
        self.kge_proj = nn.Linear(kge_dim, hidden_dim)
        self.modal_emb = nn.Embedding(2, hidden_dim)
        self.gene_emb = nn.Embedding(2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, gat_A, kge_A, gat_B, kge_B):
        device = gat_A.device
        batch_size = gat_A.size(0)

        h_gA = F.gelu(self.gat_proj(gat_A))
        h_kA = F.gelu(self.kge_proj(kge_A))
        h_gB = F.gelu(self.gat_proj(gat_B))
        h_kB = F.gelu(self.kge_proj(kge_B))

        m_gat = self.modal_emb(torch.tensor(0, device=device))
        m_kge = self.modal_emb(torch.tensor(1, device=device))
        g_A = self.gene_emb(torch.tensor(0, device=device))
        g_B = self.gene_emb(torch.tensor(1, device=device))

        t_gA = h_gA + m_gat + g_A
        t_kA = h_kA + m_kge + g_A
        t_gB = h_gB + m_gat + g_B
        t_kB = h_kB + m_kge + g_B

        seq = torch.stack([t_gA, t_kA, t_gB, t_kB], dim=1)
        out_seq = self.transformer(seq)
        out_transformer = out_seq.reshape(batch_size, -1)

        mult_gat = h_gA * h_gB
        diff_gat = torch.abs(h_gA - h_gB)
        mult_kge = h_kA * h_kB
        diff_kge = torch.abs(h_kA - h_kB)
        out_geometric = torch.cat([mult_gat, diff_gat, mult_kge, diff_kge], dim=-1)

        combined = torch.cat([out_transformer, out_geometric], dim=-1)
        logits = self.mlp(combined)

        return logits.squeeze(-1)



def extract_raw_modalities(model, X_tensor, adj, pairs_df, id_to_idx, kge_dict):

    model.eval()
    num_samples = X_tensor.shape[0]
    batch_size = 8
    node_embeddings_sum = 0
    with torch.inference_mode():
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            batch_emb = model.get_node_embeddings(X_tensor[i:end], adj)
            node_embeddings_sum += batch_emb * (end - i)

    gat_embeddings = (node_embeddings_sum / num_samples).cpu().numpy()
    gat_A, kge_A, gat_B, kge_B, labels = [], [], [], [], []

    for _, row in pairs_df.iterrows():
        gA_id, gB_id = str(row['Gene.A_ID']).strip(), str(row['Gene.B_ID']).strip()
        if gA_id in id_to_idx and gB_id in id_to_idx:
            idx_A, idx_B = id_to_idx[gA_id], id_to_idx[gB_id]
            gat_A.append(gat_embeddings[idx_A])
            kge_A.append(kge_dict[gA_id])
            gat_B.append(gat_embeddings[idx_B])
            kge_B.append(kge_dict[gB_id])
            labels.append(row['True_Label'])

    return np.array(gat_A), np.array(kge_A), np.array(gat_B), np.array(kge_B), np.array(labels)



def evaluate_deep_model(model, loader, device):

    model.eval()
    all_probs, all_preds, all_y = [], [], []

    with torch.inference_mode():
        for g_A, k_A, g_B, k_B, y in loader:
            g_A, k_A, g_B, k_B = g_A.to(device), k_A.to(device), g_B.to(device), k_B.to(device)
            logits = model(g_A, k_A, g_B, k_B)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_y.extend(y.numpy())

    all_y = np.asarray(all_y)
    all_probs = np.asarray(all_probs)
    all_preds = np.asarray(all_preds)

    if len(np.unique(all_y)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(all_y, all_probs)
        auprc = average_precision_score(all_y, all_probs)

    f1 = f1_score(all_y, all_preds, zero_division=0)
    return {"auroc": auroc, "auprc": auprc, "f1": f1}


def make_loader(gat_A, kge_A, gat_B, kge_B, y, batch_size=256, shuffle=True):
    ds = TensorDataset(
        torch.FloatTensor(gat_A),
        torch.FloatTensor(kge_A),
        torch.FloatTensor(gat_B),
        torch.FloatTensor(kge_B),
        torch.LongTensor(y)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def split_train_inner_val(train_df, val_ratio=0.15, seed=42):

    y = train_df["True_Label"].astype(int)

    class_counts = y.value_counts()
    if len(class_counts) == 2 and class_counts.min() >= 2:
        train_inner_df, val_inner_df = train_test_split(
            train_df,
            test_size=val_ratio,
            stratify=y,
            random_state=seed
        )
    else:
        print("   ⚠️ Warning: cannot stratify inner validation split; using random split.")
        train_inner_df, val_inner_df = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=seed
        )

    return train_inner_df.reset_index(drop=True), val_inner_df.reset_index(drop=True)


def train_deep_model(train_loader, val_loader, gat_dim, kge_dim, pos_weight, device, epochs=60):

    model = TransformerCrossFusionNet(gat_dim, kge_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=pos_weight, gamma=2.0).to(device)

    best_auroc = -np.inf
    best_metrics = {}
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for g_A, k_A, g_B, k_B, y in train_loader:
            g_A, k_A, g_B, k_B, y = g_A.to(device), k_A.to(device), g_B.to(device), k_B.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(g_A, k_A, g_B, k_B)
            loss = criterion(logits, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        val_metrics = evaluate_deep_model(model, val_loader, device)
        auroc = val_metrics["auroc"]

        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_metrics = val_metrics
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())


    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("   ⚠️ Warning: no valid validation AUROC found; using final epoch model.")
        best_metrics = evaluate_deep_model(model, val_loader, device)
        best_epoch = epochs

    return best_metrics, model, best_epoch


def train_and_evaluate(eval_dir, kg_raw_file):
    print("\n" + "=" * 70)
    print("🚀 PHASE 3: TRANSFORMER CROSS-MODAL NETWORK")

    print("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==== 1. 加载组学与图谱基础特征 ====
    X_raw, Y_np, genes, samples = load_data()
    adj = build_adjacency_matrix(genes).to(device)

    preprocessor = OmicsPreprocessor()
    preprocessor.load_params('results/models/scaler_params.npz')
    X_tensor = torch.FloatTensor(preprocessor.transform(X_raw)).to(device)

    id_to_idx = {str(g).split(' (')[1].replace(')', '').strip() if ' (' in str(g) else str(g).strip(): i for i, g in
                 enumerate(genes)}

    gat_model = PPI_MOGAT(num_genes=len(genes)).to(device)
    gat_model.load_state_dict(torch.load('results/models/production_model_gat_final.pth', map_location=device))

    kge_dict, kge_dim = generate_or_load_kge(list(id_to_idx.keys()), kg_raw_file=kg_raw_file)

    # ==== 2. 遍历 5 个 Fold ====
    print(f"\n>>> 正在基于固定基准数据集 [{eval_dir}] 评估 DDS 模型...")
    test_metrics_list = []
    val_metrics_list = []
    selected_epochs = []


    imbalance_ratio = 1.0

    for fold in range(1, 6):
        print("\n" + "-" * 70)
        print(f"Fold {fold}")
        print("-" * 70)

        fold_dir = os.path.join(eval_dir, f"fold_{fold}")
        train_file = os.path.join(fold_dir, "train.csv")
        test_file = os.path.join(fold_dir, "test.csv")

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)


        train_inner_df, val_inner_df = split_train_inner_val(
            train_df,
            val_ratio=0.15,
            seed=42 + fold
        )
      
        print(f"   Original train pairs: {len(train_df)}")
        print(f"   Inner train pairs:    {len(train_inner_df)}")
        print(f"   Inner val pairs:      {len(val_inner_df)}")
        print(f"   Held-out test pairs:  {len(test_df)}")


        train_gA, train_kA, train_gB, train_kB, train_y = extract_raw_modalities(
            gat_model, X_tensor, adj, train_inner_df, id_to_idx, kge_dict
        )
        val_gA, val_kA, val_gB, val_kB, val_y = extract_raw_modalities(
            gat_model, X_tensor, adj, val_inner_df, id_to_idx, kge_dict
        )
        test_gA, test_kA, test_gB, test_kB, test_y = extract_raw_modalities(
            gat_model, X_tensor, adj, test_df, id_to_idx, kge_dict
        )

        gat_dim = train_gA.shape[1]

        if fold == 1:
            num_neg = np.sum(train_y == 0)
            num_pos = np.sum(train_y == 1)
            imbalance_ratio = num_neg / num_pos if num_pos > 0 else 1.0


        train_loader = make_loader(
            train_gA, train_kA, train_gB, train_kB, train_y,
            batch_size=256,
            shuffle=True
        )
        val_loader = make_loader(
            val_gA, val_kA, val_gB, val_kB, val_y,
            batch_size=512,
            shuffle=False
        )
        test_loader = make_loader(
            test_gA, test_kA, test_gB, test_kB, test_y,
            batch_size=512,
            shuffle=False
        )

        # 训练：只根据 inner validation 选择最佳 epoch
        val_metrics, model, best_epoch = train_deep_model(
            train_loader,
            val_loader,
            gat_dim,
            kge_dim,
            imbalance_ratio,
            device,
            epochs=60
        )

        test_metrics = evaluate_deep_model(model, test_loader, device)

        val_metrics_list.append(val_metrics)
        test_metrics_list.append(test_metrics)
        selected_epochs.append(best_epoch)

        print(
            f"   Fold {fold} | Best Epoch by Val: {best_epoch} | "
            f"Val AUROC: {val_metrics['auroc']:.4f} | Val AUPRC: {val_metrics['auprc']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )
        print(
            f"   Fold {fold} | Final Test AUROC: {test_metrics['auroc']:.4f} | "
            f"Test AUPRC: {test_metrics['auprc']:.4f} | Test F1: {test_metrics['f1']:.4f}"
        )

    # ==== 3. 汇总输出 ====
    print("\n 最终性能：Held-out Test Set")
    print(f"Selected epochs by inner validation: {selected_epochs}")

    avg_auroc = np.nanmean([m['auroc'] for m in test_metrics_list])
    avg_auprc = np.nanmean([m['auprc'] for m in test_metrics_list])
    avg_f1 = np.nanmean([m['f1'] for m in test_metrics_list])

    std_auroc = np.nanstd([m['auroc'] for m in test_metrics_list])
    std_auprc = np.nanstd([m['auprc'] for m in test_metrics_list])
    std_f1 = np.nanstd([m['f1'] for m in test_metrics_list])

    print(f"Mean Test AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Mean Test AUPRC: {avg_auprc:.4f} ± {std_auprc:.4f}")
    print(f"Mean Test F1:    {avg_f1:.4f} ± {std_f1:.4f}")

    print("\nPer-fold final test metrics:")
    for i, m in enumerate(test_metrics_list, start=1):
        print(f"   Fold {i}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, F1={m['f1']:.4f}")

    print("\n✅ Evaluation finished. No SL classifier model weight was saved.")


if __name__ == "__main__":

    EVAL_DATASET_DIR = "DDS_Benchmarks/strict_cold_start_1_to_1/fold_data"

    train_and_evaluate(EVAL_DATASET_DIR, "data/raw/raw_kg.csv")
