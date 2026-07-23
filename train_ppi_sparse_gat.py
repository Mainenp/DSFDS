import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import gc
import random
import warnings

warnings.filterwarnings('ignore')

from src.dataset import load_data, build_adjacency_matrix
from src.model import PPI_MOGAT
from src.data_utils import DataSplitter, OmicsPreprocessor
from src.config import PARAMS, FILES
from src.utils import EarlyStopping, calculate_metrics


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train_epoch(model, loader, adj, optimizer, criterion, device, accumulation_steps):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)


        pred = model(batch_x, adj)


        mask = torch.isfinite(batch_y)
        if torch.sum(mask) == 0:
            continue

        loss = criterion(pred[mask], batch_y[mask])
        loss = loss / accumulation_steps
        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
    return epoch_loss / len(loader)


def validate(model, loader, adj, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    valid_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x, adj)

            mask = torch.isfinite(batch_y)
            if torch.sum(mask) == 0:
                continue

            loss = criterion(pred[mask], batch_y[mask])
            total_loss += loss.item()
            valid_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_true.append(batch_y.cpu().numpy())

    if valid_batches == 0:
        return float('nan'), 0.0, 0.0

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_true_np = np.concatenate(all_true, axis=0)
    global_pcc, median_gene_pcc = calculate_metrics(all_true_np, all_preds_np)
    return total_loss / valid_batches, global_pcc, median_gene_pcc


def get_gat_model(device, num_genes):
    model = PPI_MOGAT(num_genes=num_genes).to(device)
    transformer_path = 'results/models/production_model_final.pth'
    if os.path.exists(transformer_path):
        print("\n   [Linkage Success] 🧬 Loading Pre-trained Transformer Weights...")
        model.transformer_branch.load_state_dict(torch.load(transformer_path, map_location=device))

        # 强制冻结Transformer
        for name, param in model.transformer_branch.named_parameters():
            param.requires_grad = False

        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"   -> Trainable parameters (only GAT): {len(trainable_params)}")
    else:
        print("\n   ⚠️ WARNING: Transformer weights not found!")
    return model


def run_scientific_phase(X_train, Y_train, X_val, Y_val, adj, device, genes):
    print("\n" + "=" * 50)
    print("PHASE 1: HYPERPARAMETER TUNING & EARLY STOPPING (GAT)")
    print("=" * 50)
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    train_loader = DataLoader(train_ds, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=PARAMS['batch_size'], shuffle=False)

    model = get_gat_model(device, len(genes))


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=PARAMS['weight_decay']
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=PARAMS['patience'], verbose=True, path='results/models/checkpoint_gat_final.pth')

    for epoch in range(PARAMS['max_epochs']):
        train_loss = train_epoch(model, train_loader, adj, optimizer, criterion, device, PARAMS['accumulation_steps'])
        val_loss, global_pcc, median_pcc = validate(model, val_loader, adj, criterion, device)
        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Global PCC: {global_pcc:.4f} | Median PCC: {median_pcc:.4f}")
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            break
    return early_stopping.best_epoch


def run_production_phase(X_train, Y_train, X_val, Y_val, adj, optimal_epochs, device, genes):
    print("\n" + "=" * 50)
    print("PHASE 2: FINAL ACADEMIC GAT MODEL (Fixed Epochs)")
    print("=" * 50)
    X_final = np.concatenate([X_train, X_val], axis=0)
    Y_final = np.concatenate([Y_train, Y_val], axis=0)
    prod_ds = TensorDataset(torch.FloatTensor(X_final), torch.FloatTensor(Y_final))
    prod_loader = DataLoader(prod_ds, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)

    model = get_gat_model(device, len(genes))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=PARAMS['weight_decay']
    )
    criterion = nn.MSELoss()
    target_epochs = optimal_epochs + 5

    for epoch in range(target_epochs):
        loss = train_epoch(model, prod_loader, adj, optimizer, criterion, device, PARAMS['accumulation_steps'])
        if (epoch + 1) % 5 == 0:
            print(f"   Prod Epoch {epoch + 1:03d} | Loss: {loss:.4f}")
    torch.save(model.state_dict(), 'results/models/production_model_gat_final.pth')
    print("✅ GAT Unified Model Saved Successfully (production_model_gat_final.pth).")


def main():
    set_seed(42)
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_raw, Y_raw, genes, samples = load_data()
    if X_raw is None: return
    adj = build_adjacency_matrix(genes)
    adj = adj.to(device)


    adj_np = adj.cpu().numpy()
    num_self_loops = np.sum(np.diag(adj_np))
    num_total_edges = np.sum(adj_np)
    num_real_edges = num_total_edges - num_self_loops
    print(f"\n    PPI矩阵检查：")
    print(f"      总边数（含自环）: {int(num_total_edges)}")
    print(f"      真实互作边数（不含自环）: {int(num_real_edges)}")
    if num_real_edges < 1000:
        print(f"  警告：真实互作边太少！请先修复 ppi.csv 基因名匹配！")

    y_means_np = np.nanmean(Y_raw, axis=1)
    y_means = pd.Series(y_means_np, index=samples)
    splitter = DataSplitter(FILES['model'])
    train_ids, val_ids, test_ids = splitter.perform_stratified_split(samples, y_means)
    sample_to_idx = {id: i for i, id in enumerate(samples)}
    train_idx = [sample_to_idx[i] for i in train_ids if i in sample_to_idx]
    val_idx = [sample_to_idx[i] for i in val_ids if i in sample_to_idx]

    preprocessor = OmicsPreprocessor()
    print("\n   -> Fitting Preprocessor on Train Set ONLY...")
    X_train, Y_train = preprocessor.fit_transform(X_raw[train_idx], Y_raw[train_idx])
    print("   -> Transforming Validation Set...")
    X_val, Y_val = preprocessor.transform(X_raw[val_idx], Y_raw[val_idx])

    best_epoch = run_scientific_phase(X_train, Y_train, X_val, Y_val, adj, device, genes)
    print("   -> Flushing GPU Memory Cache...")
    torch.cuda.empty_cache()
    gc.collect()
    run_production_phase(X_train, Y_train, X_val, Y_val, adj, best_epoch, device, genes)


if __name__ == "__main__":
    main()