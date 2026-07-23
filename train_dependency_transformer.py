import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import gc
from src.dataset import load_data
from src.model import TransformerGeneDependencyModel
from src.data_utils import DataSplitter, OmicsPreprocessor
from src.config import PARAMS, FILES
from src.utils import EarlyStopping, calculate_metrics


def train_epoch(model, loader, optimizer, criterion, device, accumulation_steps):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
    return epoch_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_true.append(batch_y.cpu().numpy())

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_true_np = np.concatenate(all_true, axis=0)

    global_pcc, median_gene_pcc = calculate_metrics(all_preds_np, all_true_np)

    return total_loss / len(loader), global_pcc, median_gene_pcc


def run_scientific_phase(X_train, Y_train, X_val, Y_val, device, genes):
    print("\n" + "=" * 50)
    print("PHASE 1: HYPERPARAMETER TUNING & EARLY STOPPING")
    print("=" * 50)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))

    train_loader = DataLoader(train_ds, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=PARAMS['batch_size'], shuffle=False)

    model = TransformerGeneDependencyModel(num_genes=len(genes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['weight_decay'])
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(patience=PARAMS['patience'], verbose=True, path='results/models/checkpoint_final.pth')

    for epoch in range(PARAMS['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, PARAMS['accumulation_steps'])
        val_loss, global_pcc, median_pcc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Global PCC: {global_pcc:.4f} | Median PCC: {median_pcc:.4f}")

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            break

    return early_stopping.best_epoch


def run_production_phase(X_train, Y_train, X_val, Y_val, device, genes):
    print("\n" + "=" * 50)
    print("PHASE 2: FINAL ACADEMIC MODEL (Fixed Epochs)")
    print("=" * 50)


    X_final = np.concatenate([X_train, X_val], axis=0)
    Y_final = np.concatenate([Y_train, Y_val], axis=0)

    prod_ds = TensorDataset(torch.FloatTensor(X_final), torch.FloatTensor(Y_final))
    prod_loader = DataLoader(prod_ds, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)

    model = TransformerGeneDependencyModel(num_genes=len(genes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['weight_decay'])
    criterion = nn.MSELoss()

    fixed_epochs = PARAMS.get('production_epochs', 400)
    print(f"   -> Training Final Model for {fixed_epochs} predefined epochs...")

    for epoch in range(fixed_epochs):
        loss = train_epoch(model, prod_loader, optimizer, criterion, device, PARAMS['accumulation_steps'])
        if (epoch + 1) % 10 == 0:
            print(f"   Prod Epoch {epoch + 1:03d} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), 'results/models/production_model_final.pth')
    print("✅ Academic Production Model Saved (Test Set strictly held out).")


def main():
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load raw, unscaled data
    X_raw, Y_raw, genes, samples = load_data()
    if X_raw is None: return

    y_means_np = np.nanmean(Y_raw, axis=1)
    y_means = pd.Series(y_means_np, index=samples)

    splitter = DataSplitter(FILES['model'])
    train_ids, val_ids, test_ids = splitter.perform_stratified_split(samples, y_means)

    sample_to_idx = {id: i for i, id in enumerate(samples)}
    train_idx = [sample_to_idx[i] for i in train_ids if i in sample_to_idx]
    val_idx = [sample_to_idx[i] for i in val_ids if i in sample_to_idx]
    test_idx = [sample_to_idx[i] for i in test_ids if i in sample_to_idx]


    preprocessor = OmicsPreprocessor()
    print("\n   -> Fitting Preprocessor on Train Set ONLY to prevent Data Leakage...")
    X_train, Y_train = preprocessor.fit_transform(X_raw[train_idx], Y_raw[train_idx])

    print("   -> Transforming Validation and Test Sets...")
    X_val, Y_val = preprocessor.transform(X_raw[val_idx], Y_raw[val_idx])
    X_test, Y_test = preprocessor.transform(X_raw[test_idx], Y_raw[test_idx])

    preprocessor.save_params()


    _ = run_scientific_phase(X_train, Y_train, X_val, Y_val, device, genes)

    print("   -> Flushing GPU Memory Cache...")
    torch.cuda.empty_cache()
    gc.collect()


    run_production_phase(X_train, Y_train, X_val, Y_val, device, genes)




if __name__ == "__main__":
    main()