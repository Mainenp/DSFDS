import torch
import numpy as np
import os
from scipy.stats import pearsonr


class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'   |-- EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def calculate_metrics(all_preds, all_true):
    flat_p = all_preds.flatten()
    flat_t = all_true.flatten()
   
    mask = np.isfinite(flat_p) & np.isfinite(flat_t)
    global_pcc = pearsonr(flat_p[mask], flat_t[mask])[0] if np.sum(mask) > 0 else 0.0
    
    num_genes = all_preds.shape[1]
    gene_pccs = []

    for i in range(num_genes):
        p_gene = all_preds[:, i]
        t_gene = all_true[:, i]
        mask_gene = np.isfinite(p_gene) & np.isfinite(t_gene)

       
        if np.std(p_gene[mask_gene]) > 1e-6 and np.std(t_gene[mask_gene]) > 1e-6:
            pcc = pearsonr(p_gene[mask_gene], t_gene[mask_gene])[0]
            if not np.isnan(pcc):
                gene_pccs.append(pcc)

    median_gene_pcc = np.median(gene_pccs) if len(gene_pccs) > 0 else 0.0

    return global_pcc, median_gene_pcc
