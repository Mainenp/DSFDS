import pandas as pd
import numpy as np
import scipy.stats as stats
import os
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, model_csv_path):
        self.meta_df = pd.read_csv(model_csv_path)
        self.meta_df['ModelID'] = self.meta_df['ModelID'].astype(str).str.strip()

    def determine_ratios(self, n_samples):
        if n_samples >= 100:
            return 0.6, 0.2, 0.2  # 60% Train, 20% Val, 20% Test
        elif 30 <= n_samples < 100:
            return 0.7, 0.15, 0.15
        else:
            return 1.0, 0.0, 0.0

    def check_consistency(self, df_train, df_test, feature_col, is_categorical=False):
        if len(df_test) == 0: return 1.0
        if is_categorical:
            # Chi-square logic if needed
            return 1.0
        else:
            _, p_val = stats.ks_2samp(df_train[feature_col], df_test[feature_col])
            return p_val

    def perform_stratified_split(self, valid_model_ids, y_means):
        df = pd.DataFrame({'ModelID': valid_model_ids})
        df = df.merge(self.meta_df[['ModelID', 'OncotreePrimaryDisease']], on='ModelID', how='left')
        df['Mean_Score'] = df['ModelID'].map(y_means)

        df['Score_Quantile'] = pd.qcut(df['Mean_Score'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
        df['OncotreePrimaryDisease'] = df['OncotreePrimaryDisease'].fillna('Unknown')
        df['Strata'] = df['OncotreePrimaryDisease'].astype(str) + "_" + df['Score_Quantile'].astype(str)

        n_samples = len(df)
        if n_samples < 5:
            print(f"   ❌ Critical Error: Only {n_samples} valid samples found. Cannot split.")
            return df['ModelID'].tolist(), [], []

        r_train, r_val, r_test = self.determine_ratios(n_samples)

        try:
            train_val_df, test_df = train_test_split(df, test_size=r_test, stratify=df['Strata'], random_state=42)
            train_df, val_df = train_test_split(train_val_df, test_size=r_val / (r_train + r_val),
                                                stratify=train_val_df['Strata'], random_state=42)
        except:
            train_val_df, test_df = train_test_split(df, test_size=r_test, random_state=42)
            train_df, val_df = train_test_split(train_val_df, test_size=r_val / (r_train + r_val), random_state=42)

        return train_df['ModelID'].tolist(), val_df['ModelID'].tolist(), test_df['ModelID'].tolist()


class OmicsPreprocessor:
    def __init__(self):
        self.exp_mean = None
        self.exp_std = None
        self.dep_mean = None

    def fit_transform(self, X_train, Y_train):    
        self.exp_mean = np.nanmean(X_train[:, :, 0], axis=0)
        self.exp_std = np.nanstd(X_train[:, :, 0], axis=0) + 1e-8

       
        self.dep_mean = np.nanmean(Y_train, axis=0)

        return self.transform(X_train, Y_train)

    def transform(self, X, Y=None):
        X_proc = X.copy()

        X_proc[:, :, 0] = (X_proc[:, :, 0] - self.exp_mean) / self.exp_std

        X_proc = np.nan_to_num(X_proc, nan=0.0)

        if Y is not None:
            Y_proc = Y.copy()
            inds = np.where(np.isnan(Y_proc))
            Y_proc[inds] = np.take(self.dep_mean, inds[1])
            return X_proc, Y_proc
        return X_proc

    def save_params(self, save_dir='results/models/'):
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'scaler_params.npz'),
                 exp_mean=self.exp_mean, exp_std=self.exp_std, dep_mean=self.dep_mean)

    def load_params(self, load_path='results/models/scaler_params.npz'):
        data = np.load(load_path)
        self.exp_mean = data['exp_mean']
        self.exp_std = data['exp_std']
        self.dep_mean = data['dep_mean']
