import os

FILES = {
    'exp': os.path.join('data', 'raw', 'OmicsExpressionProteinCodingGenesTPMLogp1.csv'),
    'cnv': os.path.join('data', 'raw', 'OmicsCNGene.csv'),
    'mut': os.path.join('data', 'raw', 'OmicsSomaticMutationsMatrixDamaging.csv'),
    'dep': os.path.join('data', 'raw', 'CRISPRGeneEffect.csv'),
    'model': os.path.join('data', 'raw', 'Model.csv'),
}

OMICS_CHANNELS = 3

# === Hyperparameters ===
PARAMS = {
    'hidden_dim': 512,
    'transformer_heads': 4,
    'transformer_layers': 2,
    'dropout': 0.3,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'max_epochs': 400,
    'patience': 40,
    'production_epochs': 300,
    'batch_size': 4,
    'accumulation_steps': 4
}
