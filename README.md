# DSFDS

**Dual-Source-Fusion Dependency and Synthetic Lethality framework**

DSFDS is a hierarchical deep-learning framework for pan-cancer single-gene dependency prediction and synthetic-lethality (SL) inference. It separates baseline gene dependency modeling from pairwise SL classification and integrates multi-omics profiles, physical protein–protein interaction (PPI) topology, and leakage-filtered knowledge graph embeddings.

Web platform: https://www.tmliang.cn/DSFDS

## Overview

DSFDS contains three sequential components:

1. **Dependency Transformer**  
   Integrates gene expression, copy-number variation, and damaging somatic mutation profiles using gated multi-omics fusion, learnable gene embeddings, and a Transformer encoder to predict genome-wide CRISPR dependency scores.

2. **PPI-constrained Sparse GAT**  
   Uses the pretrained dependency Transformer as a frozen feature extractor and propagates gene representations only along valid physical PPI edges.

3. **Cross-modal SL classifier**  
   Combines PPI-aware gene representations with leakage-filtered knowledge graph embeddings. Gene-pair representations are modeled as four tokens—GAT-A, KGE-A, GAT-B, and KGE-B—and are supplemented with Hadamard-product and absolute-difference features for SL classification.

## Repository structure

```text
DSFDS/
├── README.md
├── requirements.txt
├── train_dependency_transformer.py
├── train_ppi_sparse_gat.py
├── train_cross_modal_sl_classifier.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── data/
│   └── raw/
│       ├── OmicsExpressionProteinCodingGenesTPMLogp1.csv
│       ├── OmicsCNGene.csv
│       ├── OmicsSomaticMutationsMatrixDamaging.csv
│       ├── CRISPRGeneEffect.csv
│       ├── Model.csv
│       ├── ppi.csv
│       └── raw_kg.csv
├── DDS_Benchmarks/
│   └── <evaluation_setting>/
│       └── fold_data/
│           ├── fold_1/
│           │   ├── train.csv
│           │   └── test.csv
│           └── ...
└── results/
    ├── models/
    └── predictions/
```

Large datasets, model checkpoints, generated embeddings, and benchmark splits do not need to be committed directly to GitHub. Their download locations or preparation instructions should be documented separately.

## Installation

Python 3.10 is recommended.

```bash
conda create -n dsfds python=3.10 -y
conda activate dsfds
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with the local CUDA version before installing the remaining dependencies.

## Required input files

Place the following DepMap files under `data/raw/`:

```text
OmicsExpressionProteinCodingGenesTPMLogp1.csv
OmicsCNGene.csv
OmicsSomaticMutationsMatrixDamaging.csv
CRISPRGeneEffect.csv
Model.csv
```

Additional files:

- `ppi.csv`: physical PPI edges with columns `Gene1` and `Gene2`.
- `raw_kg.csv`: knowledge graph edges with columns `x_id`, `y_id`, `relation`, and `display_relation`.
- Fold-specific `train.csv` and `test.csv` files: each file must contain `Gene.A_ID`, `Gene.B_ID`, and `True_Label`.

The repository does not redistribute third-party datasets. Users should obtain the required resources from their original databases and comply with the corresponding licenses and terms of use.

## Running DSFDS

The three training stages must be executed in order.

### 1. Train the single-gene dependency Transformer

```bash
python train_dependency_transformer.py
```

Main outputs:

```text
results/models/checkpoint_final.pth
results/models/production_model_final.pth
results/models/scaler_params.npz
```

The preprocessing parameters are fitted using the training set only and are subsequently applied to validation, test, and inference data.

### 2. Train the PPI-constrained Sparse GAT

```bash
python train_ppi_sparse_gat.py
```

This stage loads `production_model_final.pth`, freezes the dependency Transformer, and trains the PPI-constrained graph module.

Main outputs:

```text
results/models/checkpoint_gat_final.pth
results/models/production_model_gat_final.pth
```

### 3. Train and evaluate the cross-modal SL classifier

Before running the script, set `EVAL_DATASET_DIR` near the end of `train_cross_modal_sl_classifier.py` to the desired benchmark split, for example:

```python
EVAL_DATASET_DIR = "DDS_Benchmarks/strict_cold_start_1_to_1/fold_data"
```

Then run:

```bash
python train_cross_modal_sl_classifier.py
```

The script:

- loads the pretrained dependency and Sparse GAT components;
- generates or loads 256-dimensional knowledge graph embeddings;
- removes SL, synthetic-rescue, and non-SL relations before embedding generation;
- creates an inner validation split within each training fold;
- selects the best epoch using validation AUROC;
- evaluates the selected model once on the held-out test fold;
- reports fold-level and mean AUROC, AUPRC, and F1 scores.

Generated KGE file:

```text
results/models/synlethdb_kge_256d.csv
```

The current benchmark script reports evaluation metrics but does not save fold-specific SL-classifier weights.

## Evaluation settings

DSFDS supports fold directories prepared for different generalization settings:

- **Warm start:** test pairs may contain genes observed during training.
- **Semi-cold start:** each test pair contains at least one unseen gene.
- **Strict cold start:** both genes in each test pair are unseen during training.

To switch settings, update `EVAL_DATASET_DIR` in `train_cross_modal_sl_classifier.py`.

## Reproducibility

The dependency-data split and PPI-Sparse-GAT training use a random seed of 42. Inner validation splits use `42 + fold_index`.

The random-walk KGE generator does not currently set an explicit seed. For exact reuse of semantic embeddings, retain and reuse:

```text
results/models/synlethdb_kge_256d.csv
```

The following files should also be preserved for reproducible inference:

```text
results/models/scaler_params.npz
results/models/production_model_final.pth
results/models/production_model_gat_final.pth
```

## Hardware considerations

A CUDA-capable GPU is strongly recommended because the dependency model operates over the full consensus gene set and the graph stage processes genome-scale PPI information. CPU execution is supported by the scripts but may be substantially slower.

## Outputs

Depending on the stage, DSFDS produces:

- dependency Transformer checkpoints;
- preprocessing parameters;
- PPI-constrained Sparse GAT checkpoints;
- knowledge graph embeddings;
- fold-level AUROC, AUPRC, and F1 metrics printed to the console.

## Citation

A formal citation will be added after publication. Until then, please cite the GitHub repository and the associated manuscript:

```text
Zheng W, et al. DSFDS: A Dual-Source Fusion Framework for Decoding
Pan-Cancer Vulnerabilities via Dependency and Synthetic Lethality.
Manuscript in preparation.
```

## Code availability

The DSFDS source code is available at:

https://github.com/Mainenp/DSFDS

## License

Add an open-source `LICENSE` file before public release. MIT and BSD-3-Clause are commonly used permissive licenses for research software, but the final choice should be confirmed by the authors and institution.

## Contact

For questions regarding the model or repository, please open a GitHub issue.
