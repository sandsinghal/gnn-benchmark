# GNN Benchmark

A structured framework for benchmarking various Graph Neural Network (GNN) models on standard node and graph classification datasets.

## Implemented Models

*   **GCN** (Kipf & Welling, 2017) - Node Classification
*   **GAT** (Veličković et al., 2018) - Node Classification
*   **GIN** (Xu et al., 2019 - "How Powerful are GNNs?") - Node Classification
*   **ChebNet** (Defferrard et al., 2016) - Node Classification
*   **FastGCN** (Chen et al., 2018) - Node Classification (Uses custom layer-wise importance sampling during training)
*   **DGCNN_SortPool** (Zhang et al., 2018 - "An End-to-End Deep Learning Architecture for Graph Classification") - Graph Classification

## Datasets

Supports standard PyG datasets:

*   **Node Classification:**
    *   Cora
    *   Citeseer
    *   PubMed
    *   Reddit (Requires minibatching)
    *   PPI (Multi-graph dataset used for node classification, requires minibatching)
*   **Graph Classification (from TUDataset):**
    *   MUTAG
    *   PROTEINS
    *   *(Can be extended by adding names to `config.py` and `data/load_data.py`)*

## Features

*   **Modular Structure:** Code separated into `data`, `models`, `utils`, `config`.
*   **Configurable:** Uses `argparse` for command-line configuration, with defaults inspired by relevant papers. Supports optional YAML config files (`--config_file`). Overrides are applied hierarchically (Defaults -> YAML -> CLI -> Dataset -> Model -> Model+Dataset).
*   **Weights & Biases Integration:** Logs metrics, configuration, model parameters, and saves the best model checkpoint artifact (`--log_to_wandb`, `--no-log_to_wandb`).
*   **Flexible Training Modes:**
    *   Full-batch training (for small node classification datasets).
    *   Minibatch training using `NeighborLoader` (for large node classification datasets like Reddit).
    *   Minibatch training using `DataLoader` (for multi-graph node classification like PPI, and all graph classification tasks).
    *   Custom `FastGcnSampler` for layer-wise importance sampling specific to the FastGCN model.
*   **Early Stopping:** Monitors validation performance (`acc` or `f1_micro`) and stops training early based on `--patience`.
*   **Paper-Inspired Defaults:** Configuration includes default hyperparameters based on common practices and the original model papers where available, with dataset/model-specific overrides.
*   **Automatic Feature Handling:** Adds node degree features for graph classification datasets lacking inherent node features.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sandsinghal/gnn-benchmark.git
    cd gnn_benchmark
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Ensure PyTorch and PyG match your system/CUDA version.
    # It might be safer to install them manually first based on official instructions:
    # See: https://pytorch.org/get-started/locally/
    # See: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
    ```
4.  **Login to Weights & Biases (optional but recommended):**
    ```bash
    wandb login
    ```

## Running Experiments

Use `main.py` to run experiments. Key configuration options are available via command-line arguments. See `config.py` for all defaults and overrides.

**Node Classification Examples:**

```bash
# Run GCN on Cora (uses defaults, full-batch)
python main.py --model_name GCN --dataset_name Cora

# Run GAT on Citeseer (uses specific overrides from config.py, full-batch)
python main.py --model_name GAT --dataset_name Citeseer

# Run GIN on PubMed (uses specific overrides, full-batch)
python main.py --model_name GIN --dataset_name PubMed

# Run ChebNet on Cora with specific K=5 (overrides default/config K)
python main.py --model_name ChebNet --dataset_name Cora --cheb_k 5

# Run GAT on PPI (uses minibatch via DataLoader, specific overrides in config.py)
python main.py --model_name GAT --dataset_name PPI --device cuda

# Run GCN on Reddit (uses minibatch via NeighborLoader)
# Specify minibatch parameters and potentially larger model
python main.py --model_name GCN --dataset_name Reddit --hidden_channels 128 --batch_size 1024 --lr 0.01 --epochs 50 --device cuda

# Run FastGCN on PubMed (uses custom sampler, minibatch)
python main.py --model_name FastGCN --dataset_name PubMed --batch_size 256 --device cuda

# Disable WandB logging
python main.py --model_name GCN --dataset_name Cora --no-log_to_wandb

# Override learning rate and dropout via CLI
python main.py --model_name GAT --dataset_name Cora --lr 0.001 --dropout 0.7

# Load config from YAML (overrides defaults, overridden by CLI)
# Example: python main.py --config_file configs/my_cora_gat_config.yaml --seed 123
```

**Graph Classification Examples:**

```bash
# Run DGCNN_SortPool on MUTAG (uses minibatch via DataLoader)
python main.py --model_name DGCNN_SortPool --dataset_name MUTAG --device cuda

# Run DGCNN_SortPool on PROTEINS with different hyperparameters
python main.py --model_name DGCNN_SortPool --dataset_name PROTEINS --hidden_channels 128 --dgcnn_sortpool_k 40 --lr 0.0005 --batch_size 128 --device cuda

```

**Note on TUDataset Splitting:** The current implementation in `main.py` uses a basic random 80/10/10 split for graph classification datasets. For rigorous benchmarking, consider using predefined splits if available or more standard cross-validation techniques.
```
