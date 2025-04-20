# config.py
import argparse
import os
import yaml
from typing import Dict, Any, List

import torch

# Default hyperparameters inspired by common practices and papers
# These can be overridden by command-line arguments or YAML files
DEFAULTS: Dict[str, Any] = {
    # General
    "experiment_name": "gnn_experiment",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "log_to_wandb": True,
    "wandb_project": "gnn-benchmark-structured",
    # Data
    "dataset_name": "Cora",
    "data_dir": "./datasets",
    "normalize_features": True,
    "graph_classification_dataset": False,
    # Training
    "epochs": 200,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "optimizer": "Adam",
    "patience": 50,
    "batch_size": 0,
    # Model Base
    "model_name": "GCN",
    "hidden_channels": 64,
    "dropout": 0.5,
    # GAT Specific
    "gat_heads": 8,
    "gat_hidden_channels_per_head": 8,
    "gat_output_heads": 1,
    "gat_concat_output": False,
    "gat_negative_slope": 0.2,
    # ChebNet Specific
    "cheb_k": 3,
    # GIN Specific
    "gin_mlp_layers": 2,
    "gin_epsilon": 0.0,
    "gin_train_epsilon": False,
    # DGCNN_SortPool Specific
    "dgcnn_gcn_layers": 4,
    "dgcnn_sortpool_k": 30,
    "dgcnn_conv1d_channels": 16,
    "dgcnn_conv1d_kernel_size": 5,
    "dgcnn_dense_hidden": 128,
    # FastGCN Specific
    "fastgcn_layer_sample_sizes": [512, 512],
    # Dataset Specific Sampling
    "num_neighbors": [25, 10],
}

# Dataset/Model Specific Overrides
OVERRIDES: Dict[str, Any] = {
    # Dataset Overrides
    "Cora": {
        "epochs": 200,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "batch_size": 0,
        "graph_classification_dataset": False,
    },
    "Citeseer": {
        "epochs": 200,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "hidden_channels": 32,
        "batch_size": 0,
        "graph_classification_dataset": False,
    },
    "PubMed": {
        "epochs": 200,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "hidden_channels": 16,
        "batch_size": 0,
        "graph_classification_dataset": False,
    },
    # Add other dataset overrides here...
}

def get_config() -> Dict[str, Any]:
    """Parses command-line arguments, loads optional YAML, applies overrides, and returns config."""
    parser = argparse.ArgumentParser(description="GNN Benchmark Configuration")

    # General experiment setup
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML config file to override defaults.")
    parser.add_argument("--experiment_name", type=str, help="Name for the experiment run.")
    parser.add_argument("--device", type=str, help="Device to use (cuda or cpu).")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--log_to_wandb", action=argparse.BooleanOptionalAction, default=None, help="Log results to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, help="WandB project name.")

    # Dataset
    parser.add_argument("--dataset_name", type=str, choices=["Cora", "Citeseer", "PubMed", "Reddit", "PPI"], help="Dataset to use.")
    parser.add_argument("--data_dir", type=str, help="Directory for storing datasets.")
    parser.add_argument("--normalize_features", action=argparse.BooleanOptionalAction, default=None, help="Whether to normalize features.")

    # Model
    parser.add_argument("--model_name", type=str, choices=["GCN", "GAT", "GIN", "ChebNet", "DGCNN_SortPool", "FastGCN"], help="GNN model to use.")
    parser.add_argument("--hidden_channels", type=int, help="Number of hidden units.")
    parser.add_argument("--dropout", type=float, help="Dropout rate.")

    # Training
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay (L2 regularization).")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "SGD"], help="Optimizer.")
    parser.add_argument("--patience", type=int, help="Early stopping patience (0 to disable).")
    parser.add_argument("--batch_size", type=int, help="Batch size (0 for full-batch).")

    # Parse arguments
    args = parser.parse_args()
    config = DEFAULTS.copy()

    # Load from YAML if provided
    if args.config_file:
        if os.path.exists(args.config_file):
            print(f"Loading configuration from YAML file: {args.config_file}")
            try:
                with open(args.config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config.update(yaml_config)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}. Using previous defaults.")
            except Exception as e:
                print(f"Error reading YAML file: {e}. Using previous defaults.")
        else:
            print(f"Warning: Config file not found at {args.config_file}")

    # Override with command-line arguments
    cmd_line_args_dict: Dict[str, Any] = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cmd_line_args_dict)

    # Apply Dataset Specific Overrides
    final_dataset_name = config["dataset_name"]
    if final_dataset_name in OVERRIDES:
        print(f"Applying overrides for dataset: {final_dataset_name}")
        dataset_overrides = OVERRIDES[final_dataset_name]
        config.update({k: v for k, v in dataset_overrides.items() if k not in cmd_line_args_dict})

    print("\n--- Final Configuration ---")
    for key, value in sorted(config.items()):
        print(f"{key}: {value}")
    print("---------------------------\n")

    return config