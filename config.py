# config.py
import argparse
import os
import yaml
from typing import Any, Dict, List, Optional, Union

import torch

# Default hyperparameters inspired by common practices and papers
# These can be overridden by command-line arguments or YAML files

DEFAULTS: Dict[str, Any] = {
    # General
    "experiment_name": "gnn_experiment",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "log_to_wandb": True,
    "wandb_mode": "online",
    "wandb_project": "gnn-benchmark-structured",
    # Data
    "dataset_name": "Cora",
    "data_dir": "./datasets",  # Directory to store datasets
    "normalize_features": True,  # Apply T.NormalizeFeatures where appropriate
    "graph_classification_dataset": False,  # Flag to indicate graph task (auto-set by overrides)
    # Training
    "epochs": 200,  # General default, often overridden
    "lr": 0.01,  # Common starting point, often overridden
    "weight_decay": 5e-4,  # Common GCN default
    "optimizer": "Adam",  # Adam is prevalent
    "patience": 50,  # Early stopping patience (0 to disable)
    "batch_size": 0,  # Default to full-batch for node classification unless overridden
    # Model Base
    "model_name": "GCN",
    "hidden_channels": 64,  # General default
    "dropout": 0.5,  # Common default, GAT often uses 0.6
    # GAT Specific (Veličković et al., 2018)
    "gat_heads": 8,  # Common default (e.g., Cora, Citeseer)
    "gat_hidden_channels_per_head": 8,  # Results in hidden_channels=64 for first layer
    "gat_output_heads": 1,  # Default for most node datasets
    "gat_concat_output": False,  # Usually average final layer heads if > 1
    "gat_negative_slope": 0.2,  # LeakyReLU slope from paper
    # ChebNet Specific (Defferrard et al., 2016)
    "cheb_k": 3,  # Polynomial order K, can vary significantly (e.g., 2-5 common, >20 in paper)
    # GIN Specific (Xu et al., 2019 - "How Powerful are GNNs?")
    "gin_mlp_layers": 2,  # Number of layers in the MLP for GINConv
    "gin_epsilon": 0.0,  # Learnable epsilon? Paper suggests fixing often works well (GIN-0)
    "gin_train_epsilon": False,
    # DGCNN_SortPool Specific (Graph Classification - Zhang et al. 2018)
    "dgcnn_gcn_layers": 4,
    "dgcnn_sortpool_k": 30,  # K for SortPooling (can be % of avg nodes or fixed)
    "dgcnn_conv1d_channels": 16,
    "dgcnn_conv1d_kernel_size": 5,
    "dgcnn_dense_hidden": 128,
    # FastGCN specific (Node Classification - Chen et al. 2018)
    "fastgcn_layer_sample_sizes": [
        512,
        512,
    ],  # Nodes to sample per layer (t_l in paper) - NEEDS custom sampler
    # Dataset Specific Sampling (NeighborLoader / GraphSAGE style)
    "num_neighbors": [
        25,
        10,
    ],  # Default sampling depth for 2-layer models using NeighborLoader
}

# --- Dataset/Model Specific Overrides ---
# These values take precedence over DEFAULTS if the condition matches
# Order of application: DEFAULTS -> YAML -> CLI -> Dataset Override -> Model Base Override -> Model+Dataset Override

OVERRIDES: Dict[str, Any] = {
    # === Dataset Overrides (Applied FIRST after initial config) ===
    "Cora": {
        "epochs": 200,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "dropout": 0.5,  # GCN defaults
        "batch_size": 0,  # Full batch
        "graph_classification_dataset": False,
    },
    "Citeseer": {
        "epochs": 200,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "dropout": 0.5,  # GCN defaults
        "hidden_channels": 32,  # Sometimes smaller works better
        "batch_size": 0,  # Full batch
        "graph_classification_dataset": False,
    },
    "PubMed": {
        "epochs": 200,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "dropout": 0.5,  # GCN defaults
        "hidden_channels": 16,  # GCN paper PubMed
        "batch_size": 0,  # Full batch
        "graph_classification_dataset": False,
    },
    "Reddit": {
        "epochs": 200,  # Often trains faster but needs careful tuning
        "lr": 0.005,
        "weight_decay": 0,
        "dropout": 0.5,
        "hidden_channels": 128,  # Larger models often needed
        "batch_size": 1024,  # Minibatch required
        "normalize_features": False,  # Reddit features are usually used as-is
        "graph_classification_dataset": False,
    },
    "PPI": {
        "epochs": 200,
        "lr": 0.005,
        "weight_decay": 0,
        "dropout": 0.1,  # Common PPI settings
        "hidden_channels": 256,  # Larger models needed
        "batch_size": 20,  # Graph-level batching (DataLoader)
        "normalize_features": False,  # PPI features usually used as-is
        "graph_classification_dataset": False,  # Node classification task on multiple graphs
    },
    # === Graph Classification Dataset Examples ===
    "MUTAG": {
        "batch_size": 32,
        "hidden_channels": 64,
        "normalize_features": False,
        "graph_classification_dataset": True,
        "epochs": 200,
        "lr": 0.001,
        "weight_decay": 0,
    },
    "PROTEINS": {
        "batch_size": 64,
        "hidden_channels": 64,
        "normalize_features": False,
        "graph_classification_dataset": True,
        "epochs": 200,
        "lr": 0.001,
        "weight_decay": 0,
    },
    "IMDB-BINARY": {
        "batch_size": 64,
        "hidden_channels": 32,
        "normalize_features": False,
        "graph_classification_dataset": True,
        "epochs": 200,
        "lr": 0.0005,
        "weight_decay": 1e-4,
    },
    # === Model Overrides (Applied AFTER dataset overrides) ===
    "GAT": {
        "dropout": 0.6,  # GAT paper default dropout
        "lr": 0.005,  # GAT paper default LR for Cora/Citeseer
        "graph_classification_dataset": False,  # Base assumption is node classification
        # Specific GAT overrides based on dataset
        "Cora": {
            "gat_heads": 8,
            "gat_hidden_channels_per_head": 8,
            "gat_output_heads": 1,
            "lr": 0.005,
            "weight_decay": 5e-4,
            "dropout": 0.6,
        },
        "Citeseer": {
            "gat_heads": 8,
            "gat_hidden_channels_per_head": 8,
            "gat_output_heads": 1,
            "lr": 0.005,
            "weight_decay": 5e-4,
            "dropout": 0.6,
        },
        "PubMed": {
            "gat_heads": 8,
            "gat_hidden_channels_per_head": 8,
            "gat_output_heads": 8,
            "gat_concat_output": False,  # Average final heads for PubMed
            "lr": 0.005,
            "weight_decay": 5e-4,
            "dropout": 0.6,
        },
        "PPI": {  # Node classification settings for PPI
            "gat_heads": 4,
            "gat_hidden_channels_per_head": 64,  # -> 256 hidden dim
            "gat_output_heads": 6,  # Multi-label output heads
            "gat_concat_output": False,  # Average final heads
            "lr": 0.005,
            "weight_decay": 0,
            "dropout": 0.0, 
        },
        "Reddit": {
            "batch_size": 512 # Example minibatch for Reddit
        },  
    },
    "ChebNet": {
        "lr": 0.005,  # ChebNet paper used 0.01 Adam
        "optimizer": "Adam",
        "graph_classification_dataset": False,  # Base assumption is node classification
        # K value often depends heavily on dataset
        "Cora": {"cheb_k": 3},
        "Citeseer": {"batch_size": 512}, #Set according to GPU memory available
        "PubMed": {"cheb_k": 3, "batch_size": 512},
        "Reddit": {"cheb_k": 3, "batch_size": 512},  # Example
    },
    "GIN": {
        "lr": 0.005,  # GIN paper setting
        "hidden_channels": 64,  # GIN paper default hidden
        "dropout": 0.5,  # GIN paper dropout
        "graph_classification_dataset": False,  # Assume node classification unless dataset overrides        
        "Reddit": {"batch_size": 128},  # Example minibatch for Reddit
    },
    "DGCNN_SortPool": {  # Graph classification model (Zhang et al., 2018)
        # --- Base settings reflecting the paper's common parameters ---
        "graph_classification_dataset": True,
        "optimizer": "Adam",
        "dropout": 0.5,
        "dgcnn_gcn_layers": 4, # 4 layers (e.g., 32, 32, 32, 1 output channels)
        "dgcnn_conv1d_channels": 16, # First 1D conv has 16 channels
        "dgcnn_conv1d_kernel_size": 5, # Second 1D conv has kernel 5 (assume same for first)
        "batch_size": 32,  # Sensible default for these datasets
        "lr": 0.0001,      # Often requires lower LR
        "weight_decay": 0, 
        "hidden_channels": 32, # uses 32 GCN hidden channels
        "dgcnn_dense_hidden": 128, #  uses 128 dense hidden units

        # --- Dataset-Specific Settings based on Paper ---
        "MUTAG": {
            # hidden_channels=32, dgcnn_dense_hidden=128 (inherited from base DGCNN override)
            # k based on 60% percentile rule. Avg nodes=18. Max=28.
            # Need k >= 14. Estimate 60% percentile gives k around 18-20.
            "dgcnn_sortpool_k": 18,
        },
        "PROTEINS": {
            # hidden_channels=32, dgcnn_dense_hidden=128 (inherited from base DGCNN override)
            # k based on 60% percentile rule. Avg nodes=39. Max=620.
            # Need k >= 14. Estimate 60% percentile gives k around 35-40.
            "dgcnn_sortpool_k": 35, # Adjusted k based on 60% rule & constraint
        },
        "IMDB-BINARY": {
            # hidden_channels=32, dgcnn_dense_hidden=128 (inherited from base DGCNN override)
            # k based on 90% percentile rule (social network). Avg nodes=20. Max=136.
            # Need k >= 14. 90% rule means k should be small (most graphs > k).
            # Let's use the minimum viable k or slightly above.
            "dgcnn_sortpool_k": 15, # Smallest k >= 14, respecting 90% rule estimate
            "lr": 0.00005, # social networks, even lower LR
        },
    },
    "FastGCN": {  # Node classification model
        "graph_classification_dataset": False,
        "lr": 0.001,
        "Citeceer": {"batch_size": 256},
        "Cora": {"batch_size": 256},
        "PubMed": {"batch_size": 256},
        "Reddit": {
            "batch_size": 256,
            "fastgcn_layer_sample_sizes": [
                512,
                512,
            ],  # Larger samples often needed
        },
    },
}


def get_config() -> Dict[str, Any]:
    """
    Parses command-line arguments, loads optional YAML configuration,
    applies hierarchical overrides (Defaults -> YAML -> CLI -> Dataset -> Model -> Model+Dataset),
    performs final adjustments, and returns the final configuration dictionary.
    """
    parser = argparse.ArgumentParser(description="GNN Benchmark Configuration")

    # --- Define Arguments (matching keys in DEFAULTS where applicable) ---
    # General experiment setup
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to YAML config file to override defaults.",
    )
    parser.add_argument(
        "--experiment_name", type=str, help="Name for the experiment run."
    )
    parser.add_argument(
        "--device", type=str, help="Device to use (cuda or cpu)."
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--log_to_wandb",
        action=argparse.BooleanOptionalAction,
        default=None,  # Use None to detect if user explicitly set it via CLI
        help="Log results to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="WandB project name."
    )
    parser.add_argument( # <-- NEW: Argument for wandb_mode
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"], # Restrict choices
        help="W&B run mode (online, offline, disabled)."
    )

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[
            "Cora",
            "Citeseer",
            "PubMed",
            "Reddit",
            "PPI",  # Node
            "MUTAG",
            "PROTEINS",
            "IMDB-BINARY",  # Graph
            # Add other TUDataset names used in OVERRIDES here if needed
        ],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data_dir", type=str, help="Directory for storing datasets."
    )
    parser.add_argument(
        "--normalize_features",
        action=argparse.BooleanOptionalAction,
        default=None,  # Detect explicit user setting via CLI
        help="Whether to normalize features (overrides default/dataset spec).",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "GCN",
            "GAT",
            "GIN",
            "ChebNet",
            "DGCNN_SortPool",
            "FastGCN",
        ],
        help="GNN model to use.",
    )
    parser.add_argument(
        "--hidden_channels", type=int, help="Number of hidden units."
    )
    parser.add_argument("--dropout", type=float, help="Dropout rate.")

    # Training
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, help="Weight decay (L2 regularization)."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "AdamW", "SGD"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--patience", type=int, help="Early stopping patience (0 to disable)."
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size (0 for full-batch)."
    )

    # --- Model Specific Params ---
    # GAT
    parser.add_argument(
        "--gat_heads", type=int, help="Number of attention heads (GAT)."
    )
    parser.add_argument(
        "--gat_hidden_channels_per_head",
        type=int,
        help="Hidden channels per head (GAT layer 1).",
    )
    parser.add_argument(
        "--gat_output_heads",
        type=int,
        help="Number of output attention heads (GAT final layer).",
    )
    parser.add_argument(
        "--gat_concat_output",
        action=argparse.BooleanOptionalAction,
        default=None, # Detect explicit user setting via CLI
        help="Concatenate or average output heads (GAT).",
    )
    parser.add_argument(
        "--gat_negative_slope",
        type=float,
        help="LeakyReLU negative slope (GAT).",
    )
    # ChebNet
    parser.add_argument(
        "--cheb_k", type=int, help="Chebyshev filter order K (ChebNet)."
    )
    # GIN
    parser.add_argument(
        "--gin_mlp_layers", type=int, help="Number of layers in GIN MLP."
    )
    parser.add_argument(
        "--gin_epsilon", type=float, help="Initial epsilon value for GIN."
    )
    parser.add_argument(
        "--gin_train_epsilon",
        action=argparse.BooleanOptionalAction,
        default=None, # Detect explicit user setting via CLI
        help="Train GIN epsilon.",
    )
    # FastGCN
    parser.add_argument(
        "--fastgcn_layer_sample_sizes",
        nargs="+",
        type=int,
        help="Nodes to sample per layer (FastGCN). E.g., 512 512",
    )
    # NeighborLoader/GraphSAGE
    parser.add_argument(
        "--num_neighbors",
        nargs="+",
        type=int,
        help="Num neighbors to sample per layer (NeighborLoader). E.g., 25 10",
    )
    # DGCNN_SortPool
    parser.add_argument(
        "--dgcnn_gcn_layers", type=int, help="Number of GCN layers (DGCNN)."
    )
    parser.add_argument(
        "--dgcnn_sortpool_k", type=int, help="K for SortPooling (DGCNN)."
    )
    parser.add_argument(
        "--dgcnn_conv1d_channels",
        type=int,
        help="Channels for 1D Conv (DGCNN).",
    )
    parser.add_argument(
        "--dgcnn_conv1d_kernel_size",
        type=int,
        help="Kernel size for 1D Conv (DGCNN).",
    )
    parser.add_argument(
        "--dgcnn_dense_hidden", type=int, help="Dense hidden units (DGCNN)."
    )

    # --- Parsing and Merging ---
    args = parser.parse_args()
    config = DEFAULTS.copy()  # Start with base defaults

    # 1. Load from YAML if provided
    if args.config_file:
        if os.path.exists(args.config_file):
            print(f"Loading configuration from YAML file: {args.config_file}")
            try:
                with open(args.config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config.update(yaml_config)  # YAML overrides DEFAULTS
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}. Using previous defaults.")
            except Exception as e:
                print(f"Error reading YAML file: {e}. Using previous defaults.")
        else:
            print(f"Warning: Config file not found at {args.config_file}")

    # 2. Override with command-line arguments
    # Convert Namespace to dict, filtering out None values (unless it's a BooleanOptionalAction
    # where None means 'not specified by user')
    cmd_line_args_dict: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if k == "config_file":
            continue
        # If the argument's default in argparse is None, a value of None means the user didn't provide it.
        # However, for BooleanOptionalAction, the default is often also None, but the user *can*
        # explicitly pass --no-flag (False) or --flag (True). So we keep v if it's not None.
        if v is not None:
            cmd_line_args_dict[k] = v

    config.update(cmd_line_args_dict)  # Command line overrides YAML/DEFAULTS

    # --- Apply Hierarchy Overrides ---
    # Store explicitly set CLI args to prevent them from being overwritten by subsequent overrides
    cli_specified_keys = set(cmd_line_args_dict.keys())

    # 3. Apply Dataset Specific Overrides
    final_dataset_name = config["dataset_name"]
    if final_dataset_name in OVERRIDES:
        print(f"Applying overrides for dataset: {final_dataset_name}")
        dataset_overrides = OVERRIDES[final_dataset_name]
        for key, value in dataset_overrides.items():
            # Apply if key was NOT specified via command line
            if key not in cli_specified_keys:
                config[key] = value
            # Special case: graph_classification_dataset flag is primarily determined by the dataset
            # chosen, even if potentially set differently by user or default.
            if key == "graph_classification_dataset":
                config[key] = value

    # 4. Apply Model Specific Overrides (Base and Dataset-Specific)
    final_model_name = config["model_name"]
    if final_model_name in OVERRIDES:
        print(f"Applying base overrides for model: {final_model_name}")
        model_base_overrides = OVERRIDES[final_model_name]

        # Apply non-nested model defaults first
        for key, value in model_base_overrides.items():
            if not isinstance(value, dict):  # Apply only base overrides here
                # Apply if key was NOT specified via command line
                if key not in cli_specified_keys:
                    config[key] = value
                # Special case: model's intended task type takes precedence over dataset/user setting
                if key == "graph_classification_dataset":
                    config[key] = value  # Model flag takes highest priority

        # Apply nested dataset-specific model overrides
        # These override Dataset overrides and Model Base overrides if the key wasn't set by CLI
        if final_dataset_name in model_base_overrides and isinstance(
            model_base_overrides[final_dataset_name], dict
        ):
            print(
                f"Applying dataset-specific overrides for model {final_model_name} on {final_dataset_name}"
            )
            dataset_specific_model_overrides = model_base_overrides[
                final_dataset_name
            ]
            for key, value in dataset_specific_model_overrides.items():
                # Apply if key was NOT specified via command line
                if key not in cli_specified_keys:
                    config[key] = value

    # --- Final Adjustments ---
    # Ensure boolean flags that might still be None (if not set by CLI or overrides) fall back to default
    if config.get("log_to_wandb") is None:
        config["log_to_wandb"] = DEFAULTS["log_to_wandb"]
    if config.get("normalize_features") is None:
        config["normalize_features"] = DEFAULTS["normalize_features"]
    if config.get("gin_train_epsilon") is None:
        config["gin_train_epsilon"] = DEFAULTS["gin_train_epsilon"]
    if config.get("gat_concat_output") is None:
        config["gat_concat_output"] = DEFAULTS["gat_concat_output"]
    
    # Coordinate wandb_mode and log_to_wandb
    if config.get("wandb_mode") == "disabled":
        config["log_to_wandb"] = False
    elif config.get("log_to_wandb") is False: # Handle explicit --no-log_to_wandb
        config["wandb_mode"] = "disabled"

    # Ensure batch_size > 0 for graph classification tasks or models that require minibatching
    is_graph_task = config.get("graph_classification_dataset", False)
    requires_minibatch = (
        final_model_name == "FastGCN"
        or final_dataset_name == "Reddit"
        or config['dataset_name'] == 'PPI' # PPI also uses minibatch DataLoader
    )

    current_batch_size = config.get("batch_size", 0)
    if is_graph_task and current_batch_size <= 0:
        # Find a sensible default batch size for graph tasks
        model_default_bs = OVERRIDES.get(final_model_name, {}).get(
            "batch_size", 32
        )
        dataset_default_bs = OVERRIDES.get(final_dataset_name, {}).get(
            "batch_size", 32
        )
        final_bs = (
            model_default_bs
            if model_default_bs > 0
            else dataset_default_bs
            if dataset_default_bs > 0
            else 32
        ) # Fallback
        print(
            f"Warning: Graph classification task ({final_dataset_name}/{final_model_name}) "
            f"requires batch_size > 0. Setting batch_size to {final_bs}."
        )
        config["batch_size"] = final_bs
    elif requires_minibatch and current_batch_size <= 0:
        # Find a sensible default batch size for models/datasets requiring minibatching
        model_default_bs = OVERRIDES.get(final_model_name, {}).get(
            "batch_size", 128
        )
        dataset_default_bs = OVERRIDES.get(final_dataset_name, {}).get(
            "batch_size", 128
        )
        final_bs = (
             model_default_bs
            if model_default_bs > 0
            else dataset_default_bs
            if dataset_default_bs > 0
            else 128 # Fallback
        )
        print(
            f"Warning: Model/Dataset ({final_dataset_name}/{final_model_name}) "
            f"requires batch_size > 0 for minibatching. Setting batch_size to {final_bs}."
        )
        config["batch_size"] = final_bs


    # --- Print Final Configuration ---
    print("\n--- Final Configuration ---")
    # Sort config items alphabetically for consistent and readable printing
    for key, value in sorted(config.items()):
        print(f"{key}: {value}")
    print("---------------------------\n")

    return config