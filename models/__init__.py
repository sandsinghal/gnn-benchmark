# models/__init__.py
from typing import Any, Dict

import torch
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data.batch import Batch

from .chebnet import ChebNet
from .dgcnn_sortpool import DGCNN_SortPool # Graph classification
from .fastgcn import FastGCN # Node classification
from .gat import GAT # Node classification
from .gcn import GCN # Node classification
from .gin import GIN # Node classification

__all__ = [
    "GCN",
    "GAT",
    "GIN",
    "ChebNet",
    "DGCNN_SortPool",
    "FastGCN",
    "get_model",
]


def get_model(config: Dict[str, Any], dataset: PyGDataset) -> torch.nn.Module:
    """
    Factory function to instantiate a GNN model based on configuration and dataset properties.

    Args:
        config (dict): Dictionary containing the experiment configuration.
        dataset (Dataset): The loaded PyG dataset object. Used to determine
                           input/output dimensions. Can be single graph or multi-graph.

    Returns:
        torch.nn.Module: The instantiated GNN model.

    Raises:
        ValueError: If model name is unknown or dataset properties cannot be determined.
    """
    model_name: str = config["model_name"]
    num_features: int = -1
    num_classes: int = -1
    is_graph_task: bool = config.get("graph_classification_dataset", False) # Get task type from config

    # --- Determine Input Features and Output Classes ---
    # Need to handle different ways features/classes are stored in PyG datasets
    try:
        if hasattr(dataset, "num_node_features"):
            num_features = dataset.num_node_features
        elif hasattr(dataset, "num_features"): # Some datasets use this
            num_features = dataset.num_features
        elif hasattr(dataset.data, "num_node_features"): # Check inside data object
             num_features = dataset.data.num_node_features
        elif hasattr(dataset.get(0), "num_node_features"): # Check first graph element
             num_features = dataset.get(0).num_node_features
        else:
            # Last resort: infer from x shape if possible
            example_data = dataset.get(0) if isinstance(dataset, list) or isinstance(dataset, PyGDataset) else dataset.data
            if hasattr(example_data, 'x') and example_data.x is not None:
                 num_features = example_data.x.shape[1]
            else:
                 raise ValueError("Cannot determine number of node features from dataset.")

        if hasattr(dataset, "num_classes"):
            num_classes = dataset.num_classes
        elif hasattr(dataset.data, "num_classes"): # Check inside data object
             num_classes = dataset.data.num_classes
        elif hasattr(dataset.get(0), "num_classes"): # Check first graph element
             num_classes = dataset.get(0).num_classes
        else:
            # Infer from labels if possible
            example_data = dataset.get(0) if isinstance(dataset, list) or isinstance(dataset, PyGDataset) else dataset.data
            if hasattr(example_data, 'y') and example_data.y is not None:
                y = example_data.y
                if y.dim() == 1: # Single label per node/graph
                    num_classes = int(y.max().item()) + 1
                elif y.dim() == 2: # Multi-label or one-hot
                     num_classes = y.shape[1]
                else:
                     raise ValueError("Cannot determine number of classes from label shape.")
            else:
                raise ValueError("Cannot determine number of classes from dataset.")

    except Exception as e:
        print(f"Error accessing dataset properties: {e}")
        raise ValueError(
            "Could not determine num_features or num_classes from the dataset object."
        )

    if num_features <= 0:
         raise ValueError(f"Determined num_features is invalid: {num_features}")
    if num_classes <= 0:
         raise ValueError(f"Determined num_classes is invalid: {num_classes}")


    task_type = "Graph" if is_graph_task else "Node"
    print(f"\nInstantiating model: {model_name} for {task_type} classification task")
    print(f"  Input features: {num_features}, Output classes: {num_classes}")
    print(f"  Hidden channels: {config['hidden_channels']}, Dropout: {config['dropout']}")

    # --- Model Instantiation ---
    if model_name == "GCN":
        model = GCN(
            in_channels=num_features,
            hidden_channels=config["hidden_channels"],
            out_channels=num_classes,
            dropout=config["dropout"],
        )
    elif model_name == "GAT":
        print(f"  GAT Heads: {config['gat_heads']}, Hidden/Head: {config['gat_hidden_channels_per_head']}, Output Heads: {config['gat_output_heads']}")
        model = GAT(
            in_channels=num_features,
            hidden_channels_per_head=config["gat_hidden_channels_per_head"],
            out_channels=num_classes,
            heads=config["gat_heads"],
            dropout=config["dropout"],
            output_heads=config["gat_output_heads"],
            concat_output=config["gat_concat_output"],
            negative_slope=config["gat_negative_slope"],
        )
    elif model_name == "GIN":
         print(f"  GIN MLP Layers: {config['gin_mlp_layers']}, Train Epsilon: {config['gin_train_epsilon']}")
         model = GIN(
            in_channels=num_features,
            hidden_channels=config["hidden_channels"],
            out_channels=num_classes,
            dropout=config["dropout"],
            num_mlp_layers=config["gin_mlp_layers"],
            eps=config["gin_epsilon"],
            train_eps=config["gin_train_epsilon"],
        )
    elif model_name == "ChebNet":
        print(f"  ChebNet K: {config['cheb_k']}")
        model = ChebNet(
            in_channels=num_features,
            hidden_channels=config["hidden_channels"],
            out_channels=num_classes,
            dropout=config["dropout"],
            K=config["cheb_k"],
        )
    elif model_name == "FastGCN":
         # FastGCN uses GCN structure, sampling handled separately
         model = FastGCN(
            in_channels=num_features,
            hidden_channels=config["hidden_channels"],
            out_channels=num_classes,
            dropout=config["dropout"],
        )
    elif model_name == "DGCNN_SortPool":
        if not is_graph_task:
             print(f"Warning: DGCNN_SortPool is intended for graph classification, but task type is Node.")
        print(f"  DGCNN GCN Layers: {config['dgcnn_gcn_layers']}, SortPool K: {config['dgcnn_sortpool_k']}")
        print(f"  DGCNN Conv1D Channels: {config['dgcnn_conv1d_channels']}, Dense Hidden: {config['dgcnn_dense_hidden']}")
        model = DGCNN_SortPool(
            in_channels=num_features,
            hidden_channels=config["hidden_channels"], # Hidden for GCN layers
            out_channels=num_classes,
            dropout=config["dropout"],
            num_gcn_layers=config["dgcnn_gcn_layers"],
            sortpool_k=config["dgcnn_sortpool_k"],
            conv1d_channels=config["dgcnn_conv1d_channels"],
            conv1d_kernel_size=config["dgcnn_conv1d_kernel_size"],
            dense_hidden_units=config["dgcnn_dense_hidden"],
        )
    else:
        raise ValueError(f"Model '{model_name}' instantiation not implemented.")

    return model