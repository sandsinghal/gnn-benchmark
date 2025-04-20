# data/load_data.py
import os
from typing import Optional

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI, Planetoid, Reddit, TUDataset
from torch_geometric.data import Dataset


def load_dataset(
    name: str,
    path: str,
    normalize: bool = True,
    use_node_attr: bool = True, # Default to True for TUDataset
) -> Dataset:
    """
    Loads the specified PyG dataset, handling node and graph classification types.

    Args:
        name (str): Name of the dataset (e.g., 'Cora', 'MUTAG').
        path (str): Root directory to store/find the dataset.
        normalize (bool): Whether to apply feature normalization (T.NormalizeFeatures).
                          Often disabled for specific datasets like Reddit or some TUDatasets.
        use_node_attr (bool): For TUDatasets, whether to use existing node attributes as features.
                             If False or attributes are missing, degree features are added.

    Returns:
        Dataset: The loaded PyG dataset object.
    """
    # Create a specific subdirectory for each dataset
    dataset_path = os.path.join(path, name)
    os.makedirs(dataset_path, exist_ok=True)

    transform: Optional[T.BaseTransform] = T.NormalizeFeatures() if normalize else None
    pre_transform: Optional[T.BaseTransform] = None # Can be used for TUDataset feature generation
    is_graph_dataset = False  # Flag to track dataset type

    print(f"\nLoading dataset: {name} from {dataset_path}...")
    if name in ["Cora", "Citeseer", "PubMed"]:
        # Node classification: Planetoid datasets (single graph with masks)
        dataset = Planetoid(root=dataset_path, name=name, transform=transform)
        if normalize and transform is None: print("Normalization disabled by config for Planetoid.")
    elif name == "Reddit":
        # Node classification: Reddit (single graph, large, needs minibatching)
        dataset = Reddit(root=dataset_path)
        # Reddit features are usually high-dimensional and sparse, normalization often omitted.
        if normalize: print("Normalization explicitly disabled for Reddit.")
        transform = None # Override config setting
    elif name == "PPI":
        # Node classification: PPI (multiple graphs, needs minibatching via DataLoader)
        # Load only the 'train' split here; 'val' and 'test' loaded separately in main.py
        dataset = PPI(root=dataset_path, split="train")
        # PPI features are often used as-is.
        if normalize: print("Normalization explicitly disabled for PPI.")
        transform = None # Override config setting
    elif name in [
        "MUTAG", "PROTEINS", "NCI1", "PTC_MR", "IMDB-BINARY", "IMDB-MULTI",
        "COLLAB", "REDDIT-BINARY", "REDDIT-MULTI-5K",
        # Add other TUDataset names here if needed
    ]:
        # Graph classification: TUDatasets
        is_graph_dataset = True
        print(f"Attempting to load TUDataset '{name}' with use_node_attr={use_node_attr}.")
        dataset = TUDataset(
            root=dataset_path, name=name, use_node_attr=use_node_attr, pre_transform=pre_transform
        )

        # Handle missing node features in TUDatasets by adding degree features
        # This check runs after loading, before returning the dataset object
        if not use_node_attr or (hasattr(dataset.data, 'x') and dataset.data.x is None):
            if not use_node_attr: print(f"use_node_attr is False.")
            if hasattr(dataset.data, 'x') and dataset.data.x is None: print(f"Dataset '{name}' has no inherent node features.")

            print("Adding node degree features...")
            max_degree = 0
            degrees = []
            for g in dataset:
                deg = T.Degree(max_degree=None)(g).x # Calculate degree for each graph
                degrees.append(deg.max().item())
            max_degree = max(degrees) if degrees else 0

            # Use OneHotDegree if max degree is manageable, otherwise use raw Degree
            if max_degree < 1000: # Threshold for one-hot encoding dimension
                print(f"Using OneHotDegree features (max_degree={max_degree}).")
                dataset.transform = T.OneHotDegree(max_degree=max_degree)
            else:
                print(f"Using raw Degree features (max_degree={max_degree} is large).")
                dataset.transform = T.Degree(max_degree=None) # Apply transform to add degree on the fly

            # Override normalization transform if degree features were just added
            transform = None # Don't normalize degree features by default

        elif transform is not None:
             # Normalization is applied only if features exist and normalize=True
             print(f"Applying feature normalization to existing node features.")
             dataset.transform = transform # Apply normalization on the fly
             transform = None # Prevent double application msg
        else:
             # Features exist, but normalization is disabled
             transform = None # Ensure transform message isn't printed

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    print(f"Dataset '{name}' loaded successfully.")
    if transform:
        print("Applied T.NormalizeFeatures transform during loading.") # Should only apply to Planetoid now

    # Print dataset statistics
    if is_graph_dataset:
        print(f"Dataset Type: Graph Classification")
        print(f"  Number of graphs: {len(dataset)}")
        print(f"  Number of classes: {dataset.num_classes}")
        # Use dataset.num_node_features which accounts for added degree features
        print(f"  Number of node features: {dataset.num_node_features}")
        if hasattr(dataset, "num_edge_features") and dataset.num_edge_features > 0:
            print(f"  Number of edge features: {dataset.num_edge_features}")
    else: # Node classification dataset stats
        print(f"Dataset Type: Node Classification")
        if name == "PPI":
            # PPI train split loaded here
            print(f"  Number of graphs (train split): {len(dataset)}")
            print(f"  Number of node features: {dataset.num_features}")
            print(f"  Number of classes (multi-label): {dataset.num_classes}")
        else:
            # Single graph datasets (Planetoid, Reddit)
            data = dataset[0]
            print(f"  Number of graphs: {len(dataset)}")
            print(f"  Number of nodes: {data.num_nodes:,}")
            print(f"  Number of edges: {data.num_edges:,}")
            print(f"  Number of node features: {dataset.num_node_features}")
            print(f"  Number of classes: {dataset.num_classes}")
            print(f"  Graph object keys: {data.keys}")
            if hasattr(data, "train_mask"):
                print(f"  Train nodes: {data.train_mask.sum().item():,}")
            if hasattr(data, "val_mask"):
                print(f"  Val nodes:   {data.val_mask.sum().item():,}")
            if hasattr(data, "test_mask"):
                print(f"  Test nodes:  {data.test_mask.sum().item():,}")

    # Store graph task flag in dataset object for convenience
    if not hasattr(dataset, "is_graph_task"):
        dataset.is_graph_task = is_graph_dataset # type: ignore

    return dataset
