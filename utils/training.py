# utils/training.py
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader, NeighborLoader
from tqdm import tqdm

from .sampling import FastGcnSampler # Required for type hinting


# === Full-Batch Training & Evaluation (for single, small graphs) ===

def train_full_batch(
    model: Module, data: Data, optimizer: Optimizer, criterion: Module
) -> float:
    """Performs one training epoch for full-batch datasets (e.g., Planetoid)."""
    model.train()
    optimizer.zero_grad()

    # Ensure data is on the correct device (should be done in main.py)
    # Pass x, edge_index, and optional edge_weight to the model
    out = model(data.x, data.edge_index, getattr(data, "edge_weight", None))

    # Use the training mask to select nodes for loss calculation
    mask = data.train_mask
    if mask.sum() == 0:
         print("Warning: No training nodes found in train_mask.")
         return 0.0

    # Handle multi-label vs single-label targets for loss calculation
    target_y = data.y[mask]
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        target_y = target_y.float() # BCE requires float targets

    loss = criterion(out[mask], target_y)

    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: NaN/Inf loss detected during training. Skipping backward pass.")
        return 0.0 # Or handle differently

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_full_batch(
    model: Module, data: Data, criterion: Module, is_multilabel: bool = False
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Evaluates model performance on full-batch datasets (train/val/test splits).

    Args:
        model: The GNN model.
        data: The PyG Data object containing the full graph and masks.
        criterion: The loss function.
        is_multilabel: Boolean flag indicating if the task is multi-label.

    Returns:
        A dictionary containing loss, accuracy (or None), f1_macro (or None),
        and f1_micro (or None) for each split ('train', 'val', 'test').
    """
    model.eval()
    # Ensure data is on the correct device (should be done in main.py)
    out = model(data.x, data.edge_index, getattr(data, "edge_weight", None))

    # Initialize metrics dictionary
    metrics: Dict[str, Dict[str, Optional[float]]] = {
        "loss": {},
        "acc": {},      # Accuracy (usually for multi-class)
        "f1_macro": {}, # Macro F1 (usually for multi-class)
        "f1_micro": {}, # Micro F1 (usually for multi-label)
    }

    for split in ["train", "val", "test"]:
        mask = getattr(data, f"{split}_mask", None)

        # Skip split if mask doesn't exist or is empty
        if mask is None or mask.sum() == 0:
            metrics["loss"][split] = float("nan")
            metrics["acc"][split] = None
            metrics["f1_macro"][split] = None
            metrics["f1_micro"][split] = None
            continue

        # --- Calculate Loss ---
        try:
            target_y_loss = data.y[mask]
            if is_multilabel: target_y_loss = target_y_loss.float()
            loss = criterion(out[mask], target_y_loss)
            metrics["loss"][split] = loss.item()
        except Exception as e:
            # print(f"Warning: Could not compute loss for split '{split}'. Error: {e}")
            metrics["loss"][split] = float("nan")

        # --- Calculate Accuracy / F1 ---
        y_true_cpu = data.y[mask].cpu() # Move labels to CPU for sklearn metrics

        if is_multilabel:
            # Get predictions using sigmoid and threshold
            preds_cpu = (torch.sigmoid(out[mask]) > 0.5).int().cpu()
            # Micro F1 is standard for multi-label node classification
            metrics["f1_micro"][split] = f1_score(
                y_true_cpu, preds_cpu, average="micro", zero_division=0
            )
            # Accuracy and Macro F1 are less standard/meaningful here
            metrics["acc"][split] = None
            metrics["f1_macro"][split] = None
        else: # Multi-class classification
            # Get predictions using argmax
            preds_cpu = out[mask].argmax(dim=1).cpu()
            # Accuracy and Macro F1 are standard
            metrics["acc"][split] = accuracy_score(y_true_cpu, preds_cpu)
            metrics["f1_macro"][split] = f1_score(
                y_true_cpu, preds_cpu, average="macro", zero_division=0
            )
             # Micro F1 usually equals accuracy in multi-class single-label cases
            metrics["f1_micro"][split] = None


    return metrics


# === Mini-Batch Training & Evaluation (NeighborLoader / DataLoader) ===

def train_minibatch(
    model: Module,
    loader: Union[DataLoader, NeighborLoader],
    optimizer: Optimizer,
    criterion: Module,
    device: torch.device,
    is_multilabel: bool = False,
    is_graph_classification: bool = False,
    desc: str = "Train Minibatch",
) -> float:
    """Performs one training epoch for minibatch datasets (Node or Graph classification)."""
    model.train()
    total_loss = 0.0
    num_processed_items = 0 # Can be nodes or graphs depending on loader type

    for batch in tqdm(loader, desc=desc, leave=False):
        # Skip potentially empty batches, especially from NeighborLoader if masks are sparse
        if (not is_graph_classification and hasattr(batch, 'batch_size') and \
           (batch.x is None or batch.x.shape[0] == 0 or batch.batch_size == 0)) or \
           (is_graph_classification and (batch.x is None or batch.x.shape[0] == 0)):
            # print("Warning: Skipping empty batch.")
            continue

        batch = batch.to(device)
        optimizer.zero_grad()

        edge_weight = getattr(batch, "edge_weight", None)

        # --- Model Forward Pass ---
        # Adapt model call based on task and expected signature
        try:
            if is_graph_classification:
                # Graph models (like DGCNN) might need the `batch` vector
                # Assumes models accept x, edge_index, batch, [edge_weight]
                out = model(batch.x, batch.edge_index, batch.batch, edge_weight=edge_weight)
                target_out = out # Graph output corresponds to graph labels
                target_y = batch.y
                loss_count = batch.num_graphs # Loss is calculated per graph
            else: # Node classification (NeighborLoader or PPI DataLoader)
                # Node models expect x, edge_index, [edge_weight]
                out = model(batch.x, batch.edge_index, edge_weight=edge_weight)
                # For NeighborLoader, predictions/labels correspond to the first `batch_size` nodes
                if hasattr(batch, 'batch_size'):
                    target_out = out[:batch.batch_size]
                    target_y = batch.y[:batch.batch_size]
                    loss_count = batch.batch_size # Loss per node in the batch center
                else: # Assume DataLoader for PPI node classification - use all nodes
                    target_out = out
                    target_y = batch.y
                    loss_count = batch.num_nodes # Loss per node in the graph(s)
        except TypeError as e:
             print(f"Error during model forward pass: {e}. Check model signature.")
             print(f"Input shapes: x={batch.x.shape}, edge_index={batch.edge_index.shape}, batch={batch.batch.shape if batch.batch else None}")
             continue # Skip batch on error
        except Exception as e:
             print(f"Unexpected error during model forward pass: {e}")
             continue

        # --- Loss Calculation ---
        if target_y is None or target_out.shape[0] != target_y.shape[0]:
             print(f"Warning: Skipping batch due to label mismatch or missing labels. Out shape: {target_out.shape}, Label shape: {target_y.shape if target_y is not None else 'None'}")
             continue

        if is_multilabel:
            loss = criterion(target_out, target_y.float()) # BCE requires float targets
        else:
            loss = criterion(target_out, target_y) # CrossEntropy expects long targets

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss detected. Skipping batch backward pass.")
            continue # Skip update if loss is invalid

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * loss_count
        num_processed_items += loss_count

    # Return average loss over all processed items (nodes or graphs)
    return total_loss / num_processed_items if num_processed_items > 0 else 0.0


@torch.no_grad()
def evaluate_minibatch(
    model: Module,
    loader: Union[DataLoader, NeighborLoader],
    criterion: Module,
    device: torch.device,
    is_multilabel: bool = False,
    is_graph_classification: bool = False,
    desc="Evaluate Minibatch",
) -> Dict[str, Optional[float]]:
    """Evaluates model performance on minibatch datasets (Node or Graph classification)."""
    model.eval()
    all_preds: List[Tensor] = []
    all_labels: List[Tensor] = []
    total_loss = 0.0
    num_processed_items = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        # Skip empty batches
        if (not is_graph_classification and hasattr(batch, 'batch_size') and \
           (batch.x is None or batch.x.shape[0] == 0 or batch.batch_size == 0)) or \
           (is_graph_classification and (batch.x is None or batch.x.shape[0] == 0)):
            continue

        batch = batch.to(device)
        edge_weight = getattr(batch, "edge_weight", None)

        # --- Model Forward Pass ---
        try:
            if is_graph_classification:
                out = model(batch.x, batch.edge_index, batch.batch, edge_weight=edge_weight)
                target_out = out
                target_y = batch.y
                count = batch.num_graphs
            else: # Node classification
                out = model(batch.x, batch.edge_index, edge_weight=edge_weight)
                if hasattr(batch, 'batch_size'): # NeighborLoader
                    target_out = out[:batch.batch_size]
                    target_y = batch.y[:batch.batch_size]
                    count = batch.batch_size
                else: # PPI DataLoader
                    target_out = out
                    target_y = batch.y
                    count = batch.num_nodes
        except TypeError as e:
             print(f"Error during model evaluation forward pass: {e}. Check model signature.")
             continue
        except Exception as e:
             print(f"Unexpected error during model evaluation forward pass: {e}")
             continue

        # --- Loss Calculation ---
        if target_y is None or target_out.shape[0] != target_y.shape[0]:
             print(f"Warning: Skipping eval batch due to label mismatch. Out: {target_out.shape}, Label: {target_y.shape if target_y is not None else 'None'}")
             continue

        try:
            loss = criterion(target_out, target_y.float() if is_multilabel else target_y)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * count
                num_processed_items += count
            else:
                loss = torch.tensor(float("nan")) # Mark invalid loss
        except Exception:
            loss = torch.tensor(float("nan")) # Mark invalid loss on error


        # --- Store Predictions and Labels (on CPU) ---
        if is_multilabel:
            preds = (torch.sigmoid(target_out) > 0.5).int().cpu()
        else: # Multi-class (node or graph)
            preds = target_out.argmax(dim=1).cpu()

        labels = target_y.cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    avg_loss = total_loss / num_processed_items if num_processed_items > 0 else float("nan")

    # Handle case where no batches were processed
    if not all_preds:
        print("Warning: No predictions generated during evaluation (loader might be empty).")
        return {"loss": avg_loss, "acc": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}

    # Concatenate all batch results
    final_preds = torch.cat(all_preds, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    # --- Calculate Final Metrics ---
    metrics: Dict[str, Optional[float]] = {"loss": avg_loss}
    if is_multilabel: # Usually PPI Node classification
        metrics["f1_micro"] = f1_score(
            final_labels, final_preds, average="micro", zero_division=0
        )
        metrics["acc"] = None # Accuracy less meaningful
        metrics["f1_macro"] = None # Macro F1 less standard
    else: # Multi-class Node or Graph classification
        metrics["acc"] = accuracy_score(final_labels, final_preds)
        metrics["f1_macro"] = f1_score(
            final_labels, final_preds, average="macro", zero_division=0
        )
        metrics["f1_micro"] = None # Micro F1 usually equals accuracy

    return metrics


# === FastGCN Specific Training & Evaluation ===

def train_fastgcn(
    model: Module,
    loader: FastGcnSampler, # Expects the custom sampler
    optimizer: Optimizer,
    criterion: Module,
    device: torch.device,
    desc="Train FastGCN",
) -> float:
    """Performs one training epoch using FastGCN layer-wise sampling."""
    model.train()
    total_loss = 0.0
    num_processed_nodes = 0

    # --- Access Convolutional Layers ---
    # Find the convolutional layers within the model (assumes a common structure)
    model_convs: List[Module] = []
    if hasattr(model, 'convs') and isinstance(model.convs, torch.nn.ModuleList):
         model_convs = list(model.convs)
    elif hasattr(model, 'conv1'): # Fallback for 2-layer manual definition
         model_convs.append(model.conv1)
         if hasattr(model, 'conv2'):
              model_convs.append(model.conv2)
    else:
         raise AttributeError("Cannot find convolutional layers (e.g., 'conv1', 'conv2', or 'convs') in the FastGCN model.")

    num_layers = len(model_convs)
    model_dropout = getattr(model, 'dropout_p', getattr(model, 'dropout', 0.0)) # Get dropout rate

    # Iterate through batches provided by the FastGCN sampler
    # Each batch contains layer-wise sampled node features and subgraph structures
    for layerwise_batch_data in tqdm(loader, desc=desc, leave=False):
        # layerwise_batch_data is a list of dictionaries, one per layer + final labels
        # Example: [ {'x': L0_x, 'edge_index': L0_adj, 'edge_weight': L0_w, 'nodes': L0_nodes_orig_idx}, # Layer 0 input
        #            {'x': L1_x, 'edge_index': L1_adj, 'edge_weight': L1_w, 'nodes': L1_nodes_orig_idx}, # Layer 1 input
        #            ...
        #            {'nodes': Final_nodes_orig_idx, 'y': Final_y} ] # Final layer output nodes & labels

        optimizer.zero_grad()

        # --- Layer-wise Forward Pass ---
        # Start with features of nodes sampled for the *first* layer's input
        layer0_info = layerwise_batch_data[0]
        h = layer0_info['x'].to(device, non_blocking=True).float() # Ensure float features

        # The nodes corresponding to the *final* output of the network
        # are the 'nodes' from the *last dictionary* in the list.
        # These are the nodes for which we calculate the loss.
        final_output_nodes_info = layerwise_batch_data[-1]
        nodes_for_loss_orig_idx = final_output_nodes_info['nodes'] # Original indices

        # Propagate through layers
        for l in range(num_layers):
            # Get the pre-sampled subgraph structure for this layer
            # Note: The 'x' in layer_info[l] is the *input* features for conv layer 'l'.
            # The edge_index/weight are for the subgraph connecting these input nodes.
            current_layer_subgraph_info = layerwise_batch_data[l]
            current_edge_index = current_layer_subgraph_info['edge_index'].to(device, non_blocking=True)
            # Ensure edge_weight is float
            current_edge_weight = current_layer_subgraph_info['edge_weight'].to(device, non_blocking=True).float()

            # Apply the l-th GCN layer
            conv_layer = model_convs[l]
            h = conv_layer(h, current_edge_index, current_edge_weight)

            # Apply activation and dropout (except after the last layer)
            if l < num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=model_dropout, training=True)

            # For the *next* iteration (l+1), the input features 'h' must correspond
            # to the nodes sampled as input for that *next* layer.
            # We need to select the rows of 'h' that correspond to layer_info[l+1]['nodes']
            # This requires careful mapping between node indices across layers.
            # The current FastGcnSampler implementation structures the data such that:
            # - layer_info[l]['x'] are the features for nodes needed as input to conv[l]
            # - h = conv[l](...) are the output features for these same nodes.
            # - layer_info[l+1]['x'] should be derived by *selecting* the correct rows from 'h'
            #   based on the node indices provided in layer_info[l+1]['nodes'] relative to layer_info[l]['nodes'].
            # --> The provided `FastGcnSampler._create_layerwise_batch` seems to handle this by providing
            #     the correctly subsetted 'x' for each layer dictionary already.
            #     So, we just need to update 'h' for the next layer's input.
            if l < num_layers - 1:
                 next_layer_input_info = layerwise_batch_data[l+1]
                 h = next_layer_input_info['x'].to(device, non_blocking=True).float() # Use the pre-sampled features


        # Final output `out = h` corresponds to nodes `nodes_for_loss_orig_idx`
        out = h

        # --- Loss Calculation ---
        # Fetch the ground truth labels for the final output nodes
        target_y = final_output_nodes_info['y'].to(device, non_blocking=True)

        # Check for size consistency
        if out.shape[0] != target_y.shape[0]:
            print(f"FATAL ERROR in train_fastgcn: Output size ({out.shape[0]}) mismatch "
                  f"with target size ({target_y.shape[0]}). Skipping batch.")
            continue

        # Handle multi-label vs single-label loss
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            loss = criterion(out, target_y.float())
        else:
            loss = criterion(out, target_y)

        if torch.isnan(loss) or torch.isinf(loss):
             print("Warning: NaN/Inf loss detected in FastGCN. Skipping batch backward pass.")
             continue

        loss.backward()
        optimizer.step()

        batch_node_count = target_y.shape[0] # Loss calculated over final sampled nodes
        total_loss += loss.item() * batch_node_count
        num_processed_nodes += batch_node_count

    return total_loss / num_processed_nodes if num_processed_nodes > 0 else 0.0


@torch.no_grad()
def evaluate_fastgcn(
    model: Module,
    full_data_cpu: Data, # Requires full graph data on CPU
    criterion: Module,
    device: torch.device,
    is_multilabel: bool = False,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Evaluates FastGCN model using standard full-graph inference.
    (FastGCN paper uses full graph for evaluation).

    Args:
        model: The FastGCN model (structurally same as GCN).
        full_data_cpu: The full PyG Data object (on CPU).
        criterion: The loss function.
        device: The device to perform evaluation on.
        is_multilabel: Boolean flag for multi-label tasks.

    Returns:
        Metrics dictionary similar to evaluate_full_batch.
    """
    model.eval()
    print("Evaluating FastGCN using full graph inference...")
    # Move data to the target device for evaluation
    try:
        full_data_gpu = full_data_cpu.to(device)
    except Exception as e:
        print(f"Error moving full data to {device} for FastGCN evaluation: {e}")
        # Return dummy metrics or raise error
        return {
            "loss": {"train": float("nan"), "val": float("nan"), "test": float("nan")},
            "acc": {"train": None, "val": None, "test": None},
            "f1_macro": {"train": None, "val": None, "test": None},
            "f1_micro": {"train": None, "val": None, "test": None},
        }

    # Use the standard full-batch evaluation function
    return evaluate_full_batch(model, full_data_gpu, criterion, is_multilabel)