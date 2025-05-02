# utils/training.py
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm # For progress bars
import warnings
import sys

# ==============================================================================
#                    Full-Batch Training & Evaluation
# ==============================================================================
# Suitable for datasets that fit entirely into memory (e.g., Cora, CiteSeer)

def train_full_batch(model: torch.nn.Module, data: 'torch_geometric.data.Data',
                     optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
    """
    Performs one training epoch on a full-batch dataset.

    Args:
        model (torch.nn.Module): The GNN model.
        data (Data): The graph data object containing features, labels, edge_index, and masks.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.

    Returns:
        float: The training loss for the epoch.
    """
    model.train() # Set model to training mode
    optimizer.zero_grad() # Clear gradients

    # Perform forward pass using the full graph data
    # Assumes model.forward signature is (x, edge_index, edge_weight=None)
    out = model(data.x, data.edge_index, getattr(data, 'edge_weight', None))

    # Apply the training mask to select nodes for loss calculation
    mask = data.train_mask
    # Calculate loss only on training nodes
    loss = criterion(out[mask], data.y[mask])

    # Backpropagate gradients and update model weights
    loss.backward()
    optimizer.step()

    return loss.item() # Return the scalar loss value


@torch.no_grad() # Disable gradient calculations for evaluation
def evaluate_full_batch(model: torch.nn.Module, data: 'torch_geometric.data.Data',
                        criterion: torch.nn.Module, is_multilabel: bool = False) -> dict:
    """
    Evaluates model performance on full-batch datasets for train, validation, and test splits.

    Calculates loss, accuracy (for multi-class), and F1 scores (macro for multi-class,
    micro for multi-label).

    Args:
        model (torch.nn.Module): The GNN model.
        data (Data): The graph data object.
        criterion (torch.nn.Module): The loss function.
        is_multilabel (bool): Flag indicating if the task is multi-label classification.
                               Defaults to False (multi-class).

    Returns:
        dict: A dictionary containing evaluation metrics ('loss', 'acc', 'f1_macro', 'f1_micro')
              for each split ('train', 'val', 'test'). Metrics might be NaN if a split is missing
              or empty, or None if not applicable (e.g., acc for multi-label).
    """
    model.eval() # Set model to evaluation mode

    # Perform forward pass on the full graph
    out = model(data.x, data.edge_index, getattr(data, 'edge_weight', None))

    # Initialize dictionary to store metrics
    metrics = {'loss': {}, 'acc': {}, 'f1_macro': {}, 'f1_micro': {}}

    # Evaluate performance on each data split (train, validation, test)
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask', None) # Get the mask for the current split

        # Check if the mask exists and contains any nodes
        if mask is None or mask.sum() == 0:
            # Assign NaN to metrics if split is invalid or empty
            metrics['loss'][split] = float('nan')
            metrics['acc'][split] = float('nan')
            metrics['f1_macro'][split] = float('nan')
            metrics['f1_micro'][split] = float('nan')
            continue # Skip to the next split

        # Get ground truth labels for the current split (move to CPU for metrics calculation)
        y_true = data.y[mask].cpu()

        # --- Calculate Loss ---
        try:
             # For multi-label tasks using BCEWithLogitsLoss, target labels need to be float
             target_y_loss = data.y[mask].float() if is_multilabel else data.y[mask]
             loss = criterion(out[mask], target_y_loss)
             metrics['loss'][split] = loss.item()
        except Exception as e:
             # warnings.warn(f"Could not compute loss for split '{split}'. Error: {e}")
             metrics['loss'][split] = float('nan') # Indicate loss calculation failure

        # --- Calculate Accuracy / F1 Score ---
        if is_multilabel:
            # Multi-label: Apply sigmoid, threshold at 0.5, compute F1-micro
            preds = (torch.sigmoid(out[mask]) > 0.5).int().cpu()
            # Accuracy and F1-macro are typically not standard for multi-label node classification
            metrics['acc'][split] = None
            metrics['f1_macro'][split] = None
            # F1-micro is common for multi-label tasks
            metrics['f1_micro'][split] = f1_score(y_true, preds, average='micro', zero_division=0)
        else:
            # Multi-class: Apply argmax, compute accuracy and F1-macro
            preds = out[mask].argmax(dim=1).cpu()
            metrics['acc'][split] = accuracy_score(y_true, preds)
            metrics['f1_macro'][split] = f1_score(y_true, preds, average='macro', zero_division=0)
            # F1-micro is less common/interpretable for multi-class node classification
            metrics['f1_micro'][split] = None

    return metrics


# ==============================================================================
#          Mini-Batch Training & Evaluation (Standard Loaders)
# ==============================================================================
# Suitable for large graphs using NeighborLoader (node classification) or
# DataLoader (graph classification, or node classification like PPI).

def train_minibatch(model: torch.nn.Module, loader: object, optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module, device: torch.device,
                    is_multilabel: bool = False, is_graph_classification: bool = False,
                    desc: str = "Train Mini-Batch") -> float:
    """
    Performs one training epoch using a standard mini-batch loader
    (e.g., NeighborLoader, DataLoader).

    Args:
        model (torch.nn.Module): The GNN model.
        loader: The PyG DataLoader or NeighborLoader instance.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        is_multilabel (bool): Flag for multi-label classification. Defaults to False.
        is_graph_classification (bool): Flag for graph classification task. Defaults to False.
        desc (str): Description for the tqdm progress bar.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0.0
    num_processed = 0 # Tracks total nodes or graphs processed for averaging loss

    # Iterate over mini-batches provided by the loader
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device) # Move batch data to the target device
        optimizer.zero_grad() # Clear gradients

        # --- Input Validation & Skipping Empty Batches ---
        # Skip batches that might be empty (e.g., due to sampling issues)
        if not is_graph_classification and hasattr(batch, 'batch_size') and \
           (batch.x is None or batch.x.shape[0] == 0 or batch.batch_size == 0):
            continue # Skip empty node classification batches (NeighborLoader)
        if is_graph_classification and (batch.x is None or batch.x.shape[0] == 0):
            continue # Skip empty graph classification batches (DataLoader)

        # Get edge weights if available in the batch
        edge_weight = getattr(batch, 'edge_weight', None)

        # --- Model Forward Pass (Adapts to Task Type) ---
        if is_graph_classification:
            # Graph Classification (usually DataLoader)
            # Models might need the 'batch' vector for pooling.
            try:
                # Try passing all common arguments
                out = model(batch.x, batch.edge_index, batch.batch, edge_weight=edge_weight)
            except TypeError: # Handle models with different forward signatures
                 try:
                     # Try without edge_weight
                     out = model(batch.x, batch.edge_index, batch.batch)
                 except TypeError:
                     # Try minimal signature (common for simpler models)
                     out = model(batch.x, batch.edge_index)

            target_out = out # Output corresponds to graphs
            target_y = batch.y # Labels correspond to graphs
            loss_count = batch.num_graphs # Loss is averaged over graphs in the batch
        else:
            # Node Classification (NeighborLoader or DataLoader like PPI)
            # Standard node models expect x, edge_index, [edge_weight]
            out = model(batch.x, batch.edge_index, edge_weight=edge_weight)

            if hasattr(batch, 'batch_size'): # NeighborLoader convention
                # Output for the first 'batch_size' nodes corresponds to target nodes
                target_out = out[:batch.batch_size]
                target_y = batch.y[:batch.batch_size]
                loss_count = batch.batch_size # Loss is averaged over target nodes
            else: # Assume DataLoader for node classification (e.g., PPI)
                # Output corresponds to all nodes in the batch (subgraph)
                target_out = out
                target_y = batch.y
                loss_count = batch.num_nodes # Loss is averaged over all nodes in the batch

        # --- Loss Calculation ---
        if is_multilabel:
            # BCEWithLogitsLoss expects float targets
            loss = criterion(target_out, target_y.float())
        else:
            # CrossEntropyLoss expects long targets
            loss = criterion(target_out, target_y)

        # Skip batch if loss is invalid
        if torch.isnan(loss) or torch.isinf(loss):
             warnings.warn(f"NaN or Inf loss detected in batch. Skipping batch.")
             continue

        # --- Backpropagation and Optimization ---
        loss.backward()
        optimizer.step()

        # Accumulate loss and count processed items
        total_loss += loss.item() * loss_count
        num_processed += loss_count

    # Return average loss over all processed items (nodes or graphs)
    return total_loss / num_processed if num_processed > 0 else 0.0


@torch.no_grad() # Disable gradient calculations for evaluation
def evaluate_minibatch(model: torch.nn.Module, loader: object, criterion: torch.nn.Module,
                       device: torch.device, is_multilabel: bool = False,
                       is_graph_classification: bool = False,
                       desc: str = "Evaluate Mini-Batch") -> dict:
    """
    Evaluates model performance using a standard mini-batch loader.

    Accumulates predictions and labels across batches to compute overall metrics.

    Args:
        model (torch.nn.Module): The GNN model.
        loader: The PyG DataLoader or NeighborLoader instance.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to evaluate on.
        is_multilabel (bool): Flag for multi-label classification. Defaults to False.
        is_graph_classification (bool): Flag for graph classification task. Defaults to False.
        desc (str): Description for the tqdm progress bar.

    Returns:
        dict: A dictionary containing evaluation metrics ('loss', 'acc', 'f1_macro', 'f1_micro').
              Metrics might be NaN/0.0 if loader is empty, or None if not applicable.
    """
    model.eval() # Set model to evaluation mode
    all_preds = []   # List to store predictions from all batches
    all_labels = []  # List to store true labels from all batches
    total_loss = 0.0
    num_processed = 0 # Tracks total nodes or graphs processed

    # Iterate over mini-batches
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device) # Move data to device

        # --- Input Validation & Skipping Empty Batches ---
        if not is_graph_classification and hasattr(batch, 'batch_size') and \
           (batch.x is None or batch.x.shape[0] == 0 or batch.batch_size == 0):
            continue
        if is_graph_classification and (batch.x is None or batch.x.shape[0] == 0):
            continue

        # Get edge weights if available
        edge_weight = getattr(batch, 'edge_weight', None)

        # --- Model Forward Pass (Adapts to Task Type) ---
        if is_graph_classification:
            # Graph Classification
            try:
                out = model(batch.x, batch.edge_index, batch.batch, edge_weight=edge_weight)
            except TypeError: # Fallback signatures
                 try:
                     out = model(batch.x, batch.edge_index, batch.batch)
                 except TypeError:
                     out = model(batch.x, batch.edge_index)
            target_out = out
            target_y = batch.y
            count = batch.num_graphs # Items processed in this batch
        else:
            # Node Classification
            out = model(batch.x, batch.edge_index, edge_weight=edge_weight)
            if hasattr(batch, 'batch_size'): # NeighborLoader
                target_out = out[:batch.batch_size]
                target_y = batch.y[:batch.batch_size]
                count = batch.batch_size
            else: # PPI DataLoader
                try:    
                    target_out = out
                    target_y = batch.y
                    count = batch.num_nodes
                except RuntimeError as e:
                    print("Error during forward computation:")
                    print("  target_out.shape:", target_out.shape)
                    print("  target_y.unique():", torch.unique(target_y))
                    raise

        # --- Calculate Loss ---
        try:
            if is_multilabel:
                loss = criterion(target_out, target_y.float())
            else:
                loss = criterion(target_out, target_y)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * count
                num_processed += count
            else:
                loss = torch.tensor(float('nan')) # Mark loss as invalid for this batch
        except Exception as e:
            warnings.warn(f"Could not compute loss for evaluation batch. Error: {e}")
            loss = torch.tensor(float('nan')) # Mark loss as invalid

        # --- Store Predictions and Labels (Move to CPU) ---
        if is_multilabel:
            preds = (torch.sigmoid(target_out) > 0.5).int().cpu()
        elif is_graph_classification: # Graph predictions
             preds = target_out.argmax(dim=1).cpu()
        else: # Node predictions (multi-class)
             preds = target_out.argmax(dim=1).cpu()

        labels = target_y.cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    # --- Aggregate Metrics Across All Batches ---
    avg_loss = total_loss / num_processed if num_processed > 0 else float('nan')

    # Handle case where loader was empty or all batches were skipped
    if not all_preds:
        metrics = {'loss': avg_loss, 'acc': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0}
        warnings.warn("Evaluation resulted in no predictions. Returning zero metrics.")
        return metrics

    # Concatenate predictions and labels from all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate final metrics based on task type
    if is_multilabel: # Typically Node classification (e.g., PPI)
        f1_micro = f1_score(all_labels.numpy(), all_preds.numpy(), average='micro', zero_division=0)
        metrics = {'loss': avg_loss, 'acc': None, 'f1_macro': None, 'f1_micro': f1_micro}
    else: # Node or Graph classification (multi-class)
        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        f1_macro = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro', zero_division=0)
        metrics = {'loss': avg_loss, 'acc': acc, 'f1_macro': f1_macro, 'f1_micro': None}

    return metrics


# ==============================================================================
#             FastGCN Mini-Batch Training & Evaluation
# ==============================================================================
# Specific training loop for models using the FastGcnSampler.
# Evaluation typically uses the standard full-batch method.

def train_fastgcn(model: 'FastGCN', loader: 'FastGcnSampler', optimizer: torch.optim.Optimizer,
                  criterion: torch.nn.Module, device: torch.device,
                  desc: str = "Train FastGCN") -> float:
    """
    Performs one training epoch using the FastGCN layer-wise sampling strategy.

    Args:
        model (FastGCN): The FastGCN model instance. Should have `num_layers` and accessible conv layers.
        loader (FastGcnSampler): The custom FastGCN sampler instance.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to train on.
        desc (str): Description for the tqdm progress bar.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0.0
    num_processed = 0 # Tracks total number of nodes for which loss was computed

    # --- Access Convolutional Layers ---
    # Dynamically access the model's convolutional layers
    model_convs = []
    if hasattr(model, 'convs') and isinstance(model.convs, torch.nn.ModuleList):
         # Assumes layers are stored in a ModuleList named 'convs'
         model_convs = model.convs
    elif hasattr(model, 'conv1'):
         # Fallback for manually defined layers (like the example FastGCN model)
         model_convs.append(model.conv1)
         if hasattr(model, 'conv2'):
              model_convs.append(model.conv2)
         # Add more checks here if models with > 2 manually defined layers are expected
    else:
         raise AttributeError("Cannot find convolutional layers in the FastGCN model. Expected 'convs' (ModuleList) or 'conv1'/'conv2'.")

    # Sanity check: number of layers reported by model vs. found conv layers
    if model.num_layers != len(model_convs):
         warnings.warn(f"Model's `num_layers` ({model.num_layers}) does not match the number of "
                       f"found convolutional layers ({len(model_convs)}). Using {len(model_convs)} layers.")
         actual_num_layers = len(model_convs) # Use the number of layers actually found
    else:
         actual_num_layers = model.num_layers

    # Iterate over layer-wise batches provided by the FastGcnSampler
    for layerwise_batch_data in tqdm(loader, desc=desc, leave=False):
        # `layerwise_batch_data` is a list of dicts, one per layer + final info.
        # Each dict contains CPU tensors ('x', 'edge_index', 'edge_weight', 'nodes', 'y')
        optimizer.zero_grad() # Clear gradients

        # --- Layer-wise Forward Pass ---
        # Start with input features for the first layer (layer 0)
        # These correspond to the nodes sampled specifically for layer 0's input.
        layer0_info = layerwise_batch_data[0]
        # Initial hidden state `h` is the feature matrix for the first layer's input nodes
        h = layer0_info['x'].to(device, non_blocking=True) # Move features to device

        # Iterate through the actual number of convolutional layers found
        for l in range(actual_num_layers):
            # Get the pre-calculated data for the current layer `l`
            layer_info = layerwise_batch_data[l]
            # Move subgraph connectivity and weights to the device
            current_edge_index = layer_info['edge_index'].to(device, non_blocking=True)
            current_edge_weight = layer_info['edge_weight'].to(device, non_blocking=True)
            # Ensure the hidden state `h` is on the correct device (should be from previous iter or initial)
            h = h.to(device)

            # Apply the l-th convolutional layer
            conv_layer = model_convs[l]
            h = conv_layer(h, current_edge_index, current_edge_weight)

            # Apply activation and dropout (except after the last layer)
            if l < actual_num_layers - 1:
                h = F.relu(h)
                # Use model's dropout prob and ensure it's active during training
                h = F.dropout(h, p=model.dropout, training=True)

        # The final `h` contains the output embeddings for the nodes that were
        # input to the *first* layer (layer 0).
        out = h

        # --- Loss Calculation ---
        # The final output `out` corresponds to the nodes provided as input to layer 0.
        # We need the ground truth labels for *these specific nodes*.
        # The last element of `layerwise_batch_data` contains the target nodes and labels
        # for the *original* batch request (output layer nodes), but the FastGCN forward
        # pass structure means the `out` tensor aligns with the *input* nodes of layer 0.

        # Get the indices of the nodes corresponding to the output `out`
        # These are the nodes from layer 0's input dictionary.
        nodes_for_loss_cpu = layer0_info['nodes'].cpu() # Ensure indices are on CPU

        # Retrieve the corresponding labels from the sampler's full label tensor (which is on CPU)
        if not hasattr(loader, 'y'):
             raise AttributeError("FastGCN sampler ('loader') must have access to the full label tensor via a 'y' attribute.")
        try:
            target_y_cpu = loader.y[nodes_for_loss_cpu]
        except IndexError as e:
            print(f"Error indexing labels: indices shape={nodes_for_loss_cpu.shape}, loader.y shape={loader.y.shape}")
            raise e

        # Move the fetched labels to the target device
        target_y = target_y_cpu.to(device, non_blocking=True)

        # --- Size Sanity Check ---
        if out.shape[0] != target_y.shape[0]:
            # This should ideally not happen if the logic is correct.
            warnings.warn(f"Output size ({out.shape[0]}) mismatch with target label size ({target_y.shape[0]}) "
                          f"for nodes {nodes_for_loss_cpu.shape}. Skipping batch.")
            continue # Skip this batch if sizes don't align

        # Calculate loss between model output and ground truth labels
        loss = criterion(out, target_y)

        # Skip batch if loss is invalid
        if torch.isnan(loss) or torch.isinf(loss):
             warnings.warn(f"NaN or Inf loss detected during FastGCN training. Skipping batch.")
             continue

        # --- Backpropagation and Optimization ---
        loss.backward()
        optimizer.step()

        # Accumulate loss, weighted by the number of nodes in this batch's output
        batch_node_count = target_y.shape[0] # Loss computed over these nodes
        total_loss += loss.item() * batch_node_count
        num_processed += batch_node_count

    # Return the average loss over all processed nodes
    return total_loss / num_processed if num_processed > 0 else 0.0


@torch.no_grad() # Disable gradient calculations for evaluation
def evaluate_fastgcn(model: torch.nn.Module, full_data: 'torch_geometric.data.Data',
                     criterion: torch.nn.Module, device: torch.device,
                     is_multilabel: bool = False) -> dict:
    """
    Evaluates a model trained with FastGCN using standard full-graph inference.

    The FastGCN paper typically evaluates performance by running the trained
    model on the entire graph, similar to standard GCN evaluation.

    Args:
        model (torch.nn.Module): The trained GNN model.
        full_data (Data): The complete graph data object.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to evaluate on.
        is_multilabel (bool): Flag indicating if the task is multi-label classification.

    Returns:
        dict: A dictionary containing evaluation metrics from `evaluate_full_batch`.
    """
    print("Evaluating FastGCN-trained model using full graph inference...")
    # Ensure the full dataset is on the evaluation device
    full_data = full_data.to(device)
    # Use the standard full-batch evaluation function
    return evaluate_full_batch(model, full_data, criterion, is_multilabel)