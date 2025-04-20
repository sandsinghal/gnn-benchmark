# main.py
import os
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import wandb
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import PPI, Planetoid, Reddit, TUDataset
from torch_geometric.loader import DataLoader, NeighborLoader
from tqdm import tqdm

# Local imports
import config as cfg
from data import load_dataset
from models import get_model
from utils.sampling import FastGcnSampler, calculate_fastgcn_probs
from utils.training import (
    evaluate_fastgcn,
    evaluate_full_batch,
    evaluate_minibatch,
    train_fastgcn,
    train_full_batch,
    train_minibatch,
)


# --- Seeding ---
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms are used where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed}")


def main():
    # 1. Load Configuration & Setup
    config = cfg.get_config()
    set_seed(config["seed"])
    device = torch.device(config["device"])
    is_graph_classification: bool = config.get("graph_classification_dataset", False)
    task_type = "Graph Classification" if is_graph_classification else "Node Classification"

    # Generate a descriptive run name
    run_name = (
        f"{config['model_name']}-{config['dataset_name']}-{task_type[0]}"
        f"-h{config['hidden_channels']}-bs{config['batch_size']}-lr{config['lr']}-s{config['seed']}"
    )
    if config["log_to_wandb"]:
        try:
            wandb.init(
                project=config["wandb_project"],
                config=config,
                name=run_name,
                resume="allow",
                settings=wandb.Settings(start_method="fork"), # Use 'fork' for better compatibility unless CUDA issues arise
            )
            print(f"Initialized Weights & Biases run: {run_name}")
        except Exception as e:
            print(f"Error initializing W&B: {e}. Set --no-log_to_wandb to disable.")
            config['log_to_wandb'] = False # Disable if init fails

    print(f"Task Type: {task_type}")
    print(f"Run Name: {run_name}")
    print(f"Using device: {device}")

    # 2. Load Dataset
    train_dataset, val_dataset, test_dataset = None, None, None
    dataset_for_model_init = None # Dataset object used for model parameter calculation

    if is_graph_classification:
        try:
            # Load TUDataset for graph classification
            dataset = TUDataset(
                root=config["data_dir"], name=config["dataset_name"], use_node_attr=True
            )
            dataset_for_model_init = dataset

            # Basic random split (80/10/10). Consider more robust splitting methods for real benchmarks.
            num_graphs = len(dataset)
            indices = np.random.permutation(num_graphs)
            split_train = int(0.8 * num_graphs)
            split_val = int(0.9 * num_graphs)
            train_idx = torch.tensor(indices[:split_train], dtype=torch.long)
            val_idx = torch.tensor(indices[split_train:split_val], dtype=torch.long)
            test_idx = torch.tensor(indices[split_val:], dtype=torch.long)

            train_dataset = dataset[train_idx]
            val_dataset = dataset[val_idx]
            test_dataset = dataset[test_idx]

            print(
                f"Loaded Graph Classification Dataset '{config['dataset_name']}'"
                f" - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )
            print(f"  Num Node Features: {dataset.num_node_features}, Num Classes: {dataset.num_classes}")

        except Exception as e:
            print(f"Error loading/splitting graph dataset {config['dataset_name']}: {e}")
            sys.exit(1)

    else:  # Node classification
        try:
            dataset = load_dataset(
                config["dataset_name"],
                config["data_dir"],
                config["normalize_features"],
            )
            dataset_for_model_init = dataset
            # Node classification datasets often have predefined splits (masks or separate datasets like PPI)
            if config['dataset_name'] == 'PPI':
                # PPI requires loading splits separately
                train_dataset = dataset # The loaded 'train' split
                val_dataset = PPI(root=os.path.join(config['data_dir'], 'PPI'), split='val')
                test_dataset = PPI(root=os.path.join(config['data_dir'], 'PPI'), split='test')
                print(f"Loaded PPI Dataset Splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            else:
                # Planetoid/Reddit have masks within the single graph object
                # No separate datasets needed here, loaders will use masks.
                pass
        except Exception as e:
             print(f"Error loading node dataset {config['dataset_name']}: {e}")
             sys.exit(1)


    # Determine if the task involves multi-label classification
    is_multilabel = config['dataset_name'] == 'PPI' or \
                    (is_graph_classification and hasattr(dataset_for_model_init, 'data') and \
                     hasattr(dataset_for_model_init.data, 'y') and \
                     dataset_for_model_init.data.y is not None and \
                     dataset_for_model_init.data.y.dim() > 1 and \
                     dataset_for_model_init.data.y.shape[1] > 1) # Check if y exists and is multi-dim

    print(f"Multi-label task detected: {is_multilabel}")


    # 3. Initialize Model
    model = get_model(config, dataset_for_model_init).to(device)
    print("\n--- Model Architecture ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    if config["log_to_wandb"]:
        wandb.summary["Total Parameters"] = total_params
        # Log gradients and parameters, adjust log_freq as needed
        wandb.watch(model, log_freq=max(100, config.get("log_freq", 100)), log="all")


    # 4. Setup Optimizer and Loss Criterion
    try:
        optimizer_class = getattr(optim, config["optimizer"])
        optimizer = optimizer_class(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    except AttributeError:
        print(f"Error: Optimizer '{config['optimizer']}' not found in torch.optim.")
        sys.exit(1)

    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    print(
        f"Optimizer: {config['optimizer']} (LR={config['lr']}, WD={config['weight_decay']})"
    )
    print(f"Criterion: {'BCEWithLogitsLoss' if is_multilabel else 'CrossEntropyLoss'}")


    # 5. Data Handling & Training Path Selection
    train_loader, val_loader, test_loader = None, None, None
    data_full_cpu: Optional[Data] = None # Full graph data on CPU (for loaders)
    data_full_gpu: Optional[Data] = None # Full graph data on GPU (for full-batch)
    train_mode: Optional[str] = None  # 'full_batch_node', 'fastgcn', 'minibatch'

    # Determine the training mode based primarily on task type, model, and batch size
    if is_graph_classification:
        # Graph classification ALWAYS uses DataLoader for minibatching
        train_mode = "minibatch"
        loader_batch_size = config["batch_size"]
        num_workers = 4 if device.type == "cuda" else 0 # Use workers for GPU loading
        print(
            f"Using DataLoader (Minibatch) for Graph Classification."
            f" Batch size: {loader_batch_size}, Num workers: {num_workers}"
        )
        try:
            train_loader = DataLoader(
                train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = DataLoader(
                val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers
            )
            test_loader = DataLoader(
                test_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers
            )
        except Exception as e:
            print(f"Error creating graph DataLoaders: {e}")
            sys.exit(1)

    else:  # Node Classification Task
        if config["model_name"] == "FastGCN":
            train_mode = "fastgcn"
            print("Setting up FastGCN sampler...")
            try:
                data_full_cpu = dataset[0].cpu() # Requires single graph on CPU
                # Calculate importance sampling probabilities
                fastgcn_probs = calculate_fastgcn_probs(
                    data_full_cpu.edge_index, data_full_cpu.num_nodes
                )
                # Initialize the custom FastGCN sampler for training
                train_loader = FastGcnSampler(
                    data_full_cpu,
                    config["fastgcn_layer_sample_sizes"],
                    fastgcn_probs,
                    config["batch_size"],
                )
                print(f"FastGCN using custom sampler for training (Batch size: {config['batch_size']}).")
                # Note: FastGCN evaluation uses full graph (see evaluate_fastgcn)
                # Exception: If dataset is PPI, need standard loaders for val/test eval
                if config['dataset_name'] == 'PPI':
                    print("Creating standard DataLoaders for FastGCN+PPI evaluation...")
                    ppi_batch_size = config.get('batch_size', 2) # Use main batch size or PPI default
                    num_workers = 4 if device.type == 'cuda' else 0
                    val_loader = DataLoader(val_dataset, batch_size=ppi_batch_size, shuffle=False, num_workers=num_workers)
                    test_loader = DataLoader(test_dataset, batch_size=ppi_batch_size, shuffle=False, num_workers=num_workers)
            except Exception as e:
                print(f"Error setting up FastGCN sampler or PPI eval loaders: {e}")
                sys.exit(1)

        elif config["batch_size"] <= 0: # Full-batch node classification
            train_mode = "full_batch_node"
            print(f"Using Full-batch training for Node Classification ({config['dataset_name']}).")
            try:
                # Move the single graph object entirely to the target device
                data_full_gpu = dataset[0].to(device)
                print(f"Moved full graph data to {device}.")
            except Exception as e:
                print(f"Error moving data to {device} for full-batch training: {e}")
                sys.exit(1)

        else:  # Minibatch Node Classification (NeighborLoader or PPI DataLoader)
            train_mode = "minibatch"
            loader_batch_size = config["batch_size"]
            num_workers = 4 if device.type == "cuda" else 0

            if config["dataset_name"] == "PPI":
                 print(f"Using DataLoader for PPI node classification (Batch size: {loader_batch_size}, Workers: {num_workers}).")
                 try:
                     # PPI uses standard DataLoader over its graph structure
                     train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=num_workers)
                     val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers)
                     test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers)
                 except Exception as e:
                     print(f"Error setting up PPI DataLoaders: {e}")
                     sys.exit(1)
            else: # Reddit, etc. using NeighborLoader
                print(f"Using NeighborLoader for Node Classification (Batch size: {loader_batch_size}, Workers: {num_workers}).")
                try:
                    data_full_cpu = dataset[0].cpu() # Keep full data on CPU for loader
                    # Determine number of layers dynamically if possible, default to 2
                    num_layers = getattr(model, 'num_layers', 2)
                    num_neighbors_cfg = config.get('num_neighbors', [10]*num_layers) # Default neighbors per layer

                    # Adjust num_neighbors list length to match model layers
                    if len(num_neighbors_cfg) != num_layers:
                        print(f"Warning: num_neighbors length ({len(num_neighbors_cfg)}) != model layers ({num_layers}). Adjusting...")
                        if len(num_neighbors_cfg) > num_layers:
                            num_neighbors_list = num_neighbors_cfg[:num_layers]
                        else: # Repeat last element
                            num_neighbors_list = num_neighbors_cfg + [num_neighbors_cfg[-1]] * (num_layers - len(num_neighbors_cfg))
                        print(f"Using adjusted num_neighbors: {num_neighbors_list}")
                    else:
                        num_neighbors_list = num_neighbors_cfg

                    # Use a larger batch size for evaluation if desired
                    eval_batch_size = loader_batch_size * 2
                    print(f"NeighborLoader sampling: {num_neighbors_list} neighbors per layer.")

                    common_loader_params = {
                        'num_neighbors': num_neighbors_list,
                        'num_workers': num_workers,
                        'persistent_workers': (num_workers > 0), # Keep workers alive if using multiple
                    }
                    # Training loader: samples neighbors for nodes in the training mask
                    train_loader = NeighborLoader(
                        data_full_cpu,
                        input_nodes=data_full_cpu.train_mask,
                        batch_size=loader_batch_size,
                        shuffle=True,
                        **common_loader_params
                    )
                    # Validation loader
                    val_loader = NeighborLoader(
                        data_full_cpu,
                        input_nodes=data_full_cpu.val_mask,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        **common_loader_params
                    )
                    # Test loader
                    test_loader = NeighborLoader(
                        data_full_cpu,
                        input_nodes=data_full_cpu.test_mask,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        **common_loader_params
                    )
                except Exception as e:
                    print(f"Error setting up NeighborLoader: {e}")
                    sys.exit(1)

    if train_mode is None:
        print("Error: Could not determine training mode (full_batch_node, fastgcn, minibatch). Check config.")
        sys.exit(1)
    print(f"Selected Training Mode: {train_mode}")


    # --- 6. Training Loop ---
    print("\n--- Starting Training ---")
    best_val_metric = 0.0 if not is_multilabel else 0.0 # Maximize Acc/F1-Micro
    # Metric to optimize for early stopping (use micro F1 for multilabel)
    metric_to_optimize = "f1_micro" if is_multilabel else "acc"
    patience_counter = 0
    best_epoch = 0
    best_model_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()
        epoch_train_loss = 0.0
        model.train() # Ensure model is in training mode

        # --- Training Step ---
        if train_mode == "full_batch_node":
            epoch_train_loss = train_full_batch(model, data_full_gpu, optimizer, criterion)
        elif train_mode == "fastgcn":
            epoch_train_loss = train_fastgcn(
                model, train_loader, optimizer, criterion, device, desc=f"Epoch {epoch} Train"
            )
        elif train_mode == "minibatch":
            epoch_train_loss = train_minibatch(
                model, train_loader, optimizer, criterion, device, is_multilabel, is_graph_classification, desc=f"Epoch {epoch} Train"
            )
        else:
             print(f"Error: Invalid train_mode '{train_mode}' in training loop.")
             sys.exit(1)

        # --- Evaluation Step (on Validation Set) ---
        model.eval() # Ensure model is in evaluation mode
        val_metrics: Dict[str, Any] = {}
        if train_mode == "full_batch_node":
            # Evaluate on all splits (train/val/test) simultaneously
            val_metrics = evaluate_full_batch(model, data_full_gpu, criterion, is_multilabel)
        elif train_mode == "fastgcn":
            # FastGCN standard eval is full-graph, except for PPI which needs minibatch eval
            if config['dataset_name'] == 'PPI' and val_loader:
                val_metrics = evaluate_minibatch(
                    model, val_loader, criterion, device, is_multilabel, is_graph_classification=False, desc=f"Epoch {epoch} Val"
                )
            elif data_full_cpu: # Standard full-graph eval for other FastGCN cases
                 val_metrics = evaluate_fastgcn(model, data_full_cpu, criterion, device, is_multilabel)
            else:
                print("Warning: Cannot evaluate FastGCN - missing full data or validation loader.")
                val_metrics = {'loss': {'val': float('nan')}, metric_to_optimize: {'val': 0.0}, 'f1_macro': {'val': float('nan')}}

        elif train_mode == "minibatch" and val_loader:
             val_metrics = evaluate_minibatch(
                 model, val_loader, criterion, device, is_multilabel, is_graph_classification, desc=f"Epoch {epoch} Val"
             )
        elif not val_loader:
             print("Warning: No validation loader available, skipping validation.")
             val_metrics = {'loss': float('nan'), metric_to_optimize: 0.0, 'f1_macro': float('nan')}
        else:
             print(f"Error: Invalid train_mode '{train_mode}' or missing loader for validation.")
             sys.exit(1)


        # --- Logging, Printing, Early Stopping ---
        epoch_time = time.time() - start_time
        # Extract metrics carefully, handling different return structures
        val_loss = float('nan')
        current_val_metric = 0.0
        val_f1_macro = float('nan')

        # Check if metrics are nested (full_batch/fastgcn non-ppi) or flat (minibatch/fastgcn-ppi)
        is_nested_metrics = 'acc' in val_metrics and isinstance(val_metrics['acc'], dict)

        if is_nested_metrics:
            val_loss = val_metrics.get('loss', {}).get('val', float('nan'))
            current_val_metric = val_metrics.get(metric_to_optimize, {}).get('val', 0.0)
            val_f1_macro = val_metrics.get('f1_macro', {}).get('val', float('nan'))
        else: # Flat structure
            val_loss = val_metrics.get('loss', float('nan'))
            current_val_metric = val_metrics.get(metric_to_optimize, 0.0)
            val_f1_macro = val_metrics.get('f1_macro', float('nan'))

        # Prepare log dictionary for WandB
        log_dict: Dict[str, Union[int, float, str, None]] = {
            "Epoch": epoch,
            "Epoch Time (s)": epoch_time,
            "Train Loss": epoch_train_loss,
            "Val Loss": val_loss,
            f"Val {metric_to_optimize.capitalize()}": current_val_metric,
        }
        # Add aux validation metric if relevant
        if metric_to_optimize != "f1_macro" and not is_multilabel and not np.isnan(val_f1_macro):
            log_dict["Val F1-Macro"] = val_f1_macro

        # Add train/test metrics if available from full-batch eval
        if is_nested_metrics:
            if not is_multilabel:
                log_dict["Train Accuracy"] = val_metrics['acc'].get('train', float('nan'))
                log_dict["Train F1-Macro"] = val_metrics['f1_macro'].get('train', float('nan'))
                log_dict["Test Accuracy"] = val_metrics['acc'].get('test', float('nan'))
                log_dict["Test F1-Macro"] = val_metrics['f1_macro'].get('test', float('nan'))
            else: # Multilabel specific
                log_dict["Train F1-Micro"] = val_metrics['f1_micro'].get('train', float('nan'))
                log_dict["Test F1-Micro"] = val_metrics['f1_micro'].get('test', float('nan'))

        if config["log_to_wandb"]:
            wandb.log(log_dict, step=epoch)

        # Print epoch summary
        print(
            f"Epoch: {epoch:03d} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val {metric_to_optimize.capitalize()}: {current_val_metric:.4f}",
            end="",
        )
        if metric_to_optimize != "f1_macro" and not is_multilabel and not np.isnan(val_f1_macro):
            print(f" | Val F1-Macro: {val_f1_macro:.4f}", end="")
        # Print test metric from full-batch eval if available
        if is_nested_metrics:
             test_metric_key = f"Test {metric_to_optimize.capitalize()}"
             test_metric_val = log_dict.get(test_metric_key)
             if isinstance(test_metric_val, (int, float)) and not np.isnan(test_metric_val):
                 print(f" | {test_metric_key}: {test_metric_val:.4f}", end='')
        print() # Newline

        # Early Stopping Check
        if config["patience"] > 0:
            # Check if validation metric improved (allow for slight tolerance if needed)
            # Ensure metric is valid before comparison
            if not np.isnan(current_val_metric) and current_val_metric >= best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict() # Save best model state
                # print(f" * New best validation metric: {best_val_metric:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping triggered after {config['patience']} epochs without improvement.")
                print(f"Best validation {metric_to_optimize.capitalize()}: {best_val_metric:.4f} at epoch {best_epoch}")
                break
        else:
            # If no patience, always save the last model state as "best"
             best_model_state = model.state_dict()
             best_epoch = epoch
             best_val_metric = current_val_metric # Store last val metric


    # --- 7. Final Evaluation ---
    print("\n--- Training Finished ---")
    if best_model_state:
        print(f"Loading best model state from epoch {best_epoch} (Val {metric_to_optimize.capitalize()}: {best_val_metric:.4f})")
        model.load_state_dict(best_model_state)
        # Save best model artifact to WandB if enabled
        if config["log_to_wandb"] and wandb.run:
            try:
                model_path = os.path.join(wandb.run.dir, "best_model.pt")
                torch.save(best_model_state, model_path)
                artifact = wandb.Artifact(f"{run_name}-model", type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                print(f"Best model state saved to WandB artifacts.")
            except Exception as e:
                print(f"Error saving model artifact to W&B: {e}")
    else:
        print("Warning: No best model state found (e.g., training stopped early or no patience). Evaluating last state.")

    print("\n--- Final Evaluation on Test Set ---")
    model.eval() # Ensure model is in evaluation mode
    test_metrics: Dict[str, Any] = {}
    if not test_loader and train_mode not in ["full_batch_node", "fastgcn"]:
         print("Warning: No test loader available, skipping final test evaluation.")
    elif not data_full_gpu and train_mode == "full_batch_node":
         print("Warning: Missing GPU data for full-batch test evaluation, skipping.")
    elif not data_full_cpu and train_mode == "fastgcn" and config['dataset_name'] != 'PPI':
         print("Warning: Missing CPU data for FastGCN test evaluation, skipping.")
    else:
        if train_mode == "full_batch_node":
            test_metrics = evaluate_full_batch(model, data_full_gpu, criterion, is_multilabel)
        elif train_mode == "fastgcn":
            if config['dataset_name'] == 'PPI' and test_loader:
                test_metrics = evaluate_minibatch(
                    model, test_loader, criterion, device, is_multilabel, is_graph_classification=False, desc="Final Test"
                )
            elif data_full_cpu: # Standard full-graph eval
                test_metrics = evaluate_fastgcn(model, data_full_cpu, criterion, device, is_multilabel)
            else:
                 print("Warning: Cannot run final FastGCN test evaluation - missing data/loader.")
        elif train_mode == "minibatch" and test_loader:
            test_metrics = evaluate_minibatch(
                model, test_loader, criterion, device, is_multilabel, is_graph_classification, desc="Final Test"
            )
        else:
            print(f"Warning: Could not perform final test evaluation for train_mode '{train_mode}'.")


    # Extract final test metrics
    final_test_metric_name = f"Test {metric_to_optimize.capitalize()}"
    final_test_metric = float('nan')
    final_test_f1_macro = float('nan')
    final_test_loss = float('nan')

    if test_metrics:
        is_nested_metrics_test = 'acc' in test_metrics and isinstance(test_metrics['acc'], dict)
        if is_nested_metrics_test:
            final_test_loss = test_metrics.get('loss', {}).get('test', float('nan'))
            final_test_metric = test_metrics.get(metric_to_optimize, {}).get('test', float('nan'))
            final_test_f1_macro = test_metrics.get('f1_macro', {}).get('test', float('nan'))
        else: # Flat structure
            final_test_loss = test_metrics.get('loss', float('nan'))
            final_test_metric = test_metrics.get(metric_to_optimize, float('nan'))
            final_test_f1_macro = test_metrics.get('f1_macro', float('nan'))

    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final {final_test_metric_name}: {final_test_metric:.4f}")
    if not is_multilabel and not np.isnan(final_test_f1_macro):
        print(f"Final Test F1-Macro: {final_test_f1_macro:.4f}")

    # Log final metrics to WandB summary
    if config["log_to_wandb"] and wandb.run:
        wandb.summary["Best Epoch"] = best_epoch
        wandb.summary[f"Best Val {metric_to_optimize.capitalize()}"] = best_val_metric
        wandb.summary["Final Test Loss"] = final_test_loss
        wandb.summary[final_test_metric_name] = final_test_metric
        if not is_multilabel and not np.isnan(final_test_f1_macro):
            wandb.summary["Final Test F1-Macro"] = final_test_f1_macro
        wandb.finish()

    print("\n--- Experiment Complete ---")


if __name__ == "__main__":
    # Set multiprocessing start method (important for CUDA compatibility)
    # 'spawn' is generally safer, especially with CUDA. 'fork' can be faster but cause issues.
    start_method = 'spawn' if torch.cuda.is_available() or sys.platform == 'win32' else 'fork'
    try:
        current_context = mp.get_context()
        if current_context.get_start_method() != start_method:
             print(f"Setting multiprocessing start method to '{start_method}' (was '{current_context.get_start_method()}').")
             mp.set_start_method(start_method, force=True)
        else:
             print(f"Multiprocessing start method already set to '{start_method}'.")
    except RuntimeError as e:
        # Handles cases where it might already be set by another library or cannot be changed
        print(f"Info: Could not set multiprocessing start_method ('{start_method}'). It might already be running or set. Error: {e}")
    except Exception as e:
        print(f"Warning: Unexpected error setting multiprocessing start method: {e}")

    main()
