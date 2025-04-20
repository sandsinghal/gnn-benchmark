# utils/sampling.py
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree, from_scipy_sparse_matrix, to_scipy_sparse_matrix
from tqdm import tqdm


@torch.no_grad()
def calculate_fastgcn_probs(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Calculates the importance sampling probabilities for FastGCN based on Chen et al. (2018).
    Probability q(v) is proportional to the squared L2 norm of the v-th column
    of the symmetrically normalized adjacency matrix with self-loops (A_tilde in GCN paper).

    Args:
        edge_index (Tensor): Edge index in COO format [2, num_edges].
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        Tensor: Importance sampling probabilities for each node [num_nodes], on CPU.
    """
    print("Calculating FastGCN sampling probabilities (on CPU)...")
    cpu_device = torch.device("cpu")
    edge_index = edge_index.to(cpu_device)

    # 1. Create Adjacency Matrix (A) + Identity (I) -> A_hat
    # Use float32 for weights to ensure consistency
    edge_weight = torch.ones(edge_index.size(1), device=cpu_device, dtype=torch.float32)
    # Convert edge_index to sparse tensor A
    adj_sp = torch.sparse_coo_tensor(
        edge_index, edge_weight, (num_nodes, num_nodes)
    ).coalesce()

    # Create sparse identity matrix I
    identity_indices = torch.arange(num_nodes, device=cpu_device).unsqueeze(0).repeat(2, 1)
    identity_values = torch.ones(num_nodes, device=cpu_device, dtype=torch.float32)
    identity_sp = torch.sparse_coo_tensor(
        identity_indices, identity_values, (num_nodes, num_nodes)
    )

    # A_hat = A + I
    adj_hat = (adj_sp + identity_sp).coalesce()

    # 2. Calculate Degree Matrix D_hat
    row, col = adj_hat.indices()
    # Degree calculation needs float type if weights are float
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow_(-0.5)
    # Replace potential inf values (degree 0 nodes) with 0
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

    # 3. Calculate Symmetrically Normalized Adjacency A_tilde = D_hat^-0.5 @ A_hat @ D_hat^-0.5
    # Get values and indices from sparse adj_hat
    adj_hat_indices = adj_hat.indices()
    adj_hat_values = adj_hat.values()
    # Apply normalization D^-0.5 * A_val * D^-0.5
    normalized_values = deg_inv_sqrt[row] * adj_hat_values * deg_inv_sqrt[col]

    # Create the sparse A_tilde matrix
    a_tilde_sp = torch.sparse_coo_tensor(
        adj_hat_indices, normalized_values, (num_nodes, num_nodes)
    ).coalesce()

    # 4. Calculate Probabilities: q(v) proportional to ||A_tilde[:, v]||^2
    # Convert to dense (might be memory intensive for very large graphs)
    # TODO: Explore computing column norms directly from sparse format if memory becomes an issue
    try:
        a_tilde_dense = a_tilde_sp.to_dense()
        # Sum of squares along columns (dim=0)
        col_norms_sq = torch.sum(a_tilde_dense**2, dim=0)
    except RuntimeError as e:
         print(f"Error converting A_tilde to dense ({e}). Graph might be too large.")
         print("Attempting calculation via CSR format (requires scipy)...")
         try:
             a_tilde_csr = to_scipy_sparse_matrix(a_tilde_sp.indices(), a_tilde_sp.values(), num_nodes)
             col_norms_sq = torch.tensor(np.array(a_tilde_csr.power(2).sum(axis=0)).flatten(), dtype=torch.float32)
         except Exception as e_csr:
             print(f"Error calculating norms via CSR: {e_csr}")
             raise RuntimeError("Could not calculate FastGCN probabilities due to memory/conversion issues.")


    # Normalize probabilities
    total_norm_sq = torch.sum(col_norms_sq)
    if total_norm_sq <= 0:
        print("Warning: Sum of squared column norms is zero. Using uniform probabilities.")
        probs = torch.ones(num_nodes, device=cpu_device) / num_nodes
    else:
        probs = col_norms_sq / total_norm_sq

    # Ensure probabilities are non-negative and sum to 1 (handle potential float errors)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum()

    print("FastGCN probabilities calculated.")
    # Return probabilities on CPU, sampling happens on CPU anyway
    return probs.cpu()


class FastGcnSampler(IterableDataset):
    """
    Implements layer-wise importance sampling for FastGCN training as an IterableDataset.

    Samples nodes for each layer based on precomputed importance probabilities (`probs`).
    Constructs layer-wise computation graphs (features and renormalized adjacency)
    required for the `train_fastgcn` function.

    Yields batches of layer-wise data structures, all residing on CPU.
    """
    def __init__(
        self,
        data: Data, # Full graph data object (must be on CPU)
        sample_sizes: List[int], # Number of nodes to sample per layer [t_1, t_2, ...]
        probs: Tensor, # Importance sampling probabilities [num_nodes] (on CPU)
        batch_size: int, # Number of nodes in the final layer output batch
    ):
        super().__init__()
        if not (data.x is not None and data.y is not None and data.edge_index is not None):
            raise ValueError("Input data must have 'x', 'y', and 'edge_index'.")

        self.num_nodes = data.num_nodes
        # Keep data attributes on CPU, ensure features are float32
        self.x = data.x.float().cpu()
        self.y = data.y.cpu() # Labels usually long
        self.edge_index = data.edge_index.cpu()
        self.sample_sizes = sample_sizes # e.g., [512, 512] for 2 layers
        self.num_layers = len(sample_sizes)
        self.probs = probs.cpu() # Ensure probs are on CPU for sampling
        self.batch_size = batch_size # Number of nodes in the final layer (loss nodes)

        print("Building SciPy CSR adjacency matrix for FastGCN sampler (on CPU)...")
        # Build SciPy sparse adj (A) once for efficient subgraph slicing. Use float32.
        adj_sp = sp.csr_matrix(
            (np.ones(self.edge_index.shape[1], dtype=np.float32),
             (self.edge_index[0].numpy(), self.edge_index[1].numpy())),
            shape=(self.num_nodes, self.num_nodes),
        )
        # Precompute A_hat = A + I (sparse) for GCN normalization within sampling
        self.adj_hat_sp = adj_sp + sp.identity(self.num_nodes, format="csr", dtype=np.float32)
        print("Adjacency matrix built.")

    def __iter__(self):
        """Yields layer-wise batch data structures."""
        # We iterate over the *final layer* nodes in batches
        # Shuffle the nodes for which we want predictions
        node_indices = torch.randperm(self.num_nodes, device='cpu')

        for i in range(0, self.num_nodes, self.batch_size):
            # --- 1. Sample Output Nodes ---
            # These are the nodes for which loss will be calculated (layer L)
            batch_nodes_L_orig_idx = node_indices[i : i + self.batch_size]

            # --- 2. Layer-wise Importance Sampling (Backward Pass) ---
            # Start with the output nodes and sample backwards to get input nodes for each layer
            sampled_nodes_per_layer_orig_idx: List[Tensor] = [batch_nodes_L_orig_idx]
            current_layer_nodes = batch_nodes_L_orig_idx

            for layer in range(self.num_layers - 1, -1, -1):
                # Sample `t_l` nodes for the input of layer `l` using importance probs `q(v)`
                num_to_sample = self.sample_sizes[layer] # t_l from config
                # `torch.multinomial` expects probabilities for sampling WITH replacement
                sampled_indices = torch.multinomial(
                    self.probs, num_samples=num_to_sample, replacement=True
                )
                sampled_nodes_per_layer_orig_idx.insert(0, sampled_indices)
                # The nodes sampled here become the 'current' nodes for the *next* sampling step (layer l-1)
                current_layer_nodes = sampled_indices

            # `sampled_nodes_per_layer_orig_idx` now contains lists of original node indices
            # for each layer's input, from layer 0 up to layer L (output nodes).
            # e.g., [layer0_input_nodes, layer1_input_nodes, ..., layerL_output_nodes]

            # --- 3. Create Batch Data Structure (Forward Pass View) ---
            # Construct the list of dictionaries needed by `train_fastgcn`
            batch_data = self._create_layerwise_batch(sampled_nodes_per_layer_orig_idx)
            if batch_data: # Ensure batch creation was successful
                yield batch_data


    def _create_layerwise_batch(
        self, sampled_nodes_per_layer_orig_idx: List[Tensor]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Constructs the layer-wise data structure for a single batch.

        Args:
            sampled_nodes_per_layer_orig_idx: A list where each element is a Tensor of
                original node indices sampled for the *input* of that layer.
                Example: [L0_input_indices, L1_input_indices, ..., LL_output_indices]

        Returns:
            A list of dictionaries, one for each GCN layer, plus a final dict for labels.
            Returns None if sampling resulted in an empty set of nodes at any critical step.
            All tensors within the dictionaries are on CPU.
        """
        layer_data: List[Dict[str, Any]] = []
        # The nodes whose features form the *input* to the first GCN layer (layer 0)
        prev_layer_nodes_orig_idx = sampled_nodes_per_layer_orig_idx[0]

        # --- Iterate through GCN layers (0 to L-1) ---
        for l in range(self.num_layers):
            # Nodes whose features are needed *after* GCNConv layer `l` completes.
            # These are the nodes sampled as input for layer `l+1`.
            current_layer_nodes_orig_idx = sampled_nodes_per_layer_orig_idx[l+1]

            # --- a) Get Input Features for Layer l ---
            # Select features corresponding to the nodes inputting this layer (`prev_layer_nodes_orig_idx`)
            layer_input_x_cpu = self.x[prev_layer_nodes_orig_idx] # Shape [t_l, num_features]

            # --- b) Construct and Renormalize Subgraph Adjacency for Layer l ---
            # We need the subgraph of A_hat involving only nodes in `prev_layer_nodes_orig_idx`.
            # Use the precomputed CSR matrix `self.adj_hat_sp` for efficient slicing.
            num_sub_nodes = len(prev_layer_nodes_orig_idx)
            if num_sub_nodes == 0: return None # Skip if no nodes sampled for input

            # Extract the submatrix using CSR indexing: adj[rows][:, cols]
            sub_adj_sp = self.adj_hat_sp[prev_layer_nodes_orig_idx][:, prev_layer_nodes_orig_idx]

            # Convert the SciPy sparse submatrix back to PyG edge_index and edge_weight (float32)
            sub_edge_index_th_cpu, sub_edge_weight_th_cpu = from_scipy_sparse_matrix(sub_adj_sp)
            sub_edge_weight_th_cpu = sub_edge_weight_th_cpu.float() # Ensure float

            # --- c) Renormalize the Subgraph (Symmetric GCN Norm) ---
            # Calculate degrees based on the *subgraph* edge_index
            sub_row, sub_col = sub_edge_index_th_cpu
            # Use float32 for degree calculation
            sub_deg = degree(sub_row, num_nodes=num_sub_nodes, dtype=torch.float32)
            sub_deg_inv_sqrt = sub_deg.pow(-0.5)
            sub_deg_inv_sqrt.masked_fill_(sub_deg_inv_sqrt == float("inf"), 0) # Handle isolated nodes

            # Apply normalization: D^-0.5 * A_sub * D^-0.5 (element-wise on edges)
            # Ensure all components are float32
            norm_edge_weight_cpu = (
                sub_deg_inv_sqrt[sub_row] * sub_edge_weight_th_cpu * sub_deg_inv_sqrt[sub_col]
            )

            # --- d) Store Layer Information (CPU Tensors) ---
            layer_info = {
                "x": layer_input_x_cpu,           # Input features for conv[l] [t_l, C]
                "edge_index": sub_edge_index_th_cpu, # Subgraph connectivity [2, E_sub]
                "edge_weight": norm_edge_weight_cpu, # Renormalized weights [E_sub]
                "nodes": prev_layer_nodes_orig_idx # Original indices of input nodes
            }
            layer_data.append(layer_info)

            # Update for the next iteration
            prev_layer_nodes_orig_idx = current_layer_nodes_orig_idx

        # --- Add Final Output Nodes and Labels ---
        # The last element in `sampled_nodes_per_layer_orig_idx` contains the nodes
        # for which the final prediction `h` is generated and loss is computed.
        final_nodes_orig_idx = sampled_nodes_per_layer_orig_idx[-1]
        if len(final_nodes_orig_idx) == 0: return None # Skip if no output nodes

        # Get the corresponding labels
        batch_y_cpu = self.y[final_nodes_orig_idx]

        # Append the final node indices and their labels
        layer_data.append({"nodes": final_nodes_orig_idx, "y": batch_y_cpu})

        return layer_data # Return structure with CPU tensors

    def __len__(self) -> int:
        """Returns the number of batches in an epoch."""
        # Note: This is an approximation as the actual number depends on sampling
        # and potential skipping of empty batches. Used mainly by progress bars.
        return (self.num_nodes + self.batch_size - 1) // self.batch_size