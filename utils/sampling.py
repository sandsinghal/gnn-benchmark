# utils/sampling.py
import torch
import numpy as np
from torch_geometric.utils import degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data
import scipy.sparse as sp
from tqdm import tqdm  # For progress bars
from typing import Union, List, Dict

@torch.no_grad() # Disable gradient calculations for efficiency
def calculate_fastgcn_probs(adj: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Calculates the importance sampling probabilities required for FastGCN layer sampling.

    The probability q(v) for sampling node v is proportional to the squared
    L2 norm of the v-th column of the symmetrically normalized adjacency matrix Ã,
    where Ã = D̂^(-1/2) @ Â @ D̂^(-1/2) and Â = A + I.

    Args:
        adj (torch.Tensor or SparseTensor): The graph's adjacency information,
            typically expected in PyG's edge_index format ([2, num_edges])
            or as a sparse COO tensor.
        num_nodes (int): The total number of nodes (N) in the graph.

    Returns:
        torch.Tensor: A tensor of shape [N] containing the importance sampling
                      probability for each node. Returned on CPU.
    """
    print("Calculating FastGCN sampling probabilities...")
    # Perform calculations on CPU to potentially avoid GPU memory issues for large graphs
    # and ensure consistency, as sampling logic often runs on CPU.
    cpu_device = torch.device('cpu')
    adj = adj.to(cpu_device) # Ensure adjacency is on CPU

    # --- Step 1: Construct Â = A + I (Adjacency matrix with self-loops) ---
    if isinstance(adj, torch.Tensor) and adj.layout == torch.sparse_coo:
        # Ensure sparse tensor is coalesced (indices sorted, duplicates summed)
        adj_sp = adj.coalesce()
    elif isinstance(adj, torch.Tensor) and adj.dim() == 2 and adj.shape[0] == 2: # Standard edge_index format
        edge_index = adj
        # Assume uniform edge weights if none provided. Use float32 for calculations.
        edge_weight = torch.ones(edge_index.size(1), device=cpu_device, dtype=torch.float32)
        # Create a sparse COO tensor representation of A
        adj_sp = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), dtype=torch.float32).coalesce()
    else:
        raise ValueError("Unsupported adjacency format for FastGCN probability calculation. Expected edge_index or sparse COO tensor.")

    # Create the identity matrix I as a sparse tensor
    idx = torch.arange(num_nodes, device=cpu_device).unsqueeze(0).repeat(2, 1)
    val = torch.ones(num_nodes, device=cpu_device, dtype=torch.float32) # Use float32
    identity = torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes))

    # Calculate Â = A + I
    adj_hat = (adj_sp + identity).coalesce() # Â (sparse)

    # --- Step 2: Calculate D̂^(-1/2) (Inverse square root of degree matrix) ---
    # Get row and column indices from Â
    row, col = adj_hat.indices()
    # Get edge values (weights) from Â; default to 1 if not present
    adj_hat_val = adj_hat.values() if adj_hat.values is not None else torch.ones(row.size(0), device=cpu_device, dtype=torch.float32)
    # Calculate the degree D̂ using the row indices and potentially edge weights. Ensure float calculation.
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32) # Specify float dtype
    # Calculate D̂^(-1/2)
    deg_inv_sqrt = deg.pow_(-0.5)
    # Replace potential 'inf' values (from nodes with degree 0) with 0
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    # Create D̂^(-1/2) as a diagonal matrix (dense for easier matmul below, but could be sparse)
    D_inv_sqrt_diag = torch.diag(deg_inv_sqrt) # Dense diag matrix, already float32

    # --- Step 3: Calculate Ã = D̂^(-1/2) @ Â @ D̂^(-1/2) (Normalized adjacency) ---
    try:
        # Attempt sparse matrix multiplication (potentially more efficient)
        D_inv_sqrt_sp = torch.sparse_coo_tensor(idx, deg_inv_sqrt, (num_nodes, num_nodes))
        # Ensure operands are float32. Cast adj_hat just in case.
        a_tilde_sp = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt_sp, adj_hat.float()), D_inv_sqrt_sp)
        # Convert the final sparse result to a dense tensor for norm calculation
        a_tilde = a_tilde_sp.to_dense()
    except Exception as e:
        # Fallback to dense calculation if sparse matmul fails or is unsupported
        print(f"Warning: Sparse matmul failed ({e}), falling back to dense calculation for A_tilde.")
        adj_hat_dense = adj_hat.to_dense() # Convert Â to dense
        # Perform dense matrix multiplication: D̂^(-1/2) @ Â_dense @ D̂^(-1/2)
        a_tilde = D_inv_sqrt_diag @ adj_hat_dense @ D_inv_sqrt_diag # Result is dense float32

    # --- Step 4: Calculate Probabilities q(v) ∝ ||Ã[:, v]||^2 ---
    # Calculate the squared L2 norm for each column (dim=0) of Ã. Ensure float32.
    col_norms_sq = torch.sum(a_tilde.float()**2, dim=0)
    # Normalize the squared norms to get probabilities that sum to 1
    probs = col_norms_sq / torch.sum(col_norms_sq)

    print("FastGCN probabilities calculated.")
    # Return probabilities on the CPU
    return probs.cpu()


class FastGcnSampler:
    """
    Implements the layer-wise importance sampling strategy for FastGCN.

    This sampler selects nodes for each layer based on pre-calculated importance
    probabilities (derived from `calculate_fastgcn_probs`). It constructs
    mini-batches containing the necessary node features and subgraph structures
    for each layer required for a single training step.

    Note: This sampler operates primarily on the CPU.

    Args:
        data (torch_geometric.data.Data): The full graph data object.
        sample_sizes (list[int]): A list where `sample_sizes[l]` is the number
            of nodes to sample for the input of layer `l`. The list length
            should equal `num_layers`.
        probs (torch.Tensor): The importance sampling probabilities for each node,
            as calculated by `calculate_fastgcn_probs`. Shape: [num_nodes].
        batch_size (int): The number of target nodes (output nodes of the GCN)
            to include in each mini-batch.
    """
    def __init__(self, data: Data, sample_sizes: list[int], probs: torch.Tensor, batch_size: int):
        # --- Store graph properties ---
        self.num_nodes = data.num_nodes
        self.num_layers = len(sample_sizes) # Number of GCN layers determines sampling depth

        # --- Store data tensors on CPU ---
        # Node features (ensure float32 for model input)
        self.x = data.x.float().cpu()
        # Node labels (usually long type, keep on CPU until needed)
        self.y = data.y.cpu()
        # Edge index (keep on CPU)
        self.edge_index = data.edge_index.cpu()

        # --- Store sampling parameters ---
        self.sample_sizes = sample_sizes
        # Sampling probabilities (keep on CPU)
        self.probs = probs.cpu()
        self.batch_size = batch_size

        # --- Precompute Adjacency Matrix with Self-Loops (Â) on CPU ---
        # Using SciPy's sparse matrices can be efficient for certain CPU operations.
        # Ensure data types are float32 for consistency with normalization later.
        print("Building adjacency matrix with self-loops for FastGCN sampler (on CPU)...")
        adj_sp = sp.csr_matrix(
            (np.ones(self.edge_index.shape[1], dtype=np.float32), # Weights (ones), float32
             (self.edge_index[0].numpy(), self.edge_index[1].numpy())), # Indices
            shape=(self.num_nodes, self.num_nodes) # Shape
        )
        # Add identity matrix (self-loops), ensuring it's also float32
        identity_sp = sp.identity(self.num_nodes, format='csr', dtype=np.float32)
        self.adj_hat_sp = (adj_sp + identity_sp).tocsr() # Â = A + I in CSR format
        print("Adjacency matrix with self-loops built.")

    def __iter__(self):
        """Creates an iterator yielding layer-wise sampled mini-batches."""
        # Total number of nodes to potentially generate output predictions for.
        # Here, we iterate through all nodes in batches.
        num_output_nodes = self.num_nodes
        # Create a random permutation of all node indices on CPU for batching
        node_indices = torch.randperm(num_output_nodes, device='cpu')

        # Iterate through the permuted nodes in steps of batch_size
        for i in range(0, num_output_nodes, self.batch_size):
            # --- Select Target Nodes for this Batch ---
            # These are the final layer's output nodes for which loss will be computed.
            batch_target_nodes = node_indices[i : i + self.batch_size]

            # --- Layer-wise Importance Sampling ---
            # Start with the target nodes for the last layer's output.
            sampled_nodes_per_layer = [batch_target_nodes]

            # Sample nodes for the preceding layers (from num_layers-1 down to 0)
            for layer in range(self.num_layers - 1, -1, -1):
                # Get the required number of nodes to sample for this layer's input
                num_to_sample = self.sample_sizes[layer]
                # Perform importance sampling using precomputed probabilities `self.probs`
                # `replacement=True` is standard for FastGCN sampling.
                sampled_indices = torch.multinomial(self.probs, num_samples=num_to_sample, replacement=True)
                # Prepend the sampled nodes for this layer to the list
                # The list will be [nodes_for_L0_input, nodes_for_L1_input, ..., nodes_for_LN_output]
                sampled_nodes_per_layer.insert(0, sampled_indices)

            # --- Construct Batch Data Structure ---
            # Create the structured batch data needed for the layer-wise forward pass
            batch_data = self._create_layerwise_batch(sampled_nodes_per_layer)

            # Yield the batch if it's valid (not empty)
            if batch_data:
                yield batch_data

    # def _create_layerwise_batch(self, sampled_nodes_per_layer: list[torch.Tensor]) -> list[dict] | None:
    def _create_layerwise_batch(self, sampled_nodes_per_layer: List[torch.Tensor]) -> Union[List[Dict], None]:
        """
        Constructs the data structure for a single FastGCN mini-batch.

        This involves extracting node features and calculating the normalized
        subgraph adjacency matrix for each layer based on the sampled nodes.

        Args:
            sampled_nodes_per_layer (list[torch.Tensor]): A list where element `l`
                contains the CPU tensor of node indices sampled for the input of layer `l`.
                The last element contains the target node indices.

        Returns:
            list[dict] | None: A list where each element `l` is a dictionary
                containing 'x', 'edge_index', 'edge_weight', and 'nodes' for the
                input to layer `l`. The final element contains the 'nodes' (targets)
                and their 'y' labels. Returns None if any layer results in empty data.
        """
        layer_data = [] # To store data for each layer's computation
        cpu_device = torch.device('cpu') # Ensure operations are on CPU

        # Nodes providing input features to the first GCN layer (layer 0)
        prev_layer_nodes_cpu = sampled_nodes_per_layer[0].to(cpu_device)

        # Iterate through each GCN layer the model has
        for l in range(self.num_layers):
            # Nodes providing input features to the *next* GCN layer (l+1)
            # These define the output nodes needed from the current layer l
            current_layer_nodes_cpu = sampled_nodes_per_layer[l+1].to(cpu_device)

            # --- Extract Features and Subgraph for Current Layer's Input ---
            # Get the features for the input nodes of layer l (already float32 on CPU)
            layer_x_cpu = self.x[prev_layer_nodes_cpu]

            num_sub_nodes = len(prev_layer_nodes_cpu)
            # If no nodes were sampled for this layer's input, the batch is invalid
            if num_sub_nodes == 0:
                print("Warning: Empty node set encountered during batch creation. Skipping batch.")
                return None

            # Extract the sub-adjacency matrix Â_sub corresponding to the connections
            # *among* the nodes in `prev_layer_nodes_cpu`.
            # This uses fancy indexing on the precomputed SciPy CSR matrix Â.
            sub_adj_sp = self.adj_hat_sp[prev_layer_nodes_cpu][:, prev_layer_nodes_cpu]

            # Convert the SciPy sparse subgraph back to PyG's edge_index and edge_weight format.
            # Ensure edge weights derived from SciPy matrix are float32.
            sub_edge_index_th_cpu, sub_edge_weight_th_cpu = from_scipy_sparse_matrix(sub_adj_sp)
            sub_edge_weight_th_cpu = sub_edge_weight_th_cpu.float() # Explicit cast to float32

            # --- Normalize the Subgraph Adjacency (GCN Normalization) ---
            # Calculate D̂_sub^(-1/2) for the extracted subgraph
            sub_row, sub_col = sub_edge_index_th_cpu
            # Calculate degrees *within the subgraph*. Specify float dtype.
            sub_deg = degree(sub_row, num_nodes=num_sub_nodes, dtype=torch.float)
            sub_deg_inv_sqrt = sub_deg.pow(-0.5)
            sub_deg_inv_sqrt.masked_fill_(sub_deg_inv_sqrt == float('inf'), 0) # Handle 0-degree nodes

            # Apply normalization to the edge weights: D̂_sub^(-1/2) @ Â_sub @ D̂_sub^(-1/2)
            # This is done efficiently by scaling edge weights: norm_w_ij = d_i^(-1/2) * w_ij * d_j^(-1/2)
            # Ensure all components are float32 for the multiplication.
            norm_edge_weight_cpu = sub_deg_inv_sqrt[sub_row] * sub_edge_weight_th_cpu * sub_deg_inv_sqrt[sub_col]

            # Store the data required for layer l's computation
            layer_info = {
                'x': layer_x_cpu,                      # Node features (Float32 CPU)
                'edge_index': sub_edge_index_th_cpu,   # Subgraph connectivity (Long CPU)
                'edge_weight': norm_edge_weight_cpu,   # Normalized edge weights (Float32 CPU)
                'nodes': prev_layer_nodes_cpu          # Original indices of input nodes (Long CPU)
            }
            layer_data.append(layer_info)

            # Prepare for the next iteration: the output nodes of this layer become the input nodes for the next
            prev_layer_nodes_cpu = current_layer_nodes_cpu

        # --- Add Final Target Information ---
        # Get the original indices of the target nodes for this batch
        final_nodes_cpu = sampled_nodes_per_layer[-1].to(cpu_device)
        # Check if target node set is empty
        if len(final_nodes_cpu) == 0:
             print("Warning: Empty target node set encountered. Skipping batch.")
             return None

        # Extract the corresponding labels (already on CPU)
        # Ensure labels have the correct dtype (usually long for CrossEntropyLoss)
        batch_y_cpu = self.y[final_nodes_cpu]

        # Append final info needed for loss calculation
        layer_data.append({'nodes': final_nodes_cpu, 'y': batch_y_cpu})

        # Return the list of dictionaries, containing data for each layer + final targets/labels
        return layer_data

    def __len__(self) -> int:
        """Returns the number of mini-batches in one epoch."""
        # Calculate the number of batches needed to cover all nodes
        return (self.num_nodes + self.batch_size - 1) // self.batch_size