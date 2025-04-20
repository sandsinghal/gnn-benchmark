# models/dgcnn_sortpool.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
from typing import Optional, Tuple


class SortPooling(nn.Module):
    """
    The SortPooling layer from "An End-to-End Deep Learning Architecture
    for Graph Classification" (Zhang et al., 2018).

    Sorts node features based on their values in the last channel, effectively
    using graph structural information captured by the final GCN layer's output
    (interpreted as continuous WL colors). Truncates or pads to a fixed size k.
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): The fixed number of nodes to keep after pooling. Must be positive.
        """
        super().__init__()
        if k <= 0:
            raise ValueError("SortPooling k must be a positive integer.")
        self.k = k

    def forward(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x (Tensor): Node features, shape [N, C], where N is the total number
                        of nodes in the batch, C is the number of features/channels.
                        The *last channel* (x[:, -1]) is used for sorting.
            batch (LongTensor): Batch assignment vector indicating graph membership, shape [N].

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - pooled_x (Tensor): Sorted and pooled node features, shape [B, k, C],
                                     where B is the batch size. Padded with zeros if
                                     a graph has fewer than k nodes.
                - perm (Tensor): Permutation indices used for sorting relative to the
                                 dense representation, shape [B, k]. Padded with -1.
                - mask (Tensor): Boolean mask indicating valid nodes (True) vs. padded
                                 nodes (False) after pooling, shape [B, k].
        """
        num_nodes_per_graph = torch.bincount(batch)
        batch_size = num_nodes_per_graph.size(0)
        max_nodes_in_batch = int(num_nodes_per_graph.max().item())

        # Use the last feature channel as the key for sorting nodes within each graph
        sort_key = x[:, -1]

        # Convert to dense representation for easier sorting within each graph.
        # Pad graphs smaller than max_nodes_in_batch with a large negative value in the key
        # to ensure padded nodes appear last after descending sort.
        fill_value_key = -1e10 # Or torch.finfo(x.dtype).min
        dense_x, _ = to_dense_batch(x, batch, max_num_nodes=max_nodes_in_batch, fill_value=0) # [B, max_N, C]
        dense_key, key_mask = to_dense_batch( # [B, max_N], key_mask indicates real nodes
            sort_key, batch, max_num_nodes=max_nodes_in_batch, fill_value=fill_value_key
        )

        # Sort nodes within each graph based on the key (descending order)
        _, perm = dense_key.sort(dim=-1, descending=True)  # [B, max_N] indices

        # Gather sorted features using the permutation indices
        # perm needs expansion to gather along the feature dimension
        perm_expanded = perm.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [B, max_N, C]
        sorted_x = dense_x.gather(dim=1, index=perm_expanded)  # [B, max_N, C]

        # Truncate or pad to exactly k nodes
        if max_nodes_in_batch >= self.k:
            # Truncate if graphs have more than k nodes
            pooled_x = sorted_x[:, : self.k]
            pooled_perm = perm[:, : self.k]
        else:
            # Pad with zeros if graphs have fewer than k nodes
            pad_size = self.k - max_nodes_in_batch
            # Pad the feature tensor (dim 1 is node dim, dim 2 is feature dim)
            pooled_x = F.pad(sorted_x, (0, 0, 0, pad_size), value=0) # Pad nodes dim
            # Pad the permutation tensor
            pooled_perm = F.pad(perm, (0, pad_size), value=-1) # Pad with dummy index -1

        # Create the output mask based on original graph sizes, indicating real vs padded nodes
        # Shape [B, k]
        pooled_mask = torch.zeros(batch_size, self.k, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            num_valid_nodes = min(num_nodes_per_graph[i].item(), self.k)
            pooled_mask[i, :num_valid_nodes] = True

        return pooled_x, pooled_perm, pooled_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


class DGCNN_SortPool(nn.Module):
    """
    Deep Graph CNN (DGCNN) model for graph classification based on Zhang et al. (2018).
    Uses GCN layers, concatenation, SortPooling, 1D Convolutions, and Dense layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int, # Hidden dim for GCN layers
        out_channels: int, # Num classes
        dropout: float,
        num_gcn_layers: int = 4,
        sortpool_k: int = 30,
        conv1d_channels: int = 16,
        conv1d_kernel_size: int = 5,
        dense_hidden_units: int = 128,
    ):
        """
        Args:
            in_channels (int): Input node feature dimensionality.
            hidden_channels (int): Hidden dimensionality for the GCN layers.
            out_channels (int): Output dimensionality (number of graph classes).
            dropout (float): Dropout probability for dense layers.
            num_gcn_layers (int): Number of GCN layers. The last layer outputs 1 channel for sorting.
            sortpool_k (int): Number of nodes to keep after SortPooling.
            conv1d_channels (int): Number of output channels for the first 1D conv layer.
            conv1d_kernel_size (int): Kernel size for the 1D conv layers.
            dense_hidden_units (int): Number of units in the hidden dense layer.
        """
        super().__init__()
        self.sortpool_k = sortpool_k
        self.dropout_p = dropout
        self.num_layers = num_gcn_layers # Reflects GCN layers for consistency

        # --- Graph Convolution Layers ---
        self.gcn_layers = nn.ModuleList()
        current_dim = in_channels
        gcn_output_dims: List[int] = [] # Keep track of output dims for concatenation
        for i in range(num_gcn_layers):
            # Last GCN layer outputs 1 channel, used for sorting in SortPool
            out_dim = 1 if i == num_gcn_layers - 1 else hidden_channels
            # Use cached=False for graph classification batches (graphs change each batch)
            self.gcn_layers.append(GCNConv(current_dim, out_dim, cached=False))
            current_dim = out_dim
            gcn_output_dims.append(current_dim)

        # Total dimension after concatenating outputs from all GCN layers (incl. input & final sort channel)
        self.total_gcn_output_dim = (
            in_channels + sum(gcn_output_dims[:-1]) + 1 # C_in + C_h1 + ... + C_h_last-1 + 1
        )
        print(f"DGCNN: Total concatenated GCN output dim = {self.total_gcn_output_dim}")

        # --- SortPooling Layer ---
        self.sort_pool = SortPooling(k=sortpool_k)

        # --- 1D Convolutional Layers (applied after SortPooling) ---
        # Input to Conv1D is [B, C_total, k]
        self.conv1d_1 = nn.Conv1d(
            self.total_gcn_output_dim,
            conv1d_channels,
            kernel_size=conv1d_kernel_size,
            stride=1,
        )
        # Calculate output length after first 1D conv (L_out = L_in - Kernel + 1) / Stride
        conv1_output_len = self.sortpool_k - conv1d_kernel_size + 1

        # Max Pooling Layer (reduces length)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        pool1_output_len = conv1_output_len // 2

        # Second 1D Conv Layer
        self.conv1d_2 = nn.Conv1d(
            conv1d_channels,
            conv1d_channels * 2, # Often increase channels
            kernel_size=conv1d_kernel_size,
            stride=1,
        )
        conv2_output_len = pool1_output_len - conv1d_kernel_size + 1

        # --- Dense Layers (Classifier Head) ---
        # Calculate input size for the first dense layer after flattening 1D conv output
        dense_input_units = (conv1d_channels * 2) * conv2_output_len
        if dense_input_units <= 0:
            raise ValueError(
                f"Calculated negative or zero input size ({dense_input_units}) for dense layer. "
                f"Check SortPool k ({sortpool_k}), Conv1D kernel sizes ({conv1d_kernel_size}), "
                f"and pooling parameters."
            )
        print(f"DGCNN: Input units to dense layer = {dense_input_units}")

        self.dense1 = nn.Linear(dense_input_units, dense_hidden_units)
        self.dense2 = nn.Linear(dense_hidden_units, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for DGCNN graph classification.

        Args:
            x (Tensor): Node feature matrix [N, C_in].
            edge_index (LongTensor): Graph connectivity [2, E].
            batch (LongTensor): Batch vector [N].
            edge_weight (Tensor, optional): Edge weights [E]. Defaults to None.

        Returns:
            Tensor: Graph-level output logits [B, out_channels].
        """
        gcn_outputs = [x]  # Store outputs of each GCN layer, starting with input features

        # Apply GCN layers
        current_features = x
        for i, layer in enumerate(self.gcn_layers):
            current_features = layer(current_features, edge_index, edge_weight)
            # Use tanh activation as suggested in paper's Figure 2 caption.
            # The last layer (outputting 1 channel for sorting) doesn't need activation here.
            if i < len(self.gcn_layers) - 1:
                current_features = torch.tanh(current_features)
            gcn_outputs.append(current_features)

        # Concatenate features from all GCN layers (results in [N, C_total])
        z1h = torch.cat(gcn_outputs, dim=-1)

        # Apply SortPooling ([N, C_total] -> [B, k, C_total])
        pooled_x, _, _ = self.sort_pool(z1h, batch)

        # Prepare for 1D CNNs: Permute to shape [B, C_total, k]
        cnn_input = pooled_x.permute(0, 2, 1)

        # Apply 1D Convolutions and Pooling
        out = F.relu(self.conv1d_1(cnn_input))
        out = self.maxpool1d(out)
        out = F.relu(self.conv1d_2(out))

        # Flatten for Dense layers ([B, C_conv_out * L_conv_out] -> [B, DenseInputUnits])
        out = out.view(out.size(0), -1)

        # Apply Dense layers (classifier)
        out = F.relu(self.dense1(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.dense2(out)  # Output graph classification logits

        # Logits are returned, loss function (e.g., CrossEntropyLoss) will handle softmax/sigmoid
        return out
