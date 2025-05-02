# models/fastgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FastGCN(nn.Module):
    """
    Implements the FastGCN model architecture as described in Chen et al. (2018).

    This model uses standard GCNConv layers for the graph convolutions.
    The "Fast" characteristic of FastGCN originates from the layer-wise
    importance sampling strategy employed during the training phase.
    This sampling mechanism is typically implemented externally, for instance,
    within a custom data loader or the training loop itself, rather than
    being part of the model's forward pass definition.

    Args:
        in_channels (int): Dimensionality of input node features.
        hidden_channels (int): Dimensionality of the hidden layer features.
        out_channels (int): Dimensionality of the output node features (e.g., number of classes).
        dropout (float): Dropout probability applied after the activation function.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()

        # Layer 1: GCN Convolution from input features to hidden features.
        # `cached=False` is used because FastGCN training involves mini-batches
        # built from varying subgraphs (due to sampling), so caching node features
        # from a full pass isn't applicable or beneficial.
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)

        # Layer 2: GCN Convolution from hidden features to output features.
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)

        # Dropout probability.
        self.dropout = dropout

        # Number of convolutional layers in this specific architecture.
        # Useful for setting up samplers that depend on the number of layers.
        self.num_layers = 2

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the FastGCN model using standard GCN operations.

        Args:
            x (torch.Tensor): Node feature matrix (shape: [num_nodes, in_channels]).
            edge_index (torch.Tensor): Graph connectivity in COO format (shape: [2, num_edges]).
            edge_weight (torch.Tensor, optional): Edge weights (shape: [num_edges]). Defaults to None.

        Returns:
            torch.Tensor: The output node embeddings (logits), shape [num_nodes, out_channels].
        """
        # Apply first GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        # Apply ReLU activation function
        x = F.relu(x)
        # Apply dropout for regularization
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Apply second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        # Return the final node embeddings (logits)
        return x

    # --- Important Note on FastGCN Implementation ---
    # To fully leverage the benefits of FastGCN, a specific layer-wise
    # importance sampling technique must be implemented during data loading or
    # within the training loop. This typically involves calculating sampling
    # probabilities based on the squared L2 norm of the columns of the
    # normalized adjacency matrix (q(v) ∝ ||Ã(:, v)||^2).
    # The `utils/sampling.py` file provides an example sampler, but note that
    # the reference `utils/training.py` might use standard `NeighborLoader` by default,
    # which performs neighbor sampling, not the layer-wise importance sampling
    # specific to FastGCN. Ensure the correct sampler is used during training.
    # ---