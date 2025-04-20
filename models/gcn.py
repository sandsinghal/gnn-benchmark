# models/gcn.py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from typing import Optional


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model based on Kipf & Welling (2017).
    Uses two GCNConv layers. Caching is enabled, assuming full-batch training
    for datasets where this model is typically used (e.g., Planetoid).
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ):
        super().__init__()
        # cached=True is efficient for full-batch node classification on single static graphs
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout_p = dropout
        self.num_layers = 2  # Simple property for loader setup

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for GCN.

        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format [2, num_edges].
            edge_weight (Tensor, optional): Edge weights [num_edges]. Defaults to None.

        Returns:
            Tensor: Output node features (logits) [num_nodes, out_channels].
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # Output logits, nn.CrossEntropyLoss will handle softmax internally
        return x