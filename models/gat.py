# models/gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv
from typing import Optional


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) model based on Veličković et al. (2018).
    Uses multi-head attention layers. Designed for node classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels_per_head: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
        output_heads: int = 1,
        concat_output: bool = False,
        negative_slope: float = 0.2,
    ):
        """
        Args:
            in_channels (int): Input feature dimensionality.
            hidden_channels_per_head (int): Feature dimensionality per head in the first layer.
                                          Total hidden channels = heads * hidden_channels_per_head.
            out_channels (int): Output dimensionality (number of classes).
            heads (int): Number of attention heads in the first layer.
            dropout (float): Dropout probability for input, attention, and hidden features.
            output_heads (int): Number of attention heads in the final layer.
            concat_output (bool): If True and output_heads > 1, concatenate final head outputs.
                                  Otherwise (False), average them.
            negative_slope (float): LeakyReLU negative slope for the attention mechanism.
        """
        super().__init__()
        self.dropout_p = dropout
        self.negative_slope = negative_slope
        self.num_layers = 2  # Simple property for loader setup

        # First GAT layer: Multi-head attention, concatenating head outputs
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels_per_head,
            heads=heads,
            dropout=dropout, # Apply dropout to attention coefficients
            negative_slope=negative_slope,
            concat=True,
        )

        # Second GAT layer (output layer):
        # Input dimension is heads * hidden_channels_per_head
        self.conv2 = GATConv(
            in_channels=hidden_channels_per_head * heads,
            out_channels=out_channels,
            heads=output_heads,
            dropout=dropout, # Apply dropout to attention coefficients
            negative_slope=negative_slope,
            concat=concat_output, # Average or concat based on flag for the final layer
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for GAT.

        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format [2, num_edges].
            edge_weight (Tensor, optional): Edge weights. NOTE: Standard GATConv
                                            implementation in PyG does not directly use edge weights
                                            for attention calculation, but accepts the argument
                                            for API consistency. Defaults to None.

        Returns:
            Tensor: Output node features (logits) [num_nodes, out_channels].
        """
        # Apply dropout to input features - common practice from GAT paper's code
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # First layer
        # GATConv does not use edge_weight in its standard attention calculation
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU activation used in the paper

        # Apply dropout to hidden features
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Second layer (output logits)
        # GATConv does not use edge_weight here either
        x = self.conv2(x, edge_index)
        return x