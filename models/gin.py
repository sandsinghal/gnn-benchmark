# models/gin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool
from typing import Optional


class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN) model from Xu et al. (2019)
    "How Powerful are Graph Neural Networks?".
    This implementation focuses on the node classification variant.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        num_mlp_layers: int = 2,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        """
        Args:
            in_channels (int): Input feature dimensionality.
            hidden_channels (int): Hidden dimensionality used within MLPs and GIN layers.
            out_channels (int): Output dimensionality (number of classes).
            dropout (float): Dropout probability applied after the first GIN layer.
            num_mlp_layers (int): Number of layers in the MLP for each GINConv. Must be >= 1.
            eps (float): Initial value for the epsilon parameter in GINConv.
            train_eps (bool): If True, the epsilon parameter becomes learnable.
        """
        super().__init__()

        if num_mlp_layers < 1:
            raise ValueError("num_mlp_layers must be at least 1.")

        # Define MLP for the first GIN layer
        mlp1_layers = [
            Linear(in_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
        ]
        for _ in range(num_mlp_layers - 1):
            mlp1_layers.extend(
                [
                    Linear(hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    ReLU(),
                ]
            )
        mlp1 = Sequential(*mlp1_layers)
        self.conv1 = GINConv(nn=mlp1, eps=eps, train_eps=train_eps)

        # Define MLP for the second GIN layer (includes final projection)
        mlp2_layers = [
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
        ]
        for _ in range(num_mlp_layers - 1):
            mlp2_layers.extend(
                [
                    Linear(hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    ReLU(),
                ]
            )
        # Final projection layer within the MLP of the second GINConv
        mlp2_layers.append(Linear(hidden_channels, out_channels))
        mlp2 = Sequential(*mlp2_layers)
        self.conv2 = GINConv(nn=mlp2, eps=eps, train_eps=train_eps)

        self.dropout_p = dropout
        self.num_layers = 2  # Simple property for loader setup

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for GIN node classification.

        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format [2, num_edges].
            edge_weight (Tensor, optional): Edge weights. NOTE: Standard GINConv
                                            does not use edge weights. Defaults to None.

        Returns:
            Tensor: Output node features (logits) [num_nodes, out_channels].
        """
        # GINConv does not use edge_weight
        x = self.conv1(x, edge_index)
        # Activation (ReLU) and BatchNorm are typically included *within* the MLP (nn) passed to GINConv.
        # Dropout is often applied *between* GIN layers.
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        # Output logits
        return x
