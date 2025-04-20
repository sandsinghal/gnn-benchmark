# models/chebnet.py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import ChebConv
from typing import Optional


class ChebNet(torch.nn.Module):
    """
    Chebyshev Spectral CNN (ChebNet) model from Defferrard et al. (2016).
    Uses Chebyshev polynomial filters implemented by ChebConv for node classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        K: int,
    ):
        """
        Args:
            in_channels (int): Input feature dimensionality.
            hidden_channels (int): Hidden dimensionality.
            out_channels (int): Output dimensionality (number of classes).
            dropout (float): Dropout probability.
            K (int): Chebyshev filter order (number of hops).
        """
        super().__init__()
        # Note: K controls the receptive field size (hops)
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)
        self.dropout_p = dropout
        self.K = K
        self.num_layers = 2  # Simple property for loader setup

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        lambda_max: Optional[Tensor] = None,
        batch: Optional[Tensor] = None, # Add batch for potential compatibility, though usually used full-batch
    ) -> Tensor:
        """
        Forward pass for ChebNet.

        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format [2, num_edges].
            edge_weight (Tensor, optional): Edge weights [num_edges]. Defaults to None.
            lambda_max (Tensor, optional): The largest eigenvalue of the normalized
                Laplacian. If not provided, it's estimated, which can add overhead.
                Recommended to precompute for training efficiency. Shape [batch_size]
                if batch is provided, otherwise scalar. Defaults to None.
            batch (LongTensor, optional): Batch vector [num_nodes], for minibatching.
                Defaults to None.

        Returns:
            Tensor: Output node features (logits) [num_nodes, out_channels].
        """
        # ChebConv requires lambda_max for stable computation.
        # If None, it will be computed internally (less efficient).
        x = self.conv1(x, edge_index, edge_weight=edge_weight, batch=batch, lambda_max=lambda_max)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight, batch=batch, lambda_max=lambda_max)
        # Output logits
        return x
