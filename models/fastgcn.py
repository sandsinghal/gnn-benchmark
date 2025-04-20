# models/fastgcn.py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from typing import Optional


class FastGCN(torch.nn.Module):
    """
    FastGCN model structure based on Chen et al. (2018).
    Uses standard GCNConv layers for node classification.
    The "Fast" aspect comes from the layer-wise importance sampling
    strategy during training, which is handled *outside* this model definition
    by a custom sampler (e.g., `utils.sampling.FastGcnSampler`).

    For inference/evaluation, this model is used like a standard GCN, typically
    on the full graph.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ):
        super().__init__()
        # Use cached=False as FastGCN training involves mini-batching on varying subgraphs.
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)
        self.dropout_p = dropout
        self.num_layers = 2  # Simple property

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass using standard GCNConv layers.

        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels]. In FastGCN training,
                        num_nodes corresponds to the number of nodes sampled for the *input*
                        to the current layer.
            edge_index (LongTensor): Graph connectivity of the sampled subgraph for the current
                                     layer, in COO format [2, num_edges]. Indices are relative
                                     to the nodes in `x`.
            edge_weight (Tensor, optional): Renormalized edge weights for the sampled subgraph
                                            [num_edges]. Defaults to None.

        Returns:
            Tensor: Output node features (logits) [num_nodes, out_channels].
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # Output logits
        return x

    # NOTE: The performance benefits of FastGCN require the custom layer-wise sampling
    # implemented in `utils.sampling.FastGcnSampler` and used within the
    # `utils.training.train_fastgcn` function. Standard evaluation typically uses
    # the full graph (`utils.training.evaluate_fastgcn`).
