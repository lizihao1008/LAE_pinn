"""
models/gnn_encoder.py
GNN environment encoder: LAE graph → per-node environment embeddings.

The GNN's role is NOT to predict the ionization field directly.
It models the LAE environment — local overdensity, neighbour mark clustering,
group topology — and outputs an environment embedding h_i per galaxy.
This embedding is then passed to the source head to predict f_esc,i.

Architecture: GATv2Conv (graph attention, v2) or SAGEConv.
Edge features (r, dx/r, dy/r, dz/r) are used as edge attributes.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, SAGEConv, BatchNorm


class GATv2Encoder(nn.Module):
    """
    Multi-layer GATv2 encoder.

    Input:  node features (N, in_channels)
            edge attributes (E, edge_dim)
    Output: node embeddings (N, out_channels)
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        out_channels: int = 32,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 4,
    ):
        super().__init__()
        self.dropout = dropout

        layers = []
        norms  = []

        # First layer: in_channels → hidden_dim * heads
        layers.append(GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
        ))
        norms.append(BatchNorm(hidden_dim * heads))

        # Middle layers
        for _ in range(n_layers - 2):
            layers.append(GATv2Conv(
                in_channels=hidden_dim * heads,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True,
            ))
            norms.append(BatchNorm(hidden_dim * heads))

        # Last layer: aggregate heads → out_channels
        if n_layers >= 2:
            layers.append(GATv2Conv(
                in_channels=hidden_dim * heads,
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=False,
            ))
            norms.append(BatchNorm(out_channels))
        else:
            # n_layers == 1: collapse directly
            layers[0] = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=False,
            )
            norms[0] = BatchNorm(out_channels)

        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,            # (N, in_channels)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: torch.Tensor,    # (E, edge_dim)
    ) -> torch.Tensor:
        """Returns (N, out_channels) node embeddings."""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGEEncoder(nn.Module):
    """GraphSAGE encoder (no edge features, simpler, faster)."""

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        out_channels: int = 32,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        dims = [in_channels] + [hidden_dim] * (n_layers - 1) + [out_channels]
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([BatchNorm(dims[i + 1]) for i in range(n_layers)])
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,   # unused
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


GNN_REGISTRY = {"GATv2Conv": GATv2Encoder, "SAGEConv": SAGEEncoder}


def build_gnn_encoder(
    architecture: str = "GATv2Conv",
    in_channels: int = 8,
    hidden_dim: int = 64,
    out_channels: int = 32,
    n_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    edge_dim: int = 4,
) -> nn.Module:
    if architecture not in GNN_REGISTRY:
        raise ValueError(f"Unknown GNN architecture '{architecture}'. "
                         f"Available: {list(GNN_REGISTRY)}")
    cls = GNN_REGISTRY[architecture]
    if architecture == "GATv2Conv":
        return cls(in_channels, hidden_dim, out_channels, n_layers, heads, dropout, edge_dim)
    else:
        return cls(in_channels, hidden_dim, out_channels, n_layers, dropout)
