"""
data/graph_builder.py
Build a k-NN graph from LAE positions for PyG.

Handles periodic boundary conditions (PBC) via minimum-image convention.
Returns a torch_geometric.data.Data object.
"""

from __future__ import annotations
import torch
import numpy as np
from torch_geometric.data import Data


def _pbc_displacement(pos_a: np.ndarray, pos_b: np.ndarray, box: float) -> np.ndarray:
    """Minimum-image displacement vector a→b in a periodic box [0, box)."""
    d = pos_b - pos_a
    d -= np.round(d / box) * box
    return d


def build_knn_graph(
    pos: np.ndarray,          # (N, 3) positions in cMpc/h
    node_feats: torch.Tensor,  # (N, F) node feature matrix
    box_size: float,
    k: int = 16,
    r_max: float = 15.0,       # max edge length (cMpc/h)
    device: str | torch.device = "cpu",
    subsample: int | None = None,
) -> Data:
    """
    Build a k-NN graph with periodic boundary conditions.

    For large N (> ~50 000), exact pairwise distances are expensive.
    Use subsample to randomly thin the catalog first (for speed),
    or use a spatial tree approach.

    Returns a PyG Data object with:
        x:          (N, F) node features
        pos:        (N, 3) normalised positions [0,1]
        edge_index: (2, E) source/target indices
        edge_attr:  (E, 4) [r, dx/r, dy/r, dz/r] edge features
    """
    N = len(pos)

    if subsample is not None and N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        pos = pos[idx]
        node_feats = node_feats[idx]
        N = subsample

    # For large catalogs use a KD-tree; for small ones brute-force is fine.
    # We use scipy's cKDTree with PBC via repeated copies in border regions.
    from scipy.spatial import cKDTree

    # Standard KD-tree (non-PBC); then check PBC images for border halos.
    # Simple approach: wrap positions into 27 images and query once.
    # This is O(N log N) and correct for r_max << box_size/2.
    tree = cKDTree(pos, boxsize=box_size)
    dist_matrix, idx_matrix = tree.query(pos, k=k + 1, workers=-1,
                                          distance_upper_bound=r_max)
    # idx_matrix[:, 0] is self; skip it
    src_list, dst_list, r_list, d_list = [], [], [], []
    for i in range(N):
        for j_idx in range(1, k + 1):
            j = idx_matrix[i, j_idx]
            r = dist_matrix[i, j_idx]
            if j >= N or r >= r_max:
                continue
            disp = _pbc_displacement(pos[i], pos[j], box_size)
            src_list.append(i)
            dst_list.append(j)
            r_list.append(r)
            d_list.append(disp / (r + 1e-8))   # unit vector

    if len(src_list) == 0:
        # Fallback: connect each node to its nearest neighbour
        src_list, dst_list, r_list, d_list = [], [], [], []
        for i in range(N):
            j = idx_matrix[i, 1] if idx_matrix[i, 1] < N else 0
            r = float(dist_matrix[i, 1]) if dist_matrix[i, 1] < r_max * 10 else 1.0
            disp = _pbc_displacement(pos[i], pos[j], box_size)
            src_list.append(i)
            dst_list.append(j)
            r_list.append(r)
            d_list.append(disp / (r + 1e-8))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    r_tensor   = torch.tensor(r_list, dtype=torch.float32).unsqueeze(-1)
    d_tensor   = torch.tensor(np.array(d_list), dtype=torch.float32)
    edge_attr  = torch.cat([r_tensor, d_tensor], dim=-1)  # (E, 4)

    pos_norm = torch.from_numpy((pos / box_size).astype(np.float32))

    data = Data(
        x=node_feats,
        pos=pos_norm,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data.to(device)


def build_graph_from_snapshot(
    snap_dict: dict,
    k: int = 16,
    r_max: float = 15.0,
    subsample: int | None = None,
) -> Data:
    """
    Convenience wrapper that takes a preprocessed snapshot dict
    (from preprocessing.prepare_snapshot) and returns a PyG Data object
    with extra fields needed by the physics modules:
        src_weights: (N,)  raw source weights
        xi_global:  scalar
        xbox_true:  (1,1,G,G,G)
    """
    pos_raw   = snap_dict["pos_raw"].cpu().numpy()
    node_feats = snap_dict["node_feats"]
    device    = node_feats.device

    graph = build_knn_graph(
        pos=pos_raw,
        node_feats=node_feats,
        box_size=snap_dict["box_size"],
        k=k,
        r_max=r_max,
        device=device,
        subsample=subsample,
    )

    # Attach extra fields
    graph.src_weights   = snap_dict["src_weights"]
    graph.pos_raw       = snap_dict["pos_raw"]         # (N, 3) cMpc/h
    graph.xbox_true     = snap_dict["xbox_true"]       # (1,1,G,G,G)
    graph.density_basis = snap_dict["density_basis"]   # (P,G,G,G)
    graph.xi_global     = snap_dict["xi_global"]
    graph.z             = snap_dict["z"]
    graph.box_size      = snap_dict["box_size"]
    graph.grid_size     = snap_dict["grid_size"]

    return graph
