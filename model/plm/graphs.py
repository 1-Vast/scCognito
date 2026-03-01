from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _empty_graph(n: int) -> sp.coo_matrix:
    return sp.coo_matrix((n, n), dtype=np.float32)


def build_knn_graph(X: np.ndarray, k: int = 12) -> sp.coo_matrix:
    N = int(X.shape[0])
    if N < 3:
        return _empty_graph(N)

    k_eff = int(max(1, min(int(k), N - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]  # remove self

    rows = np.repeat(np.arange(N), k_eff)
    cols = idx.reshape(-1)
    data = np.ones_like(rows, dtype=np.float32)

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocoo()

def build_radius_graph(X: np.ndarray, radius: float) -> sp.coo_matrix:
    nn = NearestNeighbors(radius=radius, metric="euclidean").fit(X)
    graph = nn.radius_neighbors_graph(X, mode="connectivity")
    A = graph.tocoo()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocoo()

def coo_to_edge_index(A: sp.coo_matrix):
    import torch
    return torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)

def normalize_adj(A: sp.coo_matrix) -> sp.coo_matrix:
    # Symmetric normalization: D^-1/2 A D^-1/2
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv = sp.diags(deg_inv_sqrt)
    An = D_inv @ A @ D_inv
    return An.tocoo()
