"""Minimal GNN refiner placeholder (to be swapped with PyG later)."""
class GNNRefiner:
    def __init__(self, hidden_dim: int = 64): self.hidden_dim = hidden_dim
    def forward(self, graph): return graph  # TODO: return refined (x_norm_ref, y_norm_ref)
