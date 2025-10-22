"""Build a per-frame graph object (placeholder)."""
from dataclasses import dataclass
@dataclass
class GraphData:
    x: list   # node features
    edges: list  # list of (src, dst)

def build_frame_graph(pred_df_frame, calib: dict, skeleton) -> GraphData:
    # TODO: build nodes [x_norm,y_norm,conf] and edges: intra-skeleton + cross-view
    return GraphData(x=[], edges=[])
