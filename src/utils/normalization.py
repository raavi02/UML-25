"""Pixel <-> normalized coordinate transforms."""
def to_norm(x_px, y_px, H, W): return x_px / W, y_px / H
def to_px(x_norm, y_norm, H, W): return x_norm * W, y_norm * H
