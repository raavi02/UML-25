from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_loo_recon_plots(
    eval_dir: str | Path,
    keypoint_names: Sequence[str] | None,
    base_errors: np.ndarray,
    recon_errors: np.ndarray,
    drop_mask: np.ndarray,
    tag: str,
) -> None:
    """
    Save per-keypoint plots comparing base vs reconstruction errors for dropped joints.
    This mirrors the refine-mode pixel error plots but focuses on LOO examples only.
    """

    stats_dir = Path(eval_dir) / "loo_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    if keypoint_names is None or len(keypoint_names) == 0:
        keypoint_names = [f"kp_{i}" for i in range(base_errors.shape[1])]

    drop_mask = drop_mask > 0.5
    base_errors = np.asarray(base_errors)
    recon_errors = np.asarray(recon_errors)

    saved_any = False
    for idx, kp in enumerate(keypoint_names):
        kp_mask = drop_mask[:, idx]
        if not np.any(kp_mask):
            continue

        base_k = base_errors[kp_mask, idx]
        recon_k = recon_errors[kp_mask, idx]

        order = np.argsort(base_k)
        base_sorted = base_k[order]
        recon_sorted = recon_k[order]

        plt.figure(figsize=(7, 5))
        plt.plot(base_sorted, label="Base Error", linewidth=2)
        plt.plot(recon_sorted, label="Reconstruction Error", linewidth=2)
        plt.xlabel("Dropped frames (sorted by base error)")
        plt.ylabel("Pixel Error")
        plt.title(f"{kp} — LOO Base vs Reconstruction Error")
        plt.grid(alpha=0.3)
        plt.legend()

        kp_slug = str(kp).replace(" ", "_")
        out_plot = stats_dir / f"loo_error_compare_{tag}_{kp_slug}.png"
        plt.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close()
        saved_any = True

    if saved_any:
        print(f"Saved LOO comparison stats \u2192 {stats_dir}")
    else:
        print("No dropped joints available for LOO plotting; skipping plots.")
