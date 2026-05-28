import math
from typing import Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _spearman_ranks_correlation(vec_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """
    Fast Spearman correlation between a 1D vector `vec_a` (length G) and each row of `mat_b` (N x G),
    returning correlation per row (length N).
    Implementation: compute ranks for both and use Pearson correlation formula.
    """
    # ranks
    ra = rankdata(vec_a).astype(float)
    rb = np.apply_along_axis(rankdata, 1, mat_b).astype(float)  # shape (N, G)

    # center
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean(axis=1, keepdims=True)

    num = (rb_c * ra_c).sum(axis=1)
    den = np.sqrt(((rb_c ** 2).sum(axis=1)) * ((ra_c ** 2).sum()))
    # avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = num / den
    rho = np.nan_to_num(rho, nan=0.0)
    return rho


def compute_connection_score(
    signature: Mapping[str, float],
    cell_expression: pd.DataFrame,
    selected_cells: Optional[Iterable[str]] = None,
    normalize: Optional[str] = None,
    fisher_clip: float = 0.999999,
) -> Tuple[pd.Series, Optional[float]]:
    """
    Compute scFOCAL-style connection scores between a single compound signature and cells.

    Steps (matching scFOCAL implementation):
    1. Intersect genes between `signature` and `cell_expression` columns.
    2. Compute Spearman correlation between signature vector and each cell's expression vector.
    3. Apply Fisher z-transform: z = 0.5 * log((1 + rho) / (1 - rho))
    4. Optionally normalize the resulting z-scores with 'minmax' or 'zscore'.
    5. Optionally compute mean connectivity (mrc) over `selected_cells`.

    Args:
        signature: mapping gene -> value (e.g., L1000 TCS log-fold-change).
        cell_expression: DataFrame indexed by cell IDs, columns are gene symbols. Values are expression (scale.data).
        selected_cells: optional iterable of cell IDs to aggregate mean connectivity (mrc).
        normalize: None | 'minmax' | 'zscore' - if provided, normalize z-scores.
        fisher_clip: clipping value for rho before Fisher transform to avoid infinities.

    Returns:
        (z_scores, mrc)
        - z_scores: pd.Series indexed by cell IDs containing Fisher z-transformed connectivity.
        - mrc: float mean of z_scores over selected_cells if provided, otherwise None.
    """
    if len(signature) == 0:
        raise ValueError("Empty signature provided.")

    # prepare signature vector and intersect genes
    sig_genes = set(signature.keys())
    expr_genes = set(cell_expression.columns)
    common_genes = list(sig_genes & expr_genes)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between signature and cell_expression.")

    # build arrays: signature vector (G,), expression matrix (N_cells x G)
    sig_vec = np.array([signature[g] for g in common_genes], dtype=float)
    expr_mat = cell_expression.loc[:, common_genes].values  # shape (N_cells, G)

    # compute Spearman rho per cell (N,)
    rho = _spearman_ranks_correlation(sig_vec, expr_mat)

    # clip rho to avoid infinite Fisher transform
    rho = np.clip(rho, -fisher_clip, fisher_clip)

    # fisher z transform
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 0.5 * np.log((1.0 + rho) / (1.0 - rho))
    z = np.nan_to_num(z, nan=0.0, posinf=np.sign(z) * np.finfo(float).max, neginf=np.sign(z) * np.finfo(float).min)

    z_series = pd.Series(z, index=cell_expression.index, name="connection_z")

    # optional normalization
    if normalize is not None:
        if normalize == "minmax":
            minv = z_series.min()
            maxv = z_series.max()
            if maxv - minv > 0:
                z_series = (z_series - minv) / (maxv - minv)
            else:
                z_series = z_series * 0.0
        elif normalize == "zscore":
            mean = z_series.mean()
            std = z_series.std()
            if std > 0:
                z_series = (z_series - mean) / std
            else:
                z_series = z_series * 0.0
        else:
            raise ValueError("Unsupported normalize option: choose None, 'minmax' or 'zscore'.")

    # compute mrc if requested
    mrc = None
    if selected_cells is not None:
        sel = [c for c in selected_cells if c in z_series.index]
        if len(sel) == 0:
            mrc = float("nan")
        else:
            mrc = float(z_series.loc[sel].mean())

    return z_series, mrc




