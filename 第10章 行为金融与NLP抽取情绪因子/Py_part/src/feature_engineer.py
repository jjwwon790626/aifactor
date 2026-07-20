# -*- coding: utf-8 -*-
"""Feature‑engineering helpers: label creation & rolling window builder.

This module is aligned with the 1‑day prediction horizon specified in
`config.py`.  It exposes two public functions:

- `create_labels(df)`   → returns DataFrame with `Future_Return` & `Label`.
- `build_sequences(df, feature_cols)` → tensors ready for model training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from config import Config

cfg = Config()

# ---------------------------------------------------------------------------
# Label engineering
# ---------------------------------------------------------------------------

def create_labels(
    df: pd.DataFrame,
    forward_days: int = cfg.FORWARD_DAYS,
    threshold: float = cfg.THRESHOLD,
) -> pd.DataFrame:
    """Attach future‑return label (+1 / 0 / ‑1) to *price‑indexed* DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column ``'Adj Close'`` and be indexed by dates.
    forward_days : int, default from ``cfg``
        Look‑ahead horizon (number of trading days).
    threshold : float, default from ``cfg``
        Absolute return cut‑off such that
        * `>= threshold` → **+1**  (bullish)
        * `<= -threshold` → **‑1** (bearish)
        * otherwise       → **0**  (neutral)

    Returns
    -------
    pd.DataFrame
        Original frame with two extra columns:
        ``'Future_Return'`` and ``'Label'``.  The final *forward_days* rows
        are dropped because their future price is unknown.
    """
    _df = df.copy()

    # next‑day (or k‑day) % return
    _df["Future_Return"] = (
        _df["Adj Close"].pct_change(periods=forward_days).shift(-forward_days)
    )

    _df["Label"] = np.select(
        [
            _df["Future_Return"] >= threshold,
            _df["Future_Return"] <= -threshold,
        ],
        [1, -1],
        default=0,
    )

    # 剔除无法计算未来收益的尾部样本
    if forward_days > 0:
        _df = _df.iloc[:-forward_days].copy()

    return _df


# ---------------------------------------------------------------------------
# Sequence builder for LSTM / Transformer
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int = cfg.LOOKBACK_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Convert feature DataFrame into 3‑D tensor windows.

    Returns ``X, y, dates`` where ``dates`` is the end‑timestamp of each
    sample (useful for aligning predictions back to price series).
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    dates: List[pd.Timestamp] = []

    for end_idx in range(lookback, len(df)):
        window = df.iloc[end_idx - lookback : end_idx]

        # skip if any NaNs in the window
        if window[feature_cols].isna().any().any():
            continue

        X.append(window[feature_cols].values)
        y.append(int(df.iloc[end_idx]["Label"]))
        dates.append(df.index[end_idx])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        dates,
    )
