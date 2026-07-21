# -*- coding: utf-8 -*-
"""Global configuration for the behavioural‑sentiment trading framework.
Edit these values to run sensitivity or robustness tests.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Centralised hyper‑parameters and data settings."""
    # ----------------------- Data settings ----------------------- #
    TICKER: str = "QQQ"                               # target asset
    START_DATE: str = "2014-01-01"                    # backtest start (inclusive)
    END_DATE: str = "2024-01-01"                      # backtest end   (exclusive)

    LOOKBACK_WINDOW: int = 72                          # rolling time‑window (days)
    FORWARD_DAYS: int = 1                              # prediction horizon (days)

    # Classification thresholds (next‑day return)
    THRESHOLD_LIST: List[float] = (0.008, 0.010, 0.012) # used in robustness sweep
    THRESHOLD: float = 0.010                           # default ±1 % threshold

    # ----------------------- Model hyper‑parameters --------------- #
    INPUT_SIZE: int = 11                               # feature dimension (update if features change)

    # LSTM settings
    LSTM_HIDDEN: int = 128

    # Transformer settings
    TRANSFORMER_DIM: int = 128                         # internal d_model (align with LSTM hidden)
    NUM_HEADS: int = 4
    DROPOUT: float = 0.2

    # ----------------------- Training parameters ----------------- #
    BATCH_SIZE: int = 64
    LR: float = 3e-4                                   # learning rate
    EPOCHS: int = 100
    PATIENCE: int = 10                                 # early‑stopping patience

# expose a default instance for convenience
config = Config()
