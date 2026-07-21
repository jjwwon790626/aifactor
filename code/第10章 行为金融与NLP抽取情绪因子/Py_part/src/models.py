# -*- coding: utf-8 -*-
"""Neural network architectures for the behaviour‑sentiment project.

This file contains a single public class `HybridModel` that replicates
*exactly* the structure描述 in Chapter 4 of the manuscript: an LSTM to
capture short‑term temporal dynamics followed by an attention‑based
Transformer block for higher‑order feature interactions, and finally a
softmax head for 3‑way classification (Up / Neutral / Down).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from config import Config

cfg = Config()

class HybridModel(nn.Module):
    """LSTM ➜ Linear proj ➜ Transformer ➜ Softmax."""

    def __init__(self, cfg: Config = cfg):
        super().__init__()
        self.cfg = cfg

        # 1. Temporal encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=cfg.INPUT_SIZE,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
            dropout=cfg.DROPOUT,
        )

        # 2. Dimensionality projection (optionally identity)
        lstm_dim = cfg.LSTM_HIDDEN
        tf_dim = cfg.TRANSFORMER_DIM
        self.proj = nn.Identity() if lstm_dim == tf_dim else nn.Linear(lstm_dim, tf_dim)

        # 3. Transformer encoder for cross‑factor interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_dim,
            nhead=cfg.NUM_HEADS,
            dropout=cfg.DROPOUT,
            batch_first=True,
            dim_feedforward=tf_dim * 4,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(tf_dim),
            nn.Linear(tf_dim, 3)  # logits for 3 classes
        )

        self.softmax = nn.Softmax(dim=-1)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_prob: bool = False) -> torch.Tensor:
        """Shape: ``x`` = (B, T, F).

        If *return_prob* is True, apply softmax before returning.
        """
        # LSTM
        x, _ = self.lstm(x)              # (B, T, H)

        # (Optional) projection to transformer dimension
        x = self.proj(x)                 # (B, T, D)

        # Transformer
        x = self.transformer(x)          # (B, T, D)

        # Take last time‑step token (equivalent to [CLS] pooling)
        feat = x[:, -1, :]               # (B, D)

        logits = self.cls_head(feat)     # (B, 3)
        if return_prob:
            return self.softmax(logits)
        return logits


__all__ = ["HybridModel"]
