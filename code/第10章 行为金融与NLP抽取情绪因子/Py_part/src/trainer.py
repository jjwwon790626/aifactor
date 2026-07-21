# -*- coding: utf-8 -*-
"""Training helper with rich metrics.

The revised `Trainer` now reports **accuracy, precision, recall, and
F1‑score** on any validation / test loader, so that section 5.1 in the
book can directly quote the numbers.

Usage
-----
>>> trainer = Trainer(model, device, cfg)
>>> trainer.train_epoch(train_loader)     # one epoch
>>> loss, acc, prec, rec, f1 = trainer.evaluate(val_loader)
>>> trainer.save_model("best.pth")
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

class Trainer:
    """Lightweight wrapper for training / evaluation."""

    def __init__(self, model: nn.Module, device: torch.device, cfg):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.argmax(out, dim=1).cpu().tolist())
        avg_loss = total_loss / len(loader)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
        )
        acc = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
        return avg_loss, acc, prec, rec, f1

    # ------------------------------------------------------------------
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    # Optional prediction helper ---------------------------------------
    def predict(self, loader: DataLoader):
        """Return list[int]: model predictions using argmax."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                preds.extend(torch.argmax(self.model(X), dim=1).cpu().tolist())
        return preds
