# -*- coding: utf-8 -*-
"""End‑to‑end training & back‑test runner.

This script now **saves next‑day predictions** to
``output/predictions.csv`` so that ``backtest.py`` can run without
manual tweaks.  It is fully aligned with the updated 1‑day horizon in
`config.py` and the new feature‑engineering logic.

Run with:
>>> python main.py
"""
from __future__ import annotations

import torch
import pandas as pd
from pathlib import Path

from config import Config
from data_loader import download_data, add_technical_indicators, merge_sentiment_data
from feature_engineer import create_labels, build_sequences
from models import HybridModel
from trainer import Trainer, load_data
from sentiment_analysis import analyze_news_sentiment
from backtest import backtest

# --------------------------------------------------------------- #
# helpers                                                         #
# --------------------------------------------------------------- #

def predict_and_save(model: HybridModel, loader, dates, device: torch.device, save_path: str):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)

    df = pd.DataFrame({"date": dates, "pred_label": preds})
    Path(save_path).parent.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions → {save_path} (rows={len(df)})")
    return df

# --------------------------------------------------------------- #
# main                                                            #
# --------------------------------------------------------------- #

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data download & feature engineering ---------------------------------------------------
    print("Downloading data…")
    price_df = download_data()
    price_df = add_technical_indicators(price_df)

    print("Running sentiment analysis (FinBERT)…")
    analyze_news_sentiment("data/raw/news_2014-2024.csv")
    merged_df = merge_sentiment_data(price_df)

    labeled_df = create_labels(merged_df)
    X, y = build_sequences(labeled_df)

    # 2. Train‑validation‑test loaders ----------------------------------------------------------
    train_loader, val_loader, test_loader = load_data(X, y, cfg)
    test_dates = labeled_df.index[-len(test_loader.dataset):]

    # 3. Model training ------------------------------------------------------------------------
    model = HybridModel(cfg)
    trainer = Trainer(model, device, cfg)

    best_loss, patience = float("inf"), 0
    for epoch in range(cfg.EPOCHS):
        tr_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        print(f"Epoch {epoch+1:>3d} | Train {tr_loss:.4f} | Val {val_loss:.4f} | ValAcc {val_acc:.3f}")

        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            trainer.save_model("output/best_model.pth")
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print("Early‑Stopping ✋")
                break

    # 4. Prediction on test set ----------------------------------------------------------------
    model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
    predict_and_save(model, test_loader, test_dates, device, "output/predictions.csv")

    # 5. Run back‑test -------------------------------------------------------------------------
    backtest("output/predictions.csv", cfg)


if __name__ == "__main__":
    main()
