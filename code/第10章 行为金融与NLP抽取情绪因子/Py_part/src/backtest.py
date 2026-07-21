# -*- coding: utf-8 -*-
"""Vectorised back‑test helper for the behavioural‑sentiment strategy.

Assumptions
-----------
* Prediction file ``output/predictions.csv`` must contain at least two
  columns: ``date`` (ISO format) and ``pred_label`` in {1, 0, -1}.
* One‑day ahead returns are computed on‑the‑fly if the file does not
  already provide a column named ``future_return``.
* Transaction cost model: 0.02 % base + volatility‑adjusted slippage.
  (You may customise the lambdas `position_fn` & `slippage_fn`.)

Usage
-----
>>> from backtest import backtest
>>> equity_curve = backtest("output/predictions.csv")

It will generate ``output/backtest_result.png`` for visual inspection
and return a DataFrame with daily equity curves and key metrics.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

from config import Config

plt.rcParams["figure.dpi"] = 120

# ------------------------------------------------------------------ #
# utility helpers                                                    #
# ------------------------------------------------------------------ #

def _load_predictions(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "pred_label" not in df.columns:
        raise ValueError("prediction file must contain a 'pred_label' column")
    return df


def _attach_prices(pred_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    price_df = pd.read_csv("output/price_with_features.csv", parse_dates=["Date"])
    merged = pred_df.merge(
        price_df[["Date", "Adj Close", "VIX"]].rename(columns={"Date": "date"}),
        on="date",
        how="left",
    )
    if merged["Adj Close"].isna().any():
        raise ValueError("missing price rows – ensure price_with_features.csv contains full index range")
    return merged


def _compute_future_return(prices: pd.Series) -> pd.Series:
    return prices.pct_change().shift(-1)


def backtest(pred_path: Union[str, Path], cfg: Config | None = None) -> pd.DataFrame:
    """Run a simple next‑day long/short back‑test and return equity curves."""
    cfg = cfg or Config()
    preds = _load_predictions(pred_path)
    df = _attach_prices(preds, cfg)

    if "future_return" not in df.columns:
        df["future_return"] = _compute_future_return(df["Adj Close"])

    # position & slippage models (vectorised lambdas)
    position_fn = lambda vix: np.where(vix < 25, 1.0, 0.5)
    slippage_fn = lambda vix: np.where(vix < 25, 0.0002, 0.0005)

    df["position"] = position_fn(df["VIX"].to_numpy())
    df["slippage"] = slippage_fn(df["VIX"].to_numpy())

    preds_np = df["pred_label"].to_numpy()
    fwd_ret = df["future_return"].to_numpy()
    slip = df["slippage"].to_numpy()
    pos = df["position"].to_numpy()

    # vectorised strategy return
    long_mask = preds_np == 1
    short_mask = preds_np == -1

    strat_ret = np.zeros_like(fwd_ret)
    strat_ret[long_mask] = (fwd_ret[long_mask] - slip[long_mask]) * pos[long_mask]
    strat_ret[short_mask] = (-fwd_ret[short_mask] - slip[short_mask]) * pos[short_mask]

    df["strategy_return"] = strat_ret
    df["strategy_cum"] = (1 + df["strategy_return"]).cumprod()
    df["market_cum"] = (1 + fwd_ret).cumprod()

    # ------------- quick metrics (printable) ---------------------- #
    def _sharpe(series):
        return np.mean(series) / (np.std(series) + 1e-12) * np.sqrt(252)

    sharpe = _sharpe(df["strategy_return"])
    mdd = (df["strategy_cum"].cummax() - df["strategy_cum"]) / df["strategy_cum"].cummax()
    max_dd = mdd.max()

    print(f"Strategy Sharpe: {sharpe:.2f} ; MaxDD: {max_dd:.2%}")

    # ------------- plotting -------------------------------------- #
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["strategy_cum"], label="Strategy")
    plt.plot(df["date"], df["market_cum"], label="Buy & Hold", linestyle="--")
    plt.title("Behavioural‑Sentiment Strategy vs. Buy & Hold")
    plt.legend()
    Path("output").mkdir(exist_ok=True)
    plt.savefig("output/backtest_result.png", bbox_inches="tight")
    plt.close()

    df.to_csv("output/backtest_detailed.csv", index=False)
    return df

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    backtest("output/predictions.csv")
