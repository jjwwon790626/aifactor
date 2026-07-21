# -*- coding: utf-8 -*-
"""Hourly data loader & feature builder (2014‑2024)

This replaces the old day‑level `data_loader.py`.
Key features
------------
* **download_hourly()**
  Loop‑downloads 1‑hour bars from Yahoo Finance in 700‑day chunks,
  bypassing the ~730‑day hard limit.
* **disk cache**
  After the first run, data are saved to ``data/processed/qqq_hourly.parquet``
  and ``vix_hourly.parquet`` so subsequent runs skip API calls.
* **add_technical_indicators()**
  Computes MACD diff, RSI, Bollinger Bands, ATR on *hourly* prices.
* **merge_sentiment_data()**
  Joins hour‑level FinBERT sentiment (csv) with price DataFrame
  using 1‑hour tolerance.

Usage
-----
>>> from src.data_loader import download_data, add_technical_indicators, merge_sentiment_data
>>> price_df = add_technical_indicators(download_data())
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from datetime import timedelta
from pathlib import Path

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from config import Config

# ---------------------------------------------------------------------------
# Yahoo hourly helper
# ---------------------------------------------------------------------------

def _download_hourly_chunk(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a single chunk (<= 730 days) of 1‑hour bars."""
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1h",
        progress=False,
        threads=False,
    )


def download_hourly(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Loop‑download 1‑hour bars across the *entire* date span.

    Yahoo restricts 1‑hour interval queries to ~730 days.  We therefore
    iterate in 700‑day windows to safely cover 10 years.
    """
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    dfs: list[pd.DataFrame] = []
    cur_start = start_dt
    step = timedelta(days=700)

    while cur_start < end_dt:
        cur_end = min(cur_start + step, end_dt)
        dfs.append(
            _download_hourly_chunk(
                ticker,
                cur_start.strftime("%Y-%m-%d"),
                (cur_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
        )
        cur_start = cur_end + timedelta(hours=1)

    full = pd.concat(dfs).sort_index().drop_duplicates()
    full = full.tz_localize(None)  # remove timezone for easier merge
    return full

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def download_data(cache_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Fetch 1‑hour QQQ + VIX prices (2014‑2024) with on‑disk cache."""
    cfg = Config()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    qqq_path = cache_dir / "qqq_hourly.parquet"
    vix_path = cache_dir / "vix_hourly.parquet"

    if qqq_path.exists() and vix_path.exists():
        qqq = pd.read_parquet(qqq_path)
        vix = pd.read_parquet(vix_path)
    else:
        print("Downloading 1‑Hour QQQ…")
        qqq = download_hourly(cfg.TICKER, cfg.START_DATE, cfg.END_DATE)
        print("Downloading 1‑Hour VIX…")
        vix = download_hourly("^VIX", cfg.START_DATE, cfg.END_DATE)
        qqq.to_parquet(qqq_path)
        vix.to_parquet(vix_path)

    qqq = qqq[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    merged = qqq.merge(vix, left_index=True, right_index=True, how="left")
    return merged


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD‑diff, RSI, Bollinger‑Bands & ATR to the hourly frame."""
    df = df.copy()
    macd = MACD(close=df["Adj Close"])
    df["MACD"] = macd.macd_diff()

    df["RSI"] = RSIIndicator(close=df["Adj Close"]).rsi()

    bb = BollingerBands(close=df["Adj Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"])
    df["ATR"] = atr.average_true_range()

    return df


def merge_sentiment_data(
    price_df: pd.DataFrame,
    sentiment_path: str | Path = "output/daily_sentiment.csv",
    tolerance: str = "1h",
) -> pd.DataFrame:
    """Nearest‑merge hour‑level sentiment onto price DataFrame."""
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
    merged = pd.merge_asof(
        price_df.sort_index(),
        sentiment_df.sort_values("date"),
        left_index=True,
        right_on="date",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    for col in ("net_score", "positive_ratio", "negative_ratio"):
        merged[col] = merged[col].fillna(0)
    return merged


__all__ = [
    "download_data",
    "add_technical_indicators",
    "merge_sentiment_data",
]
