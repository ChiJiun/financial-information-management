"""
Download yfinance price data cache files for local notebook testing.

How to use in Google Colab:
1. Upload or open this script in Colab.
2. Run:
   !python download_price_cache_for_colab.py
3. Download price_cache.zip from the Colab file browser.
4. Extract the zip into the repository root on your PC, so files are under:
   financial-information-management/price_cache/

The trendlines notebook reads the same CSV cache filenames before calling
yfinance, so local runs can avoid Yahoo Finance rate limits.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "yfinance is not installed. In Colab run: !pip -q install yfinance"
    ) from exc


TICKERS = ["TSM", "AAPL", "NVDA", "MSFT", "AMD", "SPY", "QQQ"]
DATA_PERIOD = "max"
INTERVALS = ["1d"]
CACHE_DIR = Path("price_cache")
ZIP_NAME = "price_cache.zip"


def normalize_download(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns:
        raise ValueError("Downloaded data does not contain a Date column.")
    return df


def download_one(ticker: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    range_tag = "MAX" if DATA_PERIOD == "max" else DATA_PERIOD
    cache_path = CACHE_DIR / f"{ticker}_{range_tag}_{interval}.csv"

    if cache_path.exists():
        print(f"Already exists: {cache_path}")
        return cache_path

    print(f"Downloading {ticker} ({interval})...")
    df = yf.download(
        ticker,
        period=DATA_PERIOD,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"Unable to download {ticker} ({interval}).")

    df = normalize_download(df)
    df.to_csv(cache_path, index=False)
    print(f"Saved: {cache_path} rows={len(df):,}")
    return cache_path


def make_zip() -> None:
    with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(CACHE_DIR.glob("*.csv")):
            zf.write(csv_path, arcname=str(csv_path))
    print(f"Created {ZIP_NAME}")


def main() -> None:
    downloaded = []
    failed = []

    for ticker in TICKERS:
        for interval in INTERVALS:
            try:
                downloaded.append(download_one(ticker, interval))
            except Exception as exc:  # noqa: BLE001
                failed.append((ticker, interval, str(exc)))
                print(f"FAILED {ticker} ({interval}): {exc}")

    make_zip()
    print("\nSummary")
    print(f"Downloaded/available files: {len(downloaded)}")
    if failed:
        print("Failed downloads:")
        for ticker, interval, reason in failed:
            print(f"- {ticker} ({interval}): {reason}")
    else:
        print("All downloads succeeded.")

    if "COLAB_RELEASE_TAG" in os.environ:
        print("\nColab detected. Download price_cache.zip from the file browser.")


if __name__ == "__main__":
    main()
