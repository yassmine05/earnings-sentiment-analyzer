"""Daily prices (Yahoo Finance via yfinance) and Fama-French daily factors (Ken French library)."""
from __future__ import annotations

import io
import time
import zipfile

import pandas as pd
import requests

from . import config

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}


def yahoo_symbol(ticker: str) -> str:
    """SEC writes share classes with a dash or dot (BRK-B / BRK.B); Yahoo uses a dash."""
    return ticker.replace(".", "-").upper()


def fetch_prices(ticker: str, start: str = config.PRICE_START, end: str = config.PRICE_END) -> pd.DataFrame:
    """Unadjusted OHLC, dividend+split adjusted close, and volume. Empty frame if Yahoo has nothing."""
    import yfinance as yf

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    df = yf.Ticker(yahoo_symbol(ticker)).history(
        start=start, end=end_excl, interval="1d", auto_adjust=False, actions=True
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(
        columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividend", "Stock Splits": "split",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df.insert(0, "ticker", ticker)
    keep = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]
    return df[[c for c in keep if c in df.columns]]


def fetch_all_prices(tickers: list[str], pause: float = 0.6) -> tuple[pd.DataFrame, list[str]]:
    """Download each ticker once (cached as parquet). Returns the combined frame and the tickers with no data."""
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    frames, missing = [], []
    for t in tickers:
        path = config.PRICES_DIR / f"{yahoo_symbol(t)}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = pd.DataFrame()
            for attempt in range(3):
                try:
                    df = fetch_prices(t)
                    break
                except Exception as e:  # network hiccup or throttling: back off and retry
                    print(f"  {t}: {type(e).__name__}: {str(e)[:120]} (attempt {attempt + 1})")
                    time.sleep(5 * (attempt + 1))
            if not df.empty:
                df.to_parquet(path, index=False)
            time.sleep(pause)
        if df.empty:
            missing.append(t)
        else:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, missing


def _parse_french_daily(raw_zip: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    header = None
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 1 and parts[0] == "" and header is None:
            header = parts[1:]
        elif len(parts) > 1 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in header:
        df[c] = pd.to_numeric(df[c]) / 100.0  # the library reports percent
    return df


def fetch_french_factors() -> pd.DataFrame:
    """Daily Mkt-RF, SMB, HML, RF and Mom as decimals."""
    config.FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for key, fname in FRENCH_FILES.items():
        path = config.FACTORS_DIR / fname
        if not path.exists():
            r = requests.get(FRENCH_BASE + fname, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
            r.raise_for_status()
            path.write_bytes(r.content)
        parts.append(_parse_french_daily(path.read_bytes()).set_index("date"))
    df = parts[0].join(parts[1], how="left")
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
    return df.reset_index()
