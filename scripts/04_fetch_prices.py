"""Stage 1, step 4: daily prices for the basket, SPY and sector ETFs, plus Fama-French daily factors.

Input   results/stage1/basket.csv
Output  data/raw/prices/*.parquet, data/raw/factors/* (git-ignored)
        results/stage1/price_coverage.csv   rows, first/last date and gaps per ticker
"""
from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal

from earnings_sentiment import config
from earnings_sentiment.prices import fetch_all_prices, fetch_french_factors

BENCHMARKS = ["SPY", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]


def main() -> None:
    out_dir = config.RESULTS / "stage1"
    basket = pd.read_csv(out_dir / "basket.csv")
    tickers = BENCHMARKS + basket["ticker"].tolist()
    prices, missing = fetch_all_prices(tickers)

    sessions = mcal.get_calendar("NYSE").schedule(config.PRICE_START, config.PRICE_END).index.normalize()
    rows = []
    for t in tickers:
        p = prices[prices["ticker"] == t] if not prices.empty else prices
        if t in missing or p.empty:
            rows.append({"ticker": t, "rows": 0})
            continue
        have = pd.DatetimeIndex(p["date"])
        span = sessions[(sessions >= have.min()) & (sessions <= have.max())]
        rows.append(
            {
                "ticker": t, "rows": len(p), "first": have.min().date(), "last": have.max().date(),
                "expected_sessions_full_window": len(sessions),
                "missing_sessions_inside_span": len(span.difference(have)),
                "null_adj_close": int(p["adj_close"].isna().sum()),
                "zero_volume_days": int((p["volume"] == 0).sum()),
                "covers_estimation_start": bool(have.min() <= pd.Timestamp("2018-01-05")),
            }
        )
    cov = pd.DataFrame(rows)
    cov.to_csv(out_dir / "price_coverage.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(cov.to_string())
    print("tickers with no Yahoo data:", missing)

    ff = fetch_french_factors()
    ff = ff[(ff["date"] >= config.PRICE_START) & (ff["date"] <= config.PRICE_END)]
    ff.to_parquet(config.FACTORS_DIR / "ff_daily.parquet", index=False)
    print("factor rows:", len(ff), ff["date"].min().date(), "->", ff["date"].max().date(), "| columns:", list(ff.columns))

    # Sanity check that replaces the Stooq cross-check: SPY return vs the Ken French market return.
    spy = prices[prices["ticker"] == "SPY"].set_index("date")["adj_close"].pct_change()
    mkt = ff.set_index("date").eval("mkt_rf + rf")
    both = pd.concat([spy.rename("spy"), mkt.rename("ff_mkt")], axis=1).dropna()
    chk = {
        "n_days": len(both),
        "correlation": round(both["spy"].corr(both["ff_mkt"]), 5),
        "mean_abs_diff_bp": round((both["spy"] - both["ff_mkt"]).abs().mean() * 1e4, 2),
        "max_abs_diff_bp": round((both["spy"] - both["ff_mkt"]).abs().max() * 1e4, 2),
    }
    pd.DataFrame([chk]).to_csv(out_dir / "spy_vs_ff_market.csv", index=False)
    print("SPY vs Ken French market:", chk)


if __name__ == "__main__":
    main()
