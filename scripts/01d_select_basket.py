"""Stage 1, step 1d: pick the 5 largest issuers per sector and fix their identifiers.

Inputs   results/stage1/basket_candidates.csv, data/reference/basket_identity_overrides.csv
Outputs  results/stage1/basket.csv
"""
from __future__ import annotations

import pandas as pd

from earnings_sentiment import config


def main() -> None:
    out_dir = config.RESULTS / "stage1"
    cand = pd.read_csv(out_dir / "basket_candidates.csv")
    if cand["market_cap_usd"].isna().any():
        raise SystemExit("some candidates have no market cap; finish the hand-check first")
    cand["rank_in_sector"] = cand.groupby("sector")["market_cap_usd"].rank(ascending=False, method="first").astype(int)
    n = config.FIRMS_PER_SECTOR
    top = cand[cand["rank_in_sector"] <= n].copy()

    # How close was the first firm left out? (the caps are approximate, so close calls are reported)
    cut = cand[cand["rank_in_sector"].isin([n, n + 1])].pivot(index="sector", columns="rank_in_sector", values="market_cap_usd")
    names = cand[cand["rank_in_sector"] == n + 1].set_index("sector")["nq_names"]
    margin = pd.DataFrame(
        {"fifth_cap_bn": (cut[n] / 1e9).round(1), "sixth_cap_bn": (cut[n + 1] / 1e9).round(1),
         "sixth_firm": names, "gap_pct": ((cut[n] / cut[n + 1] - 1) * 100).round(1)}
    )
    margin.to_csv(out_dir / "basket_cutoff_margins.csv")
    print(margin.to_string())

    ov = pd.read_csv(config.REFERENCE / "basket_identity_overrides.csv", dtype={"extra_ciks": "string"}).set_index("cik_2018")
    ov["extra_ciks"] = ov["extra_ciks"].fillna("")
    top["ticker"] = top["tickers_today"].fillna("").str.split(",").str[0]
    top["ticker"] = top["cik"].map(ov["ticker"]).fillna(top["ticker"])
    extra = top["cik"].map(ov["extra_ciks"].to_dict()).astype("string").fillna("")
    top["cik_all"] = [";".join([str(c)] + ([e] if e else [])) for c, e in zip(top["cik"], extra)]
    top["company_2018"] = top["nq_names"].str.split(" | ", regex=False).str[0].str.replace(r"\s+(Class( [A-C])?|REIT)$", "", regex=True)
    basket = top[
        ["sector", "rank_in_sector", "ticker", "company_2018", "cik", "cik_all", "market_cap_bn",
         "price_2018_12_31", "shares_out", "shares_date", "shares_source"]
    ].sort_values(["sector", "rank_in_sector"])
    if (basket["ticker"] == "").any():
        raise SystemExit("ticker missing for: " + str(basket.loc[basket["ticker"] == "", "company_2018"].tolist()))
    basket.to_csv(out_dir / "basket.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 100, "display.max_colwidth", 40):
        print(basket.drop(columns=["shares_source", "shares_out"]).to_string(index=False))


if __name__ == "__main__":
    main()
