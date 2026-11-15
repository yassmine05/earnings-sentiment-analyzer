"""Stage 1, step 1b: approximate market cap at 2018-12-31 for the basket candidates.

market cap = cover-page shares outstanding nearest 2018-12-31 (EDGAR XBRL)
             x price implied by the SPDR N-Q (position value / shares held).

Multi-class issuers and issuers without a usable XBRL share count are taken from the
hand-checked file data/reference/multiclass_shares.csv (one row per share class).

Inputs   results/stage1/spdr_holdings_2018-12-31.csv, results/stage1/issuer_map_proposed.csv,
         data/reference/issuer_overrides.csv, data/reference/multiclass_shares.csv (optional)
Outputs  results/stage1/basket_candidates.csv
"""
from __future__ import annotations

import pandas as pd

from earnings_sentiment import config
from earnings_sentiment.basket import nearest_shares
from earnings_sentiment.edgar import EdgarClient


def main() -> None:
    out_dir = config.RESULTS / "stage1"
    client = EdgarClient()
    prop = pd.read_csv(out_dir / "issuer_map_proposed.csv")
    over = pd.read_csv(config.REFERENCE / "issuer_overrides.csv").set_index("name")["cik"]
    prop["cik"] = prop["name"].map(over).fillna(prop["cik"])
    unresolved = prop[prop["cik"].isna()]
    if len(unresolved):
        print("UNRESOLVED NAMES:\n", unresolved[["sector", "name"]].to_string())
    prop = prop.dropna(subset=["cik"]).copy()
    prop["cik"] = prop["cik"].astype(int)

    hold = pd.read_csv(out_dir / "spdr_holdings_2018-12-31.csv")
    prop = prop.merge(hold[["sector", "name", "shares_held", "implied_price"]], on=["sector", "name"], how="left")

    mc_path = config.REFERENCE / "multiclass_shares.csv"
    mc = pd.read_csv(mc_path) if mc_path.exists() else pd.DataFrame(columns=["cik"])
    if len(mc):
        nq_price = hold.drop_duplicates("name").set_index("name")["implied_price"]
        mc["price"] = mc["price_from_nq_name"].map(nq_price) * mc["multiplier"]
        if mc["price"].isna().any():
            raise ValueError("multiclass_shares.csv names not found in the N-Q: "
                             + str(mc.loc[mc["price"].isna(), "price_from_nq_name"].tolist()))

    rows = []
    for cik, g in prop.groupby("cik", sort=False):
        head, _ = client.submissions(cik)
        former = "; ".join(f["name"] for f in head.get("formerNames", [])[:3])
        rec = {
            "sector": g["sector"].iloc[0],
            "nq_names": " | ".join(g["name"]),
            "cik": cik,
            "edgar_name": head["name"],
            "former_names": former,
            "tickers_today": ",".join(head.get("tickers", [])),
            "fund_value_usd": int(g["value_usd"].sum()),
            "n_classes_in_fund": len(g),
            "price_2018_12_31": round(float(g["implied_price"].iloc[0]), 4),
        }
        m = mc[mc["cik"] == cik]
        if len(m):
            rec.update(
                {
                    "market_cap_usd": float((m["shares_out"] * m["price"]).sum()),
                    "shares_out": float(m["shares_out"].sum()),
                    "shares_date": m["shares_date"].iloc[0],
                    "shares_source": "hand-checked cover page: " + m["source_accession"].iloc[0],
                }
            )
        else:
            s = nearest_shares(client.shares_outstanding_facts(cik))
            if s is not None:
                rec.update(
                    {
                        "market_cap_usd": s["shares_out"] * rec["price_2018_12_31"],
                        "shares_out": s["shares_out"],
                        "shares_date": s["shares_date"],
                        "shares_source": f"XBRL dei {s['shares_form']} {s['shares_accession']}",
                    }
                )
            else:
                rec.update({"market_cap_usd": float("nan"), "shares_out": float("nan"), "shares_date": "", "shares_source": "MISSING"})
        rec["named_share_class"] = bool(g["name"].str.contains(r"Class|Series").any())
        rows.append(rec)

    cand = pd.DataFrame(rows)
    cand["market_cap_bn"] = (cand["market_cap_usd"] / 1e9).round(1)
    cand["rank_in_sector"] = cand.groupby("sector")["market_cap_usd"].rank(ascending=False, method="first")
    cand = cand.sort_values(["sector", "market_cap_usd"], ascending=[True, False])
    cand.to_csv(out_dir / "basket_candidates.csv", index=False)
    show = ["sector", "nq_names", "cik", "edgar_name", "tickers_today", "market_cap_bn", "shares_date", "shares_source", "named_share_class"]
    with pd.option_context("display.width", 260, "display.max_rows", 500, "display.max_colwidth", 34):
        print(cand[show].to_string())
    print("missing share counts:", int(cand["market_cap_usd"].isna().sum()), "| EDGAR requests:", client.n_requests)


if __name__ == "__main__":
    main()
