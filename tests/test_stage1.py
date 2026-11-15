import pandas as pd

from earnings_sentiment.basket import nearest_shares, parse_spdr_nq
from earnings_sentiment.events import assign_fiscal_period, select_events
from earnings_sentiment.textclean import html_to_text, word_count
from earnings_sentiment.timing import day0, nyse_sessions, timing_bucket

NQ = """
<p>THE ENERGY SELECT SECTOR SPDR FUND</p><p>SCHEDULE OF INVESTMENTS</p>
<table>
<tr><td>COMMON STOCKS &#151; 99.8%</td></tr>
<tr><td>OIL, GAS &amp; CONSUMABLE FUELS &#151; 80.0%</td></tr>
<tr><td><div>Alpha Oil Corp. (a)</div></td><td>&#160;</td><td>1,000</td><td>&#160;</td><td>$ <font>50,000</font></td></tr>
<tr><td><div>Beta Gas,
Inc.</div></td><td>&#160;</td><td>2,000</td><td>&#160;</td><td>30,000</td></tr>
<tr><td>&#160;</td><td>80,000</td></tr>
<tr><td>TOTAL COMMON STOCKS</td><td>80,000</td></tr>
<tr><td>SHORT-TERM INVESTMENTS</td></tr>
<tr><td>Cash Fund</td><td>500</td><td>500</td></tr>
</table>
"""


def test_parse_spdr_nq_keeps_only_common_stock_rows():
    df = parse_spdr_nq(NQ)
    assert df["name"].tolist() == ["Alpha Oil Corp", "Beta Gas, Inc"]
    assert df["sector"].unique().tolist() == ["Energy"]
    assert df["shares_held"].tolist() == [1000, 2000]
    assert df["implied_price"].tolist() == [50.0, 15.0]


def test_nearest_shares_picks_closest_date_within_limit():
    facts = pd.DataFrame(
        {"end": ["2018-06-30", "2019-01-18", "2019-06-30"], "val": [1, 2, 3], "form": ["10-Q"] * 3,
         "filed": ["2018-07-30", "2019-01-30", "2019-07-30"], "accn": ["a", "b", "c"]}
    )
    assert nearest_shares(facts, "2018-12-31")["shares_out"] == 2
    assert nearest_shares(facts.iloc[[2]], "2018-12-31") is None


def test_assign_fiscal_period_uses_latest_period_end_before_filing():
    ev = pd.DataFrame({"accepted_et_date": ["2019-01-02", "2019-01-29", "2019-04-30"], "accession": ["x", "y", "z"]})
    periodic = pd.DataFrame(
        {"form": ["10-K", "10-Q", "10-Q"],
         "reportDate": pd.to_datetime(["2018-09-29", "2018-12-29", "2019-03-30"]),
         "filingDate": pd.to_datetime(["2018-11-05", "2019-01-30", "2019-05-01"]),
         "periodic_accepted_et": ["2018-11-05 16:31:00", "2019-01-30 16:31:00", "2019-05-01 16:31:00"]}
    )
    out = assign_fiscal_period(ev, periodic)
    assert out["fiscal_period_end"].dt.strftime("%Y-%m-%d").tolist() == ["2018-12-29", "2018-12-29", "2019-03-30"]
    assert out["days_before_periodic"].tolist() == [28, 1, 1]


def test_day0_and_bucket_follow_the_market_clock():
    sessions = nyse_sessions("2024-01-01", "2024-12-31")
    ts = pd.Series([
        "2024-07-12 06:45:00",  # Friday pre-open  -> same day
        "2024-07-12 10:07:00",  # Friday in hours  -> same day
        "2024-07-12 16:30:00",  # Friday after close -> Monday
        "2024-07-13 08:00:00",  # Saturday -> Monday
        "2024-07-03 16:05:00",  # before the 4 July holiday -> Friday 5 July
    ])
    assert day0(ts, sessions).dt.strftime("%Y-%m-%d").tolist() == [
        "2024-07-12", "2024-07-12", "2024-07-15", "2024-07-15", "2024-07-05"]
    assert timing_bucket(ts, sessions).tolist() == [
        "pre-open", "market hours", "after close", "non-trading day", "after close"]


def _filing(acc, when, **kw):
    base = {"ticker": "AAA", "fiscal_period_end": "2019-03-31", "form": "8-K", "items": "2.02,9.01",
            "accession": acc, "accepted_utc": when, "exhibit_url": "u", "words_prose": 1000,
            "head_results": True, "head_prelim": False, "days_before_periodic": 5}
    base.update(kw)
    return base


def test_select_events_keeps_one_release_per_period_and_explains_drops():
    f = pd.DataFrame([
        _filing("pre", "2019-04-05T12:00:00Z", head_prelim=True, words_prose=3000),   # longer, but preliminary
        _filing("rel", "2019-04-25T12:00:00Z", words_prose=2500),                      # the release
        _filing("amd", "2019-04-26T12:00:00Z", form="8-K/A", words_prose=2600),        # amendment
        _filing("sup", "2019-05-20T12:00:00Z", words_prose=9000, days_before_periodic=-20),  # after the 10-Q
        _filing("nox", "2019-04-10T12:00:00Z", exhibit_url=""),                        # no exhibit
        _filing("solo", "2019-07-25T12:00:00Z", fiscal_period_end="2019-06-30",
                head_results=False, days_before_periodic=-2),                          # only filing: kept
    ])
    kept, dropped = select_events(f)
    assert sorted(kept["accession"]) == ["rel", "solo"]
    assert bool(kept.set_index("accession").loc["rel", "had_other_202_before"])
    reasons = dropped.set_index("accession")["drop_reason"]
    assert reasons["amd"].startswith("step 2") and reasons["nox"].startswith("step 3")
    assert reasons["sup"].startswith("step 5") and reasons["pre"].startswith("step 6-7")


def test_html_to_text_drops_numeric_tables_but_keeps_prose():
    html = (
        "<html><body><p>Revenue grew strongly this quarter.</p>"
        "<table><tr><td>Revenue</td><td>1,234</td><td>1,100</td></tr>"
        "<tr><td>Costs</td><td>900</td><td>870</td></tr>"
        "<tr><td>Income</td><td>334</td><td>230</td></tr></table>"
        "<p>We expect continued demand.</p></body></html>"
    )
    text = html_to_text(html)
    assert "Revenue grew strongly" in text and "continued demand" in text
    assert "1,234" not in text
    assert word_count("Net income rose 5%") == 3
