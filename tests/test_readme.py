"""The README must match the saved results. These tests read only committed files in results/."""
import re

import pandas as pd

from earnings_sentiment import config, readme

TEXT = readme.README.read_text(encoding="utf-8")


def test_readme_is_exactly_what_the_saved_results_produce():
    # Any number typed or edited by hand in README.md, or any change in a result file, makes this fail.
    assert TEXT == readme.build_readme(), "README.md is out of date: run scripts/18_readme.py"


def test_headline_rows_match_headline_results_csv():
    # Independent of the builder: parse the headline table in the README and compare it with the saved file.
    h = pd.read_csv(config.RESULTS / "stage4" / "headline_results.csv")
    for r in h.itertuples():
        line = next(l for l in TEXT.splitlines() if l.startswith(f"| {r.test} |"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        coef, ci, p, p_holm, mde = float(cells[4]), cells[5], float(cells[6]), float(cells[7]), float(cells[8])
        lo, hi = (float(x) for x in re.findall(r"[+-]\d+\.\d", ci))
        assert abs(coef - r.coef_bp) <= 0.05 and abs(lo - r.ci_low_bp) <= 0.05 and abs(hi - r.ci_high_bp) <= 0.05
        assert abs(p - r.p) <= 0.0005 and abs(p_holm - r.p_holm) <= 0.0005
        assert abs(mde - r.se_bp * (1.960 + 0.842)) <= 0.5
    assert ("None of the 4 headline tests" in TEXT) == (not h["significant_5pct_holm"].any())


def test_counts_quoted_in_the_readme_match_the_saved_files():
    s4 = config.RESULTS / "stage4"
    qs, sp = pd.read_csv(s4 / "s5_quintile_spreads.csv"), pd.read_csv(s4 / "exploratory_splits.csv")
    assert f"**{(qs['p'] < 0.05).sum()} of {len(qs)} spreads**" in TEXT
    assert f"**{(sp['p'] < 0.05).sum()} of {len(sp)}** has a p-value below 0.05" in TEXT


def test_readme_figures_exist():
    for name in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", TEXT):
        assert (config.ROOT / name).is_file(), name
