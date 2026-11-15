"""Tests for the frozen Stage 3 scoring rules. All texts below are constructed fixtures."""
import numpy as np
import pandas as pd

from earnings_sentiment import finbert_score as fb
from earnings_sentiment import lm
from earnings_sentiment.tone_surprise import prior_quarter_counts, tone_surprise
from earnings_sentiment.scoring_text import keep_sentence, paragraphs, prepare, remove_safe_harbor, split_sentences

LM = lm.LMDictionary(
    positive=frozenset({"STRONG", "GAINS", "IMPROVED"}),
    negative=frozenset({"LOSS", "DECLINE", "WEAK"}),
    vocabulary=frozenset({"STRONG", "GAINS", "IMPROVED", "LOSS", "DECLINE", "WEAK", "NOT", "NO", "WAS", "DEMAND",
                          "THE", "A", "QUARTER", "REPORTED", "WE"}),
    file_name="fixture",
)


def test_exhibit_header_lines_are_dropped():
    text = "EX-99.1\n2\na8-kex991.htm\nExhibit 99.1\nAcme Reports Third Quarter Results\nRevenue grew."
    assert paragraphs(text) == ["Acme Reports Third Quarter Results", "Revenue grew."]


def test_paragraphs_join_wrapped_lines_drop_page_numbers_and_mark_bullets():
    text = ("Certain statements made in this press release may be deemed to be statements within the meaning of\n"
            "Litigation law. Although the Company believes the expectations are reasonable, results may differ\n"
            "6\n"
            "materially.\n"
            "About Acme\n"
            "\u2022\n"
            "the outcome of research;\n"
            "\u25aa Record cash flow")
    assert paragraphs(text) == [
        "Certain statements made in this press release may be deemed to be statements within the meaning of "
        "Litigation law. Although the Company believes the expectations are reasonable, results may differ materially.",
        "About Acme",
        "\u2022 the outcome of research;",
        "\u2022 Record cash flow",
    ]


def test_safe_harbor_heading_removes_its_section_until_next_heading():
    pars = [
        "Revenue grew nine percent on strong demand.",
        "Forward-Looking Statements",
        "This release contains statements about the future.",
        "Factors that could cause a difference include competition, regulation and weather.",
        "\u2022 the outcome of research;",
        "About Acme",
        "Acme makes anvils.",
    ]
    kept, removed = remove_safe_harbor(pars)
    assert kept == [pars[0], pars[5], pars[6]] and len(removed) == 4


def test_safe_harbor_under_heading_stops_at_ordinary_text():
    pars = ["Forward-Looking Statements", "This release contains statements about the future.",
            "Net sales in Europe rose four percent on higher volumes."]
    kept, _ = remove_safe_harbor(pars)
    assert kept == [pars[2]]


def test_safe_harbor_inline_keeps_the_sentences_before_the_trigger():
    pars = [
        "Margins improved during the quarter.",
        "We expect earnings growth of six percent next year. This press release contains forward-looking statements "
        "within the meaning of the Private Securities Litigation Reform Act of 1995. They involve many assumptions.",
        "Actual results could differ materially because of risks and uncertainties described in our Form 10-K.",
        "The company undertakes no obligation to update these statements.",
        "Net sales in Europe rose four percent.",
    ]
    kept, removed = remove_safe_harbor(pars)
    assert kept == [pars[0], "We expect earnings growth of six percent next year.", pars[4]]
    assert len(removed) == 3 and removed[0].startswith("This press release contains")


def test_sentence_splitter_respects_abbreviations_decimals_and_bullets():
    par = "Acme Inc. reported revenue of $1.5 billion in the U.S. market. Margins improved. • Record cash flow"
    assert split_sentences(par) == [
        "Acme Inc. reported revenue of $1.5 billion in the U.S. market.", "Margins improved.", "Record cash flow"]


def test_sentence_filter_drops_short_and_numeric_fragments():
    assert keep_sentence("Revenue increased nine percent on strong demand.")
    assert not keep_sentence("Margins improved.")                       # fewer than 5 words
    assert not keep_sentence("Q3 2024 1,234 1,100 12% 9% 8% 7% up vs. prior year total")   # mostly digits


def test_prepare_reports_words_removed_and_keeps_robustness_version_longer():
    text = ("Acme Reports Third Quarter Results\nRevenue increased nine percent on strong demand in every region.\n"
            "Forward-Looking Statements\nThis release contains statements about future events and plans.\nAbout Acme\n"
            "Acme is a maker of anvils and other products for industry.")
    main, with_sh = prepare(text, True), prepare(text, False)
    assert main["words_safe_harbor_removed"] > 0 and with_sh["words_safe_harbor_removed"] == 0
    assert with_sh["n_sentences"] == main["n_sentences"] + 1
    assert not any("future events" in s for s in main["sentences"])


def test_lm_negation_window_and_counts():
    assert lm.score_sentence("Demand was strong", LM) == {"lm_pos": 1, "lm_neg": 0, "lm_negated_pos": 0, "lm_words": 3}
    # negator within three words before a positive word: not counted as positive, not added to negative
    assert lm.score_sentence("Demand was not strong", LM)["lm_pos"] == 0
    assert lm.score_sentence("Demand was not strong", LM)["lm_negated_pos"] == 1
    assert lm.score_sentence("Demand was not strong", LM)["lm_neg"] == 0
    # negator four words back is outside the window
    assert lm.score_sentence("No a the quarter strong", LM)["lm_pos"] == 1
    # negative words are never negated
    assert lm.score_sentence("We reported no loss", LM)["lm_neg"] == 1


def test_lm_document_tone():
    s = pd.DataFrame({"ticker": "AAA", "accession": ["x", "x", "y"], "lm_pos": [2, 1, 0], "lm_neg": [1, 0, 0],
                      "lm_negated_pos": 0, "lm_words": [10, 10, 5]})
    a = lm.aggregate(s).set_index("accession")
    assert abs(a.loc["x", "lm_tone"] - 0.5) < 1e-12 and abs(a.loc["x", "lm_tone_per_word"] - 0.1) < 1e-12
    assert np.isnan(a.loc["y", "lm_tone"])          # no sentiment word at all


def test_tone_surprise_needs_four_prior_quarters_and_uses_only_the_past():
    ends = ["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31", "2020-09-30"]
    ev = pd.DataFrame({"ticker": "AAA", "fiscal_period_end": ends, "score": [1.0, 2.0, 3.0, 4.0, 10.0, 100.0]})
    assert prior_quarter_counts(ev).tolist() == [0, 1, 2, 3, 4, 3]       # the last event follows a missing quarter
    ts = tone_surprise(ev, "score")
    assert ts.isna().tolist() == [True, True, True, True, False, True]
    assert abs(ts.iloc[4] - (10.0 - 2.5)) < 1e-12                         # 10 minus mean(1, 2, 3, 4)
    # a later score must not change an earlier surprise
    ev2 = ev.copy(); ev2.loc[5, "score"] = -999.0
    assert abs(tone_surprise(ev2, "score").iloc[4] - 7.5) < 1e-12


def test_finbert_document_aggregation_is_the_mean_of_sentence_net_scores():
    s = pd.DataFrame({"accession": ["x", "x", "x"], "p_pos": [0.9, 0.1, 0.2], "p_neg": [0.05, 0.8, 0.2],
                      "p_neu": [0.05, 0.1, 0.6], "truncated": [False, True, False]})
    a = fb.aggregate(s).iloc[0]
    assert abs(a["fb_tone"] - ((0.85 - 0.7 + 0.0) / 3)) < 1e-9
    assert abs(a["fb_share_pos"] - 1 / 3) < 1e-9 and abs(a["fb_share_neg"] - 1 / 3) < 1e-9
    assert a["fb_n_sentences"] == 3 and a["fb_n_truncated"] == 1
