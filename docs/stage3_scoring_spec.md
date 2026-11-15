# Stage 3: scoring specification (frozen, version 1.1)

Scoring rules (sections 1 to 6) are unchanged since version 1.0. Version 1.1 changes the Stage 4 design in section 7 only.

This document fixes how every earnings release is turned into a sentiment score. It was written and
committed **before any score was joined to a return**. The rules were developed by looking at release
text only: no price, return or CAR was read while writing them, and Stage 3 computes no relationship
between scores and returns.

If a rule has to change later, the version number changes, the change is listed at the end of this
file, and it is committed before the affected scores are joined to returns.

The rules are implemented in:

| Part | File |
|---|---|
| HTML to text, table removal | `src/earnings_sentiment/textclean.py` |
| Paragraphs, safe-harbor removal, sentence splitting, sentence filter | `src/earnings_sentiment/scoring_text.py` |
| FinBERT scoring and aggregation | `src/earnings_sentiment/finbert_score.py` |
| Loughran-McDonald scoring and aggregation | `src/earnings_sentiment/lm.py` |
| Tests of every rule below | `tests/test_scoring.py` |

## 1. Input

One document per event: the exhibit selected in Stage 1 (`exhibit_url` in
`results/stage1/events_selected.csv`), 1,316 events. Both methods score exactly the same sentences.

## 2. Cleaning

Applied in this order, identically for FinBERT and Loughran-McDonald.

1. **HTML to text.** Scripts, styles and inline-XBRL headers are dropped. Block elements become line breaks.
2. **Tables.** An HTML table is removed when it has at least 3 rows and at least 20% of its non-space
   characters are digits. Innermost tables are judged first, so a layout table that wraps prose is kept.
3. **Paragraphs.** The remaining lines are rebuilt into paragraphs:
   - exhibit header lines at the top (`EX-99.1`, file names, `Exhibit 99.1`) are dropped;
   - a line with no letter is dropped (page numbers, `###`), except that a lone bullet glyph marks the
     next line as a bullet;
   - a line is joined to the paragraph before it when that paragraph does not end with `.`, `!` or `?`
     and either has at least 12 words or the new line starts with a lower-case letter. This repairs
     releases whose paragraphs are stored as fixed-width lines. A bullet always starts a new paragraph.
4. **Safe-harbor / forward-looking-statements removal.** Trigger phrases: "forward-looking statement(s)"
   or "forward-looking information", "Private Securities Litigation Reform Act", "safe harbor",
   "cautionary statement / note / language / notice".
   - A **heading** (15 words or fewer, not ending in `.`, `;`, `,` or `:`) containing a trigger: the heading
     and the first paragraph under it are removed.
   - A **body paragraph** containing a trigger: the sentences before the first trigger sentence are kept;
     that sentence and the rest of the paragraph are removed.
   - In both cases the following paragraphs are also removed for as long as they continue the passage:
     they are bullets, start with a lower-case letter, contain a trigger, or contain disclaimer wording
     (for example "risks and uncertainties", "actual results", "differ materially", "undertakes no
     obligation", "risk factors", "Form 10-K", "SEC"; the full list is `CONTINUATION` in the code).
     Removal stops at the next heading or at the first paragraph that does not continue the passage.
5. **Sentence splitting.** Rule-based, with no external model: a sentence ends at `.`, `!` or `?` followed
   by white space and a capital letter, digit or `$`. No split after a listed abbreviation (Inc., Corp.,
   U.S., No., month abbreviations and others), a single initial, or dotted initials. Bullets and a spaced
   dash before a capital letter also start a new sentence. Sentences never cross a paragraph.
6. **Sentence filter (prose test).** A sentence is scored only if it has 5 to 120 alphabetic words, at
   least half of its non-space characters are letters, and at most 40% of its tokens contain a digit.
   The upper limits remove the debris of tables that were laid out as text instead of HTML tables.

**Two versions are produced:**

| Version | Step 4 | Use |
|---|---|---|
| `main` | applied | primary scores |
| `with_sh` | skipped (safe-harbor text kept) | Stage 4 robustness check |

Tables (step 2) and the sentence filter (step 6) are applied in both versions.

**Known limits of the cleaning, accepted as they are:**

- Footnotes, non-GAAP definitions, "About the company" paragraphs and conference-call logistics are not
  removed. They are mostly neutral and dilute the document mean by an amount that differs across firms
  but is stable within a firm. The tone-surprise score (section 6) and firm fixed effects absorb this.
- Some table rows written as running text pass the prose test.
- Releases with no trigger phrase lose nothing in step 4.

## 3. FinBERT

| Setting | Value |
|---|---|
| Model | `ProsusAI/finbert` |
| Revision (exact commit) | `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` |
| Weights file | `model.safetensors` |
| Label order | 0 positive, 1 negative, 2 neutral (checked at load time) |
| Tokenizer | the model's own WordPiece tokenizer at the same revision, lower-casing on |
| Unit | one sentence per input, with `[CLS]` and `[SEP]` |
| `max_length` | 128 tokens including the two special tokens |
| Truncation | on; sentences longer than 128 tokens are cut and counted per document |
| Padding | to the longest sentence in the batch; batches are formed after sorting by token length |
| Batch size | 32 |
| Precision | float32 on CPU or GPU. int8 dynamic quantization is an optional CPU fallback and is **not** used for the reported scores unless stated |

Sentence score: `s = P(positive) - P(negative)`, between -1 and 1.

Document scores:

| Column | Definition | Role |
|---|---|---|
| `fb_tone` | mean of `s` over the document's sentences | **primary** |
| `fb_tone_labels` | share of sentences labelled positive minus share labelled negative (label = highest probability) | secondary |
| `fb_share_pos`, `fb_share_neg`, `fb_n_sentences`, `fb_n_truncated` | diagnostics | |

Each distinct sentence is scored once and the result reused wherever the sentence appears.

## 4. Loughran-McDonald

| Setting | Value |
|---|---|
| File | `Loughran-McDonald_MasterDictionary_1993-2025.csv`, supplied by the user in `data/external/`, never committed or redistributed |
| SHA-256 of the file used | `e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d` |
| Word lists | `Positive` and `Negative` columns; a word is in a list when the column holds a positive year (0 = never listed, negative year = removed) |
| List sizes in this file | 354 positive words, 2,355 negative words |
| Tokens | runs of letters, upper-cased; no stemming |
| Total words | tokens that appear in the master dictionary |

**Negation rule.** A positive word is not counted as positive when one of `NO, NOT, NONE, NEITHER,
NEVER, NOBODY` appears among the 3 words before it in the same sentence. It is not added to the
negative count either. Negative words are never negated.

Document scores:

| Column | Definition | Role |
|---|---|---|
| `lm_tone` | (positive - negative) / (positive + negative); missing if the document has no sentiment word | **primary** |
| `lm_tone_per_word` | (positive - negative) / total words | secondary |
| `lm_pos`, `lm_neg`, `lm_negated_pos`, `lm_words` | diagnostics | |

## 5. Fair comparison

Both methods read the same cleaned sentences of the same events. Stage 4 compares them on the same
sample, each standardised the same way (section 7).

## 6. Tone surprise

For each method and each event:

```text
tone surprise = document score - mean(document scores of the same firm's previous four fiscal quarters)
```

- The four previous quarters are the firm's kept events whose fiscal period end falls in the 380 days
  before the current event's fiscal period end.
- All four must be present. If fewer than four are present the tone surprise is missing; it is never
  computed from a shorter history.
- Only earlier quarters of the same firm are used, so the baseline contains no future information.

Columns: `fb_tone_surprise`, `lm_tone_surprise` (and the same for the `with_sh` version).

## 7. Stage 4 design (fixed now, before any score is joined to a return)

### Outcomes

| Family | Headline window | Other windows in the family | Role |
|---|---|---|---|
| Reaction | CAR[0,+1] | CAR[0,+2] to CAR[0,+5] | association test |
| Drift | CAR[+1,+5] | AR[+1], CAR[+1,+2] to CAR[+1,+4] | prediction test |

Holm correction is applied within each family. CARs are the market-model CARs of Stage 2.

### Regressors

`fb_tone` and `lm_tone` from the `main` version, each standardised by subtracting its full-sample mean
and dividing by its full-sample standard deviation. This is a change of units only and does not alter
any t-statistic. Each score is tested alone and then both together.

### Specifications

| | Specification | Fixed effects | Standard errors | Role |
|---|---|---|---|---|
| A | CAR on standardised tone | **firm** and **calendar quarter** | clustered by firm and by event date (two-way) | **headline** |
| B | CAR on standardised tone | sector and calendar quarter (pooled across firms) | clustered by firm and by event date | secondary |
| C | CAR on standardised tone surprise (`fb_tone_surprise`, `lm_tone_surprise`) | calendar quarter | clustered by firm and by event date | secondary |

- The calendar quarter is the quarter of event day 0. The event date used for clustering is event day 0.
- Specification A identifies the effect from changes in a firm's tone over time, so differences in house
  style and in the amount of boilerplate between firms cannot drive the result.
- Specification B also uses differences between firms.
- Specification C uses the within-firm deviation from the previous four quarters and therefore has no
  firm fixed effects; it covers 1,088 of the 1,316 events (section 6).
- Controls for all three: log market capitalisation at the end of 2018 is absorbed by firm fixed effects
  in A and is included in B and C; pre-event run-up CAR[-20,-1]; estimation-window volatility; and, for the
  drift family only, the day-0 abnormal return. Earnings surprise is a secondary specification and cannot
  block the main results.

### Sorts

Quintile sorts use breakpoints from the previous calendar quarter's distribution of scores, never from
the current or later quarters.

### Robustness

`with_sh` version of the text; `fb_tone_labels`; `lm_tone_per_word`; Fama-French 4-factor and sector-ETF
CARs; excluding market-hours events; excluding events with an earlier Item 2.02 filing in the same quarter.

## 8. Reproducibility

- Sentence files and scores are cached on disk. Text stays in `data/interim/` (git-ignored); scores and
  word counts, which contain no text, go to `results/stage3/`.
- The FinBERT step is one self-contained file that runs unchanged on CPU or on a Kaggle GPU.

## Change log

- 1.0: first frozen version.
- 1.1: Stage 4 design only (section 7). The headline regression now uses firm and calendar-quarter fixed
  effects with standard errors clustered by firm and by event date. The pooled specification with sector
  fixed effects and the tone-surprise specification are secondary. No scoring rule changed, so no score
  changes. Committed before any score was joined to a return.
