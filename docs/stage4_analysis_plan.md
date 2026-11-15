# Stage 4: analysis plan (pre-specified, version 1.0)

This plan was written and committed **before any sentiment score was joined to a CAR**. It turns the
Stage 4 design frozen in `docs/stage3_scoring_spec.md` (version 1.1, section 7) into exact equations,
samples, tests and reporting rules. Nothing below may be changed after the join without an entry in the
"Deviations" section of the Stage 4 results report, stating what changed, why, and what the result was
under the original plan.

Counts and power figures come from `scripts/14_power.py`
(`results/stage4/plan_sample_counts.csv`, `results/stage4/plan_power.csv`). That script reads the
return file and the score file separately and never puts a score and a return in the same table.

## 1. Variables

| Symbol | Definition | Source | Measured |
|---|---|---|---|
| `CAR01` | market-model CAR[0,+1] | `results/stage2/event_cars.csv`, `car_0_1` | days 0 to +1 |
| `CAR15` | market-model CAR[+1,+5] | `car_1_5` | days +1 to +5 |
| `FB` | FinBERT tone `fb_tone`, main text version | `results/stage3/document_scores.csv` | release text |
| `LM` | Loughran-McDonald tone `lm_tone` (ratio), main text version | same | release text |
| `FBs`, `LMs` | tone surprises `fb_tone_surprise`, `lm_tone_surprise` | same | release text and the firm's four prior releases |
| `RUNUP` | pre-event run-up CAR[-20,-1] | `car_m20_m1` | before the event |
| `VOL` | residual standard deviation of the market model | `sigma` | estimation window [-250,-11], before the event |
| `SIZE` | log of market capitalisation at 2018-12-31 | `results/stage1/basket.csv` | before the sample |
| `AR0` | day-0 abnormal return | `car_0_0` | day 0; used in the drift family only |

**Standardisation.** Every sentiment regressor is standardised over the estimation sample of the regression
it enters (mean 0, standard deviation 1), so each coefficient reads as the change in CAR, in basis points,
for a one-standard-deviation difference in tone. This is a change of units and does not affect t-statistics.

Day 0 is `event_day0` from the approved event clock. The calendar quarter `q(e)` is the quarter of day 0.

## 2. Regression equations

Indices: firm `i`, event `e`, sector `s(i)`, calendar quarter `q(e)`. `T` stands for one standardised
sentiment score. Each equation is estimated once with `T = FB`, once with `T = LM`, and once with both
(`b1*FB + b2*LM`).

### Specification A: headline (firm and calendar-quarter fixed effects)

```text
Reaction:  CAR01[i,e] = b*T[i,e] + c1*RUNUP[i,e] + c2*VOL[i,e]               + a[i] + d[q(e)] + u[i,e]
Drift:     CAR15[i,e] = b*T[i,e] + c1*RUNUP[i,e] + c2*VOL[i,e] + c3*AR0[i,e] + a[i] + d[q(e)] + u[i,e]
```

`a[i]` are firm fixed effects and `d[q]` calendar-quarter fixed effects. `SIZE` does not vary within a firm
and is absorbed by `a[i]`. The coefficient `b` is identified from changes in a firm's tone over time.

### Specification B: pooled (sector and calendar-quarter fixed effects), secondary

```text
Reaction:  CAR01[i,e] = b*T[i,e] + c1*RUNUP + c2*VOL + c4*SIZE[i]            + g[s(i)] + d[q(e)] + u[i,e]
Drift:     CAR15[i,e] = b*T[i,e] + c1*RUNUP + c2*VOL + c3*AR0 + c4*SIZE[i]   + g[s(i)] + d[q(e)] + u[i,e]
```

### Specification C: tone surprise (calendar-quarter fixed effects), secondary

```text
Reaction:  CAR01[i,e] = b*Ts[i,e] + c1*RUNUP + c2*VOL + c4*SIZE[i]           + d[q(e)] + u[i,e]
Drift:     CAR15[i,e] = b*Ts[i,e] + c1*RUNUP + c2*VOL + c3*AR0 + c4*SIZE[i]  + d[q(e)] + u[i,e]
```

`Ts` is `FBs` or `LMs`, standardised. No firm fixed effects: the surprise is already a within-firm deviation.

### Controls and timing

`RUNUP`, `VOL` and `SIZE` are all measured before day 0. `AR0` is measured on day 0 and appears only in the
drift equations, whose outcome starts on day +1, so no regressor overlaps the outcome window. The reaction
equations contain no day-0 variable on the right-hand side.

An earnings-surprise control (seasonal random-walk EPS surprise from EDGAR XBRL) is **exploratory**. It is
added only if the EPS data can be assembled for the sample, and no headline or secondary result waits for it.

## 3. Sample rules

| Group | Events |
|---|---|
| Events kept in Stage 1 | 1,316 |
| With market-model CARs and a full estimation window | 1,316 |
| With both sentiment scores | 1,316 |
| Missing the `RUNUP` control | 1 |
| **Headline sample (specifications A and B)** | **1,315** |
| With tone surprise for both methods | 1,088 |
| **Specification C sample** | **1,087** |
| Firms / distinct event days / calendar quarters | 55 / 505 / 24 |

**In.** Every event kept in Stage 1 that has both CARs, both scores and all controls of the equation.

**Out.** The one event with a missing `RUNUP` (AT&T, day 0 on 2022-04-21): its pre-event window contains the
WarnerMedia spin-off day, which Stage 1 excluded as an unverifiable price adjustment. No other event is
removed. No observation is removed because of its CAR or its score, and CARs are **not winsorised** in the
headline tests.

**Flags.** These events stay in the headline sample. Each is removed in its own robustness run:

| Flag | Events | Reason for the check |
|---|---|---|
| Earlier Item 2.02 filing in the same quarter (`had_other_202_before`) | 51 | part of the news was pre-announced |
| Market-hours event (`event_bucket`) | 23 | day 0 holds only part of a session's reaction |
| Event clock moved to the 10-Q/10-K time (`event_time_changed`) | 15 | Berkshire (14) and Linde (1) timing adjustment |
| Filed without Item 2.02, hand-reviewed (`gap_fill`) | 3 | selected by hand |
| No tone surprise | 228 | to compare A and C on identical events, A is re-run on the 1,087 events of C |

## 4. Estimation and inference

- **Estimator.** Ordinary least squares with the fixed effects absorbed.
- **Headline standard errors.** Two-way clustered by firm (55 clusters) and by event day 0 (505 clusters),
  Cameron-Gelbach-Miller, with the small-sample correction of the software. p-values are two-sided.
- **Robustness of inference.** The same point estimates with (i) firm-only clustering and (ii)
  calendar-quarter clustering (24 clusters; reported with the caveat that 24 is a small number of clusters).
- **Multiple testing.** Holm correction within each family, across the headline tests of that family.
  A headline test is called significant when its Holm-adjusted p-value is below 0.05.
- **Reported for every test.** Coefficient in basis points per one standard deviation of tone, standard
  error, 95% confidence interval, t-statistic, unadjusted p-value, Holm-adjusted p-value, number of events,
  number of clusters.

## 5. Headline tests: four in total

| # | Family | Outcome | Regressor | Specification | Hypothesis tested (two-sided) |
|---|---|---|---|---|---|
| H1 | Reaction (association) | `CAR01` | `FB` alone | A | b = 0 |
| H2 | Reaction (association) | `CAR01` | `LM` alone | A | b = 0 |
| H3 | Drift (prediction) | `CAR15` | `FB` alone | A | b = 0 |
| H4 | Drift (prediction) | `CAR15` | `LM` alone | A | b = 0 |

Holm families: {H1, H2} and {H3, H4}. These four are run once. Their results are written to a file before
any robustness or exploratory analysis is run.

Interpretation fixed in advance: the reaction tests ask whether tone is associated with the market's
response to the same release, which is not a trading result because the release and the price move happen
together. Only the drift tests ask whether tone predicts later returns.

## 6. Pre-specified secondary analyses (not headline, no Holm claim)

| # | Analysis | Content |
|---|---|---|
| S1 | FinBERT against Loughran-McDonald | specification A with both scores together, for each family: b1, b2, the test b1 = b2, and the increase in within R2 from adding each score to the other |
| S2 | Specification B | pooled, sector and quarter fixed effects; each score alone and both together, both families |
| S3 | Specification C | tone surprise; each alone and both together, both families |
| S4 | Horizon profile | specification A, each score alone, for CAR[0,+1] to CAR[0,+5] and for AR[+1], CAR[+1,+2] to CAR[+1,+5]; Holm across the five windows of a family for each score |
| S5 | Quintile sorts | section 7 |
| S6 | Inference robustness | firm-only and calendar-quarter clustering for H1 to H4 |
| S7 | Sample robustness | H1 to H4 without each flagged group of section 3, one group at a time, and on the specification C sample |
| S8 | Measurement robustness | H1 to H4 with: safe-harbor text kept (`fb_tone_sh`, `lm_tone_sh`); `fb_tone_labels`; `lm_tone_per_word`; Fama-French 4-factor CARs; sector-ETF CARs; CARs winsorised at the 1st and 99th percentiles; buy-and-hold abnormal returns |

**Everything else is exploratory** and will be labelled so wherever it appears: the earnings-surprise control,
splits by sector, year, timing bucket or firm size, interactions, rank correlations with a bootstrap interval,
non-linear effects, and anything suggested by the results themselves. Exploratory results are never
described as tests of the hypothesis.

## 7. Quintile sorts

Run separately for raw tone (`FB`, `LM`) and for tone surprise (`FBs`, `LMs`), for both outcomes.

1. For each calendar quarter `q`, the four breakpoints are the 20th, 40th, 60th and 80th percentiles of the
   score among the events of the **previous** calendar quarter `q-1`.
2. Each event of quarter `q` is placed in a quintile using those breakpoints. Quintile sizes are therefore
   not forced to be equal.
3. The first quarter with scores has no previous quarter and is left out of the sorts: 2019Q1 for raw tone,
   and the first quarter in which tone surprises exist for the surprise sorts.
4. Reported for each quintile: number of events, mean and median CAR. Reported for the spread: mean CAR of
   quintile 5 minus quintile 1, with a standard error two-way clustered by firm and event day, from a
   regression of the CAR on quintile indicators.

Raw-tone sorts compare firms with different house styles as well as quarters, so they are read together
with the tone-surprise sorts, which compare each firm with its own recent past.

## 8. Power: the smallest effect the sample can detect

Minimum detectable effect (MDE) at 80% power, in basis points of CAR per one standard deviation of tone.
The standard error is the textbook one, residual sd of the CAR divided by (square root of n times residual
sd of the standardised score), each sd computed from its own file. It assumes independent errors;
clustering can only be assessed after the join, so the last column shows the MDE if clustering inflates the
standard error by 1.5.

| Specification | Outcome | Score | n | sd of CAR after FE and controls | sd of score after FE | MDE at 5% | MDE at 2.5% (strictest Holm step) | MDE at 5%, SE x 1.5 |
|---|---|---|---|---|---|---|---|---|
| A (headline) | `CAR01` | FinBERT | 1,315 | 4.79% | 0.69 | 53 bp | 59 bp | 80 bp |
| A (headline) | `CAR01` | Loughran-McDonald | 1,315 | 4.79% | 0.57 | 65 bp | 72 bp | 98 bp |
| A (headline) | `CAR15` | FinBERT | 1,315 | 3.18% | 0.69 | 35 bp | 39 bp | 53 bp |
| A (headline) | `CAR15` | Loughran-McDonald | 1,315 | 3.18% | 0.57 | 43 bp | 48 bp | 65 bp |
| B (pooled) | `CAR01` | FinBERT | 1,315 | 4.80% | 0.90 | 41 bp | 45 bp | 62 bp |
| B (pooled) | `CAR01` | Loughran-McDonald | 1,315 | 4.80% | 0.92 | 40 bp | 45 bp | 61 bp |
| B (pooled) | `CAR15` | FinBERT | 1,315 | 3.16% | 0.90 | 27 bp | 30 bp | 41 bp |
| B (pooled) | `CAR15` | Loughran-McDonald | 1,315 | 3.16% | 0.92 | 27 bp | 29 bp | 40 bp |
| C (surprise) | `CAR01` | FinBERT | 1,087 | 4.99% | 0.92 | 46 bp | 51 bp | 69 bp |
| C (surprise) | `CAR01` | Loughran-McDonald | 1,087 | 4.99% | 0.96 | 44 bp | 48 bp | 66 bp |
| C (surprise) | `CAR15` | FinBERT | 1,087 | 3.29% | 0.92 | 30 bp | 33 bp | 46 bp |
| C (surprise) | `CAR15` | Loughran-McDonald | 1,087 | 3.29% | 0.96 | 29 bp | 32 bp | 44 bp |

Reading: the headline drift test can detect a FinBERT effect of about 35 bp over five days per standard
deviation of tone, or about 53 bp if clustering inflates the standard error by half. Firm fixed effects
remove about half of the variance of the scores (48% of FinBERT tone and 68% of dictionary tone lies between
firms), which is why specification A is less powerful than B. **A drift effect of a few basis points, the
size often reported for large and liquid firms, is below what this sample can detect.** A headline drift
result that is not significant will therefore be reported as "no effect detected; effects smaller than the
MDE cannot be ruled out", with the confidence interval, and not as evidence of no effect.

## 9. Reporting rules, including null results

1. All four headline tests are reported in one table, in the order H1 to H4, whatever they show.
2. No test, window, specification or subsample listed in sections 5 to 7 is left out of the report. Each
   appears in a table or an appendix file even when the result is null or goes the "wrong" way.
3. The report and the README state the headline results first, with effect size, confidence interval and
   Holm-adjusted p-value. Secondary and exploratory results cannot replace or re-frame a headline result.
4. A null headline result is stated plainly as a null, next to the MDE of section 8.
5. The sign of each effect is reported as estimated. There is no one-sided test.
6. Every number in the report and the README is produced by a script in the repository and saved in
   `results/stage4/`. The scripts run the headline tests once and save them before anything else.
7. Any departure from this plan is listed under "Deviations" in the results report.

## 10. Limits known before the join

- 55 very large firms from 2019 to 2024: results may not carry over to small firms or other periods.
- Text is the earnings press release, not the earnings call.
- Day 0 comes from EDGAR acceptance times; Stage 2 found the reaction centred on day 0 in every bucket.
- Earnings news itself is not controlled for in the headline tests, so a reaction-family association may
  reflect good results being described in positive words. This is one reason the reaction family is called
  an association test.
- Market-model CARs with SPY as the benchmark; other benchmarks are robustness checks.
