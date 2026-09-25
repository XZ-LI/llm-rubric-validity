# Errata

Every defect found in this analysis, what it changed, and how it surfaced.

This file exists because the alternative is worse. In work of this shape the half-life of
an intermediate conclusion is short, and the dangerous errors are not the ones that crash —
they are the ones that produce a plausible number. Several entries below were caught only
because a downstream result looked wrong, not because anything failed.

---

## A. Defects that changed reported numbers

### A1 — Analysis restricted to the pilot subset
`pool_analysis.py` hard-coded the two genres used in the pilot. After the full run finished,
it silently analysed **80 of 840** books.

**Changed:** every pooled statistic, until caught.
**Found by:** the reported *n* not matching the run log.
**Fix:** enumerate cells by scanning actual result files instead of a literal list.

### A2 — Double-counted books inflating *n*
Two genres were scored by `qwen3-max` twice — once in the full run, once in the cross-judge
experiment. Both were loaded, so 38 books entered the pool twice.

**Changed:** `n = 878` where the true figure was 840; the detection floor was correspondingly
understated, i.e. the analysis claimed more power than it had.
**Found by:** *n* exceeding the known corpus size.
**Fix:** de-duplicate on (judge, cell, title), keeping the full-run record.

### A3 — Missing within-category standardization
`covariates.py` pooled raw scores across 34 genres without standardizing within them, so
between-genre mean differences were counted as correlation.

**Changed:** `r = +.146` where the correct figure was `+.067` — **more than half the reported
effect was artefact.**
**Found by:** disagreement with `pool_analysis.py`, which did standardize.
**Fix:** z-score within cell before pooling, covariates included.

### A4 — Prompt-format confound in the AI-vs-human comparison
For one judge, machine-written chapters had been scored under the original full-output
prompt while published novels were scored under a later trimmed prompt. The comparison mixed
two formats.

**Changed:** gap `+1.31` → `+1.50` after rescoring both sides under one prompt. The confound
was *understating* the effect, not manufacturing it.
**Found by:** tracing provenance of each figure in the comparison table rather than trusting
the table.
**Fix:** rescore; record prompt variant alongside every score.

### A5 — Zero-variance features exploding a distance metric
Burrows's Delta used a corpus-wide frequency list. Within a single genre some of those words
never appear, giving zero variance; an epsilon floor of `1e-12` was used instead of dropping
them, so any non-zero frequency produced a z-score in the millions.

**Changed:** one genre's Delta reported as `2.0e6`. The blow-up was visible there but would
have silently distorted the other six.
**Found by:** an absurd value in printed output.
**Fix:** drop zero-variance features per genre, as standard practice requires.

### A6 — Contamination set narrower than the pipeline's contact surface
Exclusions were hard-coded as top-5 ∪ bottom-2, but one generation stage drew on the top 12.
Seven books per affected cell had fed the writer while remaining in the held-out set.

**Changed:** `r = .136` → `.138` after correction — the leak was not inflating the result.
**Found by:** auditing what each pipeline stage actually reads, rather than trusting the
constants.
**Fix:** compute exclusions per cell as the union of everything that cell's pipeline touched,
keyed on artefacts on disk rather than on gating logic that may later change.

**A secondary finding worth recording:** excluding the top 12 *everywhere* — the "safer"
correction — dropped `r` to `.093` and `n` to 549. That decline is range restriction, not
decontamination. Over-correcting a contamination control can manufacture an apparent
attenuation and invite the conclusion that contamination was severe.

---

## B. Defects that would have corrupted data or wasted spend

### B1 — Service failure reported as missing data
The metadata harvester classified an unreachable service and a genuine "record not found"
identically. When the local service died mid-run, **648 books that were never queried were
recorded as missing**, producing an apparent 22.7% success rate.

This is the most dangerous class of defect here: it disguised an outage as a finding about
the platform. Fixed by classifying outcomes separately, adding a circuit breaker, and
reporting the two counts apart.

### B2 — Circuit breaker too aggressive
The first breaker aborted after 12 consecutive errors, treating a recoverable stall as
permanent failure and stopping at 49% of the run. Replaced with wait-and-resume, and the
breaker reserved for genuinely unreachable service.

### B3 — Model name leaking through a module-level constant
Rubric derivation read a module-level `REVIEWER_MODEL` rather than taking a parameter, so
running the pipeline with a different vendor requested a model that vendor does not host.
The run crashed on the fifth cell.

The subtler half: had it not crashed, rubrics would have been derived by one model and
applied by another — two different instruments in one dataset.

### B4 — One cell's failure killing the batch
No per-cell error isolation, so a single failure ended the remaining 33 cells. Fixed with
per-cell exception handling plus a failure summary at the end.

### B5 — Resume logic ignoring which judge produced a result
`--skip-done` checked only whether an output file existed. One cell's file came from an
earlier judge; skipping it would have mixed two judges in one dataset. Fixed to compare the
recorded `reviewer_model`.

### B6 — Cost ledger applying one vendor's prices to all
Per-call cost was computed with a single hard-coded rate pair, so non-OpenAI spend was
reported as "what this would have cost at OpenAI rates." Fixed to look up rates per model
and, where a rate has not been verified against the vendor's own documentation, **report
token counts and refuse to print a figure.**

### B7 — Output directories colliding for same-named genres
Three genre labels exist on both the male and female channels. Results were written to a
directory keyed on genre name alone, so each pair would have overwritten the other —
including their separately derived rubrics. Fixed by suffixing the channel.

### B8 — Silent overwrite of a completed run
One genre's summary file disagreed with its own per-round records; a later run had
overwritten the earlier one in place. Fixed by renaming rather than overwriting, and by
stamping every output with a run id, UTC timestamp and model name.

### B9 — Tokenizing whole novels before truncating
A diversity script segmented full novels and then took the first 2,500 tokens, rather than
truncating first. It hung rather than failing.

---

## C. Conclusions retracted

### C1 — "All judges correlate negatively with reader ratings"
Based on the pilot at *n* = 40, where the floor for detection was `.31`. At *n* = 840 the
figure is `+.067`. **The negative sign was noise.** Recorded because it is the clearest
instance in this project of a confident intermediate conclusion that a larger sample
reversed.

### C2 — "Realist genres show higher validity than fantasy genres"
Direction held (`+.109` vs `+.032`) but a 20,000-iteration permutation test returned
`p = .114`. Retained as a hypothesis, not a result.

### C3 — "All 840 held-out novels are still serializing"
A status field was tested against the wrong literal. The true composition is 477 serializing
and 362 completed. The defect touched a descriptive sentence only — no coefficient — but it
had already been written up as a methodological limitation.

### C4 — "Readers prefer repetition; the model's higher lexical diversity is a liability"
Proposed after finding machine output more lexically diverse than the human baseline.
Tested as a two-link chain and **falsified in reverse**: among published novels, higher
diversity predicts *higher* bookmark rate. Diversity turned out to be the one dimension on
which rubric and readers agree.

### C5 — Per-dimension analysis, retracted then reinstated
The 37 per-genre rubrics produced 208 distinct dimension names, 145 appearing in a single
cell, so dimension identity was confounded with cell identity. The original per-dimension
findings were withdrawn. Re-running all cells against one fixed ten-dimension rubric
restored the pattern on valid evidence.

**The distinction matters and should not be blurred: the early intuition was right, the
early evidence was not.**

---

## D. What this record implies

Three of the defects in section A would have survived peer review. None crashed; each
produced a number of plausible magnitude in the expected direction. They were caught by
cross-checking two independent implementations, by tracing where each figure in a table
came from, and by re-deriving the held-out sets from the corpus rather than trusting the
result files.

The practical lesson is narrow and repeatable: **for any figure that will appear in a
conclusion, compute it twice by different paths.** Where the two disagree, one of them is
wrong, and which one is not obvious in advance.
