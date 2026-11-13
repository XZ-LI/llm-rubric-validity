# Reliable but Invalid

**A criterion-validity test of LLM-as-judge evaluation: does a rubric score that passes
the standard checks (test–retest reliability, agreement between judges) actually track
an external criterion — here, what readers did with Chinese web fiction?**

The object under test is the *measuring instrument*: an LLM-built, genre-specific rubric applied by
LLM judges. The study began when a self-improvement loop, in which a model wrote opening
chapters, scored them against the rubric and rewrote its own instructions from the
lowest-scoring dimensions, plateaued while its prompt grew more than tenfold. The question
became whether the score was measuring anything readers respond to.

### Contributions

1. **A validity-testing design for LLM judges against behavioural criteria.** 840
   held-out novels across 37 genre × channel cells, six reader-response criteria with
   differing contamination profiles, and within-cell standardization so that genre and
   channel differences can't stand in for validity.
2. **Per-cell contamination control.** Each cell's exclusions are the union of every text
   the pipeline touched: rubric induction, judge calibration, and the generator's own
   prompt.
3. **Separating reliability, agreement and validity.** One judge at large n answers the
   validity question; four judges at small n answer the judge-dependence question. The
   two analyses are kept apart so neither can be mistaken for the other.
4. **A reproducible, auditable pipeline.** Resumable corpus harvesting, run IDs on every
   output, per-call cost metering that refuses to estimate unverified rates, and a
   publication sanitizer that strips copyrighted prose while keeping how the judges cited
   the text analysable.
5. **A public errata log.** Every defect found, including four retracted conclusions, is
   recorded in [`ERRATA.md`](ERRATA.md).

### Models

| Model | Vendor | Role |
|---|---|---|
| `o3` | OpenAI | Generator in the self-improvement loop; one of the four cross-judges |
| `qwen3-max` | Alibaba (Qwen) | Primary judge: large-n validity, test–retest reliability, fixed-rubric comparison |
| `deepseek-v4-pro` | DeepSeek | Cross-judge |
| `qwen-turbo` | Alibaba (Qwen) | Cross-judge (a lower capability tier from the same vendor as `qwen3-max`) |

---

## The finding in one paragraph

The rubric's validity against reader behaviour is negligible: within-cell standardized
`r = .138` against exposure-normalized bookmark rate (n = 768, r² < 2%). That null is not
noise — test–retest reliability was `ICC(1,1) = .874` and disattenuation moved the estimate
by .002. Nor is reviewer consensus a safeguard: four judges (`qwen3-max`, `o3`,
`deepseek-v4-pro`, `qwen-turbo`) agreed .45–.49 with one another while none exceeded .14
with readers, and they reversed each other's verdicts on whether machine-written chapters
beat published ones — including two capability tiers from a single vendor. **Reliability
and inter-judge agreement, the two standard warrants for LLM-as-judge evaluation, are
structurally incapable of detecting this failure mode.**

---

## Documents

Both render standalone in a browser — clone and open, or serve `docs/` statically.

| | |
|---|---|
| [`docs/case-study.html`](docs/case-study.html) | **Case study** — the finding, why it matters if you run evals, and the skills it evidences. Start here. |
| [`docs/research-design.html`](docs/research-design.html) | **Full research design** (Chinese) — nine findings with figures, the retraction log, methodology threats, and seven research questions. |

---

## What's here

```
src/data/        corpus acquisition + platform metadata harvest
                 (global throttle, circuit breaker, incremental checkpoint, resume)
src/eval/        scoring pipeline, per-cell contamination control, multi-judge harness
src/analysis/    validity, reliability, per-dimension, stylometric distance, covariates
public/          published results, prose stripped (see "Data" below)
annotation/      blind paired-comparison tool generator + human judgments
docs/            method notes and figures
ERRATA.md        every defect found in this analysis, and what it changed
```

### Design in brief

| Stage | What it does |
|---|---|
| **Corpus** | 1,110 serialized novels, reader ratings embedded per file, daily incremental refresh |
| **Rubric** | Three-stage human-in-the-loop: researcher seed dimensions → per-genre model adaptation from top-rated works → researcher review |
| **Held-out set** | 840 novels across 37 genre × channel cells. Exclusions computed **per cell** as the union of every text the pipeline touched — rubric induction, reviewer calibration, and the generator's own prompt |
| **Criteria** | Six reader-response measures with differing contamination profiles, so no conclusion rests on any one being clean |
| **Judges** | One judge at large n for the validity question; four judges at small n for the judge-dependence question. Deliberately separated |

---

## Data

**This repository does not redistribute the novels.** The corpus is commercially
published work. `public/corpus_manifest.json` lists the 840 held-out titles with their
`book_id` and factual metadata (rating, genre, channel); `src/data/` can retrieve the
corpus from those identifiers. Every analysis reproduces once the corpus is in place.

Result files are published through `make_public.py`, which strips verbatim quotation
from the reviewer's evidence fields — roughly 1.46 M characters across 2,811 sites —
while retaining `evidence_shape` (how many citations each dimension produced and how
long they were), so "did the reviewer ground its score in the text" remains analysable.

Before any publish:

```bash
python3 make_public.py --check    # lists any file still holding verbatim prose
python3 make_public.py            # regenerates public/
```

---

## Reproducing

```bash
pip install openai numpy jieba tiktoken
export DASHSCOPE_API_KEY=...      # and/or OPENAI_API_KEY, DEEPSEEK_API_KEY

python3 src/eval/validate_pilot.py --all --provider qwen --workers 8 \
        --repeat 3 --reliability-k 5        # 1,210 calls, ~80 min
python3 src/analysis/pool_analysis.py       # pooled validity + inter-judge
python3 src/analysis/criterion_matrix.py    # six-criterion convergent analysis
python3 src/analysis/fixed_rubric_analyze.py
```

Every run writes a `run_id`, UTC timestamp and model name; same-name outputs are renamed
rather than silently overwritten. Cost is metered per call against published rates, and
**where a vendor's rate has not been verified the ledger reports tokens only and refuses
to print a figure.**

---

## Limitations, stated up front

- **The criterion is behavioural, not aesthetic.** This measures what readers did —
  bookmarked, kept reading — not what they judged good. The construct is scoped to reader
  response as manifested in platform behaviour; the word *quality* appears in no
  conclusion.
- **Missing data is non-random.** 5.1% of held-out titles were no longer retrievable
  through platform search, and they skew low-rated (Cohen's *d* = −0.98). This truncates
  the sample at the bottom and attenuates correlations, so reported estimates are likely
  conservative.
- **One platform, one language, one genre family.** Serialized web fiction is highly
  formulaic, which makes it a favourable test bed rather than a representative one.
- **Human vetting did not rescue validity, which strengthens rather than weakens the
  result.** A researcher-authored fixed rubric performed equivalently to the
  human-reviewed per-genre rubrics.

---

## A note on the errata

`ERRATA.md` documents every defect found in this analysis — including the ones that
changed reported numbers, and the four conclusions retracted when better data arrived.
They are recorded rather than quietly fixed. In work like this the half-life of an
intermediate conclusion is short, and the errors hide inside figures that already look
confirmed.
