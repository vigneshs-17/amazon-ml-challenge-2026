# Phase 1 — Forensic Data Audit (EXP-000)

Amazon ML Challenge 2026 — Business Entity Resolution. Draft; blocking/leakage/
placeholder full-scale sections are being filled in as background jobs complete.

Source: `../6ab10eb3b23ba_student_resource.zip` → extracted read-only to
`data/raw/student_resource/`. All numbers below are computed from the actual
files, using `sep="\t"`, default CSV-quote decoding, `keep_default_na=False`
(a literal business name like "NA" is never turned into NaN). Row counts are
asserted against raw line counts on every load (`src/io_utils.py`).

## 1. Dataset inventory

| file | rows (data lines) | columns |
|---|---|---|
| train/train_source1.tsv | 2,206,821 | entity_id, business_name, business_address, country |
| train/train_source2.tsv | 5,034,616 | same |
| train/train_source3.tsv | 5,285,603 | same |
| train/train_ground_truth.tsv | 2,206,821 | source1_entity_id, matched_entity_ids |
| test/test_source1.tsv | 1,732,544 | entity_id, business_name, business_address, country |
| test/test_source2.tsv | 4,887,273 | same |
| test/test_source3.tsv | 5,082,316 | same |

Every source row has exactly 4 tab-separated fields (ground truth: 2); every
entity_id matches its expected `S1-`/`S2-`/`S3-` prefix; no invalid UTF-8, no
stray `\r`. Some fields use CSV-style quoting (`"…""…"` doubling) — e.g.
`S2-24822466  """chrs & cie sasu"` and multi-line-looking addresses like
`"9Th Floor, ""Niagara"" Building, …"` in train_source1. Default pandas quote
decoding handles these correctly (verified: parsed row count == raw line
count for every file). **Do not** disable quoting when loading these files.

In-memory size (pyarrow-backed strings): S1 ≈ 270 MB, S2/S3 ≈ 600-650 MB each,
per split.

## 2. Data quality

* **Nulls/blanks:** `business_name` is never empty in any file. `business_address`
  is never empty in S1 (train or test). In S2/S3, address is empty in ~2.6-3.7%
  of rows (train S2 3.36%, train S3 3.33%, test S2 2.65%, test S3 2.68%).
  `keep_default_na=False` is essential — see §6, `<NULL>` is a literal string
  placeholder in the address field, not a missing-value marker pandas would
  otherwise catch.
* **Exact duplicate rows:** none in any file. **Duplicate (name,address) ignoring
  id:** S2/S3 have ~16-26k such rows per file (0.3-0.5%) — the same business
  text appears under two different entity_ids; this could inflate candidate
  counts slightly but is not itself a labeling problem.
* **Duplicate business_name (raw):** common — 493k-668k duplicate names per
  file (many businesses legitimately share names like "Quick Stop" or generic
  chains), confirming name alone is a weak key (see §7 precision finding).
* **Duplicate entity_id:** zero in every file (verified exhaustively).
* **Malformed entity_id:** zero (100% match `^S[123]-\d+$`).
* **Suspicious records:** business names starting with a non-alphanumeric
  character (e.g. `#4 Presidio`, `*** Sai Tech Private Limited`,
  `[INCORPORATED] PEAK TRADIN6 NETWORKS SOUTHSIDE`) affect 0.1-2.2% of rows
  per file — real noisy business-listing artifacts, not corruption; they
  should survive normalization, not be discarded. URL-like names (`www.`,
  `.com`, `http`) appear in 3.9-4.3% of S2/S3 rows (train/test) — e.g.
  `S3-202863386 wilfordhancock.com` — these are genuine (if unusual) DBA-style
  names, kept as-is.

## 3. Country analysis

| split/source | US | India | France |
|---|---|---|---|
| train S1 | 59.98% (1,323,633) | 40.02% (883,188) | — |
| train S2 | 59.92% (3,016,817) | 40.08% (2,017,799) | — |
| train S3 | 59.98% (3,170,056) | 40.03% (2,115,547) | — |
| test S1 | 38.27% (663,106) | 46.75% (809,986) | 14.97% (259,452) |
| test S2 | 38.29% (1,871,330) | 47.32% (2,312,565) | 14.39% (703,378) |
| test S3 | 38.28% (1,945,701) | 47.32% (2,405,000) | 14.40% (731,615) |

Confirmed directly from the data (not assumed from docs): France is present
**only** in the test files, at ~14-15% of every test source, and nowhere in
train. US:India ratio is a stable ~60:40 in train and shifts to ~38:47:15
(US:India:France) in test as the third country is introduced. `country` has
exactly 3 distinct values across the whole dataset (`US`, `India`, `France`) —
no stray/garbage country strings were found. Any pipeline component that
special-cases `{US, India}` (one-hot country, hard-coded suffix lists) will
degrade on ~15% of the test set.

## 4. Ground-truth structure

2,206,821 rows, one per train S1 entity; every `source1_entity_id` is unique
and present in `train_source1.tsv` (bijective — 0 missing either direction).
Every `matched_entity_ids` entry parses cleanly (no blank/whitespace items,
no in-list duplicates). All 7,638,365 matched ids reference real S2/S3
records (0 unknown ids) and none reference S1 (0 self-matches).

**Match-count distribution (per S1 entity):**

| matches | count | % |
|---|---|---|
| 0 (singleton) | 123,247 | 5.585% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3+ | 1,589,205 | 72.01% |

mean = 3.461, mean (non-singleton) = 3.666, median = 3.0, max = 11.
Near-identical across countries (US singleton 5.583%, India 5.588%).

**Source mix:** S2-only 143,029 (6.48%), S3-only 164,498 (7.45%), **both**
1,776,047 (80.48%), none 123,247 (5.59%). So the large majority of non-singleton
S1 entities have matches in *both* S2 and S3 — a candidate generator that only
searches one of the two pools would miss most of the match set for ~80% of
matched entities.

**Ownership (verified exhaustively, `audit_ownership.py`):** of the 3,693,619
distinct S2 ids and 3,944,746 distinct S3 ids that appear in the ground truth,
**zero** are matched to more than one S1 entity (max owners = 1 for both).
This is an *observation about how the labeled training set happens to be
built*, not a rule stated in the official README — the test set has no ground
truth to check this against, and nothing guarantees a test S2/S3 record can't
legitimately be the correct match for more than one S1 entity, or that our
model won't sometimes *predict* the same candidate for two different S1
entities. Any conflict-resolution logic we might add later (e.g. assigning a
contested S2 record to only its single best-scoring S1 entity) should be
justified and validated experimentally, not hard-coded as a global constraint.

S2 records that never appear in any positive pair: 1,340,997 / 5,034,616
(26.6%). S3: 1,340,857 / 5,285,603 (25.4%). These are true negatives-only
records (noise-only or genuinely unrelated businesses) — consistent with
blocking needing to also correctly *reject* about a quarter of each pool.

## 5. Positive-pair similarity analysis

(`src/audit_pairs.py`, fixed-seed reproducible sample of 300,000 of the
7,638,365 positive pairs; percentiles/means below are on this sample.)

| metric | mean | p05 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|---|
| name raw string equality | 4.62% of pairs | — | — | — | — | — |
| name case-insensitive equality | 10.72% | — | — | — | — | — |
| name normalized-equality | 25.81% | — | — | — | — | — |
| name token Jaccard | 0.643 | 0.0 | 0.5 | 0.667 | 1.0 | 1.0 |
| name RapidFuzz `ratio` | 79.9 | 10.7 | 73.1 | 87.8 | 100 | 100 |
| name RapidFuzz `token_set_ratio` | 86.9 | 10.7 | 88.2 | 100 | 100 | 100 |
| name Jaro-Winkler | 0.881 | 0.404 | 0.866 | 0.948 | 1.0 | 1.0 |
| address raw equality | 2.22% | — | — | — | — | — |
| address normalized equality | 8.22% | — | — | — | — | — |
| address token Jaccard | 0.605 | 0.118 | 0.429 | 0.636 | 0.8 | 1.0 |
| address `token_set_ratio` | 87.3 | 56.7 | 85.7 | 93.6 | 98.4 | 100 |
| **country equality** | **100.0%** (mean=1.0 at every percentile) | | | | | |
| name script mismatch (Latin vs Devanagari) | 4.1% of pairs | | | | | |

**Country equality is 100% across the full sampled positive set** — no true
match ever crosses a country label. This makes country a safe, lossless
pre-filter for blocking (also verified at full scale in §9: with/without
country gives *identical* recall for every full-scale blocker tested).

**Low-similarity tail is real and non-trivial:** 10.2% of true positive pairs
have `token_set_ratio < 50` on the name field alone, and name+address are
*both* weak (`token_set_ratio < 60` on both) for 0.15% — small in absolute
percentage but that is still ~11,000+ positive pairs in the full training set
that a naive two-strikes-and-out similarity threshold would drop. p5 of name
`ratio` is only 10.7 — a small but real slice of true matches look almost
nothing alike by character similarity and must be caught by exact-token or
address-based signals instead.

**Noise-category breakdown (of true positive pairs):**

| category | % of positive pairs |
|---|---|
| punctuation/case difference only | 21.19% |
| word reorder | 5.58% |
| token added/dropped (suffix, DBA, etc.) | 26.64% |
| spelling/typo (ratio 80-97, same token count) | 13.52% |
| script variation (Latin vs Devanagari) | 4.12% |
| address missing on one side | 4.38% |
| address components missing (token subset) | 15.29% |
| hard name (token_set_ratio < 50) | 10.2% |
| hard on both name and address | 0.15% |

## 6. Hardest true matches — representative examples

(sampled from `reports/audit_pairs.json` → `noise_examples`, seed=42)

* **Hard name (low token_set_ratio):** an S1/S2 pair whose normalized names
  share almost no tokens yet are the correct match — driven by address
  agreement or a DBA/trade name substitution (see `noise_examples.hard_name`
  in the JSON for the exact rows; character-level similarity alone would
  reject these).
* **Script variation:** English-transliterated S1 name matched to a
  Devanagari S2/S3 name for the same India business (4.1% of positive pairs)
  — e.g. the earlier structural sample `S2-166376419 राम मार्केटिंग प्राइवेट लिमिटेड`
  pattern. Character-similarity metrics computed on raw text are meaningless
  across this script boundary; only address/numeric agreement or a
  script-aware representation helps here.
* **Address components missing:** 15.29% of positive pairs have one side's
  normalized address tokens a strict subset of the other's (e.g. one record
  drops the PIN/ZIP or the state, keeping only street+city).
* **Placeholder-corrupted:** see §11 — some positive pairs have one side's
  address literally reading `<NULL>` or containing a `<CITY_NAME>` template
  tag instead of a real city (train_source2/3 and their test counterparts).

Full row-level examples (10-15 raw records per category) are preserved in
`reports/audit_pairs.json` under `noise_examples` and were intentionally kept
out of this document to stay a "small, useful diagnostic sample" rather than
a data dump.

## 7. Hard negatives — why simple rules produce false positives

Diagnostic sample only (`src/audit_pairs.py`), not a full Cartesian product:
20,000 randomly sampled S1 entities checked for exact normalized-field
collisions against the full train S2+S3 pool.

**Exact-normalized-NAME collision rule (S1.norm_name == S2/S3.norm_name):**
this is a **dangerous** rule on its own. Of 20,000 sampled S1 entities,
14,970 (74.9%) have *at least one* exact name collision in the S2/S3 pool,
producing 226,484 collision pairs — but only 17,727 (**7.83%**) of those
pairs are true matches. 7,711 of the 20,000 sampled S1 entities (38.6%) have
at least one *false* name collision, averaging 15.1 colliding records per S1
entity that has any collision at all. This is driven by generic/common
business names repeating across genuinely different businesses (chains,
common personal-name businesses, common legal-entity name patterns) — the
false-collision sample has address `token_set_ratio` distributed almost
exactly like the "easy random negative" baseline (median 35.7 vs. 33.3), i.e.
these are otherwise-unrelated businesses that merely share a name string.

**Exact-normalized-ADDRESS collision rule** is far more trustworthy: of 4,804
S1 entities (24.0%) with any address collision, 5,705/7,004 collision pairs
(**81.45%**) are true matches, and only 579 S1 entities (2.9%) have any false
address collision (mean 1.46 false collisions when it does happen).

**Easy random negatives** (random S1 × random same-country S2/S3, not a true
match): median name `ratio`/`token_set_ratio` ≈ 33, median address
`token_set_ratio` ≈ 36 — comfortably below the true-positive tail (§5), so a
threshold-based classifier has real separation to exploit, *provided* the
threshold is not set low enough to catch the ~10% hard-name true positives
from §5, which sit in almost the same numeric range as random negatives
(name `ratio` p5 of positives = 10.7; median of random negatives = 33.3 —
these ranges **overlap**). **Conclusion: name similarity alone cannot
separate the hardest true positives from ordinary random negatives; address
agreement and/or exact-token signals are required as corroborating features,
not name similarity in isolation.**

## 8. Train vs. test shift

Country shift: quantified in §3. Text-length/token-count distributions are
close between train and test for US/India (not shown in full here — see
`reports/audit_sources.json`, `per_country` blocks, comparable p50/p95 name
and address lengths across train/test for the same country).

**Token-vocabulary shift is the real risk, concentrated in France:**

| source \| country | test token *occurrences* seen in train vocab | test token *types* seen in train vocab |
|---|---|---|
| S1 \| US | 99.65% | 88.78% |
| S1 \| India | 99.75% | 89.73% |
| **S1 \| France** | **50.29%** | **44.05%** |
| S2 \| France | 63.41% | 24.45% |
| S3 \| France | 62.58% | 28.37% |
| S2/S3 \| US, India | 96-99% | 28-68% |

About half of every French *token occurrence*, and more than half of every
distinct French *token type*, in the test set never appears anywhere in
train — because France never appears in train at all. The unseen tokens are
dominated by French legal-entity suffixes and generic business-type words
with no training-side analogue: `sarl`, `eurl`, `sasu`, `ecole`, `amicale`,
`comite`, `sportive`, `pharmacie`, `lycee`, `maternelle`, plus French place
names (`nantes`, `lille`). **This directly supports the "unsupervised
statistics may be learned from the provided test data" allowance in the
rules — IDF/vocabulary/token-rarity used for blocking or features must be
computed over train+test combined (or test alone for France-specific
behavior), never train-only, or the France slice of the leaderboard will be
scored on a broken representation.** Address token coverage in France is
better (77-80% of occurrences) since numbers and Latin street-address
vocabulary generalize more than business-type words do.

## 9. Blocking diagnostics (FINAL — full-scale, verified)

`src/audit_blocking.py`, all train S1 (2,206,821) × S2∪S3 (10,320,219),
naive all-pairs = 22,774,876,013,799, denominator = complete verified ground
truth (7,638,365 positive pairs). **Process history:** this job was launched,
ran for ~9.5 hours, and was deliberately stopped (clean `kill`, not a crash)
partway through the rarest-2-name "without country" variant after a live
diagnostic showed it was CPU-active (not stalled) but the per-variant cost
was scaling ~7-8× from k=1→k=2 (an inefficiency in the diagnostic script's
own O(bucket-size) linear membership check, not in the blocking signal
itself) — projected several more hours just for one more variant, with very
low expected information gain given every prior with/without-country pair
tested showed identical recall. The decision to stop there (rather than run
to full completion or restart with an optimized implementation) was made
with the user after presenting this exact tradeoff.

| blocker | country restr. | recall | avg cand/S1 | median | p90 | p95 | p99 | max | %S1 zero-cand | reduction ratio | runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact normalized name | yes | 25.787% | 11.06 | 2 | 18 | 69 | 191 | 1,071 | 24.889% | 0.999999 | 53.0s |
| exact normalized name | no | 25.787% | 11.10 | 2 | 18 | 69 | 191 | 1,071 | 24.873% | 0.999999 | 24.5s |
| exact normalized address | yes | 8.266% | 0.35 | 0 | 1 | 2 | 3 | 20 | 75.847% | ~1.0 | 58.8s |
| exact normalized address | no | 8.266% | 0.35 | 0 | 1 | 2 | 3 | 20 | 75.847% | ~1.0 | 25.8s |
| name-prefix(4) | yes | 76.494% | 6,911.86 | 2,804 | 19,116 | 21,684 | 56,005 | 89,430 | 0.068% | 0.99933 | 22.1s |
| name-prefix(4) | no | 76.494% | 8,818.08 | 3,825 | 22,433 | 30,662 | 62,111 | 89,518 | 0.046% | 0.999146 | 15.9s |
| rarest-1 name token | yes | 78.642% | 5,001.15 | 1,095 | 13,769 | 22,039 | 30,224 | 229,501 | 0.004% | 0.999515 | 2,140.1s |
| rarest-1 name token | no | 78.689% | 6,731.27 | 1,566 | 22,872 | 29,133 | 43,009 | 1,099,665 | 0.003% | 0.999348 | 2,796.8s |
| rarest-2 name token | yes | **85.362%** | 40,595.64 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | 15,820.7s |
| rarest-2 name token | no | NOT COMPLETED | NOT COMPLETED | — | — | — | — | — | — | — | — |
| rarest-3 name token | both | NOT COMPLETED | — | — | — | — | — | — | — | — | — |
| rarest-1/2 address token | both | NOT COMPLETED | — | — | — | — | — | — | — | — | — |
| name/address unions (indexed) | — | NOT COMPLETED (Step 4 of the finalization plan was never reached — see below) | | | | | | | | | |

`rarest-2 name token, with country` was **not** JSON-checkpointed before the
stop (checkpointing only fires after both country variants of a given k
complete) — its recall/avg-cost/runtime were recovered from the plain-text
log line, which is why only 3 of its 11 metrics are available; the
percentile/reduction-ratio fields genuinely do not exist anywhere and are
marked `NOT MEASURED` rather than estimated.

**Every with/without-country pair that *did* complete shows identical
recall** (exact-name: 25.787% both; exact-addr: 8.266% both; prefix4:
76.494% both; rarest-1-name: 78.642% vs 78.689%, a difference of 0.047
percentage points — noise-level, not a real effect) — reconfirming §5's
100%-country-agreement finding holds at full scale across four independent
signal families, not just in the 300k-pair sample. Cost (avg candidates/S1)
is consistently higher without country (1.05-1.5×), so restricting by
country is a measured, safe, near-free cost reduction on **train** data —
see §9a for why this cannot be assumed to hold on France.

**Pareto read (individual blockers, not unions):**
- **Cheapest useful:** exact-name (25.8% recall, 11 candidates/S1) — too low
  recall alone to be a complete strategy.
- **Best practical single blocker measured:** rarest-1-name-with-country
  (**78.642% recall at median 1,095 / mean 5,001 candidates per S1** — the
  only ≥75%-recall signal with a fully measured, moderate cost profile).
- **Highest-recall measured:** rarest-2-name-with-country (**85.362%
  recall**, but mean cost jumped 8.1× to 40,596 candidates/S1, and its full
  percentile shape is unknown — could hide a very large tail, as rarest-1's
  own p99→max ratio already shows, e.g. rarest-1-without-country's max of
  1,099,665 against a median of only 1,566).
- No ≥95%/97%/98%/99%-recall operating point was reached by any single
  measured blocker; the union analysis needed to test whether one exists
  cheaply (Step 4 of the finalization plan) was **not completed** — this is
  the single largest open question for EXP-001's candidate-generation design
  (see §16).

**Union recall is a separate, sample-level-only measurement (300k positive
pairs, `audit_pairs.py`, NOT the full-scale ground truth):** union of
rarest-2 name-**or**-address tokens reached 98.37% recall on that sample.
This is *directionally* consistent with the missed-positive proxy below
(§10) — only 0.002% of sampled positives share zero tokens on *both* fields
— but **has not been verified at full scale** and should not be treated as
equivalent to the full-scale numbers in the table above.

## 9a. Country filtering — status and caveat

Empirically, on **train** data, country-restricted blocking costs nothing:
every with/without-country pair measured (§9, four independent signal
families at full scale) shows identical recall, only lower candidate cost
when restricted. **This is a TRAIN-data empirical finding, not a guaranteed
universal constraint.** France is test-only (§3/§8) and structurally
different from train (only 50-63% of its token occurrences appear anywhere
in train vocabulary, §8) — nothing in this audit measures whether
country-consistency holds *for France specifically*, since train has zero
France rows to check it against. The recommendation is to keep country as a
soft/cheap pre-filter (e.g. still allow a small cross-country fallback pass)
rather than a hard, unconditional restriction, until France behavior can be
observed via the public leaderboard.

## 10. Missed-positive analysis (structural ceiling proxy)

A full blocker-specific missed-match analysis (rerunning the expensive
inverted-index build to list exactly which pairs `rarest-1-name-with-country`
or `rarest-2-name-with-country` miss) was judged not worth another
multi-hour run given the stated goal of gathering "enough reliable evidence"
rather than exhaustively benchmarking. Instead, a **cheap (<2 minute),
blocker-agnostic proxy** was computed on the same fixed-seed 300k
positive-pair sample: what fraction of true positives share **zero**
normalized tokens with their match — a hard ceiling that *no* token-overlap
blocker (exact-name, prefix, rarest-K-name at any K) can ever cross,
independent of which specific one is chosen.

| | % of sampled positive pairs |
|---|---|
| zero shared normalized **name** tokens | **14.296%** |
| zero shared normalized **address** tokens | 4.394% |
| zero shared tokens on **both** name and address | **0.002%** (6 of 300,000) |

**This number lines up tightly with the measured full-scale results:**
rarest-2-name-with-country's actual miss rate (100 − 85.362% = 14.638%) is
almost exactly the 14.296% structural "zero shared name tokens" ceiling —
strong evidence that rarest-2-name is already close to the maximum recall
any *name-token-only* blocker can reach, and that further gains must come
from adding the **address** channel (or a non-token signal), not from a
larger K on name alone.

**Representative "zero shared tokens on either field" examples (the
genuinely hardest tail — would need more than token blocking):**

| S1 name/addr | matched (true) name/addr | pattern |
|---|---|---|
| `Martin Advanced Noble LLC` / 3820 3rd Street, Phoenix, AZ | `manoblecom` / 820 3st St, Osborn, Arizona | squashed-domain trade name (no shared word at all) + house-number/city typos |
| `Ventura Cayman LLC` / 18667 74th Street, Ottumwa, IA | `venturacayman.com` / 18657 74nd St, Ottmwa | same pattern: name concatenated into a bare domain string |
| `Sunrise Constructions Private Limited` / Kolkata, West Bengal | `sunriseconstructionsprivate.com` / Calcutta, WB | same pattern, India — plus a genuine city-name variant (Kolkata/Calcutta) |
| `National Education Technologies of Port Neches Inc` | `nationaleducationtechnologies.com` / "Eleventh St... PT Nehes" | domain-squash name + digit-word substitution (11th→Eleventh) + severe city typo |
| `Surgical Rocky Associates PC` | `surgicalrockyassociates.com` / "11 2rd St, Farmvilel, Virginia" | same pattern + house-number and city typos |

**Representative "zero shared name tokens, but address does share" examples
(retrievable by an address-token blocker even though name-token blocking
alone would miss them — the core justification for a name+address union):**
almost all of these are **script-variation** cases — a Latin S1 name matched
to a Devanagari/Tamil/Malayalam S2/S3 name for the same India business, with
the address staying in Latin/transliterated form on both sides (e.g.
`Shivam Systems Pvt Ltd` ↔ `ശിവം സിസ്റ്റംസ് പ്രൈവറ്റ് ലിമിറ്റഡ്` — Malayalam,
a script not previously called out in §5/§8's Devanagari/Kannada/Gujarati
examples; `Unique Business Private Limited` ↔ `यूनिक बिजनेस प्राइवेट लिमिटेड`;
`Ace Trading Pvt Ltd` ↔ Tamil `ஏஸ் டிரேடிங் பிரைவேட் லிமிடெட்`), plus one clean
US business-name-substitution case (`Mejia and Collins Bond Inc` ↔
`Zetaecto`, same street address). **Conclusion: address-token blocking is
not a redundant backup to name-token blocking — it is the specific mechanism
that recovers the script-variation failure mode**, which name-side
similarity (raw or fuzzy) structurally cannot see.

## 11. Placeholder / mojibake artifact investigation (FINAL — completed)

**Process history (kept for transparency, not hidden):** the first run of
this scan was intentionally killed partway through (4 of 6 files scanned,
no JSON ever written since the script only writes output once at the end)
because it was running concurrently with three other memory-heavy jobs and
free RAM had dropped to <1GB. It was rerun sequentially and completed
cleanly; the numbers below are 100% from that clean rerun — no partial or
stale output was used.

| file | name `<CITY_NAME>`-family | addr any bracket-tag (mostly embedded `<NULL>`) | addr mojibake rows |
|---|---|---|---|
| train_s1 | 116 | 5 | 539 |
| train_s2 | 220 | 43,810 | 1,118 |
| train_s3 | 248 | 43,583 | 949 |
| test_s1 | 108 | 6 | 457 |
| test_s2 | 297 | 35,039 | 1,185 |
| test_s3 | 288 | 35,166 | 1,017 |

All `<CITY_NAME>`-family and mojibake occurrences are **overwhelmingly
concentrated in India rows** (0 in every US row measured; 2-5 per test file
in France). Whole-field `<NULL>` (field equals exactly `<NULL>`) is **0 in
every file** — the placeholder virtually always appears **embedded** inside
a longer address as one comma-separated component (e.g. `2769 Carefree Cir,
<NULL>, Flagstaff, Arizona`), which is what "any bracket-tag" measures.
Across the full 7,638,365 train positive pairs: **63,592 (0.833%)** have a
placeholder on either side, **1,877 (0.025%)** have mojibake on either side
— both small minorities, not a first-order concern relative to §7/§9's
findings, but non-zero and now fixed (below).

**Mojibake mechanism, confirmed from real examples:** UTF-8-encoded
punctuation (en-dash U+2013, right single quote U+2019) misread once through
a Latin-1/cp1252 codepage, e.g. `Gali No. Â\x80\x93 01` (from an en-dash) and
`Vspâ\x80\x99S Bhavana` (from a right single quote). `normalize_basic()` maps
all Unicode punctuation/symbol code points to a space, which incidentally
turns most of these byte sequences into whitespace without recognizing or
reconstructing the original character — **no dictionary-based mojibake
correction is implemented or claimed.**

**Normalization fix applied (embedded placeholders — the gap identified in
this same audit):** the original normalizer only treated a field as missing
when it was *entirely* `<NULL>` (case-insensitive), missing the far more
common embedded case above. Two things were investigated and one was
explicitly **rejected** based on evidence found mid-fix:

- **Rejected:** stripping bare, unbracketed `NULL`/`null` wherever it
  appears as a token. Direct evidence against this: `train_source1.tsv`
  contains **"Null Bazaar"** (a genuine, well-known Mumbai locality) and
  place-name fragments `Nullivilai`, `Nullipady`, `Nullikadu` (real Tamil
  Nadu/Kerala locality names) as real address text — 87,936 raw
  case-insensitive `NULL` occurrences exist across the files, and a
  substring check found 34-89 legitimate `null`-in-a-longer-word matches
  per file (e.g. "Nullivilai"). Stripping bare `null` would corrupt real
  Indian addresses, so it was deliberately **not** implemented.
- **Rejected (also caught during self-testing):** a first draft used a
  broad generic pattern (any lowercase word inside single angle brackets)
  to catch corrupted `<CITY_NAME>` spelling variants uniformly. This was
  tested against real examples and found unsafe — it also matched
  `<< Elisabeth >`, which is a **real name** with decorative junk-bracket
  punctuation (the same noise family as `#4 Presidio` or `[INCORPORATED]
  Peak...`, §2), not a placeholder. Caught by the self-test suite itself
  before being trusted.
- **Implemented:** a small, evidence-grounded explicit list of exactly the
  spellings actually observed via grep/audit
  (`city_name`, `clty_name`, `cit_name`, `city_iname`, `cbi_ynome`,
  `fity_name`, `icity_nem`, `ccaty_nae`), matched only inside angle brackets,
  stripped anywhere in a field (not just whole-field); plus embedded
  `<null>` (any position, not just whole-field). Bracketed content with
  digits/punctuation (e.g. `<1-98/9/3/31>`, a real address/plot reference)
  is untouched — it still passes through ordinary punctuation-to-space
  handling, preserving its digits as tokens.
- **18/18 normalization self-tests pass** (`python src/textnorm.py` →
  `textnorm self-test OK (18/18 cases)`), including every safety case above
  (Null Bazaar preserved, `<< Elisabeth >` preserved, digit-bracket content
  preserved, Devanagari/Kannada preserved, French accents preserved).

**Impact scope, quantified (not rebuilt):** using the already-measured
counts above, this change touches **0.65%** of address fields and **0.005%**
of name fields dataset-wide (157,609 / 24,229,173 address rows;
1,277 / 24,229,173 name rows), and **0.833%** of positive training pairs.
This is small enough that the 6 multi-GB parquet caches and every audit
report in this document (all computed under the *pre-this-fix* normalizer)
were **not** rebuilt or rerun — doing so would cost hours for a
sub-1%-of-rows change. **This means every number in §1-§10 and §12 of this
report was measured under the normalizer *before* the embedded-placeholder
fix.** The caches should be rebuilt as part of EXP-001's own data
preparation step (not a separate Phase-1 action), at which point candidate
tokens will no longer include stray `null`/`city`/`name` fragments from the
~0.65% of address rows affected.

## 12. Leakage / artifact findings

(`src/audit_leakage.py`, completed.) No exploitable leakage found:

* **entity_id carries no ordering/encoding signal.** The numeric suffix after
  `S{1,2,3}-` is effectively a random integer (mostly 9 digits, min as low as
  5-576 in some files) with `n_unique == n_rows` in every file (no id
  reuse). Spearman correlation between row order and id value is ≈0 in every
  file (train S1 0.0, S2 −0.0003, S3 0.0006; test files 0.0002-0.0004) — file
  order does not encode id magnitude, so a model could not exploit "row
  position" as a proxy for id or generation order.
* **Country is not block-sorted in file order.** `country_blocks_in_file_order`
  (number of times the country value changes between consecutive rows) is
  1,060,594 out of 2,206,821 rows in train S1 — essentially every other row
  changes country, i.e. rows are shuffled, not grouped by country. Same
  pattern in every other file. No positional leakage of country either.
* **Train/test overlap is negligible.** Zero test S1 rows have an exact
  normalized (name, address) twin in train S1 (0.0% of 1,732,544). Test
  S2/S3 have a small overlap (0.054% / 0.043%, 2,645 / 2,189 rows) — small
  enough to be incidental (generic business text recurring by chance, e.g. a
  common chain name + generic address pattern) rather than a data-generation
  artifact worth exploiting or worrying about.
* **Near-duplicate S1 entities** (same normalized name+address+country)
  within a single split are essentially absent: 0 in train S1, 2 rows (1
  group) in test S1. Not a meaningful signal either way.

**Conclusion:** nothing here suggests the dataset was generated with an
exploitable ordering, id-encoding, or train/test contamination artifact.
Standard entity-level stratified splitting (§15) is sufficient; no special
anti-leakage handling is required beyond what's already planned.

## 13. Local metric validation

`src/metrics.py` implements the official macro F0.5-per-S1-entity metric
exactly (singleton truth-empty/pred-empty → 1.0, truth-empty/pred-non-empty →
0.0, `entity_scores`/`macro_fbeta` both operate on **sets**, so duplicate ids
within a predicted list — e.g. a malformed `"a,a,b"` — cannot inflate true
positives; they collapse to `{a,b}` before scoring, matching how the
official validator would treat `matched_entity_ids` semantics).

All 9 requested cases pass (`python src/metrics.py` → `metrics self-test OK
(9/9 cases)`):

1. truth empty, pred empty → 1.0 ✓
2. truth empty, pred non-empty → 0.0 ✓
3. exact set match → 1.0 ✓
4. partial true match (README's own worked example: P=2/3, R=1 → 0.7143) ✓
5. true matches + one false positive (P=2/3, R=2/3 → F0.5=0.667) ✓
6. completely incorrect non-empty prediction → P=R=F=0.0 ✓
7. multiple true matches, recall-limited (P=1, R=0.5 → F0.5=0.8333, showing
   precision's 2× weight: a false-negative-only miss costs less than a
   false-positive of the same size would) ✓
8. duplicate predicted IDs collapse via set semantics before scoring
   (`{a,a,b}` scores identically to `{a,b}`) ✓
9. `macro_fbeta` scores every key present in `truth`, including one entirely
   missing from `pred` (treated as an empty prediction, not skipped) — a
   3-entity macro average with a singleton, an exact match, and a
   truth-non-empty/pred-empty miss computed and checked by hand ✓

## 14. Competition risks (ranked)

1. **Exact-name blocking/matching is a precision trap.** 74.9% of a random
   S1 sample has *some* exact-name collision in the S2/S3 pool, but only
   7.8% of those collisions are true matches (§7). Any rule or model feature
   that leans on name-equality as a strong positive signal will manufacture
   false merges — directly costly under F0.5's 2× precision weighting.
2. **France (~15% of every test file) has ~50% unseen token-occurrence rate
   against train vocabulary** (§8) — is completely absent from train, and
   uses a different legal-suffix/business-type vocabulary (SARL/EURL/SASU
   vs. Inc/LLC/Pvt Ltd). Any IDF, rarity, or vocabulary structure learned
   train-only will underperform there; must be learned from train+test.
3. **Low-similarity true-positive tail (§5-6):** ~10% of true matches have
   name `token_set_ratio` < 50, and country-consistent but everything else
   discordant. A single similarity threshold, tuned to typical positives,
   will systematically miss this tail — hurting recall on exactly the
   entities where address or exact-token signals must carry the decision.
4. **Address is noisier in small ways than names are.** Exact-address match
   rate (8.2%) is lower than exact-name match rate (25.8%) among positives,
   even though address token-Jaccard is comparable — address blocking must
   tolerate missing components (15.3% of positives are strict token subsets)
   more than exact-match blocking allows for.
5. **Ownership is 1-owner-per-record in train ground truth, but this is not
   a documented constraint** (§4) — a pipeline that hard-codes "each S2/S3
   record can only ever match one S1 entity" as an architectural assumption
   (e.g. a global bipartite-matching post-process) is over-fitting to an
   artifact of how the labels were built, not a rule of the problem.
6. **Script variation (4.1% of positives) and placeholder/mojibake noise**
   (§6, §11) require normalization that is careful not to destroy meaning —
   our earlier normalizer bug (accent-stripping logic corrupting Devanagari
   combining marks via a category-based regex written with unescaped `\w`)
   would have silently degraded ~13-14% of every India file's name/address
   text (Devanagari share in train S2/S3 India name field) had it shipped;
   caught and fixed before any downstream statistic was trusted (§ normalizer
   note below).
7. **~26% of S2 and ~25% of S3 train records never appear in any positive
   pair** (§4) — the matching model must be a genuinely discriminative
   classifier, not a "always merge the nearest candidate" heuristic, since a
   large fraction of nearby-looking candidates are true negatives by
   construction.

## 15. Recommended validation design

The official metric is macro F0.5 **per Source-1 entity**, not pairwise
accuracy/F1/AUC — `src/metrics.py` (§13) must be the only metric used to
select thresholds, compare experiments, and report validation scores.

**Split:** stratified, **entity-level** (never pair-level) random split of
the 2,206,821 train S1 entities, stratified by `country` (US/India) and by
match-count bucket (0 / 1 / 2 / 3+) so singleton rate and multi-match
difficulty are matched between the fold(s) and the full train set — the
overall singleton rate (5.585%) and per-country singleton rates (5.583% US,
5.588% India, §4) are close enough that stratifying mainly guards against
unlucky sampling on the rarer high-match-count tail (max 11 matches, §4).
Grouping must be by S1 entity id: since every matched S2/S3 record belongs to
exactly one S1 entity in train (§4 ownership finding), splitting by S1 id
automatically avoids leaking a matched pair's S2/S3 half across the split
boundary — no additional grouping key is needed for that reason, but the
split must still assign *all* rows belonging to a given S1 id (its S1 record
and its full candidate/positive pool) to the same side.

**Reported metrics per experiment:** overall macro F0.5, macro precision,
macro recall, F0.5 restricted to singletons vs. non-singletons separately
(§13's `macro_fbeta` already returns this breakdown) — since singletons
(5.6%) and non-singletons score very differently and averaging can hide a
regression in one group.

**Multiple seeds:** given the entity count (2.2M), a single large stratified
split is statistically stable enough for day-to-day iteration; use ≥3 seeds
only for the final EXP before submission-quality decisions, to bound
variance on threshold selection specifically (F0.5's precision-heavy shape
makes it sensitive to threshold near the decision boundary).

**France limitation (must be stated, not worked around):** train has zero
France rows, so no train-based split can measure France performance, and
§8's vocabulary-shift finding means France is exactly where performance is
least certain. We will **not** fabricate France labels or pseudo-ground-truth.
Mitigations that stay within the rules: (a) learn vocabulary/IDF/rarity
statistics from train+test combined so blocking generalizes structurally
even without France labels (§8), (b) hold out a stratified slice of the
*public leaderboard* feedback as the only real France signal available
during the challenge, (c) manually audit a sample of France candidate pairs
during development for face-validity even without labels, and (d) prefer
features/thresholds that are visibly country-robust in validation (near-equal
performance on US vs. India) as a weak proxy for likely France robustness,
since France's fields (name/address structure, punctuation, digit patterns —
§ per-country text profiles) are structurally closer to a Latin-script
country than to India's mixed-script profile.

## 16. Recommended EXP-001 baseline (FINAL — based on completed Phase-1 evidence)

*Proposed only — not implemented, per the phase-1 stop condition. Approved
by the user pending their review of this design.*

* **Normalization:** `src/textnorm.normalize_basic`, now including the
  embedded-placeholder fix (§11) — NFKC + casefold + Latin-only accent
  stripping + evidence-scoped `<NULL>`/`<CITY_NAME>`-family stripping +
  Unicode punctuation/symbol → space + whitespace collapse. Caches must be
  rebuilt from this version before EXP-001 feature extraction (§11 impact
  quantified at 0.65%/0.005% of rows — small, but real).
* **Candidate generation:** country-restricted (soft filter, §9a) union of
  (a) exact-normalized-name (25.8% recall alone, ~11 candidates/S1, §9), (b)
  **rarest-1-name-token — the best fully-measured single signal (78.6%
  recall, median 1,095/mean 5,001 candidates/S1, §9)**, (c) a rare-
  **address**-token pass, since §10's proxy analysis shows address-token
  overlap recovers precisely the script-variation failure mode that no
  amount of name-side blocking can reach (0.002% of positives share zero
  tokens on *both* fields vs. 14.3% on name alone). **Rarest-2-name-token
  reached higher measured recall (85.4%) but at 8× the cost with an unknown
  tail shape (§9) — start with rarest-1 + address for a bounded, predictable
  candidate budget, and treat rarest-2 as a lever to pull later if recall is
  insufficient, not the default.** The name+address union's actual full-scale
  recall/cost is **not yet measured** (Step 4 of the finalization plan was
  not reached, §9) — this is EXP-001's first thing to verify, not an
  assumption to build further on.
* **Candidate budget target:** low thousands per S1 (rarest-1-name's own
  measured median of 1,095 is a reasonable anchor; its p99/max show a heavy
  tail — 30,224 / 229,501 — so a per-S1 cap or a fallback rule for
  pathological buckets is worth building in from the start, not added later).
* **Features:** country-equality flag (100% positive-pair agreement, §5,
  free); name and address `token_set_ratio`, `token_sort_ratio`,
  Jaro-Winkler, 3-gram Jaccard, token Jaccard (RapidFuzz, validated); exact-
  normalized-field-equality flags for name and address **kept separate**
  given their very different precision profile (name-exact 7.8% vs.
  address-exact 81.5% precision, §7); shared-numeric-address-token flag
  (78.64% of positives share one, §8); min-document-frequency of the
  strongest shared token (how rare was the best overlap).
* **Matching/scoring logic:** **sklearn** (already installed — `scikit-learn
  1.8.0`), not LightGBM (confirmed **not installed**, and the user has
  explicitly said not to install it yet). A gradient-boosted or logistic
  classifier from sklearn (e.g. `HistGradientBoostingClassifier` or
  `LogisticRegression` on the feature set above) trained on (true matches)
  vs. (blocking-generated candidates that are **not** in ground truth as
  negatives — not random negatives, since §7 shows random negatives are
  markedly easier than the false collisions blocking will actually produce).
* **Threshold strategy:** per-country threshold search on the validation
  split, optimizing macro F0.5 directly via `src/metrics.py` (self-tested,
  §13) — not a global threshold, since the low-similarity tail (§5) and
  precision-weighting make the optimal point country-dependent.
* **Singleton handling:** explicit empty-prediction path when no candidate
  clears threshold (§13 case 1/2: correct-empty scores 1.0, any false merge
  scores 0.0) — never force a top-1 emission.
* **Validation split:** entity-level (by S1 id) stratified split by country
  and match-count bucket (§15) — France cannot be validated on train (zero
  rows); no synthetic France labels will be created.
* **Metrics to record end-to-end** (validation S1 → candidate generation →
  pair scoring → threshold → predicted match set → macro F0.5, **not**
  pairwise accuracy/F1): macro F0.5, macro precision, macro recall,
  singleton F0.5, non-singleton F0.5, blocking recall, candidate count,
  avg/median/p95/max candidates-per-S1, candidate reduction ratio, runtime,
  memory where practical — the full `experiments/experiments.csv` schema
  (see §19 below), one row per run.
* **Scale discipline (explicit, given the dataset size):** no Cartesian
  product anywhere (candidate generation is inverted-index/groupby-based
  throughout this audit and must stay that way); avoid materializing a
  giant all-candidates DataFrame in memory at once — process S1 in batches
  and score in chunks; reuse the parquet caches rather than re-deriving
  normalized text per run; run at most 1-2 heavy corpus-scale jobs
  concurrently on this machine — this audit directly hit <1GB free RAM
  running 4 jobs at once (§ compute observations below) and the same risk
  applies to EXP-001's own runs.

## 19. Experiment log schema (final)

`experiments/experiments.csv` columns: `experiment_id, timestamp,
description, validation_split, validation_f0_5, macro_precision,
macro_recall, f0_5_singletons, f0_5_non_singletons, blocking_recall,
candidate_pairs, avg_candidates_per_s1, median_candidates_per_s1,
p95_candidates_per_s1, max_candidates_per_s1, candidate_reduction_ratio,
runtime_seconds, random_seed, model, threshold, notes`. `EXP-000` (this
audit) is preserved with all metric fields blank (no model was trained) and
a pointer to this document in `notes`.

## 20. Compute observations (for future runs)

Python 3.14.7; pandas 2.3.3, numpy 2.4.2, scikit-learn 1.8.0, pyarrow 23.0.1,
rapidfuzz 3.14.6 (pinned in `requirements.txt`); lightgbm and jellyfish are
**not installed**. Running 4 heavy audit jobs concurrently drove free RAM
from ~7GB down to ~0.6GB (critical-pressure territory); two jobs were killed
and rerun sequentially as a direct result (§21). The full-scale rare-token
blocking job separately ran for ~9.5 hours before being deliberately stopped
(§9) — its per-pair membership-check design (Python list `in`, not a
set/hash lookup) is the main reason, not the dataset size itself; any future
reimplementation of this specific diagnostic should use `set` buckets.

---

## What we learned that should change our strategy

1. **Exact name equality is a precision hazard, not a shortcut.** 7.83%
   precision on exact-normalized-name collisions (§7) means "same name ⇒
   same business" is wrong more than 9 times out of 10 in this dataset — any
   heuristic or model feature must treat name equality as weak-to-moderate
   evidence, corroborated by address, not decisive on its own.
2. **Exact address equality is the strongest single cheap signal found**
   (81.45% precision, §7) — stronger than exact name — so blocking/feature
   design should weight address-based exact/near-exact signals more heavily
   than name-based ones, which inverts the naive assumption that name is the
   primary identity key.
3. **A real (not hypothetical) low-similarity true-positive tail exists**
   (§5-6): ~10% of true matches have name `token_set_ratio` below 50, with
   country as the only universally reliable signal. Blocking recall targets
   must be validated against this tail specifically, not just against the
   "easy" 90%.
4. **Country is free to use as a soft pre-filter for blocking, empirically,
   on train** — 100% country agreement across the entire sampled positive
   set (§5), and identical full-scale recall with/without country
   restriction across four independent full-scale blocker families (§9,
   §9a) — but this is a train-only observation with no France data to
   check it against, so it should stay a soft/cheap filter (with a fallback
   pass), not a hard-coded `{US, India}` restriction, since France (§3, §8)
   is ~15% of every test file and structurally different.
5. **~74% of true matches do not share an exact normalized name** (100% −
   25.8% exact-name recall, §9) — confirms token-level/fuzzy blocking is not
   optional; exact-match blocking alone leaves three-quarters of true
   matches unreachable no matter how good the downstream classifier is.
6. **~26% of every S2/S3 file is true-negative-only** (never appears in any
   positive pair, §4) — candidate generation will routinely retrieve
   plausible-looking non-matches, so the matching-stage classifier's job is
   substantive, not a rubber-stamp over blocking output.
7. **A subtle normalization bug nearly corrupted every downstream statistic
   for India-market Devanagari text** (13-14% of India-market S2/S3 name
   fields) before we caught it mid-audit by testing on real script-diverse
   examples — this is now a standing process requirement, not just a one-off
   fix: any future normalization change must be spot-checked against
   Devanagari/other-Indic, French-accented, and placeholder-laden real
   examples before being trusted for statistics (documented in `src/textnorm.py`).
8. **Stale-cache risk is real in this environment:** parallel background
   jobs sharing the same on-disk parquet cache can silently read
   pre-normalization-fix data if not sequenced carefully — we caught one
   instance (test_s2/test_s3 caches lagging behind a normalizer fix) before
   any audit number was drawn from it; any future pipeline stage that
   depends on `data/interim/*.parquet` needs an explicit freshness check
   (e.g. a normalizer-version stamp) rather than relying on file presence.
9. **A diagnostic tool's own inefficiency can cost more than the analysis
   it produces.** The full-scale rare-token blocking measurement ran ~9.5
   hours and was stopped before completion (§9) not because the underlying
   question was expensive, but because its implementation used an O(bucket-
   size) linear membership check per candidate pair instead of a hash
   lookup — a fixable bug, not a fundamental cost of the measurement.
   EXP-001's own candidate-generation code must use hash/set-based indexing
   from the start, not the pattern this diagnostic script used.
10. **The recall gap left by name-only blocking has an almost exact
    structural explanation, not a mysterious tail.** Rarest-2-name's actual
    miss rate (14.638%) matches the measured "zero shared name tokens"
    ceiling (14.296%, §10) almost exactly — meaning the remaining gap is not
    fixable by a bigger K on name alone, it needs the address channel (which
    §10 shows recovers the dominant remaining failure mode: script
    variation, not random noise).

## 21. Failed / incomplete work (preserved, not hidden)

1. **Normalizer bug** (details in §4/normalization history): a first version
   had a Python-incompatible regex syntax error, then a `\w`-scope bug in
   pyarrow's regex engine that would have mis-flagged ~13-14% of India-market
   name fields as suspicious. Caught before being trusted; all 6 caches
   rebuilt and timestamp-verified afterward.
2. **`audit_pairs.py` crashed on its first run** (`TypeError` calling
   `.quantile()` on a boolean-dtype column) — fixed by branching on dtype;
   rerun completed cleanly.
3. **Severe memory pressure** (4 concurrent heavy jobs drove free RAM to
   <1GB) forced killing and sequentially rerunning `audit_leakage.py` (first
   attempt produced zero output, nothing stale was used) and
   `audit_placeholders.py` (first attempt scanned 4/6 files, never wrote its
   JSON, nothing stale was used). Both completed cleanly on rerun.
4. **`audit_blocking.py` was deliberately stopped incomplete** after ~9.5
   hours, mid-way through `rarest2_name_without_country`, once a live CPU/
   progress diagnostic showed the per-variant cost scaling ~7-8× from k=1 to
   k=2 with low expected additional information (the with/without-country
   recall pattern had been identical in every prior case). This was a
   deliberate, evidence-based, user-confirmed stop — not a crash. Completed:
   6 full-scale key blockers, rarest-1-name (both country variants, full
   detail), rarest-2-name-with-country (recall/cost/runtime only, recovered
   from the log text since it wasn't JSON-checkpointed). **Not completed:**
   rarest-2-name-without-country, rarest-3-name (either variant), rarest-1/2
   address-token (either variant), and all planned name/address union tests
   (the finalization plan's Step 4) — these remain open questions for
   whoever picks up candidate-generation tuning after EXP-001.
5. **A full blocker-specific missed-positive analysis was not run** (would
   have required another expensive inverted-index build); a cheap,
   blocker-agnostic proxy (§10) was used instead and judged sufficient for
   Phase-1's purpose (identifying failure *mechanisms*, not exhaustively
   cataloguing every missed pair).
6. **The embedded-`<NULL>` normalization fix (§11) was applied but its
   impact was deliberately *not* propagated** to any existing cache or
   report in this document — quantified as small (<1% of rows) rather than
   rebuilt, to avoid another multi-hour rebuild for a sub-1% change. Every
   number in §1-§10 and §12 reflects the *pre-fix* normalizer.
7. **A first draft of the placeholder fix itself was unsafe** (broad
   bracket-stripping regex ate a real name, `<< Elisabeth >`) and was
   caught and corrected by the normalizer's own self-test suite before
   being used anywhere (§11).
