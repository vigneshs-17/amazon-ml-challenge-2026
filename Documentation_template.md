# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Antigravity ER  
**Team Members:** Open ML Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, reproducible Entity Resolution solution designed for the Amazon ML Challenge 2026. Our architecture combines a country-aware multi-key inverted-index candidate blocker (B5) achieving 95.49% raw candidate recall, a deterministic sublinear priority queue candidate ranker (Formula 2), a 47-feature gradient boosted tree classifier (HistGradientBoostingClassifier), and greedy unique-owner conflict resolution. On our 100,000 $S_1$ entity held-out validation benchmark, our champion model (EXP-004) achieves a **Macro $F_{0.5}$ of 0.92237** (Precision: 0.96012, Recall: 0.85047).

---

## 2. Methodology

### 2.1 Problem Analysis
During extensive initial exploratory data analysis (EDA) and forensic auditing (Phase 1):
- **Multilingual and Regional Variations**: Indian addresses feature extensive municipal numbering, missing postal codes, and landmark references ("near ATM"). French records contain accented characters (`é`, `à`, `ç`, `ô`) and unique legal structures (`SARL`, `SAS`).
- **Legal Entity Noise**: Frequent legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `Inc`) frequently misaligned across databases or appeared in alternate word orders.
- **Transposition and Abbreviation**: Common street and area abbreviations (`Rd`, `St`, `Ave`, `Ngr`, `Colony`) created substantial token-overlap noise.
- **Physical Uniqueness**: A crucial property discovered was that each $S_2$ or $S_3$ physical entity matches at most one $S_1$ entity, necessitating strict post-classification conflict resolution.

### 2.2 Solution Strategy
We developed a four-tier sequential pipeline:
1. **Deterministic Unicode-Safe Normalization**: NFKD unicode normalization, lowercase folding, punctuation standardization, and legal suffix normalization.
2. **B5 Candidate Blocker + Formula 2 Ranking**: Pruning $1.73 \times 10^{13}$ pairwise comparisons to top-$K$ candidates per $S_1$ entity.
3. **Pairwise Feature Engineering (47 features)**: Capturing string similarity, token overlap, and tolerant numeric/postal alignment.
4. **HistGradientBoosting Classifier + Unique-Owner Conflict Resolution**: High-precision probabilistic scoring thresholded at $\tau=0.82$, followed by greedy 1-to-1 assignment.

**Approach Type:** Hybrid Multi-Key Inverted-Index Blocking + Pairwise Gradient Boosted Decision Trees + Graph Conflict Resolution  
**Core Innovation:** Deterministic Formula 2 candidate ranking combined with tolerant numeric/postal matching and partitioned parallel country inference.

---

## 3. Candidate Generation (Blocking)

To scale inference to 1.73M reference entities without out-of-memory errors:
- **Blocking keys used**:
  1. Standardized rare name tokens (document frequency $< 0.01$).
  2. Standardized rare address tokens.
  3. Postal / PIN code exact buckets.
  4. Country-partitioned inverted indices with cross-country fallback.
- **Candidate pairs generated**: Average 185.58 candidates per $S_1$ entity (capped at 200 for validation; capped at 100 for deadline test execution). Search space reduction ratio exceeds **99.998%**.
- **Ensuring true matches were not lost**: Adaptive query expansion—if country-specific inverted indices return fewer than 10 candidates, the blocker queries the global candidate pool. Raw candidate recall achieved **95.49%** on validation data.

---

## 4. Matching Model

### Features used (47 total):
1. **Name Features (18)**:
   - Token Jaccard, Token Dice, Token Cosine.
   - Levenshtein edit similarity, Jaro-Winkler distance, Prefix similarity.
   - Token Sort Ratio and IDF-weighted token overlap.
2. **Address Features (16)**:
   - Full address Jaccard and Levenshtein similarity.
   - Token Set Ratio, Substring containment indicators.
   - IDF-weighted address token overlap.
3. **Tolerant Numeric & Postal Features (5)**:
   - Numeric token intersection count and ratio.
   - Exact postal/PIN code match indicator.
   - Numeric mismatch penalty (penalizing conflicting house/building numbers).
   - Address digit length delta.
4. **Structural & Length Statistics (8)**:
   - Token count deltas, character length ratios, token length differences.

### Model Architecture:
- **Model type**: `HistGradientBoostingClassifier` (Scikit-Learn).
- **Hyperparameters**: `max_leaf_nodes=127`, `min_samples_leaf=20`, `max_iter=150`, `learning_rate=0.08`, `l2_regularization=2.0`.
- **Threshold selection method**: Evaluated on 100,000 stratified validation entities using fine-grained grid search ($\Delta \tau = 0.01$). Optimal Macro $F_{0.5}$ peaked at $\tau = 0.82$, strongly weighting precision to avoid false merges.
- **Conflict Resolution**: Greedy unique-owner assignment ensuring every Source 2 and Source 3 candidate is claimed by at most one Source 1 record.

---

## 5. Results & Error Analysis

### Official Macro $F_{0.5}$ Validation Performance:
- **Champion (EXP-004, Cap-200)**:
  - **Macro $F_{0.5}$:** **0.92237**
  - **Macro Precision:** **0.96012**
  - **Macro Recall:** **0.85047**
  - **Singleton $F_{0.5}$:** 0.94420
  - **Non-Singleton $F_{0.5}$:** 0.89980
- **Deadline Test Configuration (Cap-100)**:
  - **Validation Benchmark (10k entities):** Macro $F_{0.5} = 0.92028$ (Precision: 0.96105, Recall: 0.84512), providing a $2\times$ inference speedup with negligible metric impact ($-0.00209$).

### Error Breakdown (EXP-004 on 100k entities):
- **Common False Positives (Wrong Merges)**: Chain stores or retail franchises (e.g., branches of the same bank or courier service) sharing identical trade names with slight street variations that passed the 0.82 probability threshold.
- **Common False Negatives (Missed Matches)**: Extreme transliteration differences in regional Indian names combined with landmark-only addresses lacking street or PIN details (e.g., "Behind Bus Stand"), which failed to clear the conservative 0.82 precision threshold.

---

## 6. Conclusion

Our solution demonstrates that principled entity resolution on massive real-world datasets is best achieved through a synergy of efficient sublinear candidate blocking, discriminative domain-aware feature engineering, and robust conflict resolution. By prioritizing precision through a high threshold ($\tau=0.82$) and unique-owner assignment, our model achieves a top-tier Macro $F_{0.5}$ score of 0.92237 while maintaining 100% compliance with official submission constraints.

---

## Appendix

### A. Code Artefacts
All runnable source code is organized under `src/`:
- `src/textnorm.py`: Unicode-safe string normalization.
- `src/blocking.py`: B5 Inverted-Index Blocker.
- `src/ranking_v2.py`: Formula 2 Candidate Priority Ranker.
- `src/features_v3.py`: 47-feature extraction engine.
- `src/run_exp004.py`: Full model training and evaluation script.
- `src/run_parallel_country_inference.py`: Multi-worker country-partitioned inference runner.
- `src/run_india_reverse.py`: Dual-worker reverse runner for India partition.
- `src/assemble_deadline_fallback.py`: Streaming output assembler and conflict resolver.

**Reproduction Entry Points**:
1. Train model: `python src/run_exp004.py`
2. Run test inference: `python src/run_parallel_country_inference.py --cap 100`
3. Validate output: `python data/raw/student_resource/utils/validate_submission.py --matching output/final/matching_results.tsv --candidate output/final/candidate_pairs.tsv --test-dir data/raw/student_resource/dataset/test`

### B. Additional Results
- **Equivalence Verification**: 0 mismatches across 1,000 entity audit between sequential baseline and optimized inference code (`experiments/FINAL-INFERENCE-PREP/equivalence_report.json`).
- **Memory Footprint**: Peak RAM constrained to ~14.3 GB across 3 concurrent worker processes on 24 GB host, with zero SSD swapping.
