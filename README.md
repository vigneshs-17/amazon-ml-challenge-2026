# Amazon ML Challenge 2026 — Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance, production-grade machine learning solution for large-scale **multi-source business entity resolution (ER)** across noisy datasets. Designed and implemented for the **Amazon ML Challenge 2026**.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Structure](#3-dataset-structure)
4. [Entity Resolution Approach](#4-entity-resolution-approach)
5. [Unicode-Safe Normalization](#5-unicode-safe-normalization)
6. [Candidate Generation / Blocking](#6-candidate-generation--blocking)
7. [Formula 2 Candidate Ranking](#7-formula-2-candidate-ranking)
8. [Feature Engineering](#8-feature-engineering)
9. [Model](#9-model)
10. [Conflict Resolution](#10-conflict-resolution)
11. [Validation Strategy](#11-validation-strategy)
12. [Results](#12-results)
13. [Test Inference Architecture](#13-test-inference-architecture)
14. [Project Structure](#14-project-structure)
15. [Setup](#15-setup)
16. [Usage](#16-usage)
17. [Reproducibility](#17-reproducibility)
18. [Output Format](#18-output-format)
19. [Limitations](#19-limitations)
20. [Competition Constraints](#20-competition-constraints)
21. [License / Dataset Notice](#21-license--dataset-notice)

---

## 1. Overview

In commercial platforms, business identity data originates from multiple disparate sources, each presenting partial, noisy fragments of real-world entities without unified keys or identifiers. This repository implements an end-to-end, reproducible Entity Resolution pipeline that links **1,732,544 test reference entities (Source 1)** against **9,970,219 candidate entities (Source 2 and Source 3)** across multiple countries (United States, India, and France).

Key highlights:
- **Scalable Candidate Blocker (B5)**: Hybrid multi-key inverted index achieving **95.49% raw candidate recall** with a **99.998% search-space reduction**.
- **Deterministic Formula 2 Candidate Ranking**: Sublinear priority queue ranker scoring token-IDF overlap, exact string matches, and postal tokens.
- **47 Discriminative Features**: Granular string similarity, tolerant numeric/postal parsing, token overlap ratios, and length statistics.
- **Histogram-based Gradient Boosting (HistGradientBoostingClassifier)**: Highly optimized leaf-wise decision trees trained on representative hard negative pairs.
- **Global Greedy Unique-Owner Conflict Resolution**: Enforces 1-to-1 matching constraints on physical entities across sources.
- **Parallel Chunked Resumable Architecture**: Zero-leakage country partitioning and atomic disk checkpointing capable of sustaining ~75 S1 entities/second.

---

## 2. Problem Statement

Given:
1. **Source 1 ($S_1$)**: The deduplicated reference source of business records.
2. **Source 2 ($S_2$) & Source 3 ($S_3$)**: Un-deduplicated, noisy business records.

Each $S_1$ entity can match zero, one, or multiple records in $S_2$ and $S_3$. Conversely, each $S_2$ or $S_3$ entity represents a distinct physical establishment and can match at most one $S_1$ entity. The test set introduces **France** as an unseen country not present in the training set (`US` and `India`), requiring strictly open-set country generalizability.

The evaluation metric is **Macro $F_{0.5}$** across all $S_1$ entities, weighting precision twice as heavily as recall:
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

---

## 3. Dataset Structure

| Dataset Split | Records ($S_1$) | Records ($S_2$) | Records ($S_3$) | Total Records | Countries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1,000,000 | 2,875,123 | 2,874,877 | 6,750,000 | US, India |
| **Test** | 1,732,544 | 4,985,109 | 4,985,110 | 11,702,763 | US, India, France |

Each record contains:
- `entity_id`: Unique identifier prefixed with `S1-`, `S2-`, or `S3-`.
- `business_name`: Free-text business trade name (subject to typos, legal suffix variations, transliterations).
- `business_address`: Free-text address (partial components, PIN codes, landmarks, municipal variations).
- `country`: Country identifier (`US`, `India`, `France`).

---

## 4. Entity Resolution Approach

Our pipeline adopts a four-stage architecture:
```
Raw Data
   │
   ▼
[ Stage 1: Unicode-Safe Normalization ]
   │
   ▼
[ Stage 2: Country-Aware B5 Blocker + Formula 2 Top-K Ranking ]
   │  (Generates ≤ 100 or ≤ 200 candidates per S1)
   ▼
[ Stage 3: 47-Feature Pair Extraction + LightGBM/HGB Classifier ]
   │  (Predicts match probability p ∈ [0, 1])
   ▼
[ Stage 4: Threshold Filtering (0.82) + Unique-Owner Conflict Resolution ]
   │
   ▼
Final Output TSVs (matching_results.tsv, candidate_pairs.tsv)
```

---

## 5. Unicode-Safe Normalization

To handle multilingual accents, special characters, and non-ASCII business names (critical for French and Indian records), normalization is performed deterministically:
1. **NFKD Decomposition**: Decomposes ligatures and accented characters (e.g., `é` $\rightarrow$ `e`, `ç` $\rightarrow$ `c`).
2. **Punctuation & Control Character Stripping**: Preserves alphanumeric characters while standardizing whitespace.
3. **Legal Entity Standardization**: Normalizes frequent legal tokens (`corp`, `inc`, `llc`, `pvt ltd`, `ltd`, `sarl`, `sa`).
4. **Case Folding**: Standardizes all text to lowercase ASCII.

---

## 6. Candidate Generation / Blocking

Comparing 1.73M test records against 9.97M candidates requires evaluating $1.73 \times 10^{13}$ potential pairs—infeasible without candidate blocking.

We developed the **B5 Inverted-Index Blocker**:
- **Country-Aware Rare Token Indexing**: Indexes tokens from business names and addresses using global and country-specific document frequencies (DF).
- **Adaptive Fallback**: Queries country-partitioned inverted indices first; if candidate count is below threshold, falls back to cross-country candidate pools.
- **Coverage**: Achieves **95.49% raw candidate recall** while filtering out 99.998% of negative pairs.

---

## 7. Formula 2 Candidate Ranking

When candidate pools contain hundreds or thousands of candidate entities, ranking candidates efficiently prior to ML scoring is essential. We designed **Formula 2**:
$$\text{Score}(S_1, C) = 15.0 \cdot \mathbb{I}_{\text{exact\_addr}} + 10.0 \cdot \mathbb{I}_{\text{exact\_name}} + \sum_{t \in N_1 \cap N_2} \text{IDF}_N(t) + \sum_{t \in A_1 \cap A_2} \text{IDF}_A(t) + 5.0 \cdot \mathbb{I}_{\text{shared\_postal}} + 4.0 \cdot \mathbb{I}_{\text{both\_overlap}} + 3.0 \cdot J_N + 3.0 \cdot J_A$$
- Evaluated with a fast, zero-allocation C/heap priority queue (`heapq.nlargest`).
- Selects the top $K$ candidates ($K=200$ for validation champion; $K=100$ for deadline-optimized test execution).

---

## 8. Feature Engineering

For each $(S_1, C)$ candidate pair, we extract **47 pairwise features**:
1. **Name Similarity (18 features)**: Token Jaccard, Token Dice, Levenshtein ratio, Jaro-Winkler, Prefix match, Token Sort Ratio, IDF-weighted overlap.
2. **Address Similarity (16 features)**: Complete address Jaccard, Token set ratio, Levenshtein distance, Substring containment, Shared token count.
3. **Tolerant Numeric & Postal Matching (5 features)**:
   - Shared numeric token ratio.
   - Exact postal/PIN code match indicator.
   - Numeric mismatch penalty.
   - Address digit length delta.
4. **Length and Structural Statistics (8 features)**: Absolute character length deltas, token count differences, token length ratios.

---

## 9. Model

- **Architecture**: `HistGradientBoostingClassifier` (Scikit-Learn).
- **Hyperparameters**:
  - `max_leaf_nodes`: 127
  - `min_samples_leaf`: 20
  - `max_iter`: 150
  - `learning_rate`: 0.08
  - `l2_regularization`: 2.0
- **Loss**: Binary Cross-Entropy with class prior reweighting.
- **Decision Threshold**: Optimized to $\tau = 0.82$ on held-out validation data to maximize Macro $F_{0.5}$ (heavily favoring precision).

---

## 10. Conflict Resolution

Because Source 2 and Source 3 entities represent physical commercial entities, no candidate $C \in S_2 \cup S_3$ may be linked to multiple Source 1 entities:
1. Candidate pairs satisfying $P(\text{Match} \mid S_1, C) \ge 0.82$ are collected.
2. If multiple $S_1$ entities claim the same candidate $C$, assignment is awarded greedily to the $S_1$ entity with the highest predicted probability $\max P(S_1, C)$.
3. Rejected pairs are cleanly dropped, eliminating cross-entity contamination.

---

## 11. Validation Strategy

Validation was conducted using a strict **100,000 $S_1$ stratified split** held out from `train_source1.tsv` (Seed 2026):
- Stratified by country (`US`, `India`) and match cardinality (singletons vs non-singletons).
- 345,938 ground-truth matching links.
- Evaluated with the official competition scoring formula (identical to the leaderboard engine).

---

## 12. Results

### Formal Validation Benchmark (100,000 Entities)

| Configuration | Candidate Cap ($K$) | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton $F_{0.5}$ | Non-Singleton $F_{0.5}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (EXP-001)** | 50 | 0.89120 | 0.93410 | 0.82104 | 0.91240 | 0.86940 |
| **Features V2 (EXP-002)** | 100 | 0.90850 | 0.94820 | 0.83510 | 0.93120 | 0.88450 |
| **Formula 2 (EXP-003)** | 200 | 0.91949 | 0.95821 | 0.84650 | 0.94180 | 0.89610 |
| **EXP-004 (Champion)** | **200** | **0.92237** | **0.96012** | **0.85047** | **0.94420** | **0.89980** |

### Execution Profiles

- **Validation Champion (EXP-004, Cap-200)**: Macro $F_{0.5} = \mathbf{0.92237}$ on 100k entities.
- **Deadline-Optimized Execution (Cap-100)**: Macro $F_{0.5} = 0.92028$ on 10k validation benchmark (a $-0.00209$ tradeoff accepted for a $2\times$ throughput boost).

---

## 13. Test Inference Architecture

Processing 1.73M entities required overcoming a 18 GB RAM bottleneck when candidate tables were loaded globally.

We implemented **Country-Partitioned Parallel Streaming**:
```
Test S1 Entities (1,732,544)
    ├── France Worker (259,452 entities, ~2.0 GB RAM) ──> 6 chunks
    ├── US Worker (663,106 entities, ~5.4 GB RAM) ──────> 14 chunks
    └── India Workers (809,986 entities, ~7.8 GB RAM) ──> 17 chunks
            ├── Forward Worker (Chunks 0000 -> 0008)
            └── Reverse Worker (Chunks 0016 -> 0009)
```
- **Total Concurrent RAM**: ~14.3 GB (comfortably inside 24 GB physical RAM, 0 MB disk swapping).
- **Atomic Persistence**: Every 50,000-entity chunk writes to temporary disk storage and renames atomically upon verification.
- **Resumability**: Interrupted runs detect finalized chunk manifests and resume instantly with zero duplicate work.

---

## 14. Project Structure

```
amazon_ml_challenge_2026/
├── README.md                      # Project documentation and reproduction guide
├── requirements.txt               # Pinned package dependencies
├── .gitignore                     # Git exclusion rules (raw data, caches, models)
├── Documentation_template.md      # Official challenge methodology report
├── experiments/
│   ├── experiments.csv            # Experiment tracking log
│   ├── EXP-001/                   # Baseline artifacts
│   ├── EXP-003/                   # Formula 2 artifacts
│   └── EXP-004/                   # Champion model artifacts and evaluations
├── src/
│   ├── blocking.py                # B5 Inverted-Index Candidate Blocker
│   ├── features_v3.py             # 47-feature extraction engine
│   ├── ranking_v2.py              # Formula 2 Candidate Priority Ranker
│   ├── textnorm.py                # Unicode-safe text normalization
│   ├── paths.py                   # Path and environment resolution
│   ├── metrics.py                 # Macro F0.5 evaluation implementation
│   ├── train_and_evaluate.py      # Model training entry point
│   ├── run_parallel_country_inference.py # Country-partitioned test runner
│   ├── run_india_reverse.py       # India dual-worker reverse runner
│   ├── assemble_deadline_fallback.py     # Lightweight streaming assembler
│   └── check_repo_secrets.py      # Repository safety audit tool
└── output/
    ├── candidate_pairs.tsv        # Evaluated candidate pairs
    ├── matching_results.tsv       # Scored entity matches
    └── final/                     # Frozen final submission copies
```

---

## 15. Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- 16 GB+ RAM recommended (24 GB optimal for full 3-worker concurrency)
- Multi-core CPU (8+ cores recommended)

### Installation
```bash
git clone https://github.com/your-org/amazon-ml-challenge-2026.git
cd amazon_ml_challenge_2026

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 16. Usage

### 1. Data Normalization & Cache Generation
```bash
python src/audit_sources.py
```

### 2. Model Training & Validation (EXP-004)
```bash
python src/run_exp004.py
```

### 3. Full Parallel Test Inference
```bash
python src/run_parallel_country_inference.py --cap 100 --countries France US India
```

### 4. Validate Submission Outputs
```bash
python data/raw/student_resource/utils/validate_submission.py \
    --matching output/final/matching_results.tsv \
    --candidate output/final/candidate_pairs.tsv \
    --test-dir data/raw/student_resource/dataset/test
```

---

## 17. Reproducibility

Every component of this solution is strictly deterministic:
- Random seeds are pinned to `2026`.
- Ties in candidate ranking are broken deterministically by entity ID string sorting.
- Feature extraction order and floating-point representations are strictly maintained.
- Exact equivalence verified: 0 mismatches across 1,000 entity validation audits (`experiments/FINAL-INFERENCE-PREP/equivalence_report.json`).

---

## 18. Output Format

Outputs strictly adhere to the official challenge schema:

### `matching_results.tsv` (Leaderboard Scored)
```tsv
source1_entity_id	matched_entity_ids
S1-0000001	S2-0048123,S3-0091823
S1-0000002	S3-0001924
S1-0000003	
```

### `candidate_pairs.tsv` (Pipeline Verification)
```tsv
source1_entity_id	candidate_entity_ids
S1-0000001	S2-0048123,S3-0091823,S2-0099120
S1-0000002	S3-0001924,S2-0082103
S1-0000003	
```

---

## 19. Limitations

1. **Unseen Languages / Scripts**: The normalization pipeline focuses on Latin script and transliterated characters; non-Latin native scripts (e.g., Devanagari) are simplified to ASCII.
2. **Extreme Address Abbreviations**: Addresses consisting solely of landmark references without street, city, or PIN numbers rely heavily on name similarity.
3. **Execution Resource Footprint**: Multi-worker country indexing requires at least 14 GB of available system memory.

---

## 20. Competition Constraints

- **Strictly Self-Contained**: No external geocoding APIs, web lookups, or unapproved pre-trained LLMs were utilized.
- **Subset Invariance**: Every predicted match in `matching_results.tsv` is guaranteed to be a member of `candidate_pairs.tsv`.
- **Integrity**: Exactly 1,732,544 rows with zero duplicates, missing IDs, or malformed delimiters.

---

## 21. License / Dataset Notice

- **Code License**: Licensed under the [MIT License](LICENSE).
- **Challenge Data**: The Amazon ML Challenge 2026 dataset is proprietary to Amazon and is excluded from this public repository in compliance with competition confidentiality rules.
