"""EXP-004 Baseline Profiling and Test Benchmark.

Measures fine-grained execution profile of the frozen EXP-004 pipeline on test data:
1. Loading S2/S3 (Test corpus: 9.97M rows)
2. Inverted-index construction
3. Global IDF construction
4. Candidate retrieval
5. Formula-2 ranking & top-200 pruning
6. 47-feature extraction
7. Model predict_proba
8. Greedy unique-owner conflict resolution
9. Output TSV serialization

Uses a deterministic stratified sample of test S1 (India, US, France) at seed 2026.
Saves profile.json and baseline_benchmark.json under experiments/FINAL-INFERENCE-PREP/.
"""

import ctypes
import heapq
import io
import json
import sys
import time
from ctypes import wintypes

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from blocking import RareTokenIndex
from features_v3 import extract_pair_features_v3a, precompute_s1_v3
from paths import PROJECT_ROOT
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)

CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
EXP4_DIR = PROJECT_ROOT / "experiments" / "EXP-004"
PREP_DIR = PROJECT_ROOT / "experiments" / "FINAL-INFERENCE-PREP"

CAP = 200
SEED = 2026
THRESHOLD = 0.82

# Windows memory tracking
psapi = ctypes.windll.psapi
kernel32 = ctypes.windll.kernel32


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def get_ram_mb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / (1024 * 1024), pmc.PeakWorkingSetSize / (1024 * 1024)


class B5Blocker:
    def __init__(self, s23: pd.DataFrame):
        self.name_idx = RareTokenIndex(s23, "name_norm")
        self.addr_idx = RareTokenIndex(s23, "address_norm")

    def candidates(self, name: str, addr: str, country: str):
        name_cand = self.name_idx.query_union(name, country, 1, country_aware=True)
        if not name_cand:
            name_cand = self.name_idx.query_union(name, country, 1, country_aware=False)
        addr_cand = self.addr_idx.query_union(addr, country, 1, country_aware=True)
        if not addr_cand:
            addr_cand = self.addr_idx.query_union(addr, country, 1, country_aware=False)
        return name_cand | addr_cand


def resolve_unique_owners(preds_prob: dict) -> dict:
    cand_to_best_s1 = {}
    for s1, items in preds_prob.items():
        for cid, p in items:
            if cid not in cand_to_best_s1 or p > cand_to_best_s1[cid][1]:
                cand_to_best_s1[cid] = (s1, p)
    unique_preds = {s1: [] for s1 in preds_prob}
    for cid, (s1, p) in cand_to_best_s1.items():
        unique_preds[s1].append(cid)
    return unique_preds


def main():
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    print("=== EXP-004 Baseline Pipeline Profiling on Test Corpus ===", flush=True)

    # 1. Model Loading
    print("\nLoading frozen EXP-004 model...", flush=True)
    clf = joblib.load(EXP4_DIR / "model_exp004.joblib")
    assert clf.n_features_in_ == 47, f"Expected 47 features, got {clf.n_features_in_}"

    # 2. Stage 1: Loading S2/S3
    print("\n[Stage 1] Loading test S2/S3 parquet tables...", flush=True)
    t0 = time.perf_counter()
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "test_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "test_s3.parquet", columns=cols)
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )
    t_load = time.perf_counter() - t0
    print(f"  Loaded {len(s23):,} candidate records in {t_load:.2f}s. RAM: {get_ram_mb()[0]:.1f} MB", flush=True)

    # 3. Stage 2: Inverted-Index Construction
    print("\n[Stage 2] Building B5 inverted indexes on test S2/S3 (9.97M rows)...", flush=True)
    t0 = time.perf_counter()
    blocker = B5Blocker(s23)
    t_index = time.perf_counter() - t0
    print(f"  Inverted indexes constructed in {t_index:.2f}s. RAM: {get_ram_mb()[0]:.1f} MB", flush=True)

    # 4. Stage 3: Global IDF Construction
    print("\n[Stage 3] Precomputing global token IDFs...", flush=True)
    t0 = time.perf_counter()
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    t_idf = time.perf_counter() - t0
    print(f"  Global IDFs precomputed in {t_idf:.2f}s.", flush=True)
    del s2, s3, s23

    # 5. Build Representative Test Sample (1,000 S1 and 5,000 S1)
    print("\nLoading test S1 entities and constructing stratified benchmark sample...", flush=True)
    test_s1_full = pd.read_parquet(CACHE / "test_s1.parquet")
    n_test_total = len(test_s1_full)
    print(f"  Total test S1 entities: {n_test_total:,}")
    print(f"  Country distribution: {test_s1_full.country.value_counts().to_dict()}")

    # Stratified sampling by country
    sample_1k = (
        test_s1_full.groupby("country", group_keys=False)
        .apply(lambda g: g.sample(frac=1000.0 / n_test_total, random_state=SEED))
        .reset_index(drop=True)
    )
    # Ensure exact 1000 count
    if len(sample_1k) > 1000:
        sample_1k = sample_1k.iloc[:1000]
    elif len(sample_1k) < 1000:
        remaining = test_s1_full[~test_s1_full.entity_id.isin(sample_1k.entity_id)]
        sample_1k = pd.concat([sample_1k, remaining.sample(n=1000 - len(sample_1k), random_state=SEED)]).reset_index(
            drop=True
        )

    sample_5k = (
        test_s1_full.groupby("country", group_keys=False)
        .apply(lambda g: g.sample(frac=5000.0 / n_test_total, random_state=SEED))
        .reset_index(drop=True)
    )
    if len(sample_5k) > 5000:
        sample_5k = sample_5k.iloc[:5000]
    elif len(sample_5k) < 5000:
        remaining = test_s1_full[~test_s1_full.entity_id.isin(sample_5k.entity_id)]
        sample_5k = pd.concat([sample_5k, remaining.sample(n=5000 - len(sample_5k), random_state=SEED)]).reset_index(
            drop=True
        )

    sample_1k[["entity_id", "country"]].to_csv(PREP_DIR / "test_benchmark_ids_1k.csv", index=False)
    sample_5k[["entity_id", "country"]].to_csv(PREP_DIR / "test_benchmark_ids_5k.csv", index=False)
    print(
        f"  Saved benchmark samples: 1k ({sample_1k.country.value_counts().to_dict()}) and 5k ({sample_5k.country.value_counts().to_dict()})"
    )

    # 6. Granular Profiling on 1,000 Test S1
    print("\n--- Profiling 1,000 Test S1 Entities (Baseline Implementation) ---", flush=True)
    t_retrieval_total = 0.0
    t_ranking_total = 0.0
    t_feature_total = 0.0
    t_scoring_total = 0.0

    raw_cand_counts = []
    top_cand_counts = []

    # Store for conflict resolution and serialization profiling
    all_stored_preds = {}
    candidate_pairs_rows = []

    t_loop0 = time.perf_counter()

    for i, (eid, name, addr, country) in enumerate(
        zip(
            sample_1k.entity_id.astype(object),
            sample_1k.name_norm.astype(object),
            sample_1k.address_norm.astype(object),
            sample_1k.country.astype(object),
        )
    ):
        # Stage 4: Candidate Retrieval
        t_sub0 = time.perf_counter()
        cand = blocker.candidates(name, addr, country)
        t_retrieval_total += time.perf_counter() - t_sub0
        n_raw = len(cand)
        raw_cand_counts.append(n_raw)

        # Stage 5: Formula-2 Candidate Ranking
        t_sub0 = time.perf_counter()
        s1_rank_pre = precompute_s1_ranking(name, addr, name_idf, addr_idf)
        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score_v2(s1_rank_pre, n2, a2)

        if len(cand_list) > CAP:
            ranked_top = heapq.nlargest(CAP, cand_list, key=lambda c: (scores[c], c))
        else:
            ranked_top = sorted(cand_list, key=lambda c: (-scores[c], c))
        t_ranking_total += time.perf_counter() - t_sub0
        top_cand_counts.append(len(ranked_top))
        candidate_pairs_rows.append((eid, ",".join(ranked_top)))

        # Stage 6: Feature Extraction
        t_sub0 = time.perf_counter()
        if ranked_top:
            s1_feat_pre = precompute_s1_v3(name, addr, country, name_df, addr_df)
            feats = np.array(
                [
                    extract_pair_features_v3a(
                        s1_feat_pre, s23_lookup[c][0], s23_lookup[c][1], s23_lookup[c][2], name_df, addr_df
                    )
                    for c in ranked_top
                ],
                dtype=np.float32,
            )
            t_feature_total += time.perf_counter() - t_sub0

            # Stage 7: Model Scoring
            t_sub0 = time.perf_counter()
            proba = clf.predict_proba(feats)[:, 1]
            stored = [(cid, float(p)) for cid, p in zip(ranked_top, proba) if p >= THRESHOLD]
            all_stored_preds[eid] = stored
            t_scoring_total += time.perf_counter() - t_sub0
        else:
            t_feature_total += time.perf_counter() - t_sub0
            all_stored_preds[eid] = []

    t_loop_total = time.perf_counter() - t_loop0

    # Stage 8: Conflict Resolution
    t_sub0 = time.perf_counter()
    unique_preds = resolve_unique_owners(all_stored_preds)
    t_conflict = time.perf_counter() - t_sub0

    # Stage 9: Output Serialization
    t_sub0 = time.perf_counter()
    buf_matching = io.StringIO()
    buf_matching.write("source1_entity_id\tmatched_entity_ids\n")
    for eid in sample_1k.entity_id.astype(object):
        m_ids = unique_preds.get(eid, [])
        buf_matching.write(f"{eid}\t{','.join(m_ids)}\n")
    buf_matching.getvalue()

    buf_cands = io.StringIO()
    buf_cands.write("source1_entity_id\tcandidate_entity_ids\n")
    for eid, c_str in candidate_pairs_rows:
        buf_cands.write(f"{eid}\t{c_str}\n")
    buf_cands.getvalue()
    t_serialization = time.perf_counter() - t_sub0

    total_pipeline_time = t_loop_total + t_conflict + t_serialization
    total_pairs = sum(top_cand_counts)
    s1_rate = len(sample_1k) / total_pipeline_time
    pairs_rate = total_pairs / total_pipeline_time
    cur_ram, peak_ram = get_ram_mb()

    # Breakdown table
    stages = [
        ("1. Loading S2/S3", t_load, "one-time setup"),
        ("2. Inverted Index Construction", t_index, "one-time setup"),
        ("3. Global IDF Construction", t_idf, "one-time setup"),
        ("4. Candidate Retrieval (B5)", t_retrieval_total, "per-S1 query"),
        ("5. Formula-2 Candidate Ranking", t_ranking_total, "per-S1 ranking"),
        ("6. 47-Feature Extraction", t_feature_total, "per-pair feature extraction"),
        ("7. Model Scoring (predict_proba)", t_scoring_total, "per-batch inference"),
        ("8. Conflict Resolution", t_conflict, "post-processing"),
        ("9. Output Serialization (TSV)", t_serialization, "IO/formatting"),
    ]

    total_per_s1 = (
        t_retrieval_total + t_ranking_total + t_feature_total + t_scoring_total + t_conflict + t_serialization
    )
    profile_data = {
        "benchmark_sample_size": len(sample_1k),
        "total_test_population": n_test_total,
        "one_time_setup": {
            "load_seconds": round(t_load, 2),
            "index_build_seconds": round(t_index, 2),
            "idf_build_seconds": round(t_idf, 2),
            "setup_total_seconds": round(t_load + t_index + t_idf, 2),
            "setup_total_minutes": round((t_load + t_index + t_idf) / 60.0, 2),
        },
        "per_s1_runtime_breakdown": {
            "4_retrieval_seconds": round(t_retrieval_total, 3),
            "4_retrieval_pct": round(100.0 * t_retrieval_total / total_per_s1, 2),
            "5_ranking_seconds": round(t_ranking_total, 3),
            "5_ranking_pct": round(100.0 * t_ranking_total / total_per_s1, 2),
            "6_features_seconds": round(t_feature_total, 3),
            "6_features_pct": round(100.0 * t_feature_total / total_per_s1, 2),
            "7_scoring_seconds": round(t_scoring_total, 3),
            "7_scoring_pct": round(100.0 * t_scoring_total / total_per_s1, 2),
            "8_conflict_resolution_seconds": round(t_conflict, 4),
            "8_conflict_resolution_pct": round(100.0 * t_conflict / total_per_s1, 2),
            "9_serialization_seconds": round(t_serialization, 4),
            "9_serialization_pct": round(100.0 * t_serialization / total_per_s1, 2),
            "total_scoring_seconds_1k": round(total_per_s1, 2),
        },
        "primary_bottleneck": (
            "Formula-2 Candidate Ranking"
            if t_ranking_total >= max(t_retrieval_total, t_feature_total, t_scoring_total)
            else (
                "47-Feature Extraction"
                if t_feature_total >= max(t_retrieval_total, t_ranking_total, t_scoring_total)
                else "Candidate Retrieval"
            )
        ),
        "throughput": {
            "s1_per_second": round(s1_rate, 2),
            "pairs_per_second": round(pairs_rate, 1),
            "avg_candidates_per_s1": round(np.mean(raw_cand_counts), 2),
            "avg_top200_pairs_per_s1": round(np.mean(top_cand_counts), 2),
            "p95_raw_candidates_per_s1": round(float(np.percentile(raw_cand_counts, 95)), 1),
            "p95_top200_pairs_per_s1": round(float(np.percentile(top_cand_counts, 95)), 1),
        },
        "memory": {
            "working_set_mb": round(cur_ram, 1),
            "peak_mb": round(peak_ram, 1),
        },
        "extrapolation_full_test_1732544_s1": {
            "setup_hours": round((t_load + t_index + t_idf) / 3600.0, 3),
            "scoring_hours": round((n_test_total / s1_rate) / 3600.0, 2),
            "total_estimated_hours": round(((t_load + t_index + t_idf) + (n_test_total / s1_rate)) / 3600.0, 2),
        },
    }

    with open(PREP_DIR / "profile.json", "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    with open(PREP_DIR / "baseline_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    print("\n=== PROFILE BREAKDOWN (1,000 S1) ===")
    print(f"Total loop time: {total_pipeline_time:.2f}s | Throughput: {s1_rate:.2f} S1/s | {pairs_rate:.1f} pairs/s")
    print(
        f"Candidate Stats: Avg Raw = {np.mean(raw_cand_counts):.1f} | Avg Top200 = {np.mean(top_cand_counts):.1f} | P95 Raw = {np.percentile(raw_cand_counts, 95):.0f}"
    )
    print(f"Memory: WorkingSet = {cur_ram:.1f} MB | Peak = {peak_ram:.1f} MB")
    print("-" * 65)
    print(f"{'Stage':<35} | {'Time (s)':<10} | {'Share (%)':<10}")
    print("-" * 65)
    for name, t_val, desc in stages[3:]:
        share = 100.0 * t_val / total_per_s1
        print(f"{name:<35} | {t_val:10.3f} | {share:9.2f}%")
    print("-" * 65)
    print(f"\nPrimary Bottleneck: {profile_data['primary_bottleneck']}")
    print(
        f"Naive Extrapolated Full-Test Runtime: {profile_data['extrapolation_full_test_1732544_s1']['total_estimated_hours']} hours"
    )


if __name__ == "__main__":
    main()
