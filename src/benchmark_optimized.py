"""Benchmark Optimized Inference Pipeline on Test Data (Step 6).

Measures granular execution profile and throughput on representative test subsets:
- 1,000 stratified test entities (test_benchmark_ids_1k.csv)
- 5,000 stratified test entities (test_benchmark_ids_5k.csv)

Saves:
- experiments/FINAL-INFERENCE-PREP/optimized_benchmark.json
- experiments/FINAL-INFERENCE-PREP/runtime_projection.json
"""

import ctypes
import heapq
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
from ranking_v2 import precompute_global_idfs
from test_exact_equivalence import compute_rank_score_unpacked, resolve_unique_owners

CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
EXP4_DIR = PROJECT_ROOT / "experiments" / "EXP-004"
PREP_DIR = PROJECT_ROOT / "experiments" / "FINAL-INFERENCE-PREP"

CAP = 200
THRESHOLD = 0.82
TOTAL_TEST_POPULATION = 1_732_544

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


def run_benchmark_on_sample(sample_df, blocker, s23_lookup, name_df, addr_df, name_idf, addr_idf, clf, label="1k"):
    print(f"\n--- Running Optimized Benchmark on {len(sample_df)} Test S1 ({label}) ---", flush=True)
    t_start = time.perf_counter()

    t_retrieval = 0.0
    t_ranking = 0.0
    t_features = 0.0
    t_scoring = 0.0
    t_conflict = 0.0

    total_candidate_pairs = 0
    total_raw_cands = 0
    preds_prob = {}

    for i, (eid, name, addr, country) in enumerate(
        zip(
            sample_df.entity_id.astype(object),
            sample_df.name_norm.astype(object),
            sample_df.address_norm.astype(object),
            sample_df.country.astype(object),
        )
    ):
        # 1. Candidate Retrieval
        t0 = time.perf_counter()
        cand = blocker.candidates(name, addr, country)
        t_retrieval += time.perf_counter() - t0
        total_raw_cands += len(cand)

        # 2. Formula 2 Candidate Ranking
        t0 = time.perf_counter()
        t1 = set(name.split()) if name else set()
        a1 = set(addr.split()) if addr else set()
        num_tokens = {t for t in a1 if any(c.isdigit() for c in t)}
        s1_postals = {t for t in num_tokens if len(t) in (5, 6) and t.isdigit()}
        name_idfs = {t: name_idf.get(t, 0.0) for t in t1}
        addr_idfs = {t: addr_idf.get(t, 0.0) for t in a1}
        len_t1 = len(t1)
        len_a1 = len(a1)

        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score_unpacked(
                name, addr, t1, a1, name_idfs, addr_idfs, s1_postals, len_t1, len_a1, n2, a2
            )

        if len(cand_list) > CAP:
            ranked_top = heapq.nlargest(CAP, cand_list, key=lambda c: (scores[c], c))
        else:
            ranked_top = sorted(cand_list, key=lambda c: (-scores[c], c))
        t_ranking += time.perf_counter() - t0

        total_candidate_pairs += len(ranked_top)

        # 3. 47-Feature Extraction
        t0 = time.perf_counter()
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
            t_features += time.perf_counter() - t0

            # 4. Model Scoring
            t0 = time.perf_counter()
            proba = clf.predict_proba(feats)[:, 1]
            t_scoring += time.perf_counter() - t0

            preds_prob[eid] = [(cid, float(p)) for cid, p in zip(ranked_top, proba) if p >= THRESHOLD]
        else:
            t_features += time.perf_counter() - t0
            preds_prob[eid] = []

    # 5. Conflict Resolution
    t0 = time.perf_counter()
    resolve_unique_owners(preds_prob)
    t_conflict += time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    ram_mb, peak_ram_mb = get_ram_mb()

    result = {
        "sample_size": len(sample_df),
        "total_seconds": round(t_total, 3),
        "s1_per_second": round(len(sample_df) / t_total, 2),
        "pairs_per_second": round(total_candidate_pairs / t_total, 1),
        "avg_raw_candidates_per_s1": round(total_raw_cands / len(sample_df), 2),
        "avg_top200_pairs_per_s1": round(total_candidate_pairs / len(sample_df), 2),
        "breakdown_seconds": {
            "retrieval_seconds": round(t_retrieval, 3),
            "retrieval_pct": round(100.0 * t_retrieval / t_total, 2),
            "ranking_seconds": round(t_ranking, 3),
            "ranking_pct": round(100.0 * t_ranking / t_total, 2),
            "feature_extraction_seconds": round(t_features, 3),
            "feature_extraction_pct": round(100.0 * t_features / t_total, 2),
            "scoring_seconds": round(t_scoring, 3),
            "scoring_pct": round(100.0 * t_scoring / t_total, 2),
            "conflict_resolution_seconds": round(t_conflict, 4),
            "conflict_resolution_pct": round(100.0 * t_conflict / t_total, 2),
        },
        "memory_mb": {
            "working_set_mb": round(ram_mb, 1),
            "peak_working_set_mb": round(peak_ram_mb, 1),
        },
    }
    print(
        f"  Total time ({label}): {t_total:.2f}s | Throughput: {result['s1_per_second']} S1/s ({result['pairs_per_second']} pairs/s)"
    )
    print(
        f"  Ranking: {t_ranking:.2f}s ({result['breakdown_seconds']['ranking_pct']}%) | Features: {t_features:.2f}s ({result['breakdown_seconds']['feature_extraction_pct']}%) | Scoring: {t_scoring:.2f}s"
    )
    return result


def main():
    print("=== STEP 6: BENCHMARK OPTIMIZED PIPELINE ON TEST DATA ===", flush=True)

    # 1. Model Loading
    print("Loading frozen EXP-004 model...", flush=True)
    clf = joblib.load(EXP4_DIR / "model_exp004.joblib")
    assert clf.n_features_in_ == 47

    # 2. Loading Test S2/S3 Parquet Tables
    print("Loading test S2/S3 candidate corpus...", flush=True)
    t_setup_0 = time.perf_counter()
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "test_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "test_s3.parquet", columns=cols)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    t_load = time.perf_counter() - t_setup_0

    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )

    # 3. Inverted Index Construction
    print("Building B5 inverted index...", flush=True)
    t_idx_0 = time.perf_counter()
    blocker = B5Blocker(s23)
    t_idx = time.perf_counter() - t_idx_0
    del s23

    # 4. Global IDF Construction
    print("Building global IDFs...", flush=True)
    t_idf_0 = time.perf_counter()
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    t_idf = time.perf_counter() - t_idf_0

    setup_total = time.perf_counter() - t_setup_0
    print(f"Setup completed in {setup_total:.2f}s (Load: {t_load:.2f}s, Index: {t_idx:.2f}s, IDF: {t_idf:.2f}s)")

    # 5. Load Stratified Test Subsets
    test_s1_full = pd.read_parquet(CACHE / "test_s1.parquet")
    sample_1k_ids = pd.read_csv(PREP_DIR / "test_benchmark_ids_1k.csv")
    sample_5k_ids = pd.read_csv(PREP_DIR / "test_benchmark_ids_5k.csv")

    s1_1k = test_s1_full[test_s1_full.entity_id.isin(set(sample_1k_ids.entity_id))].reset_index(drop=True)
    s1_5k = test_s1_full[test_s1_full.entity_id.isin(set(sample_5k_ids.entity_id))].reset_index(drop=True)
    del test_s1_full

    # 6. Run Benchmarks
    res_1k = run_benchmark_on_sample(s1_1k, blocker, s23_lookup, name_df, addr_df, name_idf, addr_idf, clf, label="1k")
    res_5k = run_benchmark_on_sample(s1_5k, blocker, s23_lookup, name_df, addr_df, name_idf, addr_idf, clf, label="5k")

    # Scaling Linearity Analysis
    sec_per_s1_1k = res_1k["total_seconds"] / 1000.0
    sec_per_s1_5k = res_5k["total_seconds"] / 5000.0
    scaling_ratio = sec_per_s1_5k / sec_per_s1_1k

    benchmark_summary = {
        "one_time_setup": {
            "load_seconds": round(t_load, 2),
            "index_build_seconds": round(t_idx, 2),
            "idf_build_seconds": round(t_idf, 2),
            "setup_total_seconds": round(setup_total, 2),
            "setup_total_minutes": round(setup_total / 60.0, 2),
        },
        "benchmark_1k": res_1k,
        "benchmark_5k": res_5k,
        "scaling_linearity": {
            "sec_per_s1_1k": round(sec_per_s1_1k, 5),
            "sec_per_s1_5k": round(sec_per_s1_5k, 5),
            "ratio_5k_to_1k": round(scaling_ratio, 4),
            "linearity_verified": bool(0.90 <= scaling_ratio <= 1.15),
        },
    }

    with open(PREP_DIR / "optimized_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    # 7. Compute Projected Full Test Runtime (1,732,544 S1)
    baseline_benchmark = json.load(open(PREP_DIR / "baseline_benchmark.json"))
    base_hours = baseline_benchmark["extrapolation_full_test_1732544_s1"]["total_estimated_hours"]

    # Use 5k rate for most reliable full-scale projection
    opt_rate = res_5k["s1_per_second"]
    opt_scoring_seconds = TOTAL_TEST_POPULATION / opt_rate
    opt_scoring_hours = opt_scoring_seconds / 3600.0
    opt_total_hours = (setup_total + opt_scoring_seconds) / 3600.0

    runtime_projection = {
        "total_test_s1_entities": TOTAL_TEST_POPULATION,
        "one_time_setup_hours": round(setup_total / 3600.0, 2),
        "baseline_scoring_hours": round(
            baseline_benchmark["per_s1_runtime_breakdown"]["total_scoring_seconds_1k"]
            * (TOTAL_TEST_POPULATION / 1000.0)
            / 3600.0,
            2,
        ),
        "baseline_total_hours": round(base_hours, 2),
        "optimized_scoring_rate_s1_per_sec": round(opt_rate, 2),
        "optimized_scoring_hours": round(opt_scoring_hours, 2),
        "optimized_total_hours": round(opt_total_hours, 2),
        "speedup_factor": round(base_hours / opt_total_hours, 2),
        "time_saved_hours": round(base_hours - opt_total_hours, 2),
        "projected_duration_formatted": f"{int(opt_total_hours)} hours {int((opt_total_hours % 1) * 60)} minutes",
    }

    with open(PREP_DIR / "runtime_projection.json", "w", encoding="utf-8") as f:
        json.dump(runtime_projection, f, indent=2)

    print("\n=== BENCHMARK & RUNTIME PROJECTION SUMMARY ===")
    print(f"Setup Time                 : {setup_total/60:.2f} minutes")
    print(f"Baseline Full Test Time    : {base_hours:.2f} hours")
    print(f"Optimized Rate (5k bench)  : {opt_rate:.2f} S1/sec")
    print(
        f"Optimized Full Test Time   : {opt_total_hours:.2f} hours ({runtime_projection['projected_duration_formatted']})"
    )
    print(f"Speedup Factor             : {runtime_projection['speedup_factor']}x")
    print(f"Hours Saved                : {runtime_projection['time_saved_hours']:.2f} hours")
    print(
        f"Scaling Linearity Ratio    : {scaling_ratio:.4f} (Linear: {benchmark_summary['scaling_linearity']['linearity_verified']})"
    )


if __name__ == "__main__":
    main()
