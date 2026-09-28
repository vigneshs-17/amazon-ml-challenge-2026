"""Emergency 10k Speed Test (Step 6).

Runs 10,000 representative test entities partitioned by country:
- 4,680 India
- 3,820 US
- 1,500 France
Total = 10,000 S1 entities.

Measures:
- Total runtime
- S1/sec
- Candidate pairs/sec
- Peak RAM
- Confirms deadline feasibility.
"""

import ctypes
import heapq
import sys
import time
from ctypes import wintypes

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features_v3 import extract_pair_features_v3a, precompute_s1_v3
from paths import PROJECT_ROOT
from ranking_v2 import precompute_global_idfs
from run_exp004 import B5Blocker
from test_exact_equivalence import compute_rank_score_unpacked

CAP = 100
THRESHOLD = 0.82
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
EXP4_DIR = PROJECT_ROOT / "experiments" / "EXP-004"

# Cross-platform memory tracking
HAS_WINDLL = hasattr(ctypes, "windll")
if HAS_WINDLL:
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
else:
    psapi = None
    kernel32 = None


def get_ram_mb():
    if HAS_WINDLL and psapi and kernel32:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / (1024 * 1024), pmc.PeakWorkingSetSize / (1024 * 1024)
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        return rss, rss
    except Exception:
        return 0.0, 0.0


def process_country_partition(country_name, s1_country_df, clf):
    t_start = time.perf_counter()
    cols = ["entity_id", "name_norm", "address_norm", "country"]

    print(f"[{country_name}] Loading candidate corpus...", flush=True)
    s2 = pd.read_parquet(CACHE / "test_s2.parquet", columns=cols)
    s2 = s2[s2.country == country_name].copy()
    s3 = pd.read_parquet(CACHE / "test_s3.parquet", columns=cols)
    s3 = s3[s3.country == country_name].copy()
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )

    print(f"[{country_name}] Building B5 index ({len(s23):,} records)...", flush=True)
    blocker = B5Blocker(s23)
    del s23

    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    t_setup = time.perf_counter() - t_start

    n_entities = len(s1_country_df)
    print(f"[{country_name}] Setup done in {t_setup:.1f}s. Scoring {n_entities:,} entities...", flush=True)

    t_score_0 = time.perf_counter()
    total_pairs = 0
    total_preds = 0

    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1_country_df.entity_id.astype(object),
            s1_country_df.name_norm.astype(object),
            s1_country_df.address_norm.astype(object),
            s1_country_df.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country)
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

        total_pairs += len(ranked_top)

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
            proba = clf.predict_proba(feats)[:, 1]
            total_preds += int((proba >= THRESHOLD).sum())

    t_score = time.perf_counter() - t_score_0
    rate = n_entities / t_score
    print(
        f"[{country_name}] COMPLETE: {n_entities} entities in {t_score:.1f}s ({rate:.1f} S1/s, {total_pairs/t_score:.0f} pairs/s)",
        flush=True,
    )
    return {
        "country": country_name,
        "count": n_entities,
        "setup_sec": t_setup,
        "scoring_sec": t_score,
        "pairs": total_pairs,
        "s1_per_sec": rate,
    }


def main():
    print("=== EMERGENCY 10K SPEED TEST (Country Partitioning + Cap 100) ===", flush=True)
    clf = joblib.load(EXP4_DIR / "model_exp004.joblib")

    # Sample 10k entities (4680 India, 3820 US, 1500 France)
    test_s1 = pd.read_parquet(CACHE / "test_s1.parquet")
    s1_ind = test_s1[test_s1.country == "India"].iloc[:4680].copy().reset_index(drop=True)
    s1_us = test_s1[test_s1.country == "US"].iloc[:3820].copy().reset_index(drop=True)
    s1_fr = test_s1[test_s1.country == "France"].iloc[:1500].copy().reset_index(drop=True)
    del test_s1

    t0_global = time.perf_counter()

    # Process partitions sequentially in benchmark (or parallel)
    res_fr = process_country_partition("France", s1_fr, clf)
    res_us = process_country_partition("US", s1_us, clf)
    res_in = process_country_partition("India", s1_ind, clf)

    time.perf_counter() - t0_global
    total_s1 = res_fr["count"] + res_us["count"] + res_in["count"]
    total_pairs = res_fr["pairs"] + res_us["pairs"] + res_in["pairs"]
    total_scoring_sec = res_fr["scoring_sec"] + res_us["scoring_sec"] + res_in["scoring_sec"]
    effective_rate = total_s1 / total_scoring_sec

    _ram_w, ram_p = get_ram_mb()

    print("\n=== 10K SPEED TEST SUMMARY ===")
    print(f"Total S1 Scored: {total_s1}")
    print(f"Total Candidate Pairs: {total_pairs}")
    print(f"Scoring Time: {total_scoring_sec:.1f}s")
    print(f"Single-Process Throughput: {effective_rate:.1f} S1/sec")
    print(f"Candidate Pairs / sec: {total_pairs/total_scoring_sec:.1f}")
    print(f"Peak RAM: {ram_p:.1f} MB")


if __name__ == "__main__":
    main()
