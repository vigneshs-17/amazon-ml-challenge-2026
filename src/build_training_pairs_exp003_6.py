"""EXP-003.6: Build Formula-2-Consistent Training Pairs (100k S1, 52 features).

Key design features:
- Samples the EXACT same 100,000 stratified training S1 entities as EXP-002 (Seed 2026).
- Uses verified B5 retrieval + deterministic Formula 2 candidate ranking at Cap-200.
- Positives = ground-truth links present in Cap-200.
- Diverse Hard Negatives (bounded up to 6 per S1):
  * Top 3 non-matches by Formula 2 score (reflects Formula 2 candidate distribution)
  * Top 2 non-matches by name similarity (targeting generic-name FP traps)
  * Top 1 non-match by address similarity
- Extracts 52 features using features_v3 (Slice V2: 42, V3a: +5, V3b: +5).
- Supports shard parallelization (--shard 0 --total-shards 2) for runtime safety (<60 min).
- Merges shards cleanly into train_X_v3.npy, train_y_v3.npy, and comparison JSON.
"""

import argparse
import ctypes
import heapq
import json
import os
import sys
import time
from ctypes import wintypes

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, "src")
from blocking import RareTokenIndex
from build_training_pairs_exp002 import load_stratified_train_s1
from features_v3 import ALL_FEATURE_NAMES_V3, extract_pair_features_v3, precompute_s1_v3
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)

EXP36_DIR = PROJECT_ROOT / "experiments" / "EXP-003.6"
EXP2_DIR = PROJECT_ROOT / "experiments" / "EXP-002"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
LOG_DIR = PROJECT_ROOT / "logs"

CAP = 200
SEED = 2026
N_TRAIN_S1 = 100_000

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


def load_s23():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    return pd.concat([s2, s3], ignore_index=True)


def run_shard(shard_id: int, total_shards: int):
    os.makedirs(EXP36_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = LOG_DIR / f"build_training_pairs_exp003_6_shard{shard_id}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log(
        f"=== EXP-003.6 Build Training Pairs (Shard {shard_id}/{total_shards}) Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ==="
    )
    t_start = time.time()

    log(f"Loading full stratified sample of {N_TRAIN_S1} training S1 entities (Seed: {SEED})...")
    s1_full = load_stratified_train_s1(N_TRAIN_S1)
    n_total = len(s1_full)
    log(f"Full stratified sample loaded: {n_total} entities.")

    # Slice shard
    shard_size = (n_total + total_shards - 1) // total_shards
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, n_total)
    s1_shard = s1_full.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    log(f"Shard {shard_id}: Processing entities {start_idx} to {end_idx} (count: {len(s1_shard)})")
    del s1_full

    log("Loading S2/S3 pools and building B5 indexes with deterministic tie-breaking...")
    t_s23_0 = time.time()
    s23 = load_s23()
    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )
    blocker = B5Blocker(s23)
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    del s23
    log(f"S2/S3 indexes built in {time.time()-t_s23_0:.1f}s.")

    log("Loading ground truth for shard entities...")
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    shard_s1_set = set(s1_shard.entity_id.astype(object))
    long = long[long.source1_entity_id.astype(object).isin(shard_s1_set)]
    truth = {}
    for sid, mid in zip(long.source1_entity_id.astype(object), long.matched_id.astype(object)):
        truth.setdefault(sid, set()).add(mid)
    del gt, long

    total_truth_links = sum(len(v) for v in truth.values())
    cur_ram, peak_ram = get_ram_mb()
    log(
        f"Shard ground truth: {len(truth)} entities have matches ({total_truth_links} true links). RAM: {cur_ram:.1f} MB (Peak: {peak_ram:.1f} MB)"
    )

    X_rows = []
    y_rows = []
    n_pos = 0
    n_neg = 0
    t_loop0 = time.time()
    log_interval = 2500

    log(f"\nBuilding Cap-{CAP} diverse hard negatives with Formula 2 candidate ranking...")

    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1_shard.entity_id.astype(object),
            s1_shard.name_norm.astype(object),
            s1_shard.address_norm.astype(object),
            s1_shard.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country)
        t = truth.get(eid, set())

        s1_rank_pre = precompute_s1_ranking(name, addr, name_idf, addr_idf)
        s1_feat_pre = precompute_s1_v3(name, addr, country, name_df, addr_df)

        cand_list = list(cand)

        # Formula 2 candidate ranking
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score_v2(s1_rank_pre, n2, a2)

        if len(cand_list) > CAP:
            ranked_top = heapq.nlargest(CAP, cand_list, key=scores.__getitem__)
        else:
            ranked_top = sorted(cand_list, key=lambda c: -scores[c])

        pos_cands = [c for c in ranked_top if c in t]
        neg_cands_pool = [c for c in ranked_top if c not in t]

        # Bounded hard negatives:
        # 1. Top 3 non-matches by Formula 2 score
        hard_negs = set(neg_cands_pool[:3])
        # 2. Top 2 by name similarity (generic-name FP traps)
        if len(neg_cands_pool) > 3:
            remaining = neg_cands_pool[3:]
            name_sims = sorted(remaining, key=lambda c: -fuzz.token_set_ratio(name, s23_lookup[c][0]))[:2]
            hard_negs.update(name_sims)
            # 3. Top 1 by address similarity
            remaining = [c for c in remaining if c not in hard_negs]
            if remaining:
                addr_sims = sorted(remaining, key=lambda c: -fuzz.token_set_ratio(addr, s23_lookup[c][1]))[:1]
                hard_negs.update(addr_sims)

        for cid in pos_cands:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_pair_features_v3(s1_feat_pre, n2, a2, c2, name_df, addr_df)
            X_rows.append(f)
            y_rows.append(1)
            n_pos += 1

        for cid in hard_negs:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_pair_features_v3(s1_feat_pre, n2, a2, c2, name_df, addr_df)
            X_rows.append(f)
            y_rows.append(0)
            n_neg += 1

        if (i + 1) % log_interval == 0 or (i + 1) == len(s1_shard):
            elapsed = time.time() - t_loop0
            rate = (i + 1) / elapsed
            eta_min = (len(s1_shard) - (i + 1)) / max(rate, 0.001) / 60.0
            cur_ram, peak_ram = get_ram_mb()
            log(
                f"[{i+1:>6}/{len(s1_shard)}] {100*(i+1)/len(s1_shard):5.1f}% | Elapsed: {elapsed/60:4.1f}m | Rate: {rate:4.1f} S1/s | Pairs: {n_pos+n_neg:7d} ({n_pos} pos, {n_neg} neg) | RAM: {cur_ram:.1f} MB | ETA: {eta_min:4.1f}m"
            )

    log(f"\nShard {shard_id} complete. Assembling numpy arrays...")
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)

    assert not np.isnan(X).any(), "NaN found in X!"
    assert not np.isinf(X).any(), "Inf found in X!"

    if total_shards == 1:
        np.save(EXP36_DIR / "train_X_v3.npy", X)
        np.save(EXP36_DIR / "train_y_v3.npy", y)
    else:
        np.save(EXP36_DIR / f"train_X_v3_shard{shard_id}.npy", X)
        np.save(EXP36_DIR / f"train_y_v3_shard{shard_id}.npy", y)

    shard_meta = {
        "shard_id": shard_id,
        "total_shards": total_shards,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "entity_count": len(s1_shard),
        "total_truth_links": total_truth_links,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "total_pairs": len(X),
        "runtime_seconds": round(time.time() - t_start, 1),
        "peak_ram_mb": round(get_ram_mb()[1], 1),
    }
    with open(EXP36_DIR / f"shard_{shard_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(shard_meta, f, indent=2)

    log(f"Shard {shard_id} saved successfully ({len(X)} pairs). Total time: {(time.time()-t_start)/60:.1f}m")


def merge_shards(total_shards: int):
    print(f"=== Merging {total_shards} Shards into Final EXP-003.6 Training Dataset ===", flush=True)
    t0 = time.time()
    X_parts = []
    y_parts = []
    total_pos = 0
    total_neg = 0
    total_truth = 0

    for s in range(total_shards):
        f_X = EXP36_DIR / f"train_X_v3_shard{s}.npy"
        f_y = EXP36_DIR / f"train_y_v3_shard{s}.npy"
        if total_shards == 1 and not f_X.exists():
            f_X = EXP36_DIR / "train_X_v3.npy"
            f_y = EXP36_DIR / "train_y_v3.npy"
        f_meta = EXP36_DIR / f"shard_{s}_meta.json"
        assert f_X.exists(), f"Missing {f_X}"
        assert f_y.exists(), f"Missing {f_y}"
        assert f_meta.exists(), f"Missing {f_meta}"

        with open(f_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        total_pos += meta["positive_count"]
        total_neg += meta["negative_count"]
        total_truth += meta["total_truth_links"]

        X_parts.append(np.load(f_X))
        y_parts.append(np.load(f_y))
        print(
            f"  Loaded Shard {s}: {len(y_parts[-1])} pairs ({meta['positive_count']} pos, {meta['negative_count']} neg)",
            flush=True,
        )

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    print(f"Combined shape: X={X_all.shape}, y={y_all.shape}", flush=True)
    np.save(EXP36_DIR / "train_X_v3.npy", X_all)
    np.save(EXP36_DIR / "train_y_v3.npy", y_all)

    # Clean up shard files
    for s in range(total_shards):
        (EXP36_DIR / f"train_X_v3_shard{s}.npy").unlink(missing_ok=True)
        (EXP36_DIR / f"train_y_v3_shard{s}.npy").unlink(missing_ok=True)

    # Compare with EXP-002 Formula 1 training distribution
    with open(EXP2_DIR / "training_report.json", "r", encoding="utf-8") as f:
        exp2_rep = json.load(f)

    cov_f2 = round(100.0 * total_pos / max(total_truth, 1), 3)
    cov_f1 = exp2_rep["candidate_retrieval_positive_coverage_pct"]

    comparison = {
        "dataset_name": "EXP-003.6 Formula-2-Consistent Training Pairs",
        "train_s1_count": N_TRAIN_S1,
        "total_truth_links_in_sample": total_truth,
        "formula_2_training": {
            "total_pairs": len(X_all),
            "positive_count": int(total_pos),
            "negative_count": int(total_neg),
            "negative_sampling_ratio": round(total_neg / max(total_pos, 1), 3),
            "candidate_retrieval_positive_coverage_pct": cov_f2,
            "feature_count": X_all.shape[1],
            "feature_names": ALL_FEATURE_NAMES_V3,
        },
        "formula_1_baseline_exp002": {
            "total_pairs": exp2_rep["total_pairs"],
            "positive_count": exp2_rep["positive_count"],
            "negative_count": exp2_rep["negative_count"],
            "negative_sampling_ratio": exp2_rep["negative_sampling_ratio"],
            "candidate_retrieval_positive_coverage_pct": cov_f1,
            "feature_count": exp2_rep["feature_count"],
        },
        "major_distribution_differences": {
            "candidate_ranking_formula": "Formula 2 (logarithmic token-IDF + postal code agreement + corroboration) replaces Formula 1",
            "candidate_positive_coverage_gain_pct": round(cov_f2 - cov_f1, 3),
            "additional_training_positives_captured": int(total_pos - exp2_rep["positive_count"]),
            "hard_negative_distribution": "Hard negatives are selected from the true Formula 2 Cap-200 candidate pool, eliminating distribution mismatch between training hard negatives and inference candidates",
            "feature_dimension": f"{X_all.shape[1]} features (Slice V2: 42, V3a: +5 tolerant numeric, V3b: +5 cross-field asymmetry) vs 42 baseline features",
        },
        "merge_time_seconds": round(time.time() - t0, 1),
    }

    with open(EXP36_DIR / "training_distribution_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(EXP36_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_features": len(ALL_FEATURE_NAMES_V3),
                "v2_baseline_features_count": 42,
                "v3a_tolerant_numeric_count": 5,
                "v3b_asymmetry_count": 5,
                "feature_names_v2": ALL_FEATURE_NAMES_V3[:42],
                "feature_names_v3a_numeric": ALL_FEATURE_NAMES_V3[42:47],
                "feature_names_v3b_asymmetry": ALL_FEATURE_NAMES_V3[47:52],
                "all_features": ALL_FEATURE_NAMES_V3,
            },
            f,
            indent=2,
        )

    print("\nMerge complete. Final dataset saved:")
    print(f"  train_X_v3.npy: shape {X_all.shape} ({X_all.nbytes / (1024*1024):.1f} MB)")
    print(f"  train_y_v3.npy: shape {y_all.shape} ({y_all.nbytes / (1024*1024):.1f} MB)")
    print(f"  Formula 2 positive coverage: {cov_f2}% (vs {cov_f1}% in Formula 1, delta: +{cov_f2 - cov_f1:.3f}%)")
    print(f"  Positives: {total_pos} | Negatives: {total_neg}")
    print(f"  Saved metadata to {EXP36_DIR / 'training_distribution_comparison.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--merge", action="store_true", help="Merge all shard outputs")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.total_shards)
    else:
        run_shard(args.shard, args.total_shards)
