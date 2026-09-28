"""EXP-002.5 Stepping-Stone Experiment.

Evaluates on the reproducible 10,000 S1 validation sample (34,650 true links):
1. Candidate Retrieval & Ranking Benchmark:
   - Formula 1 (Baseline Heuristic) vs Formula 2 (Enhanced Token-IDF + Postal + Overlap)
   - Evaluated across caps [50, 100, 200, 250, 300, 500]
2. Matcher Comparison on 10k Validation Sample:
   - Config A (EXP-002 Reference): Formula 1, Cap-200, M1_Baseline (max_depth=6, iter=100)
   - Config B (Ranking Impact)  : Formula 2, Cap-200, M1_Baseline
   - Config C (Model Impact)    : Formula 2, Cap-200, M4_Leafwise_127 (leaves=127, iter=150, l2=2.0)
   - Config D (Cap Expansion)   : Formula 2, Cap-250, M4_Leafwise_127
3. Official Macro F0.5 Threshold Sweep & Ownership Conflict Resolution.
4. Saves all benchmark tables and evaluation artifacts in experiments/EXP-002.5/.
"""

import ctypes
import json
import os
import time
from ctypes import wintypes

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from blocking import RareTokenIndex
from features_v2 import extract_pair_features_v2, precompute_s1
from io_utils import explode_ground_truth, load_ground_truth
from metrics import macro_fbeta
from paths import PROJECT_ROOT
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)

EXP1_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
EXP2_DIR = PROJECT_ROOT / "experiments" / "EXP-002"
EXP25_DIR = PROJECT_ROOT / "experiments" / "EXP-002.5"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
LOG_FILE = PROJECT_ROOT / "logs" / "run_exp002_5.log"

SAMPLE_SIZE = 10000
CAPS = [50, 100, 200, 250, 300, 500]
THRESHOLDS = np.round(np.arange(0.70, 0.95, 0.02), 2)
SEED = 2026

# Memory tracking
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


def log(msg, to_file=True):
    print(msg, flush=True)
    if to_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def compute_rank_f1(s1_name, s1_addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df):
    """Formula 1 (Baseline heuristic)."""
    t2 = set(n2.split()) if n2 else set()
    a2s = set(a2.split()) if a2 else set()
    shared_n = s1_t1 & t2
    shared_a = s1_a1s & a2s
    score = 0.0
    if s1_addr and s1_addr == a2:
        score += 5.0
    if s1_name and s1_name == n2:
        score += 1.0
    if shared_n:
        min_df = min(name_df.get(t, 1) for t in shared_n)
        score += 2.0 / (1.0 + min_df) + 0.1 * len(shared_n)
    if shared_a:
        min_df = min(addr_df.get(t, 1) for t in shared_a)
        score += 2.0 / (1.0 + min_df) + 0.1 * len(shared_a)
    if s1_n1s and (s1_n1s & shared_a):
        score += 1.5
    return score


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


def load_val_sample():
    val_ids = pd.read_csv(EXP1_DIR / "validation_s1_ids.csv")
    s1 = pd.read_parquet(CACHE / "train_s1.parquet")
    s1_val = s1[s1.entity_id.isin(set(val_ids.entity_id))].reset_index(drop=True)
    return s1_val.iloc[:SAMPLE_SIZE].copy()


def load_s23():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    return pd.concat([s2, s3], ignore_index=True)


def truth_map(s1):
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    ids = set(s1.entity_id.astype(object))
    long = long[long.source1_entity_id.astype(object).isin(ids)]
    truth = {sid: set() for sid in ids}
    for sid, mid in zip(long.source1_entity_id.astype(object), long.matched_id.astype(object)):
        truth[sid].add(mid)
    return truth


def resolve_unique_owners(preds_prob: dict) -> dict:
    cand_to_best_s1 = {}
    for s1, items in preds_prob.items():
        for cid, p in items:
            if cid not in cand_to_best_s1 or p > cand_to_best_s1[cid][1]:
                cand_to_best_s1[cid] = (s1, p)
    unique_preds = {s1: set() for s1 in preds_prob}
    for cid, (s1, p) in cand_to_best_s1.items():
        unique_preds[s1].add(cid)
    return unique_preds


def evaluate_thresholds(cand_probs: dict, truth: dict, s1_keys: list):
    """Sweep thresholds with unique-owner conflict resolution and compute macro F0.5."""
    best_f05 = -1.0
    best_metrics = None
    records = []

    for th in THRESHOLDS:
        raw_preds = {sid: [(cid, p) for cid, p in cand_probs.get(sid, []) if p >= th] for sid in s1_keys}
        unique_preds = resolve_unique_owners(raw_preds)
        m = macro_fbeta(truth, unique_preds, beta=0.5)

        tp, fp, fn = 0, 0, 0
        for sid in s1_keys:
            t = truth[sid]
            p = unique_preds[sid]
            tp += len(t & p)
            fp += len(p - t)
            fn += len(t - p)

        records.append(
            {
                "threshold": float(th),
                "macro_f0_5": float(m["f0_5"]),
                "macro_precision": float(m["precision"]),
                "macro_recall": float(m["recall"]),
                "singleton_f0_5": float(m["f0_5_singletons"]),
                "non_singleton_f0_5": float(m["f0_5_non_singletons"]),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

        if m["f0_5"] > best_f05:
            best_f05 = m["f0_5"]
            best_metrics = {
                "threshold": float(th),
                "macro_f0_5": float(m["f0_5"]),
                "macro_precision": float(m["precision"]),
                "macro_recall": float(m["recall"]),
                "singleton_f0_5": float(m["f0_5_singletons"]),
                "non_singleton_f0_5": float(m["f0_5_non_singletons"]),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

    return best_metrics, records


def main():
    os.makedirs(EXP25_DIR, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== EXP-002.5 Pipeline Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    t_start = time.time()
    log("--- Step 1: Loading 10,000 S1 Validation Sample & S2/S3 Inverted Indexes ---")
    s1_val = load_val_sample()
    s23 = load_s23()
    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )

    log(f"Building B5 blocker and global IDF tables for {len(s23)} pool records...")
    t_idx0 = time.time()
    blocker = B5Blocker(s23)
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    del s23

    truth = truth_map(s1_val)
    total_truth_links = sum(len(v) for v in truth.values())
    s1_keys = list(s1_val.entity_id.astype(object))
    cur_ram, peak_ram = get_ram_mb()
    log(f"Setup complete in {time.time()-t_idx0:.1f}s. Memory: {cur_ram:.1f} MB (Peak: {peak_ram:.1f} MB)")
    log(
        f"Validation sample: {len(s1_val)} S1 | {total_truth_links} true links ({total_truth_links/len(s1_val):.3f} links/S1)"
    )

    # ---------------------------------------------------------
    # PART 1: Candidate Ranking Benchmark (Formula 1 vs Formula 2)
    # ---------------------------------------------------------
    log("\n--- Part 1: Candidate Retrieval & Ranking Benchmark (Formula 1 vs Formula 2) ---")
    raw_hits = 0
    raw_cand_counts = []
    hits_f1 = {cap: 0 for cap in CAPS}
    hits_f2 = {cap: 0 for cap in CAPS}
    cands_f1_cap200 = {}
    cands_f2_cap250 = {}

    cands_cache_path = EXP25_DIR / "interim_10k_candidates.joblib"
    if cands_cache_path.exists():
        log(f"Loading cached candidate ranking results from {cands_cache_path}...")
        c_cache = joblib.load(cands_cache_path)
        raw_hits = c_cache["raw_hits"]
        raw_cand_counts = c_cache["raw_cand_counts"]
        hits_f1 = c_cache["hits_f1"]
        hits_f2 = c_cache["hits_f2"]
        cands_f1_cap200 = c_cache["cands_f1_cap200"]
        cands_f2_cap250 = c_cache["cands_f2_cap250"]
        t_rank_total = c_cache["t_rank_total"]
    else:
        t_rank0 = time.time()
        for i, (eid, name, addr, country) in enumerate(
            zip(
                s1_val.entity_id.astype(object),
                s1_val.name_norm.astype(object),
                s1_val.address_norm.astype(object),
                s1_val.country.astype(object),
            )
        ):
            cand = blocker.candidates(name, addr, country)
            t = truth[eid]
            raw_hits += len(t & cand)
            raw_cand_counts.append(len(cand))

            # Precompute S1 data for ranking
            s1_t1 = set(name.split()) if name else set()
            s1_a1s = set(addr.split()) if addr else set()
            s1_n1s = {tok for tok in s1_a1s if any(c.isdigit() for c in tok)}
            s1_rank_pre = precompute_s1_ranking(name, addr, name_idf, addr_idf)

            cand_list = list(cand)

            # Formula 1 scores
            scores_f1 = {}
            for cid in cand_list:
                n2, a2, _ = s23_lookup[cid]
                scores_f1[cid] = compute_rank_f1(name, addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df)
            ranked_f1 = sorted(cand_list, key=lambda c: (-scores_f1[c], c))

            # Formula 2 scores
            scores_f2 = {}
            for cid in cand_list:
                n2, a2, _ = s23_lookup[cid]
                scores_f2[cid] = compute_rank_score_v2(s1_rank_pre, n2, a2)
            ranked_f2 = sorted(cand_list, key=lambda c: (-scores_f2[c], c))

            # Record candidate sets needed for Part 2
            cands_f1_cap200[eid] = ranked_f1[:200]
            cands_f2_cap250[eid] = ranked_f2[:250]

            # Evaluate recall across all caps
            for cap in CAPS:
                hits_f1[cap] += len(t & set(ranked_f1[:cap]))
                hits_f2[cap] += len(t & set(ranked_f2[:cap]))

            if (i + 1) % 2500 == 0 or (i + 1) == SAMPLE_SIZE:
                elapsed = time.time() - t_rank0
                rate = (i + 1) / elapsed
                log(
                    f"  [{i+1:>5}/{SAMPLE_SIZE}] ({100.0*(i+1)/SAMPLE_SIZE:5.1f}%) | Rate: {rate:5.1f} S1/s | Elapsed: {elapsed/60:.1f}m"
                )

        t_rank_total = time.time() - t_rank0
        joblib.dump(
            {
                "raw_hits": raw_hits,
                "raw_cand_counts": raw_cand_counts,
                "hits_f1": hits_f1,
                "hits_f2": hits_f2,
                "cands_f1_cap200": cands_f1_cap200,
                "cands_f2_cap250": cands_f2_cap250,
                "t_rank_total": t_rank_total,
            },
            cands_cache_path,
        )
        log(f"Saved candidate ranking cache to {cands_cache_path}")

    raw_b5_recall = 100.0 * raw_hits / total_truth_links
    log(
        f"\nCandidate Ranking Benchmark Complete in {t_rank_total/60:.1f} minutes ({SAMPLE_SIZE/t_rank_total:.1f} S1/s)"
    )
    log(
        f"Raw B5 Retrieval Recall: {raw_hits}/{total_truth_links} ({raw_b5_recall:.3f}%) | Avg Raw Candidates/S1: {np.mean(raw_cand_counts):.2f}"
    )

    ranking_comparison_rows = []
    log("\n" + "=" * 80)
    log(
        f"{'Cap':>6} | {'Formula 1 Recall':>20} | {'Formula 2 Recall':>20} | {'Delta Links':>12} | {'Pruning Loss Red.':>18}"
    )
    log("-" * 80)
    for cap in CAPS:
        r1 = 100.0 * hits_f1[cap] / total_truth_links
        r2 = 100.0 * hits_f2[cap] / total_truth_links
        delta_links = hits_f2[cap] - hits_f1[cap]
        f1_pruning_miss = raw_hits - hits_f1[cap]
        f2_pruning_miss = raw_hits - hits_f2[cap]
        loss_red = (f1_pruning_miss - f2_pruning_miss) / f1_pruning_miss * 100.0 if f1_pruning_miss > 0 else 0.0

        log(
            f"{cap:>6} | {hits_f1[cap]:>6} ({r1:6.3f}%) | {hits_f2[cap]:>6} ({r2:6.3f}%) | {delta_links:>+11d} | {loss_red:>16.1f}%"
        )
        ranking_comparison_rows.append(
            {
                "cap": cap,
                "raw_b5_hits": raw_hits,
                "formula_1_hits": hits_f1[cap],
                "formula_1_recall_pct": round(r1, 3),
                "formula_2_hits": hits_f2[cap],
                "formula_2_recall_pct": round(r2, 3),
                "delta_links_recovered": delta_links,
                "pruning_loss_reduction_pct": round(loss_red, 2),
            }
        )
    log("=" * 80)

    pd.DataFrame(ranking_comparison_rows).to_csv(EXP25_DIR / "candidate_ranking_comparison.csv", index=False)
    log(f"Saved candidate ranking comparison to {EXP25_DIR / 'candidate_ranking_comparison.csv'}")

    # ---------------------------------------------------------
    # PART 2: Matcher Training & Pipeline Comparison
    # ---------------------------------------------------------
    log("\n--- Part 2: Model Training on Full 100k S1 Dataset (913,717 pairs, 42 features) ---")
    X_train = np.load(EXP2_DIR / "train_X.npy")
    y_train = np.load(EXP2_DIR / "train_y.npy")
    log(f"Loaded training data: shape={X_train.shape}, pos={(y_train==1).sum()}, neg={(y_train==0).sum()}")

    models = {
        "M1_Baseline": HistGradientBoostingClassifier(random_state=SEED, max_depth=6, max_iter=100),
        "M4_Leafwise_127": HistGradientBoostingClassifier(
            random_state=SEED,
            max_depth=None,
            max_leaf_nodes=127,
            min_samples_leaf=20,
            max_iter=150,
            l2_regularization=2.0,
        ),
    }

    fitted_models = {}
    for mname, m in models.items():
        t0 = time.time()
        log(f"Fitting {mname}...")
        m.fit(X_train, y_train)
        fitted_models[mname] = m
        log(f"  {mname} fitted in {time.time()-t0:.2f}s")
    del X_train, y_train

    # Pre-extract features for candidates
    log("\n--- Part 3: Feature Extraction on Validation Candidates ---")
    feats_cache_path = EXP25_DIR / "interim_10k_features.joblib"
    if feats_cache_path.exists():
        log(f"Loading cached validation features from {feats_cache_path}...")
        f_cache = joblib.load(feats_cache_path)
        feats_f1_cap200 = f_cache["feats_f1"]
        feats_f2_cap250 = f_cache["feats_f2"]
    else:
        t_feat0 = time.time()
        feats_f1_cap200 = {}
        feats_f2_cap250 = {}

        for i, (eid, name, addr, country) in enumerate(
            zip(
                s1_val.entity_id.astype(object),
                s1_val.name_norm.astype(object),
                s1_val.address_norm.astype(object),
                s1_val.country.astype(object),
            )
        ):
            s1_pre = precompute_s1(name, addr, country, name_df, addr_df)

            # Config A/B: F1 Cap-200 candidates
            cands_f1 = cands_f1_cap200[eid]
            if cands_f1:
                feats_f1_cap200[eid] = np.array(
                    [
                        extract_pair_features_v2(
                            s1_pre, s23_lookup[c][0], s23_lookup[c][1], s23_lookup[c][2], name_df, addr_df
                        )
                        for c in cands_f1
                    ],
                    dtype=np.float32,
                )
            else:
                feats_f1_cap200[eid] = np.zeros((0, 42), dtype=np.float32)

            # Config C/D: F2 Cap-250 candidates
            cands_f2 = cands_f2_cap250[eid]
            if cands_f2:
                feats_f2_cap250[eid] = np.array(
                    [
                        extract_pair_features_v2(
                            s1_pre, s23_lookup[c][0], s23_lookup[c][1], s23_lookup[c][2], name_df, addr_df
                        )
                        for c in cands_f2
                    ],
                    dtype=np.float32,
                )
            else:
                feats_f2_cap250[eid] = np.zeros((0, 42), dtype=np.float32)

            if (i + 1) % 2500 == 0:
                log(f"  Extracted features for [{i+1:>5}/{SAMPLE_SIZE}] entities ({time.time()-t_feat0:.1f}s)")

        log(f"Feature extraction complete in {time.time()-t_feat0:.1f}s.")
        joblib.dump({"feats_f1": feats_f1_cap200, "feats_f2": feats_f2_cap250}, feats_cache_path)
        log(f"Saved candidate features cache to {feats_cache_path}")

    # ---------------------------------------------------------
    # PART 4: Pipeline Configurations Scoring & Threshold Search
    # ---------------------------------------------------------
    log("\n--- Part 4: Pipeline Configurations Scoring & Official Evaluation ---")
    pipeline_configs = [
        ("Config_A_EXP002_Baseline", "Formula 1", 200, "M1_Baseline", cands_f1_cap200, feats_f1_cap200),
        (
            "Config_B_Formula2_Ranking",
            "Formula 2",
            200,
            "M1_Baseline",
            {eid: cands_f2_cap250[eid][:200] for eid in s1_keys},
            {eid: feats_f2_cap250[eid][:200] for eid in s1_keys},
        ),
        (
            "Config_C_F2_Leafwise127",
            "Formula 2",
            200,
            "M4_Leafwise_127",
            {eid: cands_f2_cap250[eid][:200] for eid in s1_keys},
            {eid: feats_f2_cap250[eid][:200] for eid in s1_keys},
        ),
        ("Config_D_F2_Cap250_Leafwise127", "Formula 2", 250, "M4_Leafwise_127", cands_f2_cap250, feats_f2_cap250),
    ]

    pipeline_results = []
    log("\n" + "=" * 105)
    log(
        f"{'Configuration':<32} | {'Ranking':<9} | {'Cap':<4} | {'Model':<15} | {'Best Th':<7} | {'Macro F0.5':<10} | {'Prec':<7} | {'Recall':<7}"
    )
    log("-" * 105)

    for cfg_name, r_name, cap, m_name, cands_dict, feats_dict in pipeline_configs:
        model = fitted_models[m_name]
        t_sc0 = time.time()

        # Score candidates
        cand_probs = {}
        for eid in s1_keys:
            cands = cands_dict[eid]
            feats = feats_dict[eid]
            if len(cands) == 0:
                cand_probs[eid] = []
            else:
                probs = model.predict_proba(feats)[:, 1]
                cand_probs[eid] = list(zip(cands, probs))

        best_m, _th_records = evaluate_thresholds(cand_probs, truth, s1_keys)
        scoring_sec = time.time() - t_sc0

        log(
            f"{cfg_name:<32} | {r_name:<9} | {cap:<4} | {m_name:<15} | {best_m['threshold']:<7.2f} | "
            f"{best_m['macro_f0_5']:<10.5f} | {best_m['macro_precision']:<7.4f} | {best_m['macro_recall']:<7.4f}"
        )

        pipeline_results.append(
            {
                "configuration": cfg_name,
                "ranking_formula": r_name,
                "candidate_cap": cap,
                "model": m_name,
                "best_threshold": best_m["threshold"],
                "macro_f0_5": best_m["macro_f0_5"],
                "macro_precision": best_m["macro_precision"],
                "macro_recall": best_m["macro_recall"],
                "singleton_f0_5": best_m["singleton_f0_5"],
                "non_singleton_f0_5": best_m["non_singleton_f0_5"],
                "tp": best_m["tp"],
                "fp": best_m["fp"],
                "fn": best_m["fn"],
                "scoring_runtime_seconds": round(scoring_sec, 2),
            }
        )

    log("=" * 105)
    pd.DataFrame(pipeline_results).to_csv(EXP25_DIR / "matcher_comparison.csv", index=False)
    log(f"Saved matcher comparison to {EXP25_DIR / 'matcher_comparison.csv'}")

    # Build final JSON artifact
    eval_artifact = {
        "experiment_id": "EXP-002.5",
        "benchmark_sample_size": SAMPLE_SIZE,
        "total_ground_truth_links": total_truth_links,
        "candidate_ranking_benchmark": ranking_comparison_rows,
        "pipeline_comparison": pipeline_results,
        "total_runtime_seconds": round(time.time() - t_start, 2),
    }

    with open(EXP25_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_artifact, f, indent=2)
    log(f"Saved full evaluation summary to {EXP25_DIR / 'evaluation.json'}")

    log(f"\n=== EXP-002.5 Finished Successfully in {(time.time()-t_start)/60:.2f} minutes ===")


if __name__ == "__main__":
    main()
