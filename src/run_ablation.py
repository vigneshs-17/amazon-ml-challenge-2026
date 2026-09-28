"""EXP-001 Step 6/7: blocking-channel ablation + candidate-budget curve on
the fixed validation S1 set, using the efficient set-based blocking.py
(not the Phase-1 O(bucket) diagnostic pattern).
"""

import json
import time

import pandas as pd

from blocking import Blocker
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
QTS = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)


def load_val():
    val_ids = pd.read_csv(EXP_DIR / "validation_s1_ids.csv")
    s1 = pd.read_parquet(CACHE / "train_s1.parquet")
    s1 = s1[s1.entity_id.isin(set(val_ids.entity_id))].reset_index(drop=True)
    return s1, val_ids


def load_s23():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    return pd.concat([s2, s3], ignore_index=True)


def truth_map(val_ids):
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    val_set = set(val_ids.entity_id)
    long = long[long.source1_entity_id.astype(object).isin(val_set)]
    truth = {sid: set() for sid in val_set}
    for sid, mid in zip(long.source1_entity_id.astype(object), long.matched_id.astype(object)):
        truth[sid].add(mid)
    return truth


def eval_channel(blocker, s1, truth, channels, name_k=1, addr_k=1, country_soft=True, label=""):
    t0 = time.time()
    counts = []
    total_truth_pairs = 0
    hit_pairs = 0
    non_singleton_hit = 0
    non_singleton_total = 0
    for eid, name, addr, country in zip(
        s1.entity_id.astype(object),
        s1.name_norm.astype(object),
        s1.address_norm.astype(object),
        s1.country.astype(object),
    ):
        cand = blocker.candidates(name, addr, country, name_k, addr_k, channels, country_soft)
        counts.append(len(cand))
        t = truth[eid]
        total_truth_pairs += len(t)
        retrieved = len(t & cand)
        hit_pairs += retrieved
        if t:
            non_singleton_total += len(t)
            non_singleton_hit += retrieved
    n = pd.Series(counts)
    result = {
        "channels": list(channels),
        "name_k": name_k,
        "addr_k": addr_k,
        "country_soft": country_soft,
        "pairwise_recall_pct": round(100 * hit_pairs / max(total_truth_pairs, 1), 3),
        "non_singleton_recall_pct": round(100 * non_singleton_hit / max(non_singleton_total, 1), 3),
        "candidates_per_s1": {str(q): float(n.quantile(q)) for q in QTS} | {"mean": round(float(n.mean()), 2)},
        "pct_s1_zero_candidates": round(100 * float((n == 0).mean()), 3),
        "total_candidate_pairs": int(n.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    print(
        label,
        result["pairwise_recall_pct"],
        "% recall,",
        result["non_singleton_recall_pct"],
        "% non-singleton recall,",
        result["candidates_per_s1"]["mean"],
        "avg cand,",
        result["seconds"],
        "s",
        flush=True,
    )
    return result


def main():
    s1, val_ids = load_val()
    s23 = load_s23()
    truth = truth_map(val_ids)
    blocker = Blocker(s23)
    naive_total = len(s1) * len(s23)

    rep = {
        "val_s1": len(s1),
        "s23_pool": len(s23),
        "naive_all_pairs": naive_total,
        "truth_pairs_in_val": sum(len(v) for v in truth.values()),
    }

    ablation = {}
    ablation["B1_rare_name"] = eval_channel(blocker, s1, truth, ("name",), 1, 1, True, "B1_rare_name_k1")
    ablation["B2_rare_addr"] = eval_channel(blocker, s1, truth, ("addr",), 1, 1, True, "B2_rare_addr_k1")
    ablation["B3_exact_name"] = eval_channel(blocker, s1, truth, ("exact_name",), 1, 1, True, "B3_exact_name")
    ablation["B4_exact_addr"] = eval_channel(blocker, s1, truth, ("exact_addr",), 1, 1, True, "B4_exact_addr")
    ablation["B5_name_addr"] = eval_channel(blocker, s1, truth, ("name", "addr"), 1, 1, True, "B5_name+addr")
    ablation["B6_name_addr_exactname"] = eval_channel(
        blocker, s1, truth, ("name", "addr", "exact_name"), 1, 1, True, "B6_+exact_name"
    )
    ablation["B7_all_four"] = eval_channel(
        blocker, s1, truth, ("name", "addr", "exact_name", "exact_addr"), 1, 1, True, "B7_all_four"
    )
    # also try rare-2 name within the union, since Phase-1 showed higher recall at higher cost
    ablation["B7_rare2_name"] = eval_channel(
        blocker, s1, truth, ("name", "addr", "exact_name", "exact_addr"), 2, 1, True, "B7_rare2name"
    )
    for v in ablation.values():
        v["candidate_reduction_ratio_vs_naive"] = round(1 - v["total_candidate_pairs"] / naive_total, 6)

    rep["ablation"] = ablation
    json.dump(rep, open(EXP_DIR / "blocking_results.json", "w", encoding="utf-8"), indent=1, default=str)
    print("ablation done")


if __name__ == "__main__":
    main()
