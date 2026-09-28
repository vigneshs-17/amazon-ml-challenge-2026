"""EXP-003.5 Step 2 & 3: Diagnostic of Reachable Model False Negatives.

Analyzes the reachable false negatives from EXP-003 on the 10k validation benchmark:
1. Probability bucketing (<0.10, 0.10-0.30, 0.30-0.50, 0.50-0.70, 0.70-0.80, 0.80-0.84)
2. Statistical distribution comparison between False Negatives and True Positives
3. Hard positive cluster classification into evidence-based failure categories
4. Representative sample examples per cluster
5. Saves to experiments/EXP-003.5/fn_probability_buckets.json and error_analysis.json
"""

import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features_v2 import ALL_FEATURE_NAMES
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP1_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
EXP25_DIR = PROJECT_ROOT / "experiments" / "EXP-002.5"
EXP3_DIR = PROJECT_ROOT / "experiments" / "EXP-003"
EXP35_DIR = PROJECT_ROOT / "experiments" / "EXP-003.5"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"

THRESHOLD = 0.84
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def load_val_10k():
    val_ids = pd.read_csv(EXP1_DIR / "validation_s1_ids.csv")
    s1 = pd.read_parquet(CACHE / "train_s1.parquet")
    s1_val = s1[s1.entity_id.isin(set(val_ids.entity_id))].reset_index(drop=True)
    return s1_val.iloc[:10000].copy()


def load_s23_lookup():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    s23 = pd.concat([s2, s3], ignore_index=True)
    lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )
    return lookup


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


def summarize_series(vals):
    if not vals:
        return {"mean": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0}
    s = pd.Series(vals)
    return {
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "p25": round(float(s.quantile(0.25)), 4),
        "median": round(float(s.median()), 4),
        "p75": round(float(s.quantile(0.75)), 4),
        "p90": round(float(s.quantile(0.90)), 4),
    }


def main():
    os.makedirs(EXP35_DIR, exist_ok=True)
    print("=== EXP-003.5 Error Diagnosis on 10k Validation Benchmark ===")

    print("Loading EXP-003 model...")
    clf = joblib.load(EXP3_DIR / "model_exp003.joblib")

    print("Loading cached 10k candidates and features...")
    c_cache = joblib.load(EXP25_DIR / "interim_10k_candidates.joblib")
    f_cache = joblib.load(EXP25_DIR / "interim_10k_features.joblib")

    cands_f2 = c_cache["cands_f2_cap250"]
    feats_f2 = f_cache["feats_f2"]

    s1_10k = load_val_10k()
    s23_lookup = load_s23_lookup()
    truth = truth_map(s1_10k)
    total_true_links = sum(len(v) for v in truth.values())
    s1_keys = list(s1_10k.entity_id.astype(object))
    s1_info = dict(
        zip(
            s1_10k.entity_id.astype(object),
            zip(s1_10k.name_norm.astype(object), s1_10k.address_norm.astype(object), s1_10k.country.astype(object)),
        )
    )

    print("Scoring 10k validation sample across Cap-200 candidates...")
    per_s1_probs = {}
    for eid in s1_keys:
        cands = cands_f2[eid][:200]
        feats = feats_f2[eid][:200]
        if len(cands) == 0:
            per_s1_probs[eid] = []
        else:
            probs = clf.predict_proba(feats)[:, 1]
            per_s1_probs[eid] = list(zip(cands, probs, feats))

    # Apply unique-owner conflict resolution at threshold 0.84
    raw_preds = {eid: [(cid, float(p)) for cid, p, _ in per_s1_probs[eid] if p >= THRESHOLD] for eid in s1_keys}
    unique_preds = resolve_unique_owners(raw_preds)

    # Classify pairs
    tp_pairs = []
    fn_pairs = []
    fp_pairs = []

    for eid in s1_keys:
        t = truth[eid]
        pred_set = unique_preds[eid]

        for rank, (cid, p, f) in enumerate(per_s1_probs[eid], start=1):
            is_true = cid in t
            is_pred = cid in pred_set

            pair_meta = {
                "s1_id": eid,
                "cand_id": cid,
                "prob": float(p),
                "rank": rank,
                "is_true": is_true,
                "is_pred": is_pred,
                "features": f,
            }

            if is_true and is_pred:
                tp_pairs.append(pair_meta)
            elif is_true and not is_pred:
                fn_pairs.append(pair_meta)
            elif not is_true and is_pred:
                fp_pairs.append(pair_meta)

    print(f"\nBenchmark Partition: TP={len(tp_pairs)}, Reachable FN={len(fn_pairs)}, FP={len(fp_pairs)}")
    print(
        f"Total True Links: {total_true_links} | Surviving in Cap-200: {len(tp_pairs) + len(fn_pairs)} ({100.0*(len(tp_pairs)+len(fn_pairs))/total_true_links:.2f}%)"
    )

    # -------------------------------------------------------------
    # 1. Probability Bucketing of Reachable False Negatives
    # -------------------------------------------------------------
    fn_probs = [item["prob"] for item in fn_pairs]
    buckets = {
        "<0.10": sum(1 for p in fn_probs if p < 0.10),
        "0.10-0.30": sum(1 for p in fn_probs if 0.10 <= p < 0.30),
        "0.30-0.50": sum(1 for p in fn_probs if 0.30 <= p < 0.50),
        "0.50-0.70": sum(1 for p in fn_probs if 0.50 <= p < 0.70),
        "0.70-0.80": sum(1 for p in fn_probs if 0.70 <= p < 0.80),
        "0.80-0.84": sum(1 for p in fn_probs if 0.80 <= p < 0.84),
        ">=0.84_conflict_loss": sum(1 for p in fn_probs if p >= 0.84),  # true links >= 0.84 lost to conflict resolution
    }

    bucket_pcts = {k: round(100.0 * v / max(len(fn_pairs), 1), 2) for k, v in buckets.items()}
    print("\n--- False Negative Probability Distribution ---")
    for k, v in buckets.items():
        print(f"  {k:<20}: {v:>5d} ({bucket_pcts[k]:>5.2f}%)")

    fn_bucket_artifact = {
        "total_reachable_fn": len(fn_pairs),
        "threshold": THRESHOLD,
        "counts": buckets,
        "percentages": bucket_pcts,
        "summary": (
            "Distribution indicates whether misses are near-misses (close to threshold) "
            "or severe feature divergence (sub-0.10)."
        ),
    }
    with open(EXP35_DIR / "fn_probability_buckets.json", "w", encoding="utf-8") as f:
        json.dump(fn_bucket_artifact, f, indent=2)

    # -------------------------------------------------------------
    # 2. Detailed Feature Distribution Comparison (TP vs FN)
    # -------------------------------------------------------------
    feature_comparison = {}
    for feat_idx, feat_name in enumerate(ALL_FEATURE_NAMES):
        tp_vals = [p["features"][feat_idx] for p in tp_pairs]
        fn_vals = [p["features"][feat_idx] for p in fn_pairs]
        feature_comparison[feat_name] = {
            "TP": summarize_series(tp_vals),
            "FN": summarize_series(fn_vals),
        }

    # Metadata & contextual comparisons
    rank_comp = {
        "TP": summarize_series([p["rank"] for p in tp_pairs]),
        "FN": summarize_series([p["rank"] for p in fn_pairs]),
    }
    prob_comp = {
        "TP": summarize_series([p["prob"] for p in tp_pairs]),
        "FN": summarize_series([p["prob"] for p in fn_pairs]),
    }

    # Script, country, source comparisons
    def extract_context(p):
        eid, cid = p["s1_id"], p["cand_id"]
        s1_norm_name, s1_norm_addr, c1 = s1_info[eid]
        c2_norm_name, c2_norm_addr, _c2 = s23_lookup[cid]
        s1_dev = bool(DEVANAGARI_RE.search(s1_norm_name))
        c2_dev = bool(DEVANAGARI_RE.search(c2_norm_name))
        return {
            "script_mismatch": int(s1_dev != c2_dev),
            "country_india": int(c1 == "India"),
            "source_s2": int(cid.startswith("S2-")),
            "s1_name": s1_norm_name,
            "s1_addr": s1_norm_addr,
            "cand_name": c2_norm_name,
            "cand_addr": c2_norm_addr,
        }

    tp_contexts = [extract_context(p) for p in tp_pairs]
    fn_contexts = [extract_context(p) for p in fn_pairs]

    context_comparison = {
        "script_mismatch_pct": {
            "TP": round(100.0 * np.mean([c["script_mismatch"] for c in tp_contexts]), 2),
            "FN": round(100.0 * np.mean([c["script_mismatch"] for c in fn_contexts]), 2),
        },
        "country_india_pct": {
            "TP": round(100.0 * np.mean([c["country_india"] for c in tp_contexts]), 2),
            "FN": round(100.0 * np.mean([c["country_india"] for c in fn_contexts]), 2),
        },
        "source_s2_pct": {
            "TP": round(100.0 * np.mean([c["source_s2"] for c in tp_contexts]), 2),
            "FN": round(100.0 * np.mean([c["source_s2"] for c in fn_contexts]), 2),
        },
    }

    # -------------------------------------------------------------
    # 3. Hard Positive Clusters (Classification of Reachable FNs)
    # -------------------------------------------------------------
    clusters = {
        "transliteration_script_divergence": [],
        "severe_name_corruption": [],
        "severe_address_corruption": [],
        "name_weak_address_strong": [],
        "name_strong_address_weak": [],
        "numeric_or_postal_disagreement": [],
        "short_or_generic_business_name": [],
        "missing_field": [],
        "token_order_or_squashing_issue": [],
        "candidate_rank_tail": [],
        "near_miss_probability_80_84": [],
        "other": [],
    }

    name_ratio_idx = ALL_FEATURE_NAMES.index("name_ratio")
    name_jacc_idx = ALL_FEATURE_NAMES.index("name_tok_jaccard")
    name_tok_set_idx = ALL_FEATURE_NAMES.index("name_token_set_ratio")
    name_char3_idx = ALL_FEATURE_NAMES.index("name_char3_jaccard")
    addr_ratio_idx = ALL_FEATURE_NAMES.index("addr_ratio")
    addr_jacc_idx = ALL_FEATURE_NAMES.index("addr_tok_jaccard")
    postal_conflict_idx = ALL_FEATURE_NAMES.index("postal_conflict")
    num_conflict_idx = ALL_FEATURE_NAMES.index("num_tok_conflict")
    name_missing_idx = ALL_FEATURE_NAMES.index("name_missing")
    addr_missing_idx = ALL_FEATURE_NAMES.index("addr_missing")
    name_min_df_idx = ALL_FEATURE_NAMES.index("name_min_shared_df")

    for fn_item, ctx in zip(fn_pairs, fn_contexts):
        f = fn_item["features"]
        prob = fn_item["prob"]
        rank = fn_item["rank"]

        record_summary = {
            "s1_id": fn_item["s1_id"],
            "cand_id": fn_item["cand_id"],
            "prob": round(prob, 4),
            "rank": rank,
            "s1_name": ctx["s1_name"],
            "cand_name": ctx["cand_name"],
            "s1_addr": ctx["s1_addr"],
            "cand_addr": ctx["cand_addr"],
            "name_ratio": round(float(f[name_ratio_idx]), 2),
            "addr_ratio": round(float(f[addr_ratio_idx]), 2),
            "name_jacc": round(float(f[name_jacc_idx]), 2),
            "addr_jacc": round(float(f[addr_jacc_idx]), 2),
        }

        # 1. Transliteration / script divergence
        if ctx["script_mismatch"] == 1:
            clusters["transliteration_script_divergence"].append(record_summary)
        # 2. Near miss in 0.80 - 0.84 range
        elif 0.80 <= prob < 0.84:
            clusters["near_miss_probability_80_84"].append(record_summary)
        # 3. Missing field
        elif f[name_missing_idx] == 1.0 or f[addr_missing_idx] == 1.0:
            clusters["missing_field"].append(record_summary)
        # 4. Numeric or postal disagreement
        elif f[num_conflict_idx] == 1.0 or f[postal_conflict_idx] == 1.0:
            clusters["numeric_or_postal_disagreement"].append(record_summary)
        # 5. Token order or squashing issue
        elif f[name_jacc_idx] < 0.3 and (f[name_char3_idx] > 0.45 or f[name_tok_set_idx] > 80):
            clusters["token_order_or_squashing_issue"].append(record_summary)
        # 6. Name weak / address strong
        elif f[name_jacc_idx] <= 0.2 and f[addr_jacc_idx] >= 0.5:
            clusters["name_weak_address_strong"].append(record_summary)
        # 7. Name strong / address weak
        elif f[name_jacc_idx] >= 0.5 and f[addr_jacc_idx] <= 0.2:
            clusters["name_strong_address_weak"].append(record_summary)
        # 8. Severe name corruption
        elif f[name_ratio_idx] < 40 and f[name_jacc_idx] == 0:
            clusters["severe_name_corruption"].append(record_summary)
        # 9. Severe address corruption
        elif f[addr_ratio_idx] < 40 and f[addr_jacc_idx] == 0:
            clusters["severe_address_corruption"].append(record_summary)
        # 10. Short or generic business name
        elif len(ctx["s1_name"].split()) <= 2 and f[name_min_df_idx] > 5000:
            clusters["short_or_generic_business_name"].append(record_summary)
        # 11. Candidate rank tail
        elif rank > 100:
            clusters["candidate_rank_tail"].append(record_summary)
        else:
            clusters["other"].append(record_summary)

    cluster_counts = {k: len(v) for k, v in clusters.items()}
    cluster_pcts = {k: round(100.0 * len(v) / max(len(fn_pairs), 1), 2) for k, v in clusters.items()}

    print("\n--- Hard Positive Clusters (Reachable False Negatives) ---")
    for k, v in cluster_counts.items():
        print(f"  {k:<35}: {v:>5d} ({cluster_pcts[k]:>5.2f}%)")

    # Sample examples per cluster
    sample_cluster_examples = {k: v[:5] for k, v in clusters.items() if len(v) > 0}

    # Combine into comprehensive error analysis
    error_analysis_report = {
        "benchmark_sample_size": len(s1_10k),
        "total_true_links": total_true_links,
        "cap200_retained_true_links": len(tp_pairs) + len(fn_pairs),
        "cap200_candidate_recall": round((len(tp_pairs) + len(fn_pairs)) / total_true_links, 5),
        "operating_threshold": THRESHOLD,
        "confusion_counts": {
            "TP": len(tp_pairs),
            "FN": len(fn_pairs),
            "FP": len(fp_pairs),
        },
        "fn_probability_buckets": fn_bucket_artifact,
        "hard_positive_clusters": {
            "counts": cluster_counts,
            "percentages": cluster_pcts,
            "representative_samples": sample_cluster_examples,
        },
        "distributions": {
            "rank": rank_comp,
            "prob": prob_comp,
            "context": context_comparison,
            "key_features": {
                feat: feature_comparison[feat]
                for feat in [
                    "name_ratio",
                    "name_token_set_ratio",
                    "name_tok_jaccard",
                    "name_weighted_jaccard",
                    "name_char3_jaccard",
                    "addr_ratio",
                    "addr_token_set_ratio",
                    "addr_tok_jaccard",
                    "addr_weighted_jaccard",
                    "postal_match",
                    "postal_conflict",
                    "num_tok_conflict",
                ]
            },
        },
    }

    with open(EXP35_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis_report, f, indent=2)

    print(f"\nSaved complete error analysis to {EXP35_DIR / 'error_analysis.json'}")
    print(f"Saved FN probability buckets to {EXP35_DIR / 'fn_probability_buckets.json'}")


if __name__ == "__main__":
    main()
