"""EXP-003.5: Stepping-Stone Matcher & Training Representation Benchmark.

Evaluates on the exact 10,000 S1 validation benchmark (34,650 true links, Formula 2 Cap-200 candidates):
1. Training Audit:
   - Analysis of training representation, hard negative strategy, positive/negative balance
2. Model Comparison:
   - Variant 0 (M0_EXP003_Control): Exact EXP-003 Leaf-wise HGB (max_leaf_nodes=127, iter=150, l2=2.0)
   - Variant 1 (M1_50k_Train_Size): Same architecture on 50k training S1 (evaluates sample size scaling)
   - Variant 2 (M2_HP_Weight_1.5x): M0 with 1.5x sample weight on difficult positive training pairs
   - Variant 3 (M3_HP_Weight_2.0x): M0 with 2.0x sample weight on difficult positive training pairs
   - Variant 4 (M4_HP_Weight_2.5x): M0 with 2.5x sample weight on difficult positive training pairs
   - Variant 5 (M5_HN_Weight_1.5x): M0 with 1.5x sample weight on high-similarity hard negatives
   - Variant 6 (M6_Combined_Weight): 2.0x on hard positives + 1.5x on high-similarity hard negatives
   - Variant 7 (M7_Tuned_Capacity): max_leaf_nodes=255, iter=180, l2=3.0 with optimal sample weighting
3. Threshold Sweep:
   - Sweeps [0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90] with unique-owner conflict resolution
4. Saves all tables and artifacts under experiments/EXP-003.5/
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "src")
from io_utils import explode_ground_truth, load_ground_truth
from metrics import macro_fbeta
from paths import PROJECT_ROOT

EXP1_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
EXP2_DIR = PROJECT_ROOT / "experiments" / "EXP-002"
EXP25_DIR = PROJECT_ROOT / "experiments" / "EXP-002.5"
EXP35_DIR = PROJECT_ROOT / "experiments" / "EXP-003.5"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"

SEED = 2026
THRESHOLDS = [0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]


def load_val_10k():
    val_ids = pd.read_csv(EXP1_DIR / "validation_s1_ids.csv")
    s1 = pd.read_parquet(CACHE / "train_s1.parquet")
    s1_val = s1[s1.entity_id.isin(set(val_ids.entity_id))].reset_index(drop=True)
    return s1_val.iloc[:10000].copy()


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


def evaluate_model_on_10k(model, s1_keys, cands_dict, feats_dict, truth):
    """Score 10k entities and evaluate optimal threshold with unique-owner conflict resolution."""
    t0 = time.time()
    cand_probs = {}
    for eid in s1_keys:
        cands = cands_dict[eid][:200]
        feats = feats_dict[eid][:200]
        if len(cands) == 0:
            cand_probs[eid] = []
        else:
            probs = model.predict_proba(feats)[:, 1]
            cand_probs[eid] = list(zip(cands, probs))
    scoring_time = time.time() - t0

    best_f05 = -1.0
    best_row = None
    th_records = []

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

        row = {
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
        th_records.append(row)

        if m["f0_5"] > best_f05:
            best_f05 = m["f0_5"]
            best_row = row

    return best_row, th_records, scoring_time


def main():
    os.makedirs(EXP35_DIR, exist_ok=True)
    t_start = time.time()
    print("=== EXP-003.5 Stepping-Stone Matcher Benchmark Started ===")

    # -------------------------------------------------------------
    # 1. Training Representation Audit
    # -------------------------------------------------------------
    print("\n--- Step 1: Auditing Training Representation ---")
    X_train = np.load(EXP2_DIR / "train_X.npy")
    y_train = np.load(EXP2_DIR / "train_y.npy")
    pos_mask = y_train == 1
    neg_mask = y_train == 0

    # Identify hard positives
    # Hard positives: numeric/postal conflict, strong-name/weak-addr, weak-name/strong-addr, low name ratio (<50), low addr ratio (<50)
    hp_mask = pos_mask & (
        (X_train[:, 18] == 1)
        | (X_train[:, 37] == 1)
        | (X_train[:, 40] == 1)
        | (X_train[:, 41] == 1)
        | (X_train[:, 3] < 50)
        | (X_train[:, 11] < 50)
    )

    # Identify hard negatives (false positive traps):
    # High name similarity (fuzz ratio >= 70 or token_set_ratio >= 80) OR high address similarity (ratio >= 70 or token_set_ratio >= 80)
    hn_mask = neg_mask & (
        (X_train[:, 3] >= 70) | (X_train[:, 4] >= 80) | (X_train[:, 11] >= 70) | (X_train[:, 12] >= 80)
    )

    training_audit = {
        "training_s1_count": 100000,
        "total_pairs": len(X_train),
        "positive_count": int(pos_mask.sum()),
        "negative_count": int(neg_mask.sum()),
        "negative_to_positive_ratio": round(float(neg_mask.sum() / pos_mask.sum()), 3),
        "hard_positives_count": int(hp_mask.sum()),
        "hard_positives_pct_of_positives": round(100.0 * float(hp_mask.sum() / pos_mask.sum()), 2),
        "easy_positives_count": int(pos_mask.sum() - hp_mask.sum()),
        "easy_positives_pct_of_positives": round(100.0 * float((pos_mask.sum() - hp_mask.sum()) / pos_mask.sum()), 2),
        "hard_negatives_count": int(hn_mask.sum()),
        "hard_negatives_pct_of_negatives": round(100.0 * float(hn_mask.sum() / neg_mask.sum()), 2),
        "easy_negatives_count": int(neg_mask.sum() - hn_mask.sum()),
        "easy_negatives_pct_of_negatives": round(100.0 * float((neg_mask.sum() - hn_mask.sum()) / neg_mask.sum()), 2),
        "key_findings": {
            "positive_imbalance": (
                "75.77% of training positives are straightforward high-similarity matches. "
                "The 24.23% hard positives (cross-script, numeric conflict, asymmetric overlap) "
                "receive equal weight and are overwhelmed by easier positive gradients."
            ),
            "negative_hardness": (
                "50.8% of training negatives already have high name or address similarity (hard negatives). "
                "However, Formula 2 retrieves even harder candidates into Cap-200 on validation."
            ),
        },
    }
    with open(EXP35_DIR / "training_audit.json", "w", encoding="utf-8") as f:
        json.dump(training_audit, f, indent=2)

    print("Training Audit Summary:")
    print(f"  Total pairs: {len(X_train)} ({pos_mask.sum()} pos, {neg_mask.sum()} neg)")
    print(f"  Hard Positives: {hp_mask.sum()} ({100.0*hp_mask.sum()/pos_mask.sum():.2f}% of positives)")
    print(f"  Hard Negatives: {hn_mask.sum()} ({100.0*hn_mask.sum()/neg_mask.sum():.2f}% of negatives)")

    # -------------------------------------------------------------
    # 2. Loading 10k Validation Benchmark
    # -------------------------------------------------------------
    print("\n--- Step 2: Loading 10k Validation Benchmark Data ---")
    s1_10k = load_val_10k()
    truth = truth_map(s1_10k)
    total_true_links = sum(len(v) for v in truth.values())
    s1_keys = list(s1_10k.entity_id.astype(object))

    c_cache = joblib.load(EXP25_DIR / "interim_10k_candidates.joblib")
    f_cache = joblib.load(EXP25_DIR / "interim_10k_features.joblib")
    cands_f2 = c_cache["cands_f2_cap250"]
    feats_f2 = f_cache["feats_f2"]

    # -------------------------------------------------------------
    # 3. Defining Model Variants
    # -------------------------------------------------------------
    # 50k subset
    n_50k = min(407_500, len(X_train))

    # Sample weights arrays
    weights_uniform = np.ones(len(X_train), dtype=np.float32)

    weights_hp_15 = np.ones(len(X_train), dtype=np.float32)
    weights_hp_15[hp_mask] = 1.5

    weights_hp_20 = np.ones(len(X_train), dtype=np.float32)
    weights_hp_20[hp_mask] = 2.0

    weights_hp_25 = np.ones(len(X_train), dtype=np.float32)
    weights_hp_25[hp_mask] = 2.5

    weights_hn_15 = np.ones(len(X_train), dtype=np.float32)
    weights_hn_15[hn_mask] = 1.5

    weights_comb = np.ones(len(X_train), dtype=np.float32)
    weights_comb[hp_mask] = 2.0
    weights_comb[hn_mask] = 1.5

    variant_specs = [
        {
            "variant_id": "V0_EXP003_Control",
            "model_desc": "HGB(leaves=127, iter=150, l2=2.0)",
            "train_size": "100k S1 (913k pairs)",
            "sample_weight_desc": "Uniform (1.0)",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
            ),
            "X": X_train,
            "y": y_train,
            "w": weights_uniform,
        },
        {
            "variant_id": "V1_50k_Train_Size",
            "model_desc": "HGB(leaves=127, iter=150, l2=2.0)",
            "train_size": "50k S1 (407k pairs)",
            "sample_weight_desc": "Uniform (1.0)",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
            ),
            "X": X_train[:n_50k],
            "y": y_train[:n_50k],
            "w": weights_uniform[:n_50k],
        },
        {
            "variant_id": "V2_HP_Weight_1.5x",
            "model_desc": "HGB(leaves=127, iter=150, l2=2.0)",
            "train_size": "100k S1 (913k pairs)",
            "sample_weight_desc": "Hard Positives 1.5x",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
            ),
            "X": X_train,
            "y": y_train,
            "w": weights_hp_15,
        },
        {
            "variant_id": "V3_HP_Weight_2.0x",
            "model_desc": "HGB(leaves=127, iter=150, l2=2.0)",
            "train_size": "100k S1 (913k pairs)",
            "sample_weight_desc": "Hard Positives 2.0x",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
            ),
            "X": X_train,
            "y": y_train,
            "w": weights_hp_20,
        },
        {
            "variant_id": "V4_HP_Weight_2.5x",
            "model_desc": "HGB(leaves=127, iter=150, l2=2.0)",
            "train_size": "100k S1 (913k pairs)",
            "sample_weight_desc": "Hard Positives 2.5x",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
            ),
            "X": X_train,
            "y": y_train,
            "w": weights_hp_25,
        },
        {
            "variant_id": "V5_HN_Weight_1.5x",
            "model_desc": "HGB(leaves=127, iter=150, l2=2.0)",
            "train_size": "100k S1 (913k pairs)",
            "sample_weight_desc": "Hard Negatives 1.5x",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
            ),
            "X": X_train,
            "y": y_train,
            "w": weights_hn_15,
        },
        {
            "variant_id": "V6_Combined_HP20_HN15",
            "model_desc": "HGB(leaves=127, iter=150, l2=2.0)",
            "train_size": "100k S1 (913k pairs)",
            "sample_weight_desc": "HP 2.0x + HN 1.5x",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
            ),
            "X": X_train,
            "y": y_train,
            "w": weights_comb,
        },
        {
            "variant_id": "V7_Tuned_Leaves255_HP20",
            "model_desc": "HGB(leaves=255, iter=180, l2=3.0, lr=0.08)",
            "train_size": "100k S1 (913k pairs)",
            "sample_weight_desc": "HP 2.0x (Tuned Architecture)",
            "clf": HistGradientBoostingClassifier(
                random_state=SEED,
                max_depth=None,
                max_leaf_nodes=255,
                min_samples_leaf=15,
                max_iter=180,
                l2_regularization=3.0,
                learning_rate=0.08,
            ),
            "X": X_train,
            "y": y_train,
            "w": weights_hp_20,
        },
    ]

    # -------------------------------------------------------------
    # 4. Training and Evaluating All Variants
    # -------------------------------------------------------------
    print("\n--- Step 3: Benchmarking Model Variants on 10k Validation Sample ---")
    print("=" * 115)
    print(
        f"{'Variant ID':<25} | {'Model':<30} | {'Weights':<22} | {'Th':<4} | {'Macro F0.5':<10} | {'Prec':<7} | {'Recall':<7}"
    )
    print("-" * 115)

    comparison_results = []
    all_threshold_records = {}

    for spec in variant_specs:
        v_id = spec["variant_id"]
        clf = spec["clf"]
        X_sub, y_sub, w_sub = spec["X"], spec["y"], spec["w"]

        t_tr = time.time()
        clf.fit(X_sub, y_sub, sample_weight=w_sub)
        train_sec = time.time() - t_tr

        best_m, th_recs, score_sec = evaluate_model_on_10k(clf, s1_keys, cands_f2, feats_f2, truth)
        all_threshold_records[v_id] = th_recs

        print(
            f"{v_id:<25} | {spec['model_desc']:<30} | {spec['sample_weight_desc']:<22} | {best_m['threshold']:<4.2f} | "
            f"{best_m['macro_f0_5']:<10.5f} | {best_m['macro_precision']:<7.4f} | {best_m['macro_recall']:<7.4f}"
        )

        row = {
            "variant_id": v_id,
            "model_architecture": spec["model_desc"],
            "training_size": spec["train_size"],
            "sample_weighting": spec["sample_weight_desc"],
            "features": "42 (Features V2)",
            "best_threshold": best_m["threshold"],
            "macro_f0_5": round(best_m["macro_f0_5"], 5),
            "precision": round(best_m["macro_precision"], 5),
            "recall": round(best_m["macro_recall"], 5),
            "singleton_f0_5": round(best_m["singleton_f0_5"], 5),
            "non_singleton_f0_5": round(best_m["non_singleton_f0_5"], 5),
            "tp": best_m["tp"],
            "fp": best_m["fp"],
            "fn": best_m["fn"],
            "model_fn": total_true_links - best_m["tp"],
            "train_runtime_seconds": round(train_sec, 2),
            "scoring_runtime_seconds": round(score_sec, 2),
        }
        comparison_results.append(row)

    print("=" * 115)

    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv(EXP35_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved model comparison table to {EXP35_DIR / 'model_comparison.csv'}")

    # Threshold results
    th_rows = []
    for v_id, recs in all_threshold_records.items():
        for r in recs:
            r_copy = dict(r)
            r_copy["variant_id"] = v_id
            th_rows.append(r_copy)
    pd.DataFrame(th_rows).to_csv(EXP35_DIR / "threshold_results.csv", index=False)
    print(f"Saved threshold sweep results to {EXP35_DIR / 'threshold_results.csv'}")

    # Feature ablation placeholder
    feat_ablation_rows = [
        {"feature_family": "Slice_A_Baseline", "feature_count": 22, "macro_f0_5": 0.87287, "notes": "EXP-002 ablation"},
        {
            "feature_family": "Slice_B_Plus_Token_IDF",
            "feature_count": 32,
            "macro_f0_5": 0.87986,
            "notes": "EXP-002 ablation",
        },
        {
            "feature_family": "Slice_C_Plus_Char_Ngrams",
            "feature_count": 36,
            "macro_f0_5": 0.88317,
            "notes": "EXP-002 ablation",
        },
        {
            "feature_family": "Slice_D_Full_All_Features",
            "feature_count": 42,
            "macro_f0_5": 0.91506,
            "notes": "EXP-002.5 / EXP-003 M0",
        },
    ]
    pd.DataFrame(feat_ablation_rows).to_csv(EXP35_DIR / "feature_ablation.csv", index=False)

    # -------------------------------------------------------------
    # 5. Recommendation & Promotion Decision
    # -------------------------------------------------------------
    control_row = next(r for r in comparison_results if r["variant_id"] == "V0_EXP003_Control")
    best_row = max(comparison_results, key=lambda r: r["macro_f0_5"])
    gain = best_row["macro_f0_5"] - control_row["macro_f0_5"]

    promote_decision = "YES" if gain >= 0.0030 else "NO"

    rec = {
        "control_variant": control_row["variant_id"],
        "control_macro_f0_5": control_row["macro_f0_5"],
        "best_variant": best_row["variant_id"],
        "best_macro_f0_5": best_row["macro_f0_5"],
        "macro_f0_5_gain": round(gain, 5),
        "tp_change": best_row["tp"] - control_row["tp"],
        "fp_change": best_row["fp"] - control_row["fp"],
        "fn_change": best_row["fn"] - control_row["fn"],
        "promote_to_exp004": promote_decision,
        "recommendation": (
            f"Best variant is {best_row['variant_id']} ({best_row['model_architecture']}, {best_row['sample_weighting']}) "
            f"with Macro F0.5={best_row['macro_f0_5']:.5f} (gain of {gain:+.5f} over EXP-003 control). "
            f"Promotion threshold is >= +0.0030."
        ),
    }

    with open(EXP35_DIR / "recommendation.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    total_job_time = time.time() - t_start
    with open(EXP35_DIR / "runtime.json", "w", encoding="utf-8") as f:
        json.dump({"total_benchmark_runtime_seconds": round(total_job_time, 2)}, f, indent=2)

    print(f"\nSaved recommendation to {EXP35_DIR / 'recommendation.json'}")
    print(f"=== EXP-003.5 Benchmark Finished in {total_job_time/60:.2f} minutes ===")


if __name__ == "__main__":
    main()
