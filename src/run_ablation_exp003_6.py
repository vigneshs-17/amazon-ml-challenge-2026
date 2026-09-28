"""EXP-003.6: Controlled Model Ablation & Threshold Optimization.

Evaluates 4 controlled variants on the exact 10,000 S1 validation benchmark:
- V0: Control (EXP-003 architecture trained on old Formula 1 pairs, 42 features)
- V1: Formula-2 consistent training data (42 baseline features)
- V2: Formula-2 training data + 5 tolerant numeric & postal features (47 features)
- V3: Formula-2 training data + 5 tolerant numeric + 5 cross-field asymmetry features (52 features)

Applies identical unique-owner conflict resolution across fine threshold sweep [0.78 - 0.90].
Measures specific error recovery on diagnosed numeric/postal FNs, asymmetric FNs, and new FPs.
Saves all required tables and JSON reports under experiments/EXP-003.6/.
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
EXP36_DIR = PROJECT_ROOT / "experiments" / "EXP-003.6"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"

SEED = 2026
THRESHOLDS = [0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]


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


def evaluate_variant(
    model, n_features, s1_keys, cands_dict, val_feats, truth, total_truth_links, candidate_reachable_links
):
    """Score 10k entities, evaluate thresholds, and return best metrics."""
    t0 = time.time()
    cand_probs = {}
    for eid in s1_keys:
        cands = cands_dict[eid][:200]
        feats = val_feats[eid][:200, :n_features]
        if len(cands) == 0:
            cand_probs[eid] = []
        else:
            probs = model.predict_proba(feats)[:, 1]
            cand_probs[eid] = list(zip(cands, probs))
    score_time = time.time() - t0

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

        total_fn = total_truth_links - tp
        reachable_model_fn = candidate_reachable_links - tp

        row = {
            "threshold": th,
            "macro_f05": round(m["f0_5"], 5),
            "precision": round(m["precision"], 5),
            "recall": round(m["recall"], 5),
            "tp": tp,
            "fp": fp,
            "total_fn": total_fn,
            "reachable_model_fn": reachable_model_fn,
        }
        th_records.append(row)

        if m["f0_5"] > best_f05:
            best_f05 = m["f0_5"]
            best_row = row
            best_preds = unique_preds
            best_probs = cand_probs

    return best_row, th_records, best_preds, best_probs, score_time


def main():
    os.makedirs(EXP36_DIR, exist_ok=True)
    print("=== EXP-003.6 Controlled Model Ablation & Threshold Optimization ===", flush=True)
    t_global0 = time.time()

    print("Loading 10k validation ground truth and candidate metadata...", flush=True)
    s1_val = load_val_10k()
    s1_keys = list(s1_val.entity_id.astype(object))
    truth = truth_map(s1_val)
    total_truth_links = sum(len(v) for v in truth.values())

    cands_blob = joblib.load(EXP25_DIR / "interim_10k_candidates.joblib")
    cands_f2 = cands_blob["cands_f2_cap250"]

    # Candidate reachable links
    candidate_reachable_links = 0
    for sid in s1_keys:
        top200 = set(cands_f2[sid][:200])
        candidate_reachable_links += len(truth[sid] & top200)

    print(f"Validation Ground Truth: {len(s1_val)} S1 | {total_truth_links} true links", flush=True)
    print(
        f"Formula 2 Cap-200 Reachable True Links: {candidate_reachable_links} ({100*candidate_reachable_links/total_truth_links:.3f}%)",
        flush=True,
    )

    print("\nLoading precomputed 52 features for 10k validation benchmark...", flush=True)
    val_feats = joblib.load(EXP36_DIR / "val_10k_features_v3.joblib")
    print(f"Loaded validation features for {len(val_feats)} entities.", flush=True)

    print("\nLoading training datasets...", flush=True)
    # Old Formula 1 training pairs for V0 control
    X_train_f1 = np.load(EXP2_DIR / "train_X.npy")
    y_train_f1 = np.load(EXP2_DIR / "train_y.npy")
    print(f"  Old Formula-1 pairs (EXP-002): {X_train_f1.shape}", flush=True)

    # Regenerated Formula 2 training pairs
    X_train_f2 = np.load(EXP36_DIR / "train_X_v3.npy")
    y_train_f2 = np.load(EXP36_DIR / "train_y_v3.npy")
    print(f"  Regenerated Formula-2 pairs:   {X_train_f2.shape}", flush=True)

    variants_config = [
        {
            "variant_id": "V0_Control",
            "name": "EXP-003 Control (Formula-1 Training, 42 feats)",
            "X_train": X_train_f1,
            "y_train": y_train_f1,
            "n_feats": 42,
            "formula_train": "Formula 1",
            "feats_desc": "Features V2 (42)",
        },
        {
            "variant_id": "V1_F2_Training",
            "name": "Formula-2 Training Data (42 feats)",
            "X_train": X_train_f2[:, :42],
            "y_train": y_train_f2,
            "n_feats": 42,
            "formula_train": "Formula 2",
            "feats_desc": "Features V2 (42)",
        },
        {
            "variant_id": "V2_Tolerant_Numeric",
            "name": "Formula-2 Training + Tolerant Numeric (47 feats)",
            "X_train": X_train_f2[:, :47],
            "y_train": y_train_f2,
            "n_feats": 47,
            "formula_train": "Formula 2",
            "feats_desc": "Features V2 + Tolerant Numeric (47)",
        },
        {
            "variant_id": "V3_Full_V3",
            "name": "Formula-2 Training + Tolerant Numeric + Asymmetry (52 feats)",
            "X_train": X_train_f2[:, :52],
            "y_train": y_train_f2,
            "n_feats": 52,
            "formula_train": "Formula 2",
            "feats_desc": "Features V2 + Numeric + Asymmetry (52)",
        },
    ]

    all_best_rows = []
    all_th_rows = []
    trained_models = {}
    variant_preds = {}
    variant_probs = {}
    runtimes = {}

    for cfg in variants_config:
        vid = cfg["variant_id"]
        model_path = EXP36_DIR / f"model_{vid.lower()}.joblib"
        t_tr0 = time.time()
        if model_path.exists():
            print(f"\n--- Loading pre-trained {cfg['name']} from {model_path.name} ---", flush=True)
            clf = joblib.load(model_path)
            train_time = 0.0
        else:
            print(f"\n--- Training {cfg['name']} ({cfg['n_feats']} features) ---", flush=True)
            clf = HistGradientBoostingClassifier(
                max_leaf_nodes=127,
                min_samples_leaf=20,
                max_iter=150,
                l2_regularization=2.0,
                random_state=SEED,
            )
            clf.fit(cfg["X_train"], cfg["y_train"])
            train_time = time.time() - t_tr0
            print(f"  Trained HGB in {train_time:.2f}s.", flush=True)
            joblib.dump(clf, model_path)
        trained_models[vid] = clf

        print(f"  Evaluating {vid} on 10k validation benchmark...", flush=True)
        best_row, th_records, best_preds, best_probs, score_time = evaluate_variant(
            clf, cfg["n_feats"], s1_keys, cands_f2, val_feats, truth, total_truth_links, candidate_reachable_links
        )
        variant_preds[vid] = best_preds
        variant_probs[vid] = best_probs
        runtimes[vid] = {"train_seconds": round(train_time, 2), "score_seconds": round(score_time, 2)}

        best_row["variant"] = vid
        best_row["formula_train"] = cfg["formula_train"]
        best_row["features"] = cfg["feats_desc"]
        best_row["feature_count"] = cfg["n_feats"]
        all_best_rows.append(best_row)

        for r in th_records:
            r["variant"] = vid
            all_th_rows.append(r)

        print(
            f"  Best Threshold: {best_row['threshold']:.2f} | Macro F0.5: {best_row['macro_f05']:.5f} | Precision: {best_row['precision']:.5f} | Recall: {best_row['recall']:.5f} | TP: {best_row['tp']} | FP: {best_row['fp']} | Reachable Model FN: {best_row['reachable_model_fn']}",
            flush=True,
        )

    # Save summary tables
    df_best = pd.DataFrame(all_best_rows)
    df_best.to_csv(EXP36_DIR / "ablation_results.csv", index=False)
    print(f"\nSaved ablation results to {EXP36_DIR / 'ablation_results.csv'}")

    df_th = pd.DataFrame(all_th_rows)
    df_th.to_csv(EXP36_DIR / "threshold_results.csv", index=False)
    print(f"Saved threshold sweep results to {EXP36_DIR / 'threshold_results.csv'}")

    # Specific Error Impact Analysis (V0 vs Best Variant)
    # Find best variant by Macro F0.5
    best_v_idx = np.argmax([r["macro_f05"] for r in all_best_rows])
    best_variant_row = all_best_rows[best_v_idx]
    best_vid = best_variant_row["variant"]
    print(f"\nTop Performing Variant: {best_vid} (Macro F0.5 = {best_variant_row['macro_f05']:.5f})", flush=True)

    # Load numeric error diagnosis
    with open(EXP36_DIR / "numeric_error_analysis.json", "r", encoding="utf-8") as f:
        num_diag = json.load(f)

    # Error recovery tracking
    v0_preds = variant_preds["V0_Control"]
    best_preds = variant_preds[best_vid]

    # Collect pairs
    v0_tps = {(sid, cid) for sid in s1_keys for cid in v0_preds[sid] if cid in truth[sid]}
    v0_fps = {(sid, cid) for sid in s1_keys for cid in v0_preds[sid] if cid not in truth[sid]}

    best_tps = {(sid, cid) for sid in s1_keys for cid in best_preds[sid] if cid in truth[sid]}
    best_fps = {(sid, cid) for sid in s1_keys for cid in best_preds[sid] if cid not in truth[sid]}

    new_tps_recovered = best_tps - v0_tps
    lost_tps = v0_tps - best_tps
    new_fps_introduced = best_fps - v0_fps
    eliminated_fps = v0_fps - best_fps

    # Analyze numeric recovery
    numeric_recovered_count = 0
    # From numeric_error_analysis examples
    for data in num_diag.get("subcategories", {}).values():
        for ex in data.get("sample_examples", []):
            pair = (ex["s1_id"], ex["cand_id"])
            if pair in new_tps_recovered:
                numeric_recovered_count += 1

    # Broad numeric recovery: check how many newly recovered TPs had numeric mismatch flags
    num_conf_feat_idx = 18  # num_tok_conflict
    postal_conf_feat_idx = 37  # postal_conflict
    numeric_recovered_broad = 0
    for sid, cid in new_tps_recovered:
        c_list = cands_f2[sid][:200]
        if cid in c_list:
            c_idx = c_list.index(cid)
            feats = val_feats[sid][c_idx]
            if feats[num_conf_feat_idx] == 1.0 or feats[postal_conf_feat_idx] == 1.0:
                numeric_recovered_broad += 1

    # Strong name weak addr recovery (feat 40)
    # Weak name strong addr recovery (feat 41)
    strong_name_weak_addr_recovered = 0
    weak_name_strong_addr_recovered = 0
    for sid, cid in new_tps_recovered:
        c_list = cands_f2[sid][:200]
        if cid in c_list:
            c_idx = c_list.index(cid)
            feats = val_feats[sid][c_idx]
            if feats[40] == 1.0:
                strong_name_weak_addr_recovered += 1
            if feats[41] == 1.0:
                weak_name_strong_addr_recovered += 1

    error_recovery = {
        "baseline_variant": "V0_Control",
        "best_variant": best_vid,
        "baseline_macro_f05": all_best_rows[0]["macro_f05"],
        "best_macro_f05": best_variant_row["macro_f05"],
        "f05_gain_over_control": round(best_variant_row["macro_f05"] - all_best_rows[0]["macro_f05"], 5),
        "precision_comparison": {
            "baseline": all_best_rows[0]["precision"],
            "best": best_variant_row["precision"],
            "delta": round(best_variant_row["precision"] - all_best_rows[0]["precision"], 5),
        },
        "recall_comparison": {
            "baseline": all_best_rows[0]["recall"],
            "best": best_variant_row["recall"],
            "delta": round(best_variant_row["recall"] - all_best_rows[0]["recall"], 5),
        },
        "true_positives": {
            "baseline": all_best_rows[0]["tp"],
            "best": best_variant_row["tp"],
            "net_tp_gain": len(best_tps) - len(v0_tps),
            "new_tp_recovered": len(new_tps_recovered),
            "previous_tp_lost": len(lost_tps),
        },
        "false_positives": {
            "baseline": all_best_rows[0]["fp"],
            "best": best_variant_row["fp"],
            "net_fp_change": len(best_fps) - len(v0_fps),
            "new_fp_introduced": len(new_fps_introduced),
            "previous_fp_eliminated": len(eliminated_fps),
        },
        "specific_fn_categories_recovered": {
            "numeric_or_postal_disagreement_recovered": numeric_recovered_broad,
            "name_strong_address_weak_recovered": strong_name_weak_addr_recovered,
            "name_weak_address_strong_recovered": weak_name_strong_addr_recovered,
        },
    }

    with open(EXP36_DIR / "error_recovery.json", "w", encoding="utf-8") as f:
        json.dump(error_recovery, f, indent=2)
    print(f"Saved error recovery breakdown to {EXP36_DIR / 'error_recovery.json'}")

    runtimes["total_experiment_seconds"] = round(time.time() - t_global0, 1)
    with open(EXP36_DIR / "runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtimes, f, indent=2)

    # Recommendation / Promotion decision
    gain = best_variant_row["macro_f05"] - 0.91530
    promote = bool(gain >= 0.00300)
    best_cfg = next(c for c in variants_config if c["variant_id"] == best_vid)
    recommendation = {
        "experiment_id": "EXP-003.6",
        "control_macro_f05": 0.91530,
        "best_variant": best_vid,
        "best_macro_f05": best_variant_row["macro_f05"],
        "macro_f05_gain": round(gain, 5),
        "meets_promotion_threshold": promote,
        "promotion_decision": "YES" if promote else "NO",
        "rationale": (
            f"Variant {best_vid} achieved Macro F0.5 = {best_variant_row['macro_f05']:.5f} "
            f"(+{gain:+.5f} vs control 0.91530) with precision {best_variant_row['precision']:.5f}."
        ),
        "frozen_configuration": {
            "candidate_generator": "B5 Blocker (rare-name + rare-address union with soft country fallback)",
            "candidate_ranking": "Formula 2 (logarithmic token-IDF + postal code agreement + cross-field corroboration)",
            "candidate_cap": 200,
            "features": best_cfg["feats_desc"],
            "feature_count": best_variant_row["feature_count"],
            "training_distribution": "Formula 2 bounded hard negatives",
            "model": "HistGradientBoostingClassifier(max_leaf_nodes=127, min_samples_leaf=20, max_iter=150, l2_regularization=2.0)",
            "threshold": best_variant_row["threshold"],
            "conflict_resolution": "Unique-owner greedy assignment",
        },
        "next_direction_if_no_promotion": "RAW RETRIEVAL" if not promote else None,
    }

    with open(EXP36_DIR / "recommendation.json", "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)
    print(f"Saved final recommendation to {EXP36_DIR / 'recommendation.json'}")

    print("\n=== EXP-003.6 ABLATION SUMMARY TABLE ===")
    print(
        df_best[
            [
                "variant",
                "formula_train",
                "features",
                "threshold",
                "precision",
                "recall",
                "macro_f05",
                "tp",
                "fp",
                "reachable_model_fn",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
