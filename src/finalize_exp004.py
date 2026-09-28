"""Finalize EXP-004 summary artifacts and update experiments.csv."""

import json
import time

import pandas as pd

from features_v3 import FEATURE_NAMES_V3A
from paths import PROJECT_ROOT

EXP3_DIR = PROJECT_ROOT / "experiments" / "EXP-003"
EXP4_DIR = PROJECT_ROOT / "experiments" / "EXP-004"
EXP_CSV = PROJECT_ROOT / "experiments" / "experiments.csv"


def main():
    # Load threshold results
    df_th = pd.read_csv(EXP4_DIR / "threshold_results.csv")
    best_row = df_th.loc[df_th["macro_f0_5"].idxmax()]
    best_th = float(best_row["threshold"])

    # Load error breakdown
    with open(EXP4_DIR / "error_breakdown.json", "r", encoding="utf-8") as f:
        error_breakdown = json.load(f)

    # Load EXP-003 reference
    with open(EXP3_DIR / "evaluation.json", "r", encoding="utf-8") as f:
        exp3_eval = json.load(f)

    total_truth_links = error_breakdown["total_ground_truth_links"]
    f05_delta = round(float(best_row["macro_f0_5"]) - exp3_eval["official_macro_f0_5"], 5)
    prec_delta = round(float(best_row["macro_precision"]) - exp3_eval["macro_precision"], 5)
    rec_delta = round(float(best_row["macro_recall"]) - exp3_eval["macro_recall"], 5)
    tp_delta = int(best_row["tp"]) - exp3_eval["tp"]
    fp_delta = int(best_row["fp"]) - exp3_eval["fp"]
    total_fn_delta = int(best_row["total_fn"]) - (total_truth_links - exp3_eval["tp"])
    reachable_fn_delta = (
        int(best_row["reachable_model_fn"]) - exp3_eval["error_breakdown"]["C_model_false_negative_miss"]
    )

    # Generalization comparison
    comparison = {
        "exp003_reference": {
            "macro_f0_5": exp3_eval["official_macro_f0_5"],
            "macro_precision": exp3_eval["macro_precision"],
            "macro_recall": exp3_eval["macro_recall"],
            "tp": exp3_eval["tp"],
            "fp": exp3_eval["fp"],
            "total_fn": total_truth_links - exp3_eval["tp"],
            "reachable_model_fn": exp3_eval["error_breakdown"]["C_model_false_negative_miss"],
            "threshold": exp3_eval["best_threshold"],
        },
        "exp004_result": {
            "macro_f0_5": float(best_row["macro_f0_5"]),
            "macro_precision": float(best_row["macro_precision"]),
            "macro_recall": float(best_row["macro_recall"]),
            "tp": int(best_row["tp"]),
            "fp": int(best_row["fp"]),
            "total_fn": int(best_row["total_fn"]),
            "reachable_model_fn": int(best_row["reachable_model_fn"]),
            "threshold": best_th,
        },
        "deltas": {
            "f0_5_delta": f05_delta,
            "precision_delta": prec_delta,
            "recall_delta": rec_delta,
            "tp_delta": tp_delta,
            "fp_delta": fp_delta,
            "total_fn_delta": total_fn_delta,
            "reachable_model_fn_delta": reachable_fn_delta,
        },
        "generalization": {
            "stepping_stone_10k_gain": 0.00288,
            "full_100k_gain": f05_delta,
            "generalized": bool(f05_delta > 0),
            "champion": "EXP-004" if float(best_row["macro_f0_5"]) > exp3_eval["official_macro_f0_5"] else "EXP-003",
        },
    }
    with open(EXP4_DIR / "generalization_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Configuration
    with open(EXP4_DIR / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "EXP-004",
                "description": "Full 100k confirmation of V2: Formula 2 Training + 47 Features (Tolerant Numeric/Postal) + Leafwise HGB",
                "candidate_generator": "Deterministic Formula 2 B5 Blocker",
                "candidate_cap": 200,
                "training_candidate_generator": "Formula 2",
                "training_s1": "100,000 stratified training S1 entities (Seed 2026)",
                "features": "47 features (Features V2 [42] + 5 Tolerant Numeric/Postal features)",
                "feature_count": 47,
                "model": "HistGradientBoostingClassifier(max_depth=None, max_leaf_nodes=127, min_samples_leaf=20, max_iter=150, l2_regularization=2.0)",
                "conflict_resolution": "Unique-owner greedy assignment",
                "random_seed": 2026,
                "validation_split": "100,000 S1 validation entities (345,938 ground-truth links)",
            },
            f,
            indent=2,
        )

    # Feature list
    with open(EXP4_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_count": len(FEATURE_NAMES_V3A),
                "feature_names": FEATURE_NAMES_V3A,
                "slice_v2_count": 42,
                "tolerant_numeric_count": 5,
            },
            f,
            indent=2,
        )

    # Runtime
    runtime_data = {
        "model_train_seconds": 0.0,
        "setup_seconds": 252.0,
        "scoring_seconds": 4499.4,
        "total_job_seconds": 4765.0,
        "peak_ram_mb": 18232.3,
        "final_ram_mb": 8266.8,
    }
    with open(EXP4_DIR / "runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime_data, f, indent=2)

    # Evaluation summary
    eval_summary = {
        "experiment_id": "EXP-004",
        "validation_split": "100k_stratified_s1",
        "candidate_cap": 200,
        "candidate_pairs_scored": 18558349,
        "best_threshold": best_th,
        "unique_owner_applied": True,
        "official_macro_f0_5": float(best_row["macro_f0_5"]),
        "macro_precision": float(best_row["macro_precision"]),
        "macro_recall": float(best_row["macro_recall"]),
        "f0_5_singletons": float(best_row["singleton_f0_5"]),
        "f0_5_non_singletons": float(best_row["non_singleton_f0_5"]),
        "tp": int(best_row["tp"]),
        "fp": int(best_row["fp"]),
        "total_fn": int(best_row["total_fn"]),
        "reachable_model_fn": int(best_row["reachable_model_fn"]),
        "raw_b5_blocking_recall": 0.95489,
        "cap200_candidate_recall": 0.94148,
        "error_breakdown": error_breakdown,
        "runtimes": runtime_data,
    }
    with open(EXP4_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Update experiments.csv
    exp_row = {
        "experiment_id": "EXP-004",
        "timestamp": time.strftime("%Y-%m-%d"),
        "description": "B5 + Formula 2 + Cap-200 + 47 feats (V2 + Tolerant Numeric) + Leafwise HGB (127 leaves, l2=2.0)",
        "validation_split": "100k_stratified_s1",
        "validation_f0_5": float(best_row["macro_f0_5"]),
        "macro_precision": float(best_row["macro_precision"]),
        "macro_recall": float(best_row["macro_recall"]),
        "f0_5_singletons": float(best_row["singleton_f0_5"]),
        "f0_5_non_singletons": float(best_row["non_singleton_f0_5"]),
        "blocking_recall": 0.94148,
        "candidate_pairs": 18558349.0,
        "avg_candidates_per_s1": 185.58,
        "median_candidates_per_s1": 200.0,
        "p95_candidates_per_s1": 200.0,
        "max_candidates_per_s1": 200.0,
        "candidate_reduction_ratio": 0.999982,
        "runtime_seconds": 4765.0,
        "random_seed": 2026.0,
        "model": "HistGradientBoostingClassifier(max_leaf_nodes=127, l2=2.0, 47 feats)",
        "threshold": float(best_th),
        "notes": f"Macro F0.5={float(best_row['macro_f0_5']):.5f} (Formula 2 Training + 47 Feats)",
    }
    df_exp = pd.read_csv(EXP_CSV)
    df_exp = df_exp[df_exp.experiment_id != "EXP-004"]
    df_exp = pd.concat([df_exp, pd.DataFrame([exp_row])], ignore_index=True)
    df_exp.to_csv(EXP_CSV, index=False)
    print("EXP-004 finalized successfully and written to experiments.csv!")


if __name__ == "__main__":
    main()
