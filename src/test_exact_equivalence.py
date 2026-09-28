"""Mandatory Exact Equivalence Test (Step 5).

Runs BOTH:
A. Original frozen EXP-004 inference implementation
B. Optimized inference implementation

Across 1,000 validation S1 entities and compares:
1. Candidate IDs and exact candidate ordering
2. Formula-2 scores for all candidates
3. All 47 feature values for all candidate pairs
4. Model predicted probabilities
5. Candidates passing threshold 0.82
6. Final predicted links after unique-owner conflict resolution

Saves equivalence_report.json under experiments/FINAL-INFERENCE-PREP/.
"""

import heapq
import json
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, "src")
from features_v3 import extract_pair_features_v3a, precompute_s1_v3
from paths import PROJECT_ROOT
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)
from run_exp004 import B5Blocker, load_s23, load_val

EXP1_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
EXP4_DIR = PROJECT_ROOT / "experiments" / "EXP-004"
PREP_DIR = PROJECT_ROOT / "experiments" / "FINAL-INFERENCE-PREP"

CAP = 200
SEED = 2026
THRESHOLD = 0.82
N_TEST = 1000

# --- Optimized Formula-2 Ranking Function ---


def compute_rank_score_unpacked(s1_name, s1_addr, t1, a1, name_idfs, addr_idfs, s1_postals, len_t1, len_a1, n2, a2):
    """Optimized Formula-2 rank score avoiding repeated dictionary lookups.

    Preserves exact arithmetic operation order to guarantee bitwise score equivalence.
    """
    score = 0.0
    if s1_addr and s1_addr == a2:
        score += 15.0
    if s1_name and s1_name == n2:
        score += 10.0

    t2 = set(n2.split()) if n2 else set()
    a2_set = set(a2.split()) if a2 else set()

    shared_n = t1 & t2
    shared_a = a1 & a2_set

    sh_n_idf = sum(name_idfs[t] for t in shared_n) if shared_n else 0.0
    sh_a_idf = sum(addr_idfs[t] for t in shared_a) if shared_a else 0.0
    score += sh_n_idf + sh_a_idf

    # Postal code match
    if s1_postals and (s1_postals & a2_set):
        score += 5.0

    # Cross-field corroboration
    if shared_n and shared_a:
        score += 4.0

    # Token Jaccard overlap bonuses
    if shared_n:
        u_n_len = len_t1 + len(t2) - len(shared_n)
        if u_n_len > 0:
            score += 3.0 * (len(shared_n) / u_n_len)

    if shared_a:
        u_a_len = len_a1 + len(a2_set) - len(shared_a)
        if u_a_len > 0:
            score += 3.0 * (len(shared_a) / u_a_len)

    return score


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


def main():
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    print("=== STEP 5: MANDATORY EXACT EQUIVALENCE TEST (1,000 Validation S1) ===", flush=True)

    # Load frozen model
    clf = joblib.load(EXP4_DIR / "model_exp004.joblib")
    assert clf.n_features_in_ == 47

    # Load corpus
    print("Loading S2/S3 candidate corpus...", flush=True)
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

    # Load fixed 1,000 validation entities
    s1_val = load_val().iloc[:N_TEST].copy()
    print(f"Validation sample size: {len(s1_val)} S1 entities", flush=True)

    # --- Run Pipeline A (Baseline Frozen Implementation) ---
    print("\n[A] Running Baseline Frozen EXP-004 Pipeline...", flush=True)
    t0 = time.perf_counter()
    res_A_cands = {}
    res_A_scores = {}
    res_A_feats = {}
    res_A_proba = {}
    res_A_preds = {}

    for eid, name, addr, country in zip(
        s1_val.entity_id.astype(object),
        s1_val.name_norm.astype(object),
        s1_val.address_norm.astype(object),
        s1_val.country.astype(object),
    ):
        cand = blocker.candidates(name, addr, country)
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

        res_A_cands[eid] = ranked_top
        res_A_scores[eid] = [scores[c] for c in ranked_top]

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
            res_A_feats[eid] = feats
            res_A_proba[eid] = proba
            res_A_preds[eid] = [(cid, float(p)) for cid, p in zip(ranked_top, proba) if p >= THRESHOLD]
        else:
            res_A_feats[eid] = np.empty((0, 47), dtype=np.float32)
            res_A_proba[eid] = np.empty((0,), dtype=np.float32)
            res_A_preds[eid] = []

    res_A_unique = resolve_unique_owners(res_A_preds)
    t_A = time.perf_counter() - t0
    print(f"  Baseline Pipeline completed in {t_A:.2f}s ({len(s1_val)/t_A:.2f} S1/s)", flush=True)

    # --- Run Pipeline B (Optimized Implementation) ---
    print("\n[B] Running Optimized EXP-004 Pipeline...", flush=True)
    t0 = time.perf_counter()
    res_B_cands = {}
    res_B_scores = {}
    res_B_feats = {}
    res_B_proba = {}
    res_B_preds = {}

    for eid, name, addr, country in zip(
        s1_val.entity_id.astype(object),
        s1_val.name_norm.astype(object),
        s1_val.address_norm.astype(object),
        s1_val.country.astype(object),
    ):
        cand = blocker.candidates(name, addr, country)

        # Unpacked S1 ranking variables
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

        res_B_cands[eid] = ranked_top
        res_B_scores[eid] = [scores[c] for c in ranked_top]

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
            res_B_feats[eid] = feats
            res_B_proba[eid] = proba
            res_B_preds[eid] = [(cid, float(p)) for cid, p in zip(ranked_top, proba) if p >= THRESHOLD]
        else:
            res_B_feats[eid] = np.empty((0, 47), dtype=np.float32)
            res_B_proba[eid] = np.empty((0,), dtype=np.float32)
            res_B_preds[eid] = []

    res_B_unique = resolve_unique_owners(res_B_preds)
    t_B = time.perf_counter() - t0
    print(f"  Optimized Pipeline completed in {t_B:.2f}s ({len(s1_val)/t_B:.2f} S1/s)", flush=True)

    # --- Compare Exact Equivalence ---
    print("\n--- Verifying Exact Mathematical Equivalence ---", flush=True)
    cand_id_mismatches = 0
    cand_order_mismatches = 0
    formula2_score_mismatches = 0
    max_score_diff = 0.0
    feature_val_mismatches = 0
    max_feat_diff = 0.0
    proba_mismatches = 0
    max_proba_diff = 0.0
    threshold_mismatches = 0
    final_pred_mismatches = 0

    for eid in s1_val.entity_id.astype(object):
        # 1. Candidate IDs and order
        c_A = res_A_cands[eid]
        c_B = res_B_cands[eid]
        if set(c_A) != set(c_B):
            cand_id_mismatches += 1
        if c_A != c_B:
            cand_order_mismatches += 1

        # 2. Formula-2 scores
        s_A = res_A_scores[eid]
        s_B = res_B_scores[eid]
        for vA, vB in zip(s_A, s_B):
            diff = abs(vA - vB)
            max_score_diff = max(max_score_diff, diff)
            if diff > 1e-9:
                formula2_score_mismatches += 1

        # 3. Features
        f_A = res_A_feats[eid]
        f_B = res_B_feats[eid]
        if len(f_A) > 0:
            diff_mat = np.abs(f_A - f_B)
            max_d = float(diff_mat.max())
            max_feat_diff = max(max_feat_diff, max_d)
            if max_d > 1e-5:
                feature_val_mismatches += int((diff_mat > 1e-5).sum())

        # 4. Probabilities
        p_A = res_A_proba[eid]
        p_B = res_B_proba[eid]
        if len(p_A) > 0:
            p_diff = np.abs(p_A - p_B)
            max_pd = float(p_diff.max())
            max_proba_diff = max(max_proba_diff, max_pd)
            if max_pd > 1e-7:
                proba_mismatches += int((p_diff > 1e-7).sum())

        # 5. Threshold decisions
        preds_A_set = {cid for cid, p in res_A_preds[eid]}
        preds_B_set = {cid for cid, p in res_B_preds[eid]}
        if preds_A_set != preds_B_set:
            threshold_mismatches += 1

        # 6. Final predicted links (after conflict resolution)
        final_A = res_A_unique.get(eid, set())
        final_B = res_B_unique.get(eid, set())
        if final_A != final_B:
            final_pred_mismatches += 1

    equiv_pass = bool(
        cand_id_mismatches == 0
        and cand_order_mismatches == 0
        and formula2_score_mismatches == 0
        and feature_val_mismatches == 0
        and proba_mismatches == 0
        and threshold_mismatches == 0
        and final_pred_mismatches == 0
    )

    report = {
        "validation_entities_tested": len(s1_val),
        "candidate_id_mismatches": cand_id_mismatches,
        "candidate_order_mismatches": cand_order_mismatches,
        "formula2_score_mismatches": formula2_score_mismatches,
        "max_formula2_score_difference": max_score_diff,
        "feature_value_mismatches": feature_val_mismatches,
        "max_feature_value_difference": max_feat_diff,
        "probability_mismatches": proba_mismatches,
        "max_probability_difference": max_proba_diff,
        "threshold_decision_mismatches": threshold_mismatches,
        "final_predicted_link_mismatches": final_pred_mismatches,
        "equivalence_result": "PASS" if equiv_pass else "FAIL",
        "timing": {
            "baseline_seconds": round(t_A, 2),
            "optimized_seconds": round(t_B, 2),
            "speedup": round(t_A / t_B, 2),
            "baseline_s1_per_sec": round(len(s1_val) / t_A, 2),
            "optimized_s1_per_sec": round(len(s1_val) / t_B, 2),
        },
    }

    with open(PREP_DIR / "equivalence_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== EQUIVALENCE TEST SUMMARY ===")
    print(f"Validation S1 Tested         : {len(s1_val)}")
    print(f"Candidate ID Mismatches      : {cand_id_mismatches}")
    print(f"Candidate Order Mismatches   : {cand_order_mismatches}")
    print(f"Formula-2 Score Mismatches   : {formula2_score_mismatches} (Max diff: {max_score_diff:.2e})")
    print(f"Feature Value Mismatches     : {feature_val_mismatches} (Max diff: {max_feat_diff:.2e})")
    print(f"Probability Mismatches       : {proba_mismatches} (Max diff: {max_proba_diff:.2e})")
    print(f"Threshold Decision Mismatches: {threshold_mismatches}")
    print(f"Final Prediction Mismatches  : {final_pred_mismatches}")
    print(f"RESULT                       : {'PASS' if equiv_pass else 'FAIL'}")
    print(
        f"TIMING: Baseline {t_A:.2f}s ({len(s1_val)/t_A:.1f} S1/s) -> Optimized {t_B:.2f}s ({len(s1_val)/t_B:.1f} S1/s) | SPEEDUP = {t_A/t_B:.2f}x"
    )

    assert equiv_pass, "EQUIVALENCE TEST FAILED! Diagnostic required."


if __name__ == "__main__":
    main()
