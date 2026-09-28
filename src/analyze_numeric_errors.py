"""Deep Analysis of the 906 Numeric / Postal Disagreement False Negatives.

Subcategorizes reachable model false negatives with numeric/postal conflict:
A. exact numeric mismatch (no overlapping digits or base numbers)
B. numeric suffix variation (e.g. 12 vs 12A, 5 vs 5B)
C. formatting difference (e.g. 12-3 vs 12/3, room 1 vs 1)
D. partial/sub-building disagreement (one side has extra unit/floor/suite)
E. postal disagreement (5/6 digit postal codes conflict)
F. multiple numeric tokens with partial overlap (e.g. {12, 4, 560001} vs {12, 560001})
G. genuine strong numeric contradiction (multiple conflicting numbers with zero base overlap)

Saves to experiments/EXP-003.6/numeric_error_analysis.json.
"""

import json
import os
import re
import sys

import joblib
import pandas as pd

sys.path.insert(0, "src")
from features_v2 import ALL_FEATURE_NAMES
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP1_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
EXP25_DIR = PROJECT_ROOT / "experiments" / "EXP-002.5"
EXP3_DIR = PROJECT_ROOT / "experiments" / "EXP-003"
EXP36_DIR = PROJECT_ROOT / "experiments" / "EXP-003.6"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
THRESHOLD = 0.84

DIGIT_RE = re.compile(r"\d+")
NON_ALPHANUM_RE = re.compile(r"[^\w\s]")


def get_base_number(tok: str) -> str:
    """Extract pure digits from alphanumeric token like '12A' -> '12'."""
    m = DIGIT_RE.search(tok)
    return m.group(0) if m else ""


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
    return dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )


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


def main():
    os.makedirs(EXP36_DIR, exist_ok=True)
    print("=== Analyzing 906 Numeric / Postal Disagreement FNs ===")

    clf = joblib.load(EXP3_DIR / "model_exp003.joblib")
    c_cache = joblib.load(EXP25_DIR / "interim_10k_candidates.joblib")
    f_cache = joblib.load(EXP25_DIR / "interim_10k_features.joblib")
    cands_f2 = c_cache["cands_f2_cap250"]
    feats_f2 = f_cache["feats_f2"]

    s1_10k = load_val_10k()
    s23_lookup = load_s23_lookup()
    truth = truth_map(s1_10k)
    s1_keys = list(s1_10k.entity_id.astype(object))
    s1_info = dict(
        zip(
            s1_10k.entity_id.astype(object),
            zip(s1_10k.name_norm.astype(object), s1_10k.address_norm.astype(object), s1_10k.country.astype(object)),
        )
    )

    # Score and get predictions
    cand_probs = {}
    for eid in s1_keys:
        cands = cands_f2[eid][:200]
        feats = feats_f2[eid][:200]
        if len(cands) == 0:
            cand_probs[eid] = []
        else:
            probs = clf.predict_proba(feats)[:, 1]
            cand_probs[eid] = list(zip(cands, probs, feats))

    raw_preds = {eid: [(cid, float(p)) for cid, p, _ in cand_probs[eid] if p >= THRESHOLD] for eid in s1_keys}
    unique_preds = resolve_unique_owners(raw_preds)

    # Filter to reachable FNs with numeric or postal conflict
    num_conflict_idx = ALL_FEATURE_NAMES.index("num_tok_conflict")
    postal_conflict_idx = ALL_FEATURE_NAMES.index("postal_conflict")

    numeric_fn_cases = []
    for eid in s1_keys:
        t = truth[eid]
        p_set = unique_preds[eid]
        for rank, (cid, p, f) in enumerate(cand_probs[eid], start=1):
            if cid in t and cid not in p_set:
                has_num_conflict = f[num_conflict_idx] == 1.0
                has_postal_conflict = f[postal_conflict_idx] == 1.0
                if has_num_conflict or has_postal_conflict:
                    s1_name, s1_addr, _c1 = s1_info[eid]
                    c2_name, c2_addr, _c2 = s23_lookup[cid]
                    numeric_fn_cases.append(
                        {
                            "s1_id": eid,
                            "cand_id": cid,
                            "prob": float(p),
                            "rank": rank,
                            "s1_name": s1_name,
                            "s1_addr": s1_addr,
                            "c2_name": c2_name,
                            "c2_addr": c2_addr,
                            "has_num_conflict": has_num_conflict,
                            "has_postal_conflict": has_postal_conflict,
                            "features": f,
                        }
                    )

    print(f"Total reachable FNs with numeric/postal conflict: {len(numeric_fn_cases)}")

    # Classify each case into subcategories
    subcategories = {
        "A_exact_numeric_mismatch": [],
        "B_numeric_suffix_variation": [],
        "C_formatting_difference": [],
        "D_partial_unit_or_floor_variation": [],
        "E_postal_conflict_only": [],
        "F_multiple_numbers_partial_overlap": [],
        "G_genuine_contradiction": [],
    }

    for case in numeric_fn_cases:
        a1_toks = set(case["s1_addr"].split())
        a2_toks = set(case["c2_addr"].split())

        nums1 = {t for t in a1_toks if any(c.isdigit() for c in t)}
        nums2 = {t for t in a2_toks if any(c.isdigit() for c in t)}

        postals1 = {t for t in nums1 if len(t) in (5, 6) and t.isdigit()}
        postals2 = {t for t in nums2 if len(t) in (5, 6) and t.isdigit()}

        pure_nums1 = nums1 - postals1
        pure_nums2 = nums2 - postals2

        base_nums1 = {get_base_number(t) for t in nums1 if get_base_number(t)}
        base_nums2 = {get_base_number(t) for t in nums2 if get_base_number(t)}

        shared_tokens = a1_toks & a2_toks
        shared_nums = nums1 & nums2
        shared_base = base_nums1 & base_nums2

        meta = {
            "s1_id": case["s1_id"],
            "cand_id": case["cand_id"],
            "prob": round(case["prob"], 4),
            "rank": case["rank"],
            "s1_name": case["s1_name"],
            "cand_name": case["c2_name"],
            "s1_addr": case["s1_addr"],
            "cand_addr": case["c2_addr"],
            "nums1": list(nums1),
            "nums2": list(nums2),
            "postals1": list(postals1),
            "postals2": list(postals2),
        }

        # 1. Postal conflict only (pure building/street numbers agree or no other numbers)
        if case["has_postal_conflict"] and not case["has_num_conflict"]:
            subcategories["E_postal_conflict_only"].append(meta)
        # 2. Multiple numeric tokens with partial overlap
        elif len(shared_nums) > 0 and (nums1 != nums2):
            subcategories["F_multiple_numbers_partial_overlap"].append(meta)
        # 3. Base numbers agree (suffix variation e.g. 12 vs 12A, or plot 5 vs 5B)
        elif len(shared_base) > 0 and len(shared_nums) == 0:
            subcategories["B_numeric_suffix_variation"].append(meta)
        # 4. Formatting difference (e.g. hyphenated or slash numbers like 12-1 vs 12/1)
        elif any("-" in t or "/" in t for t in (nums1 | nums2)):
            subcategories["C_formatting_difference"].append(meta)
        # 5. One side has extra unit/floor/suite/room number
        elif abs(len(pure_nums1) - len(pure_nums2)) >= 1 and (len(pure_nums1) == 0 or len(pure_nums2) == 0):
            subcategories["D_partial_unit_or_floor_variation"].append(meta)
        # 6. Exact numeric mismatch with high text overlap
        elif len(shared_tokens - nums1 - nums2) >= 3:
            subcategories["A_exact_numeric_mismatch"].append(meta)
        # 7. Genuine contradiction (different street number and different text)
        else:
            subcategories["G_genuine_contradiction"].append(meta)

    counts = {k: len(v) for k, v in subcategories.items()}
    pcts = {k: round(100.0 * len(v) / len(numeric_fn_cases), 2) for k, v in subcategories.items()}

    print("\n--- Numeric Subcategory Breakdown ---")
    for k, v in counts.items():
        print(f"  {k:<42}: {v:>4d} ({pcts[k]:>5.2f}%)")

    report = {
        "total_numeric_postal_reachable_fn": len(numeric_fn_cases),
        "pct_of_all_reachable_fn": round(100.0 * len(numeric_fn_cases) / 3654, 2),
        "subcategories": {
            k: {
                "count": len(v),
                "pct": pcts[k],
                "sample_examples": v[:5],
            }
            for k, v in subcategories.items()
        },
        "evidence_based_recommendations": [
            "1. Implement base-number matching (stripping alphanumeric suffixes like 12A -> 12). Directly targets Subcategory B (suffix variation).",
            "2. Implement numeric partial overlap ratio: len(shared_nums) / min(len(nums1), len(nums2)). Directly targets Subcategory F (multiple numbers with partial overlap).",
            "3. Implement address alpha similarity without numbers: RapidFuzz token_set_ratio on address after stripping all digits. Distinguishes Subcategory A/D (street/city agree, unit differs) from Subcategory G (genuine contradiction).",
            "4. Implement postal partial agreement: 3-digit prefix agreement on 5/6 digit postal codes (matching postal district). Directly targets Subcategory E.",
        ],
    }

    with open(EXP36_DIR / "numeric_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved numeric error analysis to {EXP36_DIR / 'numeric_error_analysis.json'}")


if __name__ == "__main__":
    main()
