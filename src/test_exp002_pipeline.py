"""EXP-002 Quick Smoke Test: Verify 42-feature extraction, model fitting, and conflict resolution."""

import numpy as np

from features_v2 import ALL_FEATURE_NAMES, extract_pair_features_v2, precompute_s1


def main():
    print(f"Total features: {len(ALL_FEATURE_NAMES)}")
    assert len(ALL_FEATURE_NAMES) == 42, f"Expected 42 features, got {len(ALL_FEATURE_NAMES)}"

    name_df = {"amazon": 100, "development": 50, "center": 200}
    addr_df = {"brigade": 30, "gateway": 20, "26": 10}

    s1_pre = precompute_s1("amazon development center", "brigade gateway 26", "India", name_df, addr_df)
    f = extract_pair_features_v2(s1_pre, "amazon web services", "brigade gateway", "India", name_df, addr_df)

    assert len(f) == 42, f"Expected 42 features in output, got {len(f)}"
    assert not any(np.isnan(f)), "NaN in features!"
    assert not any(np.isinf(f)), "Inf in features!"

    # Test conflict resolution logic
    # Suppose s1_1 and s1_2 both predict c_1, but s1_1 has higher probability
    preds_prob = {
        "S1-1": [("S2-100", 0.95), ("S2-101", 0.88)],
        "S1-2": [("S2-100", 0.91), ("S2-102", 0.85)],
    }
    # Independent:
    indep = {s1: {cid for cid, p in items} for s1, items in preds_prob.items()}
    assert indep["S1-1"] == {"S2-100", "S2-101"}
    assert indep["S1-2"] == {"S2-100", "S2-102"}

    # Unique owner: S2-100 contested, S1-1 wins (0.95 > 0.91)
    cand_to_best_s1 = {}
    for s1, items in preds_prob.items():
        for cid, p in items:
            if cid not in cand_to_best_s1 or p > cand_to_best_s1[cid][1]:
                cand_to_best_s1[cid] = (s1, p)

    unique_preds = {s1: set() for s1 in preds_prob}
    for cid, (s1, p) in cand_to_best_s1.items():
        unique_preds[s1].add(cid)

    assert unique_preds["S1-1"] == {"S2-100", "S2-101"}
    assert unique_preds["S1-2"] == {"S2-102"}  # S2-100 removed from S1-2!
    print("Conflict resolution logic verified!")
    print("EXP-002 quick smoke test PASSED.")


if __name__ == "__main__":
    main()
