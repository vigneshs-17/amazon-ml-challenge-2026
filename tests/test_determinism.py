"""Tests ensuring strict determinism across candidate ranking and feature extraction."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features_v3 import extract_pair_features_v3a, precompute_s1_v3
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)


def test_ranking_determinism():
    """Verify Formula 2 ranking scores and sorted candidate ordering are 100% identical across runs."""
    name_df = {"alpha": 10, "beta": 50, "gamma": 100}
    addr_df = {"street": 20, "city": 30, "12345": 5}
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)

    candidates = [
        ("S2-001", "alpha store", "street city 12345"),
        ("S2-002", "beta store", "street city 12345"),
        ("S2-003", "gamma store", "street city"),
        ("S2-004", "alpha store", "different street"),
    ]

    def run_rank():
        s1_pre = precompute_s1_ranking("alpha store", "street city 12345", name_idf, addr_idf)
        scores = {}
        for cid, name2, addr2 in candidates:
            scores[cid] = compute_rank_score_v2(s1_pre, name2, addr2)
        ranked = sorted(scores.keys(), key=lambda c: (-scores[c], c))
        return ranked, [scores[c] for c in ranked]

    ranked1, scores1 = run_rank()
    ranked2, scores2 = run_rank()

    assert ranked1 == ranked2
    assert scores1 == scores2


def test_feature_extraction_determinism():
    """Verify 47-feature extraction vectors are identical across multiple runs."""
    name_df = {"global": 10, "logistics": 40}
    addr_df = {"port": 15, "road": 25, "110001": 2}

    def run_extract():
        s1_pre = precompute_s1_v3("global logistics", "port road 110001", "India", name_df, addr_df)
        feats = extract_pair_features_v3a(s1_pre, "global logistics pvt", "port road 110001", "India", name_df, addr_df)
        return np.asarray(feats, dtype=np.float32)

    feats1 = run_extract()
    feats2 = run_extract()

    np.testing.assert_array_equal(feats1, feats2)
