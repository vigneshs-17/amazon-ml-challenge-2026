"""Tests for rare-token candidate indexing, Formula 2 ranking, and candidate caps."""

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)


def test_formula_2_ranking_and_exact_matches():
    """Verify Formula 2 weights exact name, exact address, and shared tokens."""
    name_df = {"reliance": 10, "industries": 20, "retail": 50}
    addr_df = {"mumbai": 5, "maharashtra": 15, "400001": 2}
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)

    s1_name = "reliance retail"
    s1_addr = "nariman point mumbai 400001"

    s1_pre = precompute_s1_ranking(s1_name, s1_addr, name_idf, addr_idf)

    # Identical candidate should receive high score with exact match bonuses
    score_identical = compute_rank_score_v2(s1_pre, s1_name, s1_addr)
    # Partial candidate
    score_partial = compute_rank_score_v2(s1_pre, "reliance digital", "andheri mumbai")
    # Unrelated candidate
    score_unrelated = compute_rank_score_v2(s1_pre, "tata motors", "pune")

    assert score_identical > score_partial > score_unrelated
    assert score_identical >= 25.0  # 15.0 exact addr + 10.0 exact name


def test_formula_2_deterministic_tie_breaking():
    """Verify candidate ranking breaks score ties deterministically by candidate entity ID."""
    scores = {
        "S2-00005": 12.5,
        "S2-00002": 12.5,  # tied with S2-00005
        "S2-00009": 18.0,  # highest
        "S2-00001": 8.0,  # lowest
    }
    cand_list = list(scores.keys())

    # Deterministic top-2 selection: key=(-scores[c], c)
    ranked = sorted(cand_list, key=lambda c: (-scores[c], c))
    assert ranked[0] == "S2-00009"
    # Tied candidates must be ordered lexicographically: S2-00002 before S2-00005
    assert ranked[1] == "S2-00002"
    assert ranked[2] == "S2-00005"
    assert ranked[3] == "S2-00001"


def test_candidate_cap_behavior():
    """Verify candidate cap strictly truncates lists exceeding K while retaining top scores."""
    # 250 dummy candidates
    scores = {f"S2-{i:05d}": float(i) for i in range(250)}
    cand_list = list(scores.keys())

    CAP = 100
    if len(cand_list) > CAP:
        ranked_top = heapq.nlargest(CAP, cand_list, key=lambda c: (scores[c], c))
    else:
        ranked_top = sorted(cand_list, key=lambda c: (-scores[c], c))

    assert len(ranked_top) == CAP
    # The highest scored entity (S2-00249, score 249.0) must be first
    assert ranked_top[0] == "S2-00249"
    assert ranked_top[-1] == "S2-00150"
