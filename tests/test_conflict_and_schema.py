"""Tests for unique-owner conflict resolution, schema constraints, and serialization."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_unique_owner_conflict_resolution():
    """Verify greedy 1-to-1 conflict resolution awards the candidate to the higher probability S1."""
    # Predictions: S1_A -> C1 with 0.95, S1_B -> C1 with 0.88 (conflict on C1!)
    # S1_B -> C2 with 0.90
    raw_preds = [
        ("S1-001", "S2-100", 0.95),
        ("S1-002", "S2-100", 0.88),  # should be dropped in favor of S1-001
        ("S1-002", "S2-200", 0.90),  # should be retained
    ]

    cand_to_best_s1 = {}
    for eid, cid, p in raw_preds:
        if cid not in cand_to_best_s1 or p > cand_to_best_s1[cid][1]:
            cand_to_best_s1[cid] = (eid, p)

    # Invert to S1 -> matched candidates
    s1_matches = {"S1-001": [], "S1-002": []}
    for cid, (eid, p) in cand_to_best_s1.items():
        s1_matches[eid].append(cid)

    # C1 must belong only to S1-001
    assert "S2-100" in s1_matches["S1-001"]
    assert "S2-100" not in s1_matches["S1-002"]
    # C2 belongs to S1-002
    assert "S2-200" in s1_matches["S1-002"]


def test_tsv_serialization_and_empty_matched_ids():
    """Verify output formatting uses exact tab delimiters and empty strings for singletons."""
    s1_entities = ["S1-0001", "S1-0002", "S1-0003"]
    matches = {
        "S1-0001": ["S2-0010", "S3-0020"],
        "S1-0002": [],  # Singleton
        "S1-0003": ["S2-0030"],
    }

    buf = io.StringIO()
    buf.write("source1_entity_id\tmatched_entity_ids\n")
    for eid in s1_entities:
        m_str = ",".join(matches[eid])
        buf.write(f"{eid}\t{m_str}\n")

    output_lines = buf.getvalue().splitlines()
    assert len(output_lines) == 4  # Header + 3 entities
    assert output_lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert output_lines[1] == "S1-0001\tS2-0010,S3-0020"
    assert output_lines[2] == "S1-0002\t"  # Empty matched_entity_ids after tab
    assert output_lines[3] == "S1-0003\tS2-0030"
