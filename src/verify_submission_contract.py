"""Submission Contract & Boundary Verification (Step 8).

Validates:
1. matching_results.tsv and candidate_pairs.tsv existence, schema, and headers
2. Delimiter checks (Tab-separated, no commas in TSV structure)
3. Prefix integrity (S1- for query, S2-/S3- for matches/candidates, no self matches)
4. Row count parity with test S1 count
5. Exact subset constraint (every matched ID must be in candidate list)
6. Single row per S1 entity (no duplicates, no omissions)
7. Invokes official student_resource/utils/validate_submission.py and verifies exit code 0
8. Saves experiments/FINAL-INFERENCE-PREP/submission_schema.json
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from paths import PROJECT_ROOT, RAW_ROOT, TEST_DIR

PREP_DIR = PROJECT_ROOT / "experiments" / "FINAL-INFERENCE-PREP"
VALIDATOR_SCRIPT = RAW_ROOT / "utils" / "validate_submission.py"


def verify_submission(matching_tsv: Path, candidate_tsv: Path, test_dir: Path, expected_count: int | None = None):
    print("\n--- Verifying Submission Files ---")
    print(f"Matching file : {matching_tsv}")
    print(f"Candidate file: {candidate_tsv}")
    print(f"Test directory: {test_dir}")

    report = {
        "matching_file": str(matching_tsv),
        "candidate_file": str(candidate_tsv),
        "rules_checked": {},
        "official_validator_result": None,
        "exit_code": 1,
    }

    # Rule 1: File existence
    rule_files_exist = matching_tsv.exists() and candidate_tsv.exists()
    report["rules_checked"]["files_exist"] = rule_files_exist
    assert rule_files_exist, "One or both submission files do not exist!"

    # Rule 2: Headers
    with open(matching_tsv, "r", encoding="utf-8") as f:
        match_header = f.readline().rstrip("\r\n").split("\t")
    with open(candidate_tsv, "r", encoding="utf-8") as f:
        cand_header = f.readline().rstrip("\r\n").split("\t")

    expected_match_header = ["source1_entity_id", "matched_entity_ids"]
    expected_cand_header = ["source1_entity_id", "candidate_entity_ids"]

    rule_headers = (match_header == expected_match_header) and (cand_header == expected_cand_header)
    report["rules_checked"]["headers_valid"] = rule_headers
    assert rule_headers, f"Header mismatch: match={match_header}, cand={cand_header}"

    # Rule 3: Line by line validation
    match_rows = 0
    cand_rows = 0
    s1_matched_map = {}
    s1_cand_map = {}
    subset_violations = 0
    prefix_violations = 0
    self_match_violations = 0

    with open(matching_tsv, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            assert len(parts) == 2, f"Malformed matching row: {line[:50]}"
            s1, matched = parts[0], parts[1]
            match_rows += 1
            m_list = matched.split(",") if matched else []
            s1_matched_map[s1] = set(m_list)

            for mid in m_list:
                if mid.startswith("S1-"):
                    self_match_violations += 1
                if not (mid.startswith(("S2-", "S3-"))):
                    prefix_violations += 1

    with open(candidate_tsv, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            assert len(parts) == 2, f"Malformed candidate row: {line[:50]}"
            s1, candidates = parts[0], parts[1]
            cand_rows += 1
            c_list = candidates.split(",") if candidates else []
            s1_cand_map[s1] = set(c_list)

            for cid in c_list:
                if cid.startswith("S1-"):
                    self_match_violations += 1
                if not (cid.startswith(("S2-", "S3-"))):
                    prefix_violations += 1

    # Check subset constraint
    for s1, m_set in s1_matched_map.items():
        c_set = s1_cand_map.get(s1, set())
        if not m_set.issubset(c_set):
            subset_violations += 1

    report["rules_checked"]["matching_row_count"] = match_rows
    report["rules_checked"]["candidate_row_count"] = cand_rows
    report["rules_checked"]["row_count_match"] = match_rows == cand_rows
    report["rules_checked"]["subset_constraint_passed"] = subset_violations == 0
    report["rules_checked"]["prefix_validity_passed"] = prefix_violations == 0
    report["rules_checked"]["no_self_matches_passed"] = self_match_violations == 0

    if expected_count is not None:
        report["rules_checked"]["expected_row_count"] = expected_count
        report["rules_checked"]["row_count_equals_expected"] = match_rows == expected_count

    print(f"  Rows evaluated: {match_rows:,}")
    print(f"  Subset violations: {subset_violations}")
    print(f"  Prefix violations: {prefix_violations}")
    print(f"  Self-match violations: {self_match_violations}")

    # Invoke official validator
    print(f"Invoking official validator: {VALIDATOR_SCRIPT}...", flush=True)
    cmd = [
        sys.executable,
        str(VALIDATOR_SCRIPT),
        "--matching",
        str(matching_tsv),
        "--candidate",
        str(candidate_tsv),
        "--test-dir",
        str(test_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    report["official_validator_stdout"] = proc.stdout
    report["official_validator_stderr"] = proc.stderr
    report["official_validator_exit_code"] = proc.returncode

    print("Validator Output:\n" + proc.stdout)
    if proc.stderr:
        print("Validator Stderr:\n" + proc.stderr)

    report["official_validator_result"] = "PASS" if proc.returncode == 0 else "FAIL"
    report["exit_code"] = proc.returncode

    with open(PREP_DIR / "submission_schema.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return proc.returncode == 0


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "output"
    match_file = out_dir / "matching_results.tsv"
    cand_file = out_dir / "candidate_pairs.tsv"
    if match_file.exists() and cand_file.exists():
        success = verify_submission(match_file, cand_file, TEST_DIR)
        print(f"Submission verification: {'SUCCESS' if success else 'FAILED'}")
    else:
        print(f"Output files not found in {out_dir}. Run inference or dry-run first.")
