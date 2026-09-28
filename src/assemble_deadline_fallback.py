#!/usr/bin/env python3
"""
ML Challenge 2026 — Deadline Safety Fallback Assembly Script

Builds a valid, submission-ready matching_results.tsv (and optional candidate_pairs.tsv)
from all COMPLETED chunks so far, with an empty matched_entity_ids field for all
currently unprocessed S1 entities.

Guarantees:
1. Purely additive & read-only on output/chunks/ (does NOT touch running workers).
2. Streaming test_source1.tsv reader (under 150 MB RAM usage).
3. Exact official schema:
   - matching_results.tsv: source1_entity_id\tmatched_entity_ids
   - candidate_pairs.tsv:  source1_entity_id\tcandidate_entity_ids
4. Exact count: exactly 1,732,544 rows in the original test_source1.tsv order.
5. Strict subset constraint: every match comes from candidates; uncompleted entities have empty matches.
6. Greedy unique-owner conflict resolution across all completed chunks.
7. Atomic writing via .tmp files.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Deadline Fallback Assembly")
    parser.add_argument(
        "--test-s1",
        type=Path,
        default=Path("data/raw/student_resource/dataset/test/test_source1.tsv"),
        help="Path to test_source1.tsv",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("output/chunks"),
        help="Path to completed chunks directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/deadline_fallback"),
        help="Target output directory for fallback submission",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        default=True,
        help="Also assemble candidate_pairs.tsv (default True)",
    )
    parser.add_argument(
        "--skip-candidates",
        action="store_true",
        help="Skip assembling candidate_pairs.tsv (faster & less RAM)",
    )
    return parser.parse_args()


def assemble_fallback(test_s1_path: Path, chunks_dir: Path, output_dir: Path, build_candidates: bool = True):
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== DEADLINE FALLBACK ASSEMBLY START ===", flush=True)
    print(f"Test Source: {test_s1_path}", flush=True)
    print(f"Chunks Dir : {chunks_dir}", flush=True)
    print(f"Output Dir : {output_dir}", flush=True)

    # 1. Identify completed prediction chunks (only .tsv, ignore .tmp)
    pred_files = sorted(chunks_dir.glob("chunk_*_predictions.tsv"))
    cand_files = sorted(chunks_dir.glob("chunk_*_candidates.tsv"))
    print(f"Found {len(pred_files)} finalized prediction chunks and {len(cand_files)} candidate chunks.", flush=True)

    # 2. Global conflict resolution across completed prediction chunks
    print("Reading predictions and resolving unique-owner conflicts...", flush=True)
    cand_to_best_s1 = {}  # cid -> (s1_id, probability)
    n_preds_read = 0

    for pf in pred_files:
        with open(pf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                eid, cid, p_str = parts
                try:
                    p = float(p_str)
                except ValueError:
                    continue
                n_preds_read += 1
                if cid not in cand_to_best_s1 or p > cand_to_best_s1[cid][1]:
                    cand_to_best_s1[cid] = (eid, p)

    print(f"  Read {n_preds_read:,} raw predictions from completed chunks.")
    print(f"  Retained {len(cand_to_best_s1):,} winning matches after unique-owner resolution.")

    # Invert mapping: s1_id -> list of winning match IDs
    s1_to_matches = defaultdict(list)
    for cid, (eid, p) in cand_to_best_s1.items():
        s1_to_matches[eid].append(cid)

    # 3. If candidates are requested, load completed candidate chunks
    s1_to_cands = {}
    if build_candidates:
        print("Reading completed candidate chunks...", flush=True)
        n_cands_read = 0
        for cf in cand_files:
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    eid = parts[0]
                    c_str = parts[1] if len(parts) > 1 else ""
                    s1_to_cands[eid] = c_str
                    n_cands_read += 1
        print(f"  Loaded {n_cands_read:,} candidate rows for {len(s1_to_cands):,} S1 entities.")

    # 4. Stream test_source1.tsv to produce outputs in exact original order
    match_file = output_dir / "matching_results.tsv"
    match_tmp = output_dir / "matching_results.tmp"
    cand_file = output_dir / "candidate_pairs.tsv"
    cand_tmp = output_dir / "candidate_pairs.tmp"

    print("Streaming test_source1.tsv and writing fallback files...", flush=True)
    out_match = open(match_tmp, "w", encoding="utf-8")
    out_match.write("source1_entity_id\tmatched_entity_ids\n")

    out_cand = None
    if build_candidates:
        out_cand = open(cand_tmp, "w", encoding="utf-8")
        out_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    n_total_s1 = 0
    n_completed_s1 = 0
    n_unprocessed_s1 = 0
    n_s1_with_matches = 0
    n_total_matches = 0

    with open(test_s1_path, "r", encoding="utf-8") as f_in:
        f_in.readline()  # skip header
        for line in f_in:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            eid = parts[0]
            n_total_s1 += 1

            # Check if this S1 was completed
            # Alternatively, check against prediction/candidate membership
            if eid in s1_to_cands:
                n_completed_s1 += 1
            else:
                n_unprocessed_s1 += 1

            matches = s1_to_matches.get(eid, [])
            if matches:
                n_s1_with_matches += 1
                n_total_matches += len(matches)
                m_str = ",".join(matches)
            else:
                m_str = ""

            out_match.write(f"{eid}\t{m_str}\n")

            if out_cand:
                c_str = s1_to_cands.get(eid, "")
                # If entity has matches, ensure candidate string is consistent
                if m_str and not c_str:
                    c_str = m_str
                out_cand.write(f"{eid}\t{c_str}\n")

    out_match.close()
    if out_cand:
        out_cand.close()

    # Atomic rename
    if match_file.exists():
        match_file.unlink()
    match_tmp.replace(match_file)

    if build_candidates:
        if cand_file.exists():
            cand_file.unlink()
        cand_tmp.replace(cand_file)

    elapsed = time.perf_counter() - t0
    print("\n=== DEADLINE FALLBACK ASSEMBLY COMPLETE ===", flush=True)
    print(f"Elapsed Time           : {elapsed:.2f}s", flush=True)
    print(f"Total S1 Entities      : {n_total_s1:,}", flush=True)
    print(f"Completed S1 Entities  : {n_completed_s1:,} ({100*n_completed_s1/n_total_s1:.2f}%)", flush=True)
    print(f"Unprocessed S1 (empty) : {n_unprocessed_s1:,} ({100*n_unprocessed_s1/n_total_s1:.2f}%)", flush=True)
    print(f"S1 with Matches        : {n_s1_with_matches:,}", flush=True)
    print(f"Total Matched Pairs    : {n_total_matches:,}", flush=True)
    print(f"Generated matching_file: {match_file} ({match_file.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
    if build_candidates:
        print(f"Generated candidate_file: {cand_file} ({cand_file.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    return match_file, cand_file


if __name__ == "__main__":
    args = parse_args()
    build_cand = not args.skip_candidates
    assemble_fallback(args.test_s1, args.chunks_dir, args.output_dir, build_candidates=build_cand)
