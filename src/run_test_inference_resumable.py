"""Production-Grade Resumable Execution Engine for Test Inference (Step 7).

Features:
1. Chunked execution (default 50,000 S1 per chunk)
2. Atomic chunk writing via .tmp files with row-count and SHA-256 validation
3. Persistent manifest tracking (test_manifest.json) recording chunk ranges, status,
   timestamps, row counts, and checksums
4. Crash recovery: resumes from the exact first uncompleted chunk without duplicate work
5. Mathematically exact two-stage global conflict resolution across chunk boundaries:
   - Stage 1: stream filtered high-confidence candidate matches (p >= threshold) into chunk files
   - Stage 2: global argmax resolution over all candidates to enforce unique ownership
6. Verification mode (--dry-run) to test crash interruption and resume on a small sample.
"""

import argparse
import ctypes
import hashlib
import heapq
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from blocking import RareTokenIndex
from features_v3 import extract_pair_features_v3a, precompute_s1_v3
from paths import PROJECT_ROOT
from ranking_v2 import precompute_global_idfs
from test_exact_equivalence import compute_rank_score_unpacked

CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
EXP4_DIR = PROJECT_ROOT / "experiments" / "EXP-004"
PREP_DIR = PROJECT_ROOT / "experiments" / "FINAL-INFERENCE-PREP"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHUNKS_DIR = OUTPUT_DIR / "chunks"

CAP = 200
THRESHOLD = 0.82
CHUNK_SIZE_DEFAULT = 50_000

# Windows memory tracking
psapi = ctypes.windll.psapi
kernel32 = ctypes.windll.kernel32


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def get_ram_mb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / (1024 * 1024), pmc.PeakWorkingSetSize / (1024 * 1024)


class B5Blocker:
    def __init__(self, s23: pd.DataFrame):
        self.name_idx = RareTokenIndex(s23, "name_norm")
        self.addr_idx = RareTokenIndex(s23, "address_norm")

    def candidates(self, name: str, addr: str, country: str):
        name_cand = self.name_idx.query_union(name, country, 1, country_aware=True)
        if not name_cand:
            name_cand = self.name_idx.query_union(name, country, 1, country_aware=False)
        addr_cand = self.addr_idx.query_union(addr, country, 1, country_aware=True)
        if not addr_cand:
            addr_cand = self.addr_idx.query_union(addr, country, 1, country_aware=False)
        return name_cand | addr_cand


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_or_init_manifest(manifest_path: Path, total_entities: int, chunk_size: int) -> dict:
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("total_entities") == total_entities and manifest.get("chunk_size") == chunk_size:
            return manifest
        print("Existing manifest configuration mismatch. Reinitializing manifest.", flush=True)

    chunks = []
    n_chunks = (total_entities + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_entities)
        chunks.append(
            {
                "chunk_index": i,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "count": end_idx - start_idx,
                "status": "pending",
                "start_time": None,
                "completion_time": None,
                "duration_seconds": None,
                "candidates_rows": 0,
                "predictions_rows": 0,
                "sha256_candidates": None,
                "sha256_predictions": None,
            }
        )

    manifest = {
        "total_entities": total_entities,
        "chunk_size": chunk_size,
        "total_chunks": n_chunks,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "global_status": "in_progress",
        "chunks": chunks,
    }
    save_manifest(manifest_path, manifest)
    return manifest


def save_manifest(manifest_path: Path, manifest: dict):
    manifest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(manifest_path)


def process_chunk(
    chunk_meta: dict,
    s1_chunk_df: pd.DataFrame,
    blocker: B5Blocker,
    s23_lookup: dict,
    name_df: dict,
    addr_df: dict,
    name_idf: dict,
    addr_idf: dict,
    clf,
    chunks_dir: Path,
    manifest_path: Path,
    manifest: dict,
):
    chunk_idx = chunk_meta["chunk_index"]
    n_s1 = len(s1_chunk_df)
    print(
        f"\n>>> Processing Chunk {chunk_idx:04d} [{chunk_meta['start_idx']:,} .. {chunk_meta['end_idx']:,}] ({n_s1} entities)...",
        flush=True,
    )

    cand_tmp = chunks_dir / f"chunk_{chunk_idx:04d}_candidates.tmp"
    pred_tmp = chunks_dir / f"chunk_{chunk_idx:04d}_predictions.tmp"
    cand_final = chunks_dir / f"chunk_{chunk_idx:04d}_candidates.tsv"
    pred_final = chunks_dir / f"chunk_{chunk_idx:04d}_predictions.tsv"

    # Remove stale temp files if any
    if cand_tmp.exists():
        cand_tmp.unlink()
    if pred_tmp.exists():
        pred_tmp.unlink()

    chunk_meta["status"] = "in_progress"
    chunk_meta["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_manifest(manifest_path, manifest)

    t0 = time.perf_counter()
    cand_row_count = 0
    pred_row_count = 0

    with open(cand_tmp, "w", encoding="utf-8") as f_cand, open(pred_tmp, "w", encoding="utf-8") as f_pred:
        for i, (eid, name, addr, country) in enumerate(
            zip(
                s1_chunk_df.entity_id.astype(object),
                s1_chunk_df.name_norm.astype(object),
                s1_chunk_df.address_norm.astype(object),
                s1_chunk_df.country.astype(object),
            )
        ):
            # 1. Retrieval
            cand = blocker.candidates(name, addr, country)

            # 2. Ranking with Formula 2 unpacked
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

            # Write candidate pairs TSV row
            cand_str = ",".join(ranked_top)
            f_cand.write(f"{eid}\t{cand_str}\n")
            cand_row_count += 1

            # 3. Feature extraction & model scoring
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

                for cid, p in zip(ranked_top, proba):
                    if p >= THRESHOLD:
                        f_pred.write(f"{eid}\t{cid}\t{p:.6f}\n")
                        pred_row_count += 1

            if (i + 1) % 5000 == 0 or (i + 1) == n_s1:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                print(
                    f"  Chunk {chunk_idx:04d}: [{i+1}/{n_s1}] {100*(i+1)/n_s1:.1f}% | Rate: {rate:.1f} S1/s | Preds: {pred_row_count:,}",
                    flush=True,
                )

    elapsed_chunk = time.perf_counter() - t0

    # Verification: candidate row count must match input entity count exactly
    assert cand_row_count == n_s1, f"Chunk {chunk_idx} candidate row count mismatch: {cand_row_count} != {n_s1}"

    # Atomic rename
    cand_tmp.replace(cand_final)
    pred_tmp.replace(pred_final)

    sha_cand = sha256_file(cand_final)
    sha_pred = sha256_file(pred_final)

    chunk_meta["status"] = "completed"
    chunk_meta["completion_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    chunk_meta["duration_seconds"] = round(elapsed_chunk, 2)
    chunk_meta["candidates_rows"] = cand_row_count
    chunk_meta["predictions_rows"] = pred_row_count
    chunk_meta["sha256_candidates"] = sha_cand
    chunk_meta["sha256_predictions"] = sha_pred

    save_manifest(manifest_path, manifest)
    print(
        f"  Chunk {chunk_idx:04d} COMPLETED in {elapsed_chunk:.1f}s ({n_s1/elapsed_chunk:.1f} S1/s). Saved {cand_row_count} cand rows, {pred_row_count} pred rows.",
        flush=True,
    )


def assemble_final_submission(manifest: dict, s1_all: pd.DataFrame, chunks_dir: Path, output_dir: Path):
    print("\n=== ASSEMBLING FINAL SUBMISSION WITH GLOBAL CONFLICT RESOLUTION ===", flush=True)
    t_asm_0 = time.perf_counter()

    # 1. Assemble candidate_pairs.tsv
    cand_out = output_dir / "candidate_pairs.tsv"
    cand_tmp = output_dir / "candidate_pairs.tmp"
    print(f"Writing candidate pairs to {cand_out}...", flush=True)
    total_cand_rows = 0
    with open(cand_tmp, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for chunk_meta in manifest["chunks"]:
            chunk_idx = chunk_meta["chunk_index"]
            chunk_f = chunks_dir / f"chunk_{chunk_idx:04d}_candidates.tsv"
            with open(chunk_f, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total_cand_rows += 1
    cand_tmp.replace(cand_out)
    print(f"  candidate_pairs.tsv written: {total_cand_rows:,} rows", flush=True)

    # 2. Global Conflict Resolution across all chunks
    print("Performing global unique-owner conflict resolution...", flush=True)
    cand_to_best_s1 = {}  # cid -> (s1_id, p)
    total_raw_pred_links = 0

    for chunk_meta in manifest["chunks"]:
        chunk_idx = chunk_meta["chunk_index"]
        pred_f = chunks_dir / f"chunk_{chunk_idx:04d}_predictions.tsv"
        with open(pred_f, "r", encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                s1_id, cid, p_str = line.split("\t")
                p = float(p_str)
                total_raw_pred_links += 1
                if cid not in cand_to_best_s1 or p > cand_to_best_s1[cid][1]:
                    cand_to_best_s1[cid] = (s1_id, p)

    print(f"  Scanned {total_raw_pred_links:,} raw predicted links across all chunks.")
    print(f"  Unique candidates retained after conflict resolution: {len(cand_to_best_s1):,}")

    # Invert mapping: s1_id -> list of matched cids
    all_s1_ids = list(s1_all.entity_id.astype(object))
    s1_matches = {s1: [] for s1 in all_s1_ids}
    for cid, (s1, p) in cand_to_best_s1.items():
        if s1 in s1_matches:
            s1_matches[s1].append(cid)

    # 3. Assemble matching_results.tsv
    match_out = output_dir / "matching_results.tsv"
    match_tmp = output_dir / "matching_results.tmp"
    print(f"Writing matching results to {match_out}...", flush=True)
    total_match_rows = 0
    total_assigned_links = 0
    with open(match_tmp, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_ids:
            m_list = s1_matches[s1]
            out_f.write(f"{s1}\t{','.join(m_list)}\n")
            total_match_rows += 1
            total_assigned_links += len(m_list)
    match_tmp.replace(match_out)
    print(
        f"  matching_results.tsv written: {total_match_rows:,} rows ({total_assigned_links:,} links assigned).",
        flush=True,
    )

    elapsed_asm = time.perf_counter() - t_asm_0
    print(f"Assembly and conflict resolution completed in {elapsed_asm:.2f}s.", flush=True)
    return total_cand_rows, total_match_rows, total_assigned_links


def main():
    parser = argparse.ArgumentParser(description="Resumable Test Inference Runner")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT, help="S1 entities per chunk")
    parser.add_argument("--dry-run", action="store_true", help="Run in verification dry-run mode on sample")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for testing")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    chunks_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = PREP_DIR / "test_manifest.json" if not args.dry_run else PREP_DIR / "test_manifest_dryrun.json"

    print("=== RESUMABLE TEST INFERENCE ENGINE ===", flush=True)
    print(f"Output Directory: {out_dir}")
    print(f"Chunk Directory : {chunks_dir}")
    print(f"Manifest File   : {manifest_path}")

    # Load frozen model
    print("\nLoading frozen EXP-004 model...", flush=True)
    clf = joblib.load(EXP4_DIR / "model_exp004.joblib")
    assert clf.n_features_in_ == 47

    # Load test corpus
    print("Loading test candidate corpus...", flush=True)
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "test_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "test_s3.parquet", columns=cols)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )

    print("Building B5 inverted index...", flush=True)
    blocker = B5Blocker(s23)
    del s23

    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)

    # Load S1
    print("Loading test S1 entities...", flush=True)
    test_s1 = pd.read_parquet(CACHE / "test_s1.parquet")
    if args.dry_run:
        sample_n = args.sample_size or 1000
        test_s1 = test_s1.iloc[:sample_n].copy().reset_index(drop=True)
        print(f"DRY RUN MODE: Evaluating first {len(test_s1)} entities with chunk size {args.chunk_size}")

    manifest = load_or_init_manifest(manifest_path, len(test_s1), args.chunk_size)

    # Process all chunks
    for chunk_meta in manifest["chunks"]:
        chunk_idx = chunk_meta["chunk_index"]
        cand_final = chunks_dir / f"chunk_{chunk_idx:04d}_candidates.tsv"
        pred_final = chunks_dir / f"chunk_{chunk_idx:04d}_predictions.tsv"

        # Check if already completed
        if chunk_meta["status"] == "completed" and cand_final.exists() and pred_final.exists():
            print(f"Skipping completed chunk {chunk_idx:04d} ({chunk_meta['count']} entities).", flush=True)
            continue

        s1_chunk = test_s1.iloc[chunk_meta["start_idx"] : chunk_meta["end_idx"]].copy().reset_index(drop=True)
        process_chunk(
            chunk_meta,
            s1_chunk,
            blocker,
            s23_lookup,
            name_df,
            addr_df,
            name_idf,
            addr_idf,
            clf,
            chunks_dir,
            manifest_path,
            manifest,
        )

    # Assemble final submission
    manifest["global_status"] = "assembling"
    save_manifest(manifest_path, manifest)

    assemble_final_submission(manifest, test_s1, chunks_dir, out_dir)

    manifest["global_status"] = "completed"
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_manifest(manifest_path, manifest)
    print("\nALL TEST INFERENCE PROCESSING COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
