"""Production Country-Partitioned Multi-Worker Test Inference Runner.

Partitions the 1,732,544 test S1 entities strictly by country:
1. France : 259,452 S1 entities -> 1,434,993 France candidates (~2.0 GB RAM)
2. US     : 663,106 S1 entities -> 3,817,031 US candidates (~5.5 GB RAM)
3. India  : 809,986 S1 entities -> 4,717,565 India candidates (~6.8 GB RAM)

Total combined RAM across all 3 workers: ~14.3 GB (safely within 24 GB physical RAM).
Aggregate throughput: ~115 - 125 S1/sec.
Total runtime: ~3.8 to 4.2 hours -> Finishes before 9:30 PM IST.
"""

import argparse
import hashlib
import heapq
import multiprocessing as mp
import os
import shutil
import sys
import time
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

CAP_DEFAULT = 100
THRESHOLD = 0.82
CHUNK_SIZE_DEFAULT = 50_000


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


def country_worker(country_name: str, cap: int, chunk_size: int):
    print(f"[{country_name}] Worker process started (PID: {os.getpid()})...", flush=True)

    # 1. Load model
    clf = joblib.load(EXP4_DIR / "model_exp004.joblib")

    # 2. Load country-specific candidate corpus
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    t0_setup = time.perf_counter()
    s2 = pd.read_parquet(CACHE / "test_s2.parquet", columns=cols)
    s2 = s2[s2.country == country_name].copy()
    s3 = pd.read_parquet(CACHE / "test_s3.parquet", columns=cols)
    s3 = s3[s3.country == country_name].copy()
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )

    blocker = B5Blocker(s23)
    del s23

    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    t_setup = time.perf_counter() - t0_setup
    print(
        f"[{country_name}] Candidate index initialized in {t_setup:.1f}s ({len(s23_lookup):,} candidate records).",
        flush=True,
    )

    # 3. Load country S1 entities
    test_s1 = pd.read_parquet(CACHE / "test_s1.parquet")
    s1_country = test_s1[test_s1.country == country_name].copy().reset_index(drop=True)
    del test_s1
    n_total_country = len(s1_country)

    n_chunks = (n_total_country + chunk_size - 1) // chunk_size
    print(f"[{country_name}] Total entities: {n_total_country:,} across {n_chunks} chunks (cap={cap}).", flush=True)

    t0_worker = time.perf_counter()

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_total_country)
        s1_chunk = s1_country.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        n_s1 = len(s1_chunk)

        cand_final = CHUNKS_DIR / f"chunk_{country_name}_{chunk_idx:04d}_candidates.tsv"
        pred_final = CHUNKS_DIR / f"chunk_{country_name}_{chunk_idx:04d}_predictions.tsv"
        cand_tmp = CHUNKS_DIR / f"chunk_{country_name}_{chunk_idx:04d}_candidates.tmp"
        pred_tmp = CHUNKS_DIR / f"chunk_{country_name}_{chunk_idx:04d}_predictions.tmp"

        # Checkpoint check
        if cand_final.exists() and pred_final.exists():
            print(f"[{country_name}] Chunk {chunk_idx:04d} already completed. Skipping.", flush=True)
            continue

        cand_tmp.unlink(missing_ok=True)
        pred_tmp.unlink(missing_ok=True)

        t0_chunk = time.perf_counter()
        cand_rows = 0
        pred_rows = 0

        with open(cand_tmp, "w", encoding="utf-8") as f_cand, open(pred_tmp, "w", encoding="utf-8") as f_pred:
            for i, (eid, name, addr, country) in enumerate(
                zip(
                    s1_chunk.entity_id.astype(object),
                    s1_chunk.name_norm.astype(object),
                    s1_chunk.address_norm.astype(object),
                    s1_chunk.country.astype(object),
                )
            ):
                cand = blocker.candidates(name, addr, country)
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

                if len(cand_list) > cap:
                    ranked_top = heapq.nlargest(cap, cand_list, key=lambda c: (scores[c], c))
                else:
                    ranked_top = sorted(cand_list, key=lambda c: (-scores[c], c))

                f_cand.write(f"{eid}\t{','.join(ranked_top)}\n")
                cand_rows += 1

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
                            pred_rows += 1

                if (i + 1) % 10000 == 0:
                    el = time.perf_counter() - t0_chunk
                    print(
                        f"[{country_name}] Chunk {chunk_idx:04d}: [{i+1}/{n_s1}] | Rate: {(i+1)/el:.1f} S1/s | Preds: {pred_rows:,}",
                        flush=True,
                    )

        assert cand_rows == n_s1
        cand_tmp.replace(cand_final)
        pred_tmp.replace(pred_final)

        chunk_time = time.perf_counter() - t0_chunk
        total_time_so_far = time.perf_counter() - t0_worker
        processed_total = end_idx
        overall_rate = processed_total / total_time_so_far
        eta_min = (n_total_country - processed_total) / max(overall_rate, 0.1) / 60.0

        print(
            f"[{country_name}] Chunk {chunk_idx:04d} DONE ({n_s1} S1 in {chunk_time:.1f}s, {n_s1/chunk_time:.1f} S1/s). "
            f"Progress: {processed_total:,}/{n_total_country:,} ({100*processed_total/n_total_country:.1f}%) | "
            f"Overall Rate: {overall_rate:.1f} S1/s | ETA: {eta_min:.1f}m",
            flush=True,
        )

    print(
        f"[{country_name}] ALL {n_total_country:,} ENTITIES COMPLETED in {(time.perf_counter() - t0_worker)/60:.1f}m!",
        flush=True,
    )


def assemble_all_countries(test_s1_all: pd.DataFrame, chunks_dir: Path, output_dir: Path):
    print("\n=== FINAL GLOBAL SUBMISSION ASSEMBLY & CONFLICT RESOLUTION ===", flush=True)
    t0 = time.perf_counter()

    # 1. Assemble candidate_pairs.tsv in exact test_s1 original order
    cand_file = output_dir / "candidate_pairs.tsv"
    cand_tmp = output_dir / "candidate_pairs.tmp"
    match_file = output_dir / "matching_results.tsv"
    match_tmp = output_dir / "matching_results.tmp"

    print("Loading candidate pair chunks...", flush=True)
    s1_to_candidates = {}
    for f in chunks_dir.glob("*_candidates.tsv"):
        with open(f, "r", encoding="utf-8") as in_f:
            for line in in_f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                eid, cands = line.split("\t")
                s1_to_candidates[eid] = cands

    print(f"  Loaded {len(s1_to_candidates):,} candidate rows.")

    print("Resolving global conflicts across all predictions...", flush=True)
    cand_to_best_s1 = {}
    n_raw_preds = 0
    for f in chunks_dir.glob("*_predictions.tsv"):
        with open(f, "r", encoding="utf-8") as in_f:
            for line in in_f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                eid, cid, p_str = line.split("\t")
                p = float(p_str)
                n_raw_preds += 1
                if cid not in cand_to_best_s1 or p > cand_to_best_s1[cid][1]:
                    cand_to_best_s1[cid] = (eid, p)

    print(f"  Scanned {n_raw_preds:,} predictions. Retained {len(cand_to_best_s1):,} unique matches.")

    all_s1_ids = list(test_s1_all.entity_id.astype(object))
    s1_matches = {s1: [] for s1 in all_s1_ids}
    for cid, (s1, p) in cand_to_best_s1.items():
        if s1 in s1_matches:
            s1_matches[s1].append(cid)

    # Write output files
    print(f"Writing {cand_file}...", flush=True)
    with open(cand_tmp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in all_s1_ids:
            c_str = s1_to_candidates.get(eid, "")
            f.write(f"{eid}\t{c_str}\n")
    cand_tmp.replace(cand_file)
    print(f"Writing {match_file}...", flush=True)
    with open(match_tmp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in all_s1_ids:
            m_str = ",".join(s1_matches[eid])
            f.write(f"{eid}\t{m_str}\n")
    match_tmp.replace(match_file)

    # Also copy to output/final/
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cand_file, final_dir / "candidate_pairs.tsv")
    shutil.copyfile(match_file, final_dir / "matching_results.tsv")
    print(f"Copied final files to {final_dir}.", flush=True)

    print(f"Assembly completed in {time.perf_counter() - t0:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Parallel Country Test Inference")
    parser.add_argument("--cap", type=int, default=CAP_DEFAULT, help="Candidate cap (default 100)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT, help="Chunk size")
    parser.add_argument("--countries", nargs="+", default=["France", "US", "India"], help="Countries to process")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "final").mkdir(parents=True, exist_ok=True)

    print("=== EMERGENCY PARALLEL COUNTRY TEST INFERENCE ===", flush=True)
    print(f"Candidate Cap : {args.cap}")
    print(f"Chunk Size    : {args.chunk_size}")
    print(f"Countries     : {args.countries}")

    # Launch country workers
    processes = []
    countries = args.countries

    t0_start = time.perf_counter()

    for c in countries:
        p = mp.Process(target=country_worker, args=(c, args.cap, args.chunk_size), name=f"Worker-{c}")
        p.start()
        processes.append(p)
        print(f"Spawned {p.name} (PID: {p.pid}).", flush=True)

    # Monitor processes
    for p in processes:
        p.join()
        print(f"{p.name} exited with code {p.exitcode}.", flush=True)
        assert p.exitcode == 0, f"Process {p.name} failed!"

    t_inference = time.perf_counter() - t0_start
    print(
        f"\nALL 3 COUNTRY WORKERS COMPLETED IN {t_inference/60:.1f} MINUTES! Aggregate speed: {1732544/t_inference:.1f} S1/sec",
        flush=True,
    )

    # Assemble
    test_s1_full = pd.read_parquet(CACHE / "test_s1.parquet")
    assemble_all_countries(test_s1_full, CHUNKS_DIR, OUTPUT_DIR)
    print("\nALL INFERENCE AND ASSEMBLY FINISHED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
