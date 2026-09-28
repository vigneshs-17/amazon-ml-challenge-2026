"""India Reverse Worker for Accelerated Test Inference.

Processes India test chunks in REVERSE order (Chunk 16 down to Chunk 0).
Works in parallel with Worker-India-Forward (Chunk 0 up to Chunk 16).
When they cross in the middle, both detect completed chunks and safely terminate.
Cuts India remaining time by 50%!
"""

import heapq
import os
import sys
import time

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
OUTPUT_DIR = PROJECT_ROOT / "output"
CHUNKS_DIR = OUTPUT_DIR / "chunks"

CAP = 100
THRESHOLD = 0.82
CHUNK_SIZE = 50_000
COUNTRY_NAME = "India"


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


def main():
    print(f"[India-Reverse] Worker process started (PID: {os.getpid()})...", flush=True)

    clf = joblib.load(EXP4_DIR / "model_exp004.joblib")

    cols = ["entity_id", "name_norm", "address_norm", "country"]
    t0_setup = time.perf_counter()
    s2 = pd.read_parquet(CACHE / "test_s2.parquet", columns=cols)
    s2 = s2[s2.country == COUNTRY_NAME].copy()
    s3 = pd.read_parquet(CACHE / "test_s3.parquet", columns=cols)
    s3 = s3[s3.country == COUNTRY_NAME].copy()
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
        f"[India-Reverse] Candidate index initialized in {t_setup:.1f}s ({len(s23_lookup):,} candidate records).",
        flush=True,
    )

    test_s1 = pd.read_parquet(CACHE / "test_s1.parquet")
    s1_country = test_s1[test_s1.country == COUNTRY_NAME].copy().reset_index(drop=True)
    del test_s1
    n_total_country = len(s1_country)

    n_chunks = (n_total_country + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(
        f"[India-Reverse] Total entities: {n_total_country:,} across {n_chunks} chunks (Processing in REVERSE order from {n_chunks-1} down to 0).",
        flush=True,
    )

    t0_worker = time.perf_counter()

    for chunk_idx in reversed(range(n_chunks)):
        cand_final = CHUNKS_DIR / f"chunk_{COUNTRY_NAME}_{chunk_idx:04d}_candidates.tsv"
        pred_final = CHUNKS_DIR / f"chunk_{COUNTRY_NAME}_{chunk_idx:04d}_predictions.tsv"
        cand_tmp = CHUNKS_DIR / f"chunk_{COUNTRY_NAME}_{chunk_idx:04d}_candidates.rev.tmp"
        pred_tmp = CHUNKS_DIR / f"chunk_{COUNTRY_NAME}_{chunk_idx:04d}_predictions.rev.tmp"

        # Check if already completed by forward worker or previous run
        if cand_final.exists() and pred_final.exists():
            print(f"[India-Reverse] Chunk {chunk_idx:04d} already completed. Skipping.", flush=True)
            continue

        cand_tmp.unlink(missing_ok=True)
        pred_tmp.unlink(missing_ok=True)

        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, n_total_country)
        s1_chunk = s1_country.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        n_s1 = len(s1_chunk)

        print(
            f"[India-Reverse] >>> Starting Chunk {chunk_idx:04d} [{start_idx:,} .. {end_idx:,}] ({n_s1} entities)...",
            flush=True,
        )

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

                if len(cand_list) > CAP:
                    ranked_top = heapq.nlargest(CAP, cand_list, key=lambda c: (scores[c], c))
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
                        f"[India-Reverse] Chunk {chunk_idx:04d}: [{i+1}/{n_s1}] | Rate: {(i+1)/el:.1f} S1/s | Preds: {pred_rows:,}",
                        flush=True,
                    )

        assert cand_rows == n_s1
        cand_tmp.replace(cand_final)
        pred_tmp.replace(pred_final)

        chunk_time = time.perf_counter() - t0_chunk
        print(
            f"[India-Reverse] Chunk {chunk_idx:04d} DONE ({n_s1} S1 in {chunk_time:.1f}s, {n_s1/chunk_time:.1f} S1/s).",
            flush=True,
        )

    print(f"[India-Reverse] COMPLETED ALL ASSIGNED CHUNKS in {(time.perf_counter() - t0_worker)/60:.1f}m!", flush=True)


if __name__ == "__main__":
    main()
