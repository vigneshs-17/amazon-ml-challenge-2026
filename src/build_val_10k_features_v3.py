"""Precompute 52-feature representation for the 10,000 S1 validation benchmark.

Extracts all 52 features (Slice V2: 42, Slice V3a: +5, Slice V3b: +5) for each
Formula 2 Cap-200 candidate pair on the exact 10,000 S1 benchmark entities.
Saves to experiments/EXP-003.6/val_10k_features_v3.joblib.
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features_v3 import ALL_FEATURE_NAMES_V3, extract_pair_features_v3, precompute_s1_v3
from paths import PROJECT_ROOT

EXP1_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
EXP25_DIR = PROJECT_ROOT / "experiments" / "EXP-002.5"
EXP36_DIR = PROJECT_ROOT / "experiments" / "EXP-003.6"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"


def load_s23():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    return pd.concat([s2, s3], ignore_index=True)


def load_val_10k():
    val_ids = pd.read_csv(EXP1_DIR / "validation_s1_ids.csv")
    s1 = pd.read_parquet(CACHE / "train_s1.parquet")
    s1_val = s1[s1.entity_id.isin(set(val_ids.entity_id))].reset_index(drop=True)
    return s1_val.iloc[:10000].copy()


def main():
    os.makedirs(EXP36_DIR, exist_ok=True)
    print("=== Precomputing 52 Features on 10,000 S1 Validation Benchmark ===", flush=True)
    t0 = time.time()

    print("Loading 10,000 validation S1 entities and Formula 2 candidates...", flush=True)
    s1_val = load_val_10k()
    cands_blob = joblib.load(EXP25_DIR / "interim_10k_candidates.joblib")
    cands_f2 = cands_blob["cands_f2_cap250"]

    print("Loading S2/S3 lookup dictionary and token frequency tables...", flush=True)
    s23 = load_s23()
    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )

    # Compute global document frequencies once
    name_df = {}
    addr_df = {}
    for n, a in zip(s23.name_norm.astype(object), s23.address_norm.astype(object)):
        if n:
            for tok in set(n.split()):
                name_df[tok] = name_df.get(tok, 0) + 1
        if a:
            for tok in set(a.split()):
                addr_df[tok] = addr_df.get(tok, 0) + 1
    del s23

    print(f"Setup complete in {time.time()-t0:.1f}s. Beginning feature extraction...", flush=True)
    t_feat0 = time.time()

    val_feats_dict = {}
    total_pairs = 0

    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1_val.entity_id.astype(object),
            s1_val.name_norm.astype(object),
            s1_val.address_norm.astype(object),
            s1_val.country.astype(object),
        )
    ):
        # Formula 2 Cap-200 candidates
        cands = cands_f2[eid][:200]
        if not cands:
            val_feats_dict[eid] = np.empty((0, len(ALL_FEATURE_NAMES_V3)), dtype=np.float32)
            continue

        s1_feat_pre = precompute_s1_v3(name, addr, country, name_df, addr_df)
        pair_matrix = np.array(
            [
                extract_pair_features_v3(
                    s1_feat_pre, s23_lookup[cid][0], s23_lookup[cid][1], s23_lookup[cid][2], name_df, addr_df
                )
                for cid in cands
            ],
            dtype=np.float32,
        )

        val_feats_dict[eid] = pair_matrix
        total_pairs += len(cands)

        if (i + 1) % 2500 == 0:
            elapsed = time.time() - t_feat0
            rate = total_pairs / elapsed
            print(f"[{i+1:>5}/10000] Processed {total_pairs} pairs ({rate:.0f} pairs/s)", flush=True)

    feat_time = time.time() - t_feat0
    print(f"\nFeature extraction complete in {feat_time:.1f}s ({total_pairs/feat_time:.0f} pairs/s).", flush=True)

    out_file = EXP36_DIR / "val_10k_features_v3.joblib"
    print(f"Saving to {out_file}...", flush=True)
    joblib.dump(val_feats_dict, out_file, compress=3)
    print(f"Saved successfully! File size: {out_file.stat().st_size / (1024*1024):.1f} MB", flush=True)


if __name__ == "__main__":
    main()
