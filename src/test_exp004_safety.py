"""Smoke and throughput test for EXP-004 scoring on 200 entities."""

import heapq
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, "src")
from features_v3 import extract_pair_features_v3a, precompute_s1_v3
from paths import PROJECT_ROOT
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)
from run_exp004 import B5Blocker, get_ram_mb, load_s23, load_val


def main():
    print("Loading model and metadata...", flush=True)
    clf = joblib.load(PROJECT_ROOT / "experiments" / "EXP-003.6" / "model_v2_tolerant_numeric.joblib")
    assert clf.n_features_in_ == 47

    s23 = load_s23()
    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )
    blocker = B5Blocker(s23)
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    del s23

    s1_val = load_val().iloc[:200]
    print("Testing 200 entities...", flush=True)

    t0 = time.time()
    n_pairs = 0
    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1_val.entity_id.astype(object),
            s1_val.name_norm.astype(object),
            s1_val.address_norm.astype(object),
            s1_val.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country)
        s1_rank_pre = precompute_s1_ranking(name, addr, name_idf, addr_idf)
        s1_feat_pre = precompute_s1_v3(name, addr, country, name_df, addr_df)
        cand_list = list(cand)
        scores = {c: compute_rank_score_v2(s1_rank_pre, s23_lookup[c][0], s23_lookup[c][1]) for c in cand_list}
        if len(cand_list) > 200:
            ranked = heapq.nlargest(200, cand_list, key=lambda c: (scores[c], c))
        else:
            ranked = sorted(cand_list, key=lambda c: (-scores[c], c))
        n_pairs += len(ranked)
        if ranked:
            feats = np.array(
                [
                    extract_pair_features_v3a(
                        s1_feat_pre, s23_lookup[c][0], s23_lookup[c][1], s23_lookup[c][2], name_df, addr_df
                    )
                    for c in ranked
                ],
                dtype=np.float32,
            )
            clf.predict_proba(feats)[:, 1]

    dt = time.time() - t0
    cur_ram, peak_ram = get_ram_mb()
    s1_rate = len(s1_val) / dt
    pair_rate = n_pairs / dt
    projected_min = (100_000 / s1_rate) / 60.0

    print("\n--- BENCHMARK RESULTS (200 S1) ---")
    print(f"Time: {dt:.2f}s")
    print(f"Throughput: {s1_rate:.2f} S1/sec | {pair_rate:.1f} candidate pairs/sec")
    print(f"RAM: Working Set = {cur_ram:.1f} MB | Peak = {peak_ram:.1f} MB")
    print(f"Projected 100k Scoring Time: {projected_min:.1f} minutes")


if __name__ == "__main__":
    main()
