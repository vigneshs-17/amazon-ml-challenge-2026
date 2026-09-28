"""EXP-001 Step 2: reproducible, entity-level, stratified validation split.

Splits train S1 entities (never pairs) by a fixed seed, stratified by
(country, match_count_bucket), so singleton rate and multi-match difficulty
are matched between validation and the remainder. All ground-truth links
for a validation S1 stay attached to it (no pair-level splitting). Saves the
exact validation S1 id list so every later experiment reuses the same split.
"""

import json

import numpy as np
import pandas as pd

from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

SEED = 42
N_VAL = 100_000
OUT_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bucket(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    return 3  # "3+"


def main():
    cache = PROJECT_ROOT / "data" / "interim" / "exp001" / "train_s1.parquet"
    s1 = pd.read_parquet(cache, columns=["entity_id", "country"])
    gt = load_ground_truth()
    n_matches = gt.matched_entity_ids.astype(object).map(lambda x: 0 if not x else len(x.split(",")))
    gt_df = pd.DataFrame({"entity_id": gt.source1_entity_id.astype(object), "n_matches": n_matches})
    s1 = s1.merge(gt_df, on="entity_id", how="left")
    assert s1.n_matches.isna().sum() == 0, "every train S1 must have a ground-truth row"
    s1["bucket"] = s1.n_matches.map(bucket)
    s1["strata"] = s1.country.astype(str) + "|" + s1.bucket.astype(str)

    rng = np.random.default_rng(SEED)
    val_parts = []
    frac = N_VAL / len(s1)
    for strata, grp in s1.groupby("strata"):
        k = max(1, round(len(grp) * frac)) if len(grp) > 0 else 0
        k = min(k, len(grp))
        idx = rng.choice(grp.index.values, size=k, replace=False)
        val_parts.append(grp.loc[idx])
    val = pd.concat(val_parts)
    # trim/pad to exactly N_VAL if rounding drifted, staying within-strata
    if len(val) > N_VAL:
        val = val.sample(N_VAL, random_state=SEED)
    val_ids = set(val.entity_id.tolist())
    train_remaining = s1[~s1.entity_id.isin(val_ids)]

    # report
    long = explode_ground_truth(gt)
    val_links = long[long.source1_entity_id.astype(object).isin(val_ids)]
    rep = {
        "seed": SEED,
        "target_val_size": N_VAL,
        "actual_val_size": len(val),
        "train_remaining_size": len(train_remaining),
        "val_country_dist": val.country.value_counts().to_dict(),
        "full_country_dist": s1.country.value_counts().to_dict(),
        "val_singleton_pct": round(100 * float((val.n_matches == 0).mean()), 3),
        "full_singleton_pct": round(100 * float((s1.n_matches == 0).mean()), 3),
        "val_match_count_hist": val.n_matches.value_counts().sort_index().to_dict(),
        "val_bucket_dist": val.bucket.value_counts().sort_index().to_dict(),
        "val_true_link_count": len(val_links),
        "no_overlap_check": len(val_ids & set(train_remaining.entity_id.tolist())) == 0,
    }
    json.dump(rep, open(OUT_DIR / "validation_split_report.json", "w", encoding="utf-8"), indent=1, default=str)
    val[["entity_id", "country", "n_matches", "bucket"]].to_csv(OUT_DIR / "validation_s1_ids.csv", index=False)
    train_remaining[["entity_id"]].to_csv(OUT_DIR / "train_remaining_s1_ids.csv", index=False)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("hist") and not k.endswith("dist")}, indent=1))
    print("val_country_dist:", rep["val_country_dist"])
    print("val_bucket_dist:", rep["val_bucket_dist"])


if __name__ == "__main__":
    main()
