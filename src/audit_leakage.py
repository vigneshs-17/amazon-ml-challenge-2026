"""Leakage / synthetic-artifact checks. Observation only — nothing here is
exploited in a pipeline; we just document what's found.
"""

import json

import numpy as np
import pandas as pd

from paths import INTERIM, REPORTS


def numeric_suffix(ids):
    return ids.str.extract(r"-(\d+)$")[0].astype("Int64")


def main():
    rep = {}

    # 1. entity_id numeric-suffix structure: any ordering/encoding signal?
    for split in ("train", "test"):
        for s in (1, 2, 3):
            df = pd.read_parquet(INTERIM / f"{split}_s{s}.parquet", columns=["entity_id", "country"])
            n = numeric_suffix(df.entity_id.astype(object))
            rep[f"{split}_s{s}_id_suffix"] = {
                "min": int(n.min()),
                "max": int(n.max()),
                "n_unique": int(n.nunique()),
                "n_rows": len(df),
                "looks_sequential": bool(n.nunique() == len(df)),
                "digit_length_distribution": df.entity_id.astype(object)
                .str.slice(3)
                .str.len()
                .value_counts()
                .to_dict(),
            }
            # correlation between id and row order (Spearman via rank corr) -
            # tests whether original file order encodes id/country info
            order = np.arange(len(df))
            rank_corr = pd.Series(n.astype(float).values).corr(pd.Series(order, dtype=float), method="spearman")
            rep[f"{split}_s{s}_id_suffix"]["row_order_vs_id_spearman"] = (
                None if pd.isna(rank_corr) else round(float(rank_corr), 4)
            )
            # is country contiguous in file order (block-sorted)?
            c = df.country.astype(object)
            change_pts = int((c != c.shift()).sum())
            rep[f"{split}_s{s}_id_suffix"]["country_blocks_in_file_order"] = change_pts

    # 2. exact normalized (name, address) overlap between train and test, per source
    for s in (1, 2, 3):
        tr = pd.read_parquet(
            INTERIM / f"train_s{s}.parquet", columns=["entity_id", "norm_name", "norm_addr", "country"]
        )
        te = pd.read_parquet(INTERIM / f"test_s{s}.parquet", columns=["entity_id", "norm_name", "norm_addr", "country"])
        tr_key = set(zip(tr.norm_name.astype(object), tr.norm_addr.astype(object)))
        te_key = pd.Series(list(zip(te.norm_name.astype(object), te.norm_addr.astype(object))))
        overlap = te_key.isin(tr_key)
        rep[f"s{s}_train_test_exact_norm_overlap"] = {
            "test_rows": len(te),
            "test_rows_with_exact_train_twin": int(overlap.sum()),
            "pct": round(100 * float(overlap.mean()), 3),
        }

    # 3. duplicate S1 entities (same normalized name+address) within train / within test
    for split in ("train", "test"):
        s1 = pd.read_parquet(
            INTERIM / f"{split}_s1.parquet", columns=["entity_id", "norm_name", "norm_addr", "country"]
        )
        dup = s1.duplicated(["norm_name", "norm_addr", "country"], keep=False)
        rep[f"{split}_s1_near_duplicate_entities"] = {
            "rows_in_a_duplicate_group": int(dup.sum()),
            "distinct_groups": int(s1[dup].groupby(["norm_name", "norm_addr", "country"]).ngroups),
        }

    json.dump(rep, open(REPORTS / "audit_leakage.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
