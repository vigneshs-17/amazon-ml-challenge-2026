"""EXP-000 step 3: ground-truth integrity + match-count distribution."""

import json

import pandas as pd

from io_utils import explode_ground_truth, load_ground_truth
from paths import INTERIM, REPORTS


def main():
    gt = load_ground_truth()
    s1 = pd.read_parquet(INTERIM / "train_s1.parquet", columns=["entity_id", "country"])
    s2_ids = set(pd.read_parquet(INTERIM / "train_s2.parquet", columns=["entity_id"]).entity_id.astype(object))
    s3_ids = set(pd.read_parquet(INTERIM / "train_s3.parquet", columns=["entity_id"]).entity_id.astype(object))
    s1_ids = set(s1.entity_id.astype(object))
    rep = {}

    gids = gt.source1_entity_id.astype(object)
    rep["gt_rows"] = len(gt)
    rep["duplicate_source1_entity_id"] = int(gids.duplicated().sum())
    rep["gt_s1_not_in_source1"] = len(set(gids) - s1_ids)
    rep["source1_not_in_gt"] = len(s1_ids - set(gids))
    rep["gt_s1_bad_prefix"] = int((~gids.str.startswith("S1-")).sum())

    lists = gt.matched_entity_ids.astype(object).map(lambda x: [t for t in x.split(",")] if x else [])
    rep["lists_with_blank_items"] = int(lists.map(lambda lst: any(t.strip() == "" for t in lst)).sum())
    rep["lists_with_whitespace_items"] = int(lists.map(lambda lst: any(t != t.strip() for t in lst)).sum())
    rep["lists_with_internal_duplicates"] = int(lists.map(lambda lst: len(lst) != len(set(lst))).sum())

    long = explode_ground_truth(gt)
    mid = long.matched_id
    rep["total_positive_pairs"] = len(long)
    rep["matched_ids_prefix_counts"] = mid.str.slice(0, 3).value_counts().to_dict()
    rep["matched_ids_referencing_S1"] = int(mid.str.startswith("S1-").sum())
    is2, is3 = mid.str.startswith("S2-"), mid.str.startswith("S3-")
    rep["unknown_S2_ids"] = int((is2 & ~mid.isin(s2_ids)).sum())
    rep["unknown_S3_ids"] = int((is3 & ~mid.isin(s3_ids)).sum())

    # Is each S2/S3 record assigned to at most one S1 entity?
    owners = long.groupby("matched_id").source1_entity_id.nunique()
    rep["s2s3_records_matched_to_multiple_s1"] = int((owners > 1).sum())
    rep["distinct_matched_records"] = len(owners)
    rep["s2_records_total"] = len(s2_ids)
    rep["s3_records_total"] = len(s3_ids)
    rep["s2_records_never_matched"] = len(s2_ids - set(mid[is2]))
    rep["s3_records_never_matched"] = len(s3_ids - set(mid[is3]))

    # Distribution of matches per S1
    n = lists.map(len)
    n2 = lists.map(lambda lst: sum(t.startswith("S2-") for t in lst))
    n3 = lists.map(lambda lst: sum(t.startswith("S3-") for t in lst))
    len(n)
    rep["match_count"] = {
        "zero": int((n == 0).sum()),
        "zero_pct": round(100 * (n == 0).mean(), 3),
        "one": int((n == 1).sum()),
        "two": int((n == 2).sum()),
        "three_plus": int((n >= 3).sum()),
        "max": int(n.max()),
        "mean": round(float(n.mean()), 4),
        "median": float(n.median()),
        "mean_nonzero": round(float(n[n > 0].mean()), 4),
        "histogram": {int(k): int(v) for k, v in n.value_counts().sort_index().items()},
    }
    rep["match_source_mix"] = {
        "none": int(((n2 == 0) & (n3 == 0)).sum()),
        "s2_only": int(((n2 > 0) & (n3 == 0)).sum()),
        "s3_only": int(((n2 == 0) & (n3 > 0)).sum()),
        "both": int(((n2 > 0) & (n3 > 0)).sum()),
        "s2_count_hist": {int(k): int(v) for k, v in n2.value_counts().sort_index().items()},
        "s3_count_hist": {int(k): int(v) for k, v in n3.value_counts().sort_index().items()},
    }
    # by country
    g = pd.DataFrame({"id": gids, "n": n, "n2": n2, "n3": n3}).merge(
        s1.astype({"entity_id": object}), left_on="id", right_on="entity_id", how="left"
    )
    rep["by_country"] = {
        c: {
            "s1": len(d),
            "singleton_pct": round(100 * (d.n == 0).mean(), 3),
            "mean_matches": round(float(d.n.mean()), 3),
            "mean_s2": round(float(d.n2.mean()), 3),
            "mean_s3": round(float(d.n3.mean()), 3),
            "hist": {int(k): int(v) for k, v in d.n.value_counts().sort_index().head(12).items()},
        }
        for c, d in g.groupby("country")
    }
    # Country consistency across a positive pair
    json.dump(rep, open(REPORTS / "audit_ground_truth.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
