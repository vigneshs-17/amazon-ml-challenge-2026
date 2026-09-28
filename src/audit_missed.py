"""Missed-true-match analysis for a chosen full-scale blocker.

Reuses the rare-token inverted-index construction from audit_blocking.py but,
instead of only aggregating recall, keeps a sample of MISSED positive pairs
(true matches the blocker's candidate set does not contain) for inspection.

Usage: edit STRATEGY below to match the blocker being analyzed, then run.
"""

import json
import sys

import pandas as pd

from io_utils import explode_ground_truth, load_ground_truth
from paths import INTERIM, REPORTS

SEED = 42
N_EXAMPLES = 40


def load_train():
    cols = ["entity_id", "norm_name", "norm_addr", "country"]
    s1 = pd.read_parquet(INTERIM / "train_s1.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(INTERIM / f"train_s{i}.parquet", columns=cols) for i in (2, 3)], ignore_index=True)
    return s1, s23


def rarest_k_missed(s1, s23, truth_pairs, k, country_aware, field="name"):
    """Rebuild the rarest-K blocker's hit/miss mask and return missed pairs
    joined back to raw text for inspection, plus a coarse cause tag.
    """
    col = "norm_name" if field == "name" else "norm_addr"
    df_map = {}
    if country_aware:
        for c, s in zip(s23.country.astype(object), s23[col].astype(object)):
            for t in set(s.split()):
                df_map[(c, t)] = df_map.get((c, t), 0) + 1
    else:
        for s in s23[col].astype(object):
            for t in set(s.split()):
                df_map[t] = df_map.get(t, 0) + 1

    def rare_tokens(country, s):
        toks = s.split()
        if not toks:
            return []
        keyf = (lambda t: df_map.get((country, t), 0)) if country_aware else (lambda t: df_map.get(t, 0))
        return sorted(set(toks), key=keyf)[:k]

    s1_rare = [rare_tokens(c, s) for c, s in zip(s1.country.astype(object), s1[col].astype(object))]
    needed = {t for toks in s1_rare for t in toks}

    idx = {}
    if country_aware:
        for c, s, eid in zip(s23.country.astype(object), s23[col].astype(object), s23.entity_id.astype(object)):
            for t in set(s.split()):
                if t in needed:
                    idx.setdefault((c, t), set()).add(eid)
    else:
        for s, eid in zip(s23[col].astype(object), s23.entity_id.astype(object)):
            for t in set(s.split()):
                if t in needed:
                    idx.setdefault(t, set()).add(eid)

    s1_ids = s1.entity_id.astype(object).values
    s1_country = pd.Series(s1.country.astype(object).values, index=s1_ids)
    s1_rare_map = pd.Series(s1_rare, index=s1_ids)

    tp = truth_pairs.copy()
    tp_toks = tp.source1_entity_id.map(s1_rare_map)
    tp_country = tp.source1_entity_id.map(s1_country)
    exploded = tp[["source1_entity_id", "matched_id"]].copy()
    exploded["toks"] = tp_toks.values
    exploded["country"] = tp_country.values
    ex = exploded.explode("toks")
    ex = ex[ex.toks.notna()]
    if country_aware:
        ex["key"] = list(zip(ex.country, ex.toks))
    else:
        ex["key"] = ex.toks
    ex["in_bucket"] = [m2 in idx.get(k_, ()) for k_, m2 in zip(ex.key, ex.matched_id)]
    hit_ids = set(zip(ex[ex.in_bucket].source1_entity_id, ex[ex.in_bucket].matched_id))
    tp["hit"] = list(zip(tp.source1_entity_id, tp.matched_id))
    tp["hit"] = tp["hit"].isin(hit_ids)
    missed = tp[~tp.hit].drop(columns=["hit"])
    return missed, len(tp)


def classify_and_report(missed, total_pairs, s1, s23, out_key):
    s1c = s1.set_index(s1.entity_id.astype(object))[
        ["business_name", "business_address", "norm_name", "norm_addr", "country"]
    ]
    s23c = s23.set_index(s23.entity_id.astype(object))[["business_name", "business_address", "norm_name", "norm_addr"]]

    joined = missed.merge(s1c, left_on="source1_entity_id", right_index=True).merge(
        s23c, left_on="matched_id", right_index=True, suffixes=("_1", "_2")
    )

    def shared_tokens(a, b):
        return bool(set(a.split()) & set(b.split()))

    name_shared = [
        shared_tokens(a, b) for a, b in zip(joined.norm_name_1.astype(object), joined.norm_name_2.astype(object))
    ]
    addr_shared = [
        shared_tokens(a, b) for a, b in zip(joined.norm_addr_1.astype(object), joined.norm_addr_2.astype(object))
    ]
    joined["name_token_overlap"] = name_shared
    joined["addr_token_overlap"] = addr_shared

    n = len(joined)
    cats = {
        "zero_shared_name_and_addr_tokens": int(sum(not a and not b for a, b in zip(name_shared, addr_shared))),
        "zero_shared_name_tokens_only": int(sum((not a) and b for a, b in zip(name_shared, addr_shared))),
        "zero_shared_addr_tokens_only": int(sum(a and (not b) for a, b in zip(name_shared, addr_shared))),
        "shares_some_of_both": int(sum(a and b for a, b in zip(name_shared, addr_shared))),
    }
    empty_addr_either = (
        (joined.business_address_1.astype(object) == "") | (joined.business_address_2.astype(object) == "")
    ).sum()
    placeholder_either = (
        joined.business_address_1.astype(object).str.contains("<", regex=False)
        | joined.business_address_2.astype(object).str.contains("<", regex=False)
    ).sum()

    rep = {
        "blocker": out_key,
        "total_true_pairs": total_pairs,
        "missed_pairs": n,
        "missed_pct": round(100 * n / total_pairs, 3),
        "cause_categories": cats,
        "cause_categories_pct": {k: round(100 * v / max(n, 1), 2) for k, v in cats.items()},
        "missed_with_empty_address_either_side": int(empty_addr_either),
        "missed_with_placeholder_either_side": int(placeholder_either),
        "examples_zero_shared_both": (
            joined[(~joined.name_token_overlap) & (~joined.addr_token_overlap)][
                [
                    "source1_entity_id",
                    "business_name_1",
                    "business_address_1",
                    "matched_id",
                    "business_name_2",
                    "business_address_2",
                    "country",
                ]
            ]
            .sample(min(N_EXAMPLES, int(cats["zero_shared_name_and_addr_tokens"])), random_state=SEED)
            .astype(str)
            .values.tolist()
            if cats["zero_shared_name_and_addr_tokens"] > 0
            else []
        ),
    }
    return rep


if __name__ == "__main__":
    s1, s23 = load_train()
    gt = load_ground_truth()
    truth_pairs = explode_ground_truth(gt)

    # STRATEGY: rarest-2 name token, with country (edit if a different
    # blocker is chosen as "best practical" once full-scale numbers land)
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    country_aware = (sys.argv[2].lower() != "false") if len(sys.argv) > 2 else True
    field = sys.argv[3] if len(sys.argv) > 3 else "name"
    tag = f"rarest{k}_{field}_{'with' if country_aware else 'without'}_country"

    missed, total = rarest_k_missed(s1, s23, truth_pairs, k, country_aware, field)
    rep = classify_and_report(missed, total, s1, s23, tag)
    json.dump(
        rep,
        open(REPORTS / f"audit_missed_{tag}.json", "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
        default=str,
    )
    print(tag, "missed:", rep["missed_pairs"], "/", rep["total_true_pairs"], f"({rep['missed_pct']}%)")
    print(json.dumps(rep["cause_categories_pct"], indent=1))
