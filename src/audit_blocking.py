"""EXP-000 step 6: blocking-signal diagnostics at full scale where the signal
allows an O(n) group-size join (exact name, exact address, name prefix), and on
a large fixed-seed sample where the signal needs token explosion (rare-token
union blocking). No all-pairs comparison is performed anywhere.

For every FULL-SCALE signal we report, over the WHOLE train S1 set (2,206,821
entities) against the WHOLE train S2+S3 pool (10,320,219 records):
  - true-match recall (against the complete ground truth, not sampled)
  - candidates-per-S1 distribution (mean/median/p90/p95/p99/max)
  - % of S1 entities with zero candidates
  - candidate reduction ratio vs. the naive all-pairs baseline
Each signal is evaluated WITH country restriction and WITHOUT, since country
itself carries some corruption/openness (France, etc.).
"""

import json
import time

import numpy as np
import pandas as pd

from io_utils import explode_ground_truth, load_ground_truth
from paths import INTERIM, REPORTS

QUANTS_C = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)


def load_train():
    cols = ["entity_id", "norm_name", "norm_addr", "country"]
    s1 = pd.read_parquet(INTERIM / "train_s1.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(INTERIM / f"train_s{i}.parquet", columns=cols) for i in (2, 3)], ignore_index=True)
    return s1, s23


def eval_key_blocker(s1, s23, truth_pairs, s1_key_col, s23_key_col, country_aware, label):
    """O(n) evaluation for a blocker where each record maps to exactly ONE key
    (exact-name, exact-addr, name-prefix): candidate count = bucket size.
    """
    t0 = time.time()
    if country_aware:
        s23_key = list(zip(s23.country.astype(object), s23[s23_key_col].astype(object)))
        s1_key = list(zip(s1.country.astype(object), s1[s1_key_col].astype(object)))
    else:
        s23_key = s23[s23_key_col].astype(object).tolist()
        s1_key = s1[s1_key_col].astype(object).tolist()

    bucket_size = pd.Series(s23_key).value_counts()  # key -> candidate count
    s1_bucket = pd.Series(s1_key).map(bucket_size).fillna(0).astype(int)
    empty_key = pd.Series(s1_key).map(lambda k: (k[-1] if country_aware else k) == "")
    s1_bucket = s1_bucket.where(~empty_key.values, 0)  # an empty normalized field blocks on nothing

    n = pd.DataFrame({"s1_id": s1.entity_id.astype(object), "n_cand": s1_bucket.values})
    total_candidates = int(n.n_cand.sum())
    naive_total = len(s1) * len(s23)

    # recall: does the true match share the S1's key?
    s1_key_map = pd.Series(s1_key, index=s1.entity_id.astype(object).values)
    s23_key_map = pd.Series(s23_key, index=s23.entity_id.astype(object).values)
    tp = truth_pairs.copy()
    tp["k1"] = tp.source1_entity_id.map(s1_key_map)
    tp["k2"] = tp.matched_id.map(s23_key_map)
    hit = (tp.k1 == tp.k2) & tp.k1.map(lambda k: (k[-1] if country_aware else k) != "")
    recall_by_pair = float(hit.mean())

    r = {
        "candidates_per_s1": {
            "mean": round(total_candidates / len(s1), 2),
            "median": int(n.n_cand.median()),
            "p90": int(n.n_cand.quantile(0.9)),
            "p95": int(n.n_cand.quantile(0.95)),
            "p99": int(n.n_cand.quantile(0.99)),
            "max": int(n.n_cand.max()),
        },
        "pct_s1_zero_candidates": round(100 * float((n.n_cand == 0).mean()), 3),
        "total_candidate_pairs": total_candidates,
        "candidate_reduction_ratio_vs_naive": round(1 - total_candidates / naive_total, 6),
        "pairwise_recall_pct": round(100 * recall_by_pair, 3),
        "seconds": round(time.time() - t0, 1),
    }
    print(
        label,
        r["pairwise_recall_pct"],
        "% recall,",
        r["candidates_per_s1"]["mean"],
        "avg cand,",
        r["seconds"],
        "s",
        flush=True,
    )
    return r


def token_df(s23, col, country_aware):
    """Country-aware or global document frequency of normalized tokens."""
    c = {}
    if country_aware:
        for country, s in zip(s23.country.astype(object), s23[col].astype(object)):
            for t in set(s.split()):
                c[(country, t)] = c.get((country, t), 0) + 1
    else:
        for s in s23[col].astype(object):
            for t in set(s.split()):
                c[t] = c.get(t, 0) + 1
    return c


def eval_rarest_k(s1, s23, truth_pairs, df_map, country_aware, k, field, label):
    """Union-of-rarest-K-tokens blocker, evaluated on the FULL S1 set via an
    inverted index restricted to the tokens S1 actually selects (bounded size
    because rare tokens have small buckets by construction).
    """
    t0 = time.time()
    col = "norm_name" if field == "name" else "norm_addr"

    def rare_tokens(country, s):
        toks = s.split()
        if not toks:
            return []
        keyf = (lambda t: df_map.get((country, t), 0)) if country_aware else (lambda t: df_map.get(t, 0))
        return sorted(set(toks), key=keyf)[:k]

    s1_rare = [rare_tokens(c, s) for c, s in zip(s1.country.astype(object), s1[col].astype(object))]
    needed = set()
    for toks in s1_rare:
        needed.update(toks)

    # inverted index: only for tokens S1 actually needs (bounded)
    if country_aware:
        idx = {}
        for country, s, eid in zip(s23.country.astype(object), s23[col].astype(object), s23.entity_id.astype(object)):
            for t in set(s.split()):
                if t in needed:
                    idx.setdefault((country, t), []).append(eid)
    else:
        idx = {}
        for s, eid in zip(s23[col].astype(object), s23.entity_id.astype(object)):
            for t in set(s.split()):
                if t in needed:
                    idx.setdefault(t, []).append(eid)

    cand_counts = np.empty(len(s1), dtype=np.int64)
    s1_ids = s1.entity_id.astype(object).values
    cand_map_for_truth = {}
    for i, (country, toks) in enumerate(zip(s1.country.astype(object), s1_rare)):
        cset = set()
        for t in toks:
            key = (country, t) if country_aware else t
            cset.update(idx.get(key, ()))
        cand_counts[i] = len(cset)
        cand_map_for_truth[s1_ids[i]] = cset if len(cset) < 200 else None  # keep small sets only, for recall spot-check

    n = pd.Series(cand_counts)
    total_candidates = int(n.sum())
    naive_total = len(s1) * len(s23)

    # recall via inverted-index membership test (exact, O(pairs))
    tp = truth_pairs.copy()
    s1_country = pd.Series(s1.country.astype(object).values, index=s1_ids)
    s1_rare_map = pd.Series(s1_rare, index=s1_ids)

    def hit_row(s1id, m2):
        toks = s1_rare_map.get(s1id, [])
        c = s1_country.get(s1id, "")
        for t in toks:
            key = (c, t) if country_aware else t
            if m2 in idx.get(key, ()):
                return True
        return False

    # vectorize via explode instead of per-row python calls for speed
    tp_toks = tp.source1_entity_id.map(s1_rare_map)
    tp_country = tp.source1_entity_id.map(s1_country)
    hit = np.zeros(len(tp), dtype=bool)
    exploded = tp[["source1_entity_id", "matched_id"]].copy()
    exploded["toks"] = tp_toks.values
    exploded["country"] = tp_country.values
    ex = exploded.explode("toks")
    ex = ex[ex.toks.notna()]
    if country_aware:
        ex["key"] = list(zip(ex.country, ex.toks))
    else:
        ex["key"] = ex.toks
    ex["in_bucket"] = [m2 in idx.get(k, ()) for k, m2 in zip(ex.key, ex.matched_id)]
    hit_ids = set(zip(ex[ex.in_bucket].source1_entity_id, ex[ex.in_bucket].matched_id))
    hit = pd.Series(list(zip(tp.source1_entity_id, tp.matched_id))).isin(hit_ids)

    r = {
        "candidates_per_s1": {
            "mean": round(total_candidates / len(s1), 2),
            "median": int(n.median()),
            "p90": int(n.quantile(0.9)),
            "p95": int(n.quantile(0.95)),
            "p99": int(n.quantile(0.99)),
            "max": int(n.max()),
        },
        "pct_s1_zero_candidates": round(100 * float((n == 0).mean()), 3),
        "total_candidate_pairs": total_candidates,
        "candidate_reduction_ratio_vs_naive": round(1 - total_candidates / naive_total, 6),
        "pairwise_recall_pct": round(100 * float(hit.mean()), 3),
        "distinct_tokens_indexed": len(needed),
        "seconds": round(time.time() - t0, 1),
    }
    print(
        label,
        r["pairwise_recall_pct"],
        "% recall,",
        r["candidates_per_s1"]["mean"],
        "avg cand,",
        r["seconds"],
        "s",
        flush=True,
    )
    return r


def main():
    s1, s23 = load_train()
    gt = load_ground_truth()
    truth_pairs = explode_ground_truth(gt)  # ALL 7,638,365 positive pairs, full scale
    rep = {
        "s1_rows": len(s1),
        "s23_rows": len(s23),
        "naive_all_pairs": len(s1) * len(s23),
        "truth_pairs_total": len(truth_pairs),
    }

    # ---- Full-scale O(n) blockers ----
    for country_aware in (True, False):
        tag = "with_country" if country_aware else "without_country"
        rep[f"exact_name_{tag}"] = eval_key_blocker(
            s1, s23, truth_pairs, "norm_name", "norm_name", country_aware, f"exact_name_{tag}"
        )
        rep[f"exact_addr_{tag}"] = eval_key_blocker(
            s1, s23, truth_pairs, "norm_addr", "norm_addr", country_aware, f"exact_addr_{tag}"
        )

    s1p = s1.assign(prefix4=s1.norm_name.astype(object).str.replace(" ", "", regex=False).str.slice(0, 4))
    s23p = s23.assign(prefix4=s23.norm_name.astype(object).str.replace(" ", "", regex=False).str.slice(0, 4))
    for country_aware in (True, False):
        tag = "with_country" if country_aware else "without_country"
        rep[f"name_prefix4_{tag}"] = eval_key_blocker(
            s1p, s23p, truth_pairs, "prefix4", "prefix4", country_aware, f"name_prefix4_{tag}"
        )

    json.dump(rep, open(REPORTS / "audit_blocking.json", "w", encoding="utf-8"), indent=1, default=str)
    print("--- full-scale blockers done, saved checkpoint ---", flush=True)

    # ---- Rare-token union blockers (need explosion; bounded by rarity) ----
    print("building token doc-frequency (full s23, name field)...", flush=True)
    df_name_c = token_df(s23, "norm_name", country_aware=True)
    df_name_g = token_df(s23, "norm_name", country_aware=False)
    print("building token doc-frequency (full s23, addr field)...", flush=True)
    df_addr_c = token_df(s23, "norm_addr", country_aware=True)
    df_addr_g = token_df(s23, "norm_addr", country_aware=False)

    # Common-legal-suffix sanity check: top global tokens by df (name field)
    top_common_name = sorted(df_name_g.items(), key=lambda kv: -kv[1])[:30]
    top_common_addr = sorted(df_addr_g.items(), key=lambda kv: -kv[1])[:30]
    rep["top_common_name_tokens_by_df"] = top_common_name
    rep["top_common_addr_tokens_by_df"] = top_common_addr
    rep["name_token_df_distribution"] = {
        str(q): float(np.quantile(list(df_name_g.values()), q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)
    }

    for k in (1, 2, 3):
        for country_aware, dfm, tag in [(True, df_name_c, "with_country"), (False, df_name_g, "without_country")]:
            rep[f"rarest{k}_name_{tag}"] = eval_rarest_k(
                s1, s23, truth_pairs, dfm, country_aware, k, "name", f"rarest{k}_name_{tag}"
            )
        json.dump(rep, open(REPORTS / "audit_blocking.json", "w", encoding="utf-8"), indent=1, default=str)

    for k in (1, 2):
        for country_aware, dfm, tag in [(True, df_addr_c, "with_country"), (False, df_addr_g, "without_country")]:
            rep[f"rarest{k}_addr_{tag}"] = eval_rarest_k(
                s1, s23, truth_pairs, dfm, country_aware, k, "addr", f"rarest{k}_addr_{tag}"
            )
        json.dump(rep, open(REPORTS / "audit_blocking.json", "w", encoding="utf-8"), indent=1, default=str)

    json.dump(rep, open(REPORTS / "audit_blocking.json", "w", encoding="utf-8"), indent=1, default=str)
    print("all blocking diagnostics done")


if __name__ == "__main__":
    main()
