"""EXP-000 step 4: positive-pair similarity analysis + diagnostic negatives.

Positive pairs come from train_ground_truth (analysis only). Negatives are a
SMALL diagnostic sample: random easy negatives plus hard negatives found by
exact normalized-name / normalized-address collisions and shared rare tokens
for a fixed-seed subset of S1 entities. No all-pairs comparison is done.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from io_utils import explode_ground_truth, load_ground_truth
from paths import INTERIM, REPORTS

SEED = 42
N_POS_SAMPLE = 300_000
N_S1_HARDNEG = 20_000
DEVA = "[ऀ-ॿ]"


def load_train():
    cols = ["entity_id", "business_name", "business_address", "country", "norm_name", "norm_addr"]
    s1 = pd.read_parquet(INTERIM / "train_s1.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(INTERIM / f"train_s{i}.parquet", columns=cols) for i in (2, 3)], ignore_index=True)
    return s1, s23


def jacc(a, b):
    a, b = set(a.split()), set(b.split())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def ngram_jacc(a, b, n=3):
    A = {a[i : i + n] for i in range(max(len(a) - n + 1, 1))}
    B = {b[i : i + n] for i in range(max(len(b) - n + 1, 1))}
    return len(A & B) / max(len(A | B), 1)


def pair_features(df, left="_1", r="_2"):
    """Similarity features for a dataframe of pairs (columns suffixed _1/_2)."""
    out = pd.DataFrame(index=df.index)
    for fld, raw, norm in [("name", "business_name", "norm_name"), ("addr", "business_address", "norm_addr")]:
        a_raw, b_raw = df[raw + left].astype(object), df[raw + r].astype(object)
        a, b = df[norm + left].astype(object).tolist(), df[norm + r].astype(object).tolist()
        out[f"{fld}_raw_eq"] = a_raw.values == b_raw.values
        out[f"{fld}_ci_eq"] = a_raw.str.casefold().values == b_raw.str.casefold().values
        out[f"{fld}_norm_eq"] = [x == y for x, y in zip(a, b)]
        out[f"{fld}_tok_jacc"] = [jacc(x, y) for x, y in zip(a, b)]
        out[f"{fld}_ratio"] = [fuzz.ratio(x, y) for x, y in zip(a, b)]
        out[f"{fld}_tsort"] = [fuzz.token_sort_ratio(x, y) for x, y in zip(a, b)]
        out[f"{fld}_tset"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a, b)]
        out[f"{fld}_partial"] = [fuzz.partial_ratio(x, y) for x, y in zip(a, b)]
        out[f"{fld}_jw"] = [JaroWinkler.similarity(x, y) for x, y in zip(a, b)]
        out[f"{fld}_3gram_jacc"] = [ngram_jacc(x, y) for x, y in zip(a, b)]
        out[f"{fld}_empty_any"] = [(x == "") or (y == "") for x, y in zip(a, b)]

    # numeric tokens in address
    def nums(s):
        return {t for t in s.split() if any(c.isdigit() for c in t)}

    na = [nums(x) for x in df["norm_addr" + left].astype(object)]
    nb = [nums(x) for x in df["norm_addr" + r].astype(object)]
    out["addr_num_both_have"] = [bool(x) and bool(y) for x, y in zip(na, nb)]
    out["addr_num_overlap"] = [bool(x & y) for x, y in zip(na, nb)]
    out["addr_num_equal"] = [x == y and bool(x) for x, y in zip(na, nb)]
    out["country_eq"] = df["country" + left].values == df["country" + r].values
    dl = df["business_name" + left].str.contains(DEVA).values
    dr = df["business_name" + r].str.contains(DEVA).values
    out["name_script_mismatch"] = dl != dr
    return out


QUANTS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def summarize(feat):
    res = {}
    for c in feat.columns:
        v = feat[c]
        if v.dtype == bool:
            res[c] = round(100 * float(v.mean()), 2)
        else:
            v = v.astype("float64")
            res[c] = {q: round(float(v.quantile(q)), 3) for q in QUANTS} | {"mean": round(float(v.mean()), 3)}
    return res


def diff_tokens(pairs):
    """Tokens dropped/added between matched names and 1-for-1 substitutions."""
    dropped, subs = Counter(), Counter()
    for a, b in zip(pairs.norm_name_1.astype(object), pairs.norm_name_2.astype(object)):
        A, B = a.split(), b.split()
        sa, sb = set(A), set(B)
        for t in sa ^ sb:
            dropped[t] += 1
        da, db = sa - sb, sb - sa
        if len(da) == 1 and len(db) == 1:
            subs[(next(iter(da)), next(iter(db)))] += 1
    return dropped, subs


def diff_tokens_addr(pairs):
    subs, only1, only2 = Counter(), Counter(), Counter()
    for a, b in zip(pairs.norm_addr_1.astype(object), pairs.norm_addr_2.astype(object)):
        sa, sb = set(a.split()), set(b.split())
        da, db = sa - sb, sb - sa
        only1.update(da)
        only2.update(db)
        if len(da) == 1 and len(db) == 1:
            subs[(next(iter(da)), next(iter(db)))] += 1
    return subs, only1, only2


def examples(pairs, feat, mask, k=6, seed=SEED):
    idx = feat.index[mask]
    if len(idx) == 0:
        return []
    pick = pd.Series(idx).sample(min(k, len(idx)), random_state=seed)
    cols = [
        "source1_entity_id",
        "business_name_1",
        "business_address_1",
        "matched_id",
        "business_name_2",
        "business_address_2",
        "country_1",
    ]
    return pairs.loc[pick, cols].astype(str).values.tolist()


def doc_freq(df, col):
    """Country-aware document frequency of normalized tokens: {(country, tok): n_records}."""
    c = Counter()
    for country, s in zip(df.country.astype(object), df[col].astype(object)):
        for t in set(s.split()):
            c[(country, t)] += 1
    return c


def blocking_diagnostics(pos, s23):
    """Recall/cost of simple observable blocking signals on positive pairs.

    For each positive pair take the k rarest tokens of the S1 record (rarity =
    country-aware df over S2+S3). recall@k = pair shares one of them;
    cost@k = sum of df of those k tokens (upper bound on block size).
    """
    res = {}
    for fld in ["norm_name", "norm_addr"]:
        df = doc_freq(s23, fld)
        hit = {k: [] for k in (1, 2, 3)}
        cost = {k: [] for k in (1, 2, 3)}
        min_shared_df = []
        for c, a, b in zip(
            pos.country_1.astype(object), pos[fld + "_1"].astype(object), pos[fld + "_2"].astype(object)
        ):
            ta = sorted(set(a.split()), key=lambda t: df.get((c, t), 0))
            tb = set(b.split())
            shared = [df.get((c, t), 0) for t in ta if t in tb]
            min_shared_df.append(min(shared) if shared else np.nan)
            for k in (1, 2, 3):
                top = ta[:k]
                hit[k].append(any(t in tb for t in top))
                cost[k].append(sum(df.get((c, t), 0) for t in top))
        msd = pd.Series(min_shared_df)
        r = {
            "pairs_sharing_any_token_pct": round(100 * float(msd.notna().mean()), 2),
            "min_df_of_shared_token_quantiles": {q: float(msd.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)},
            "pairs_sharing_token_df_le": {K: round(100 * float((msd <= K).mean()), 2) for K in (10, 100, 1000, 10000)},
        }
        for k in (1, 2, 3):
            cs = pd.Series(cost[k])
            r[f"rarest{k}_recall_pct"] = round(100 * float(np.mean(hit[k])), 2)
            r[f"rarest{k}_block_cost"] = {
                "median": float(cs.median()),
                "p95": float(cs.quantile(0.95)),
                "p99": float(cs.quantile(0.99)),
                "mean": round(float(cs.mean()), 1),
            }
        res[fld] = r
        res[fld + "_by_country"] = {
            c: {
                "rarest2_recall_pct": round(100 * float(np.mean(np.array(hit[2])[pos.country_1.values == c])), 2),
                "sharing_any_pct": round(100 * float(msd[pos.country_1.values == c].notna().mean()), 2),
            }
            for c in pos.country_1.unique()
        }
        pos["_hit2_" + fld] = hit[2]
        pos["_any_" + fld] = msd.notna().values
    res["union_rarest2_name_or_addr_recall_pct"] = round(
        100 * float((pos["_hit2_norm_name"] | pos["_hit2_norm_addr"]).mean()), 2
    )
    res["union_any_shared_name_or_addr_token_pct"] = round(
        100 * float((pos["_any_norm_name"] | pos["_any_norm_addr"]).mean()), 2
    )

    # numeric address tokens
    def nums(s):
        return {t for t in s.split() if any(ch.isdigit() for ch in t)}

    shared_num = [
        bool(nums(a) & nums(b)) for a, b in zip(pos.norm_addr_1.astype(object), pos.norm_addr_2.astype(object))
    ]
    res["pairs_sharing_numeric_addr_token_pct"] = round(100 * float(np.mean(shared_num)), 2)
    # name prefix
    res["name_first4_equal_pct"] = round(
        100
        * float(
            np.mean(
                [
                    a.replace(" ", "")[:4] == b.replace(" ", "")[:4]
                    for a, b in zip(pos.norm_name_1.astype(object), pos.norm_name_2.astype(object))
                ]
            )
        ),
        2,
    )
    return res


def main():
    np.random.default_rng(SEED)
    s1, s23 = load_train()
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    rep = {}

    pos = long.sample(min(N_POS_SAMPLE, len(long)), random_state=SEED)
    pos = pos.merge(s1.astype({"entity_id": object}), left_on="source1_entity_id", right_on="entity_id")
    pos = pos.merge(
        s23.astype({"entity_id": object}), left_on="matched_id", right_on="entity_id", suffixes=("_1", "_2")
    ).reset_index(drop=True)
    pos["src"] = pos.matched_id.str.slice(0, 2)
    feat = pair_features(pos)
    rep["positive_sample_size"] = len(pos)
    rep["positive_overall"] = summarize(feat)
    rep["positive_by_country"] = {c: summarize(feat[pos.country_1.values == c]) for c in pos.country_1.unique()}
    rep["positive_by_source"] = {s: summarize(feat[pos.src.values == s]) for s in ["S2", "S3"]}
    rep["positive_country_pairs"] = (
        pos.groupby(["country_1", "country_2"]).size().astype(int).reset_index().values.tolist()
    )

    dropped, subs = diff_tokens(pos)
    rep["name_tokens_most_often_differing"] = dropped.most_common(80)
    rep["name_1for1_substitutions"] = [(f"{a} -> {b}", c) for (a, b), c in subs.most_common(80)]
    asubs, only1, only2 = diff_tokens_addr(pos)
    rep["addr_1for1_substitutions"] = [(f"{a} -> {b}", c) for (a, b), c in asubs.most_common(80)]
    rep["addr_tokens_only_in_s1"] = only1.most_common(50)
    rep["addr_tokens_only_in_s23"] = only2.most_common(50)

    # Noise-category examples
    n1 = pos.norm_name_1.astype(object)
    n2 = pos.norm_name_2.astype(object)
    sorted_eq = np.array([sorted(a.split()) == sorted(b.split()) for a, b in zip(n1, n2)])
    subset = np.array([set(a.split()) < set(b.split()) or set(b.split()) < set(a.split()) for a, b in zip(n1, n2)])
    cats = {
        "punctuation_or_case_only": (~feat.name_raw_eq) & feat.name_norm_eq,
        "word_reorder": pd.Series(sorted_eq, index=feat.index) & ~feat.name_norm_eq,
        "token_added_or_dropped (suffix etc.)": pd.Series(subset, index=feat.index),
        "spelling_typo (ratio 80-97, same #tokens)": (feat.name_ratio.between(80, 97))
        & pd.Series([len(a.split()) == len(b.split()) for a, b in zip(n1, n2)], index=feat.index)
        & ~pd.Series(sorted_eq, index=feat.index),
        "script_variation": feat.name_script_mismatch,
        "address_missing": feat.addr_empty_any,
        "address_components_missing (token subset)": pd.Series(
            [
                set(a.split()) < set(b.split()) or set(b.split()) < set(a.split())
                for a, b in zip(pos.norm_addr_1.astype(object), pos.norm_addr_2.astype(object))
            ],
            index=feat.index,
        ),
        "hard_name (tset<50)": feat.name_tset < 50,
        "hard_both (name tset<60 & addr tset<60)": (feat.name_tset < 60) & (feat.addr_tset < 60),
    }
    rep["noise_category_pct"] = {k: round(100 * float(m.mean()), 2) for k, m in cats.items()}
    rep["noise_examples"] = {k: examples(pos, feat, m.values) for k, m in cats.items()}

    rep["blocking_diagnostics"] = blocking_diagnostics(pos, s23)
    json.dump(rep, open(REPORTS / "audit_pairs.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("positives done", flush=True)

    # ---------- Negatives ----------
    neg = {}
    # Easy negatives: random S1 x random S2/S3 (same country), not in truth
    truth = set(zip(long.source1_entity_id, long.matched_id))
    rs1 = s1.sample(20_000, random_state=SEED).reset_index(drop=True)
    rs23 = s23.sample(200_000, random_state=SEED).reset_index(drop=True)
    easy_rows = []
    for c in rs1.country.unique():
        a = rs1[rs1.country == c]
        b = rs23[rs23.country == c]
        bi = b.sample(len(a), replace=True, random_state=SEED)
        easy_rows.append(
            pd.concat([a.reset_index(drop=True).add_suffix("_1"), bi.reset_index(drop=True).add_suffix("_2")], axis=1)
        )
    easy = pd.concat(easy_rows, ignore_index=True)
    easy = easy[[(x, y) not in truth for x, y in zip(easy.entity_id_1, easy.entity_id_2)]].reset_index(drop=True)
    neg["easy_random_same_country"] = summarize(pair_features(easy))

    # Hard negatives: same normalized name (or address) but not a true match
    sub = s1.sample(N_S1_HARDNEG, random_state=SEED)
    for key, label in [("norm_name", "same_norm_name"), ("norm_addr", "same_norm_addr")]:
        j = sub[sub[key] != ""].merge(s23[s23[key] != ""], on=key, suffixes=("_1", "_2"))
        j[key + "_1"] = j[key]
        j[key + "_2"] = j[key]
        j["is_true"] = [(x, y) in truth for x, y in zip(j.entity_id_1, j.entity_id_2)]
        n_s1_with_collision = j.entity_id_1.nunique()
        n_s1_with_false = j[~j.is_true].entity_id_1.nunique()
        neg[label] = {
            "s1_sampled": len(sub),
            "s1_with_any_exact_collision": int(n_s1_with_collision),
            "collision_pairs": len(j),
            "collision_pairs_true": int(j.is_true.sum()),
            "precision_of_exact_rule": round(float(j.is_true.mean()), 4) if len(j) else None,
            "s1_with_false_collision": int(n_s1_with_false),
            "mean_collisions_per_s1_with_any": round(len(j) / max(n_s1_with_collision, 1), 2),
        }
        hn = j[~j.is_true].reset_index(drop=True)
        if len(hn):
            f = pair_features(hn)
            neg[label]["feature_summary"] = summarize(f)
            ex = hn.sample(min(12, len(hn)), random_state=SEED)
            neg[label]["examples"] = (
                ex[
                    [
                        "entity_id_1",
                        "business_name_1",
                        "business_address_1",
                        "entity_id_2",
                        "business_name_2",
                        "business_address_2",
                        "country_1",
                    ]
                ]
                .astype(str)
                .values.tolist()
            )
    rep["negatives"] = neg

    json.dump(rep, open(REPORTS / "audit_pairs.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("done")


if __name__ == "__main__":
    main()
