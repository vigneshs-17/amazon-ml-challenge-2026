"""EXP-002 Step 1: Diagnose EXP-001 Errors.

Loads the trained EXP-001 model and evaluates candidate predictions on a
reproducible stratified sample of 10,000 validation S1 entities.
Partitions candidate pairs into:
- TRUE POSITIVES (TP): true links with model_prob >= 0.90
- FALSE NEGATIVES (FN): true links surviving Cap-200 with model_prob < 0.90
- FALSE POSITIVES (FP): non-match candidates with model_prob >= 0.90
- TRUE NEGATIVES (TN): non-match candidates with model_prob < 0.90

Computes statistical distributions and identifies key error patterns.
Saves to experiments/EXP-002/exp001_error_analysis.json.
"""

import heapq
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from blocking import RareTokenIndex
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
EXP2_DIR = PROJECT_ROOT / "experiments" / "EXP-002"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
CAP = 200
THRESHOLD = 0.90
N_VAL_SAMPLE = 10_000
SEED = 2026

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def compute_rank_score(s1_name, s1_addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df):
    t2 = set(n2.split())
    a2s = set(a2.split())
    shared_n = s1_t1 & t2
    shared_a = s1_a1s & a2s
    score = 0.0
    if s1_addr and s1_addr == a2:
        score += 5.0
    if s1_name and s1_name == n2:
        score += 1.0
    if shared_n:
        min_df = min(name_df.get(t, 1) for t in shared_n)
        score += 2.0 / (1.0 + min_df) + 0.1 * len(shared_n)
    if shared_a:
        min_df = min(addr_df.get(t, 1) for t in shared_a)
        score += 2.0 / (1.0 + min_df) + 0.1 * len(shared_a)
    if s1_n1s and (s1_n1s & shared_a):
        score += 1.5
    return score


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


def extract_features(name1, addr1, c1, s1_t1, s1_a1s, s1_n1s, la1, la_addr1, name2, addr2, c2, name_df, addr_df):
    t2 = set(name2.split())
    a2 = set(addr2.split())
    shared_name = s1_t1 & t2
    shared_addr = s1_a1s & a2
    n2 = {t for t in a2 if any(c.isdigit() for c in t)}

    min_df_n = float(min(name_df.get(t, 0) for t in shared_name)) if shared_name else 0.0
    min_df_a = float(min(addr_df.get(t, 0) for t in shared_addr)) if shared_addr else 0.0

    la2 = len(name2)
    name_len_ratio = min(la1, la2) / max(la1, la2) if max(la1, la2) else 0.0
    la_addr2 = len(addr2)
    addr_len_ratio = min(la_addr1, la_addr2) / max(la_addr1, la_addr2) if max(la_addr1, la_addr2) else 0.0

    u_n = s1_t1 | t2
    u_a = s1_a1s | a2
    u_num = s1_n1s | n2
    i_num = s1_n1s & n2

    return [
        1.0 if (name1 and name1 == name2) else 0.0,
        len(shared_name) / len(u_n) if u_n else 1.0,
        len(shared_name) / min(len(s1_t1), len(t2)) if (s1_t1 and t2) else 0.0,
        fuzz.ratio(name1, name2) if (name1 or name2) else 0.0,
        fuzz.token_set_ratio(name1, name2) if (name1 or name2) else 0.0,
        name_len_ratio,
        float(len(shared_name)),
        min_df_n,
        1.0 if (addr1 and addr1 == addr2) else 0.0,
        len(shared_addr) / len(u_a) if u_a else 1.0,
        len(shared_addr) / min(len(s1_a1s), len(a2)) if (s1_a1s and a2) else 0.0,
        fuzz.ratio(addr1, addr2) if (addr1 or addr2) else 0.0,
        fuzz.token_set_ratio(addr1, addr2) if (addr1 or addr2) else 0.0,
        addr_len_ratio,
        float(len(shared_addr)),
        min_df_a,
        float(len(i_num)),
        len(i_num) / len(u_num) if u_num else 1.0,
        1.0 if (s1_n1s and n2 and not i_num) else 0.0,
        1.0 if c1 == c2 else 0.0,
        1.0 if not name1 else 0.0,
        1.0 if not addr1 else 0.0,
    ]


def summarize_col(series):
    s = pd.Series(series)
    return {
        "mean": round(float(s.mean()), 3),
        "median": round(float(s.median()), 3),
        "p25": round(float(s.quantile(0.25)), 3),
        "p75": round(float(s.quantile(0.75)), 3),
    }


def main():
    print("=== EXP-002 Step 1: Diagnose EXP-001 Model Errors ===")
    time.time()

    # Load Model
    clf = joblib.load(EXP_DIR / "model_hgb.joblib")
    print(f"Loaded trained EXP-001 model: {clf}")

    # Load S2/S3
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    s23_lookup = dict(
        zip(
            s23.entity_id.astype(object),
            zip(s23.name_norm.astype(object), s23.address_norm.astype(object), s23.country.astype(object)),
        )
    )
    blocker = B5Blocker(s23)
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    del s23

    # Load ground truth
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    truth_all = {}
    for sid, mid in zip(long.source1_entity_id.astype(object), long.matched_id.astype(object)):
        truth_all.setdefault(sid, set()).add(mid)
    del gt, long

    # Load stratified 10k validation sample
    val_ids_df = pd.read_csv(EXP_DIR / "validation_s1_ids.csv")
    s1_all = pd.read_parquet(CACHE / "train_s1.parquet")
    val_s1_full = s1_all[s1_all.entity_id.isin(set(val_ids_df.entity_id))].reset_index(drop=True)

    # Stratified 10k sample
    val_s1_full = val_s1_full.merge(val_ids_df[["entity_id", "bucket"]], on="entity_id", how="left")
    val_s1_full["strata"] = val_s1_full.country.astype(str) + "|" + val_s1_full.bucket.astype(str)
    rng = np.random.default_rng(SEED)
    sample_parts = []
    frac = N_VAL_SAMPLE / len(val_s1_full)
    for _, grp in val_s1_full.groupby("strata"):
        k = max(1, round(len(grp) * frac))
        k = min(k, len(grp))
        idx = rng.choice(grp.index.values, size=k, replace=False)
        sample_parts.append(grp.loc[idx])
    val_s1 = pd.concat(sample_parts).iloc[:N_VAL_SAMPLE].reset_index(drop=True)
    print(f"Sampled {len(val_s1)} validation S1 entities stratified by country and match bucket.")

    # Tracking lists for categories
    records = {"TP": [], "FN": [], "FP": [], "TN": []}
    examples = {"FN": [], "FP": []}

    print("Scoring candidate pairs and partitioning errors...")
    t_score0 = time.time()
    for i, (eid, name, addr, country) in enumerate(
        zip(
            val_s1.entity_id.astype(object),
            val_s1.name_norm.astype(object),
            val_s1.address_norm.astype(object),
            val_s1.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country)
        t = truth_all.get(eid, set())

        s1_t1 = set(name.split())
        s1_a1s = set(addr.split())
        s1_n1s = {tok for tok in s1_a1s if any(c.isdigit() for c in tok)}
        la1, la_addr1 = len(name), len(addr)
        s1_has_dev = bool(DEVANAGARI_RE.search(name))

        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score(name, addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df)

        if len(cand_list) > CAP:
            ranked_top = heapq.nlargest(CAP, cand_list, key=scores.__getitem__)
        else:
            ranked_top = sorted(cand_list, key=lambda c: -scores[c])

        if not ranked_top:
            continue

        feats = [
            extract_features(
                name,
                addr,
                country,
                s1_t1,
                s1_a1s,
                s1_n1s,
                la1,
                la_addr1,
                s23_lookup[c][0],
                s23_lookup[c][1],
                s23_lookup[c][2],
                name_df,
                addr_df,
            )
            for c in ranked_top
        ]
        proba = clf.predict_proba(np.array(feats, dtype=np.float32))[:, 1]

        for rank, (cid, p, f) in enumerate(zip(ranked_top, proba, feats), start=1):
            is_true = cid in t
            pred = p >= THRESHOLD
            n2, a2, _c2 = s23_lookup[cid]
            c2_has_dev = bool(DEVANAGARI_RE.search(n2))
            script_mismatch = s1_has_dev != c2_has_dev
            source = "S2" if cid.startswith("S2-") else "S3"

            item = {
                "name_ratio": f[3],
                "name_token_set_ratio": f[4],
                "addr_ratio": f[11],
                "addr_token_set_ratio": f[12],
                "name_tok_jaccard": f[1],
                "addr_tok_jaccard": f[9],
                "name_shared_count": f[6],
                "addr_shared_count": f[14],
                "num_intersection": f[16],
                "num_conflict": f[18],
                "name_min_shared_df": f[7],
                "addr_min_shared_df": f[15],
                "s1_name_len": la1,
                "s1_addr_len": la_addr1,
                "cand_name_len": len(n2),
                "cand_addr_len": len(a2),
                "prob": round(float(p), 4),
                "rank": rank,
                "script_mismatch": int(script_mismatch),
                "source": source,
                "zero_shared_name": int(f[6] == 0),
                "zero_shared_addr": int(f[14] == 0),
            }

            if is_true and pred:
                records["TP"].append(item)
            elif is_true and not pred:
                records["FN"].append(item)
                if len(examples["FN"]) < 20:
                    examples["FN"].append(
                        {
                            "s1_id": eid,
                            "s1_name": name,
                            "s1_addr": addr,
                            "cand_id": cid,
                            "cand_name": n2,
                            "cand_addr": a2,
                            "prob": round(float(p), 4),
                            "rank": rank,
                            "name_ratio": f[3],
                            "name_token_set_ratio": f[4],
                            "addr_ratio": f[11],
                            "addr_token_set_ratio": f[12],
                        }
                    )
            elif not is_true and pred:
                records["FP"].append(item)
                if len(examples["FP"]) < 20:
                    examples["FP"].append(
                        {
                            "s1_id": eid,
                            "s1_name": name,
                            "s1_addr": addr,
                            "cand_id": cid,
                            "cand_name": n2,
                            "cand_addr": a2,
                            "prob": round(float(p), 4),
                            "rank": rank,
                            "name_ratio": f[3],
                            "name_token_set_ratio": f[4],
                            "addr_ratio": f[11],
                            "addr_token_set_ratio": f[12],
                        }
                    )
            else:
                # Subsample TN to keep memory bounded
                if len(records["TN"]) < 50_000:
                    records["TN"].append(item)

        if (i + 1) % 2500 == 0:
            print(
                f"[{i+1}/{len(val_s1)}] processed in {time.time()-t_score0:.1f}s | TP: {len(records['TP'])}, FN: {len(records['FN'])}, FP: {len(records['FP'])}"
            )

    print(
        f"\nPartition counts: TP={len(records['TP'])}, FN={len(records['FN'])}, FP={len(records['FP'])}, TN={len(records['TN'])}"
    )

    # Aggregate distribution comparisons
    analysis = {
        "sample_size_s1": len(val_s1),
        "threshold": THRESHOLD,
        "counts": {k: len(v) for k, v in records.items()},
        "distributions": {},
        "proportions": {},
        "sample_examples": examples,
    }

    numeric_keys = [
        "name_ratio",
        "name_token_set_ratio",
        "addr_ratio",
        "addr_token_set_ratio",
        "name_tok_jaccard",
        "addr_tok_jaccard",
        "name_shared_count",
        "addr_shared_count",
        "name_min_shared_df",
        "addr_min_shared_df",
        "rank",
        "prob",
    ]

    for key in numeric_keys:
        analysis["distributions"][key] = {
            group: summarize_col([r[key] for r in records[group]]) for group in ["TP", "FN", "FP", "TN"]
        }

    prop_keys = ["script_mismatch", "zero_shared_name", "zero_shared_addr", "num_conflict"]
    for key in prop_keys:
        analysis["proportions"][key] = {
            group: round(float(pd.Series([r[key] for r in records[group]]).mean()), 4)
            for group in ["TP", "FN", "FP", "TN"]
        }

    analysis["proportions"]["source_s2_pct"] = {
        group: round(float(pd.Series([r["source"] == "S2" for r in records[group]]).mean()), 4)
        for group in ["TP", "FN", "FP", "TN"]
    }

    # Save to EXP-002
    out_file = EXP2_DIR / "exp001_error_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nError analysis saved to {out_file}")

    print("\n--- Key Distribution Comparison (TP vs FN vs FP) ---")
    print(f"{'Feature':<22} | {'TP (Mean)':>10} | {'FN (Mean)':>10} | {'FP (Mean)':>10} | {'TN (Mean)':>10}")
    print("-" * 72)
    for k in [
        "name_ratio",
        "name_token_set_ratio",
        "addr_token_set_ratio",
        "name_tok_jaccard",
        "addr_tok_jaccard",
        "rank",
        "prob",
    ]:
        tp_m = analysis["distributions"][k]["TP"]["mean"]
        fn_m = analysis["distributions"][k]["FN"]["mean"]
        fp_m = analysis["distributions"][k]["FP"]["mean"]
        tn_m = analysis["distributions"][k]["TN"]["mean"]
        print(f"{k:<22} | {tp_m:10.2f} | {fn_m:10.2f} | {fp_m:10.2f} | {tn_m:10.2f}")

    print("\n--- Key Proportions (TP vs FN vs FP) ---")
    for k in prop_keys:
        tp_p = analysis["proportions"][k]["TP"]
        fn_p = analysis["proportions"][k]["FN"]
        fp_p = analysis["proportions"][k]["FP"]
        print(f"{k:<22} | TP: {tp_p*100:5.2f}% | FN: {fn_p*100:5.2f}% | FP: {fp_p*100:5.2f}%")


if __name__ == "__main__":
    main()
