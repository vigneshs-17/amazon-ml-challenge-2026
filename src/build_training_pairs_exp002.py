"""EXP-002: Build Diverse Hard-Negative Training Pairs (100k S1, 42 features).

Samples 100,000 training S1 entities from train_remaining_s1_ids.csv (stratified
by country and match bucket, fixed seed 2026).
For each training S1:
- Retrieves candidates via verified B5 with deterministic tie-breaking.
- Slices to Cap-200.
- Positives = ground-truth links present in Cap-200.
- Diverse Hard Negatives (up to 6 per S1):
  * Top 3 non-matches by deterministic candidate score
  * Top 2 non-matches by name similarity (targeting generic-name FP traps)
  * Top 1 non-match by address similarity
- Extracts 42 interpretable features.
- Saves train_X.npy, train_y.npy, and training_report.json to experiments/EXP-002/.
"""

import ctypes
import heapq
import json
import time
from ctypes import wintypes

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from blocking import RareTokenIndex
from features_v2 import ALL_FEATURE_NAMES, extract_pair_features_v2, precompute_s1
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP2_DIR = PROJECT_ROOT / "experiments" / "EXP-002"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
LOG_FILE = PROJECT_ROOT / "logs" / "build_training_pairs_exp002.log"
CAP = 200
SEED = 2026
N_TRAIN_S1 = 100_000

psapi = ctypes.windll.psapi
kernel32 = ctypes.windll.kernel32


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def get_ram_mb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / (1024 * 1024), pmc.PeakWorkingSetSize / (1024 * 1024)


def log(msg, to_file=True):
    print(msg, flush=True)
    if to_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


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


def bucket(n):
    return 0 if n == 0 else (1 if n == 1 else (2 if n == 2 else 3))


def load_stratified_train_s1(n_samples=N_TRAIN_S1):
    train_ids_df = pd.read_csv(PROJECT_ROOT / "experiments" / "EXP-001" / "train_remaining_s1_ids.csv")
    s1_all = pd.read_parquet(CACHE / "train_s1.parquet")
    s1_train = s1_all[s1_all.entity_id.isin(set(train_ids_df.entity_id))].reset_index(drop=True)

    gt = load_ground_truth()
    n_matches = gt.matched_entity_ids.astype(object).map(lambda x: 0 if not x else len(x.split(",")))
    gt_df = pd.DataFrame({"entity_id": gt.source1_entity_id.astype(object), "n_matches": n_matches})
    s1_train = s1_train.merge(gt_df, on="entity_id", how="left")
    s1_train["bucket"] = s1_train.n_matches.map(bucket)
    s1_train["strata"] = s1_train.country.astype(str) + "|" + s1_train.bucket.astype(str)

    rng = np.random.default_rng(SEED)
    sample_parts = []
    frac = n_samples / len(s1_train)
    for _, grp in s1_train.groupby("strata"):
        k = max(1, round(len(grp) * frac))
        k = min(k, len(grp))
        idx = rng.choice(grp.index.values, size=k, replace=False)
        sample_parts.append(grp.loc[idx])
    out = pd.concat(sample_parts).iloc[:n_samples].reset_index(drop=True)
    return out


def load_s23():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    return pd.concat([s2, s3], ignore_index=True)


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== EXP-002 Build Training Pairs Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    t_start = time.time()
    log(f"Sampling {N_TRAIN_S1} stratified training S1 entities (Seed: {SEED})...")
    s1 = load_stratified_train_s1(N_TRAIN_S1)
    log(f"Sampled {len(s1)} training S1 entities.")

    log("Loading S2/S3 pools and building B5 indexes with deterministic tie-breaking...")
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
    del s23

    log("Loading ground truth for training entities...")
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    s1_set = set(s1.entity_id.astype(object))
    long = long[long.source1_entity_id.astype(object).isin(s1_set)]
    truth = {}
    for sid, mid in zip(long.source1_entity_id.astype(object), long.matched_id.astype(object)):
        truth.setdefault(sid, set()).add(mid)
    del gt, long

    total_truth_links = sum(len(v) for v in truth.values())
    log(f"Training ground truth: {len(truth)} entities have matches ({total_truth_links} total true links).")

    X_rows, y_rows = [], []
    n_pos = n_neg = 0
    t_loop0 = time.time()
    log_interval = 5000

    log(f"\nBuilding Cap-{CAP} diverse hard-negative training pairs...")

    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1.entity_id.astype(object),
            s1.name_norm.astype(object),
            s1.address_norm.astype(object),
            s1.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country)
        t = truth.get(eid, set())

        s1_t1 = set(name.split())
        s1_a1s = set(addr.split())
        s1_n1s = {tok for tok in s1_a1s if any(c.isdigit() for c in tok)}
        s1_pre = precompute_s1(name, addr, country, name_df, addr_df)

        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score(name, addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df)

        if len(cand_list) > CAP:
            ranked_top = heapq.nlargest(CAP, cand_list, key=scores.__getitem__)
        else:
            ranked_top = sorted(cand_list, key=lambda c: -scores[c])

        # Positives in Cap-200
        pos_cands = [c for c in ranked_top if c in t]
        # Non-matches in Cap-200
        neg_cands_pool = [c for c in ranked_top if c not in t]

        # Diverse Hard Negatives:
        # 1. Top 3 by candidate rank score
        hard_negs = set(neg_cands_pool[:3])
        # 2. Top 2 by name token set ratio (FP traps)
        if len(neg_cands_pool) > 3:
            remaining = neg_cands_pool[3:]
            name_sims = sorted(remaining, key=lambda c: -fuzz.token_set_ratio(name, s23_lookup[c][0]))[:2]
            hard_negs.update(name_sims)
            # 3. Top 1 by address token set ratio
            remaining = [c for c in remaining if c not in hard_negs]
            if remaining:
                addr_sims = sorted(remaining, key=lambda c: -fuzz.token_set_ratio(addr, s23_lookup[c][1]))[:1]
                hard_negs.update(addr_sims)

        for cid in pos_cands:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_pair_features_v2(s1_pre, n2, a2, c2, name_df, addr_df)
            X_rows.append(f)
            y_rows.append(1)
            n_pos += 1

        for cid in hard_negs:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_pair_features_v2(s1_pre, n2, a2, c2, name_df, addr_df)
            X_rows.append(f)
            y_rows.append(0)
            n_neg += 1

        if (i + 1) % log_interval == 0 or (i + 1) == len(s1):
            elapsed = time.time() - t_loop0
            s1_rate = (i + 1) / elapsed
            eta_min = (len(s1) - (i + 1)) / max(s1_rate, 0.001) / 60.0
            cur_ram, _peak_ram = get_ram_mb()
            log(
                f"[{i+1:6d}/{len(s1)}] {100*(i+1)/len(s1):5.1f}% | Elapsed: {elapsed/60:4.1f}m | Rate: {s1_rate:4.1f} S1/s | Pairs: {n_pos+n_neg:7d} ({n_pos} pos, {n_neg} neg) | RAM: {cur_ram:.1f} MB | ETA: {eta_min:4.1f}m"
            )

    log("\nAssembling feature array and saving artifacts...")
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)

    assert not np.isnan(X).any(), "NaN found in X!"
    assert not np.isinf(X).any(), "Inf found in X!"

    np.save(EXP2_DIR / "train_X.npy", X)
    np.save(EXP2_DIR / "train_y.npy", y)

    total_time = time.time() - t_start
    _final_ram, final_peak_ram = get_ram_mb()

    rep = {
        "train_s1_count": len(s1),
        "total_truth_links_in_sample": total_truth_links,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "total_pairs": len(X),
        "negative_sampling_ratio": round(n_neg / max(n_pos, 1), 3),
        "candidate_retrieval_positive_coverage_pct": round(100.0 * n_pos / max(total_truth_links, 1), 3),
        "candidate_cap": CAP,
        "random_seed": SEED,
        "runtime_seconds": round(total_time, 1),
        "peak_ram_mb": round(final_peak_ram, 1),
        "feature_count": len(ALL_FEATURE_NAMES),
        "feature_names": ALL_FEATURE_NAMES,
    }
    with open(EXP2_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    log("\nTraining pairs build complete:")
    log(f"  Positives: {n_pos}")
    log(f"  Negatives: {n_neg}")
    log(f"  Total pairs: {len(X)} (Shape: {X.shape})")
    log(f"  Neg:Pos ratio: {rep['negative_sampling_ratio']}")
    log(f"  Training candidate positive coverage: {rep['candidate_retrieval_positive_coverage_pct']}%")
    log(f"  Total time: {total_time/60:.2f} min (Peak RAM: {final_peak_ram:.1f} MB)")


if __name__ == "__main__":
    main()
