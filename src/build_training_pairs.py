"""EXP-001 Step 10: build training pairs from Cap-200 candidate generator.

Samples 50,000 training S1 entities from train_remaining_s1_ids.csv (disjoint
from validation). For each training S1:
- Generates B5 candidates.
- Ranks candidates using the deterministic evidence score.
- Slices to Cap-200.
- Positives = ground-truth links present in Cap-200.
- Hard Negatives = top-ranking non-matches in Cap-200 (up to MAX_NEG_PER_S1=5).
- Extracts interpretable pair features (22 features).
- Saves train_X.npy, train_y.npy, and training_pairs_report.json.
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
from features import FEATURE_NAMES
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
LOG_FILE = PROJECT_ROOT / "logs" / "build_training_pairs.log"
CAP = 200
SEED = 42
N_TRAIN_S1 = 50_000
MAX_NEG_PER_S1 = 5

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


def load_train_s1():
    train_ids_df = pd.read_csv(EXP_DIR / "train_remaining_s1_ids.csv")
    s1_all = pd.read_parquet(CACHE / "train_s1.parquet")
    s1_train = s1_all[s1_all.entity_id.isin(set(train_ids_df.entity_id))].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    if len(s1_train) > N_TRAIN_S1:
        idx = rng.choice(s1_train.index.values, size=N_TRAIN_S1, replace=False)
        s1_train = s1_train.loc[idx].reset_index(drop=True)
    return s1_train


def load_s23():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    return pd.concat([s2, s3], ignore_index=True)


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== EXP-001 Build Training Pairs Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    t_start = time.time()
    log(f"Sampling {N_TRAIN_S1} training S1 entities (Seed: {SEED})...")
    s1 = load_train_s1()
    log(f"Sampled {len(s1)} training S1 entities.")

    log("Loading S2/S3 pools and building B5 indexes...")
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
    total_raw_cands = 0
    t_loop0 = time.time()
    log_interval = 2500

    log(f"\nBuilding Cap-{CAP} training pairs (Max hard negs per S1: {MAX_NEG_PER_S1})...")

    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1.entity_id.astype(object),
            s1.name_norm.astype(object),
            s1.address_norm.astype(object),
            s1.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country)
        total_raw_cands += len(cand)
        t = truth.get(eid, set())

        s1_t1 = set(name.split())
        s1_a1s = set(addr.split())
        s1_n1s = {tok for tok in s1_a1s if any(c.isdigit() for c in tok)}
        la1, la_addr1 = len(name), len(addr)

        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score(name, addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df)

        if len(cand_list) > CAP:
            ranked_top = heapq.nlargest(CAP, cand_list, key=scores.__getitem__)
        else:
            ranked_top = sorted(cand_list, key=lambda c: -scores[c])

        # Positives: ground-truth links present in Cap-200
        pos_cands = [c for c in ranked_top if c in t]
        # Hard Negatives: top non-matches in Cap-200
        neg_cands = [c for c in ranked_top if c not in t][:MAX_NEG_PER_S1]

        for cid in pos_cands:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_features(
                name, addr, country, s1_t1, s1_a1s, s1_n1s, la1, la_addr1, n2, a2, c2, name_df, addr_df
            )
            X_rows.append(f)
            y_rows.append(1)
            n_pos += 1

        for cid in neg_cands:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_features(
                name, addr, country, s1_t1, s1_a1s, s1_n1s, la1, la_addr1, n2, a2, c2, name_df, addr_df
            )
            X_rows.append(f)
            y_rows.append(0)
            n_neg += 1

        if (i + 1) % log_interval == 0 or (i + 1) == len(s1):
            elapsed = time.time() - t_loop0
            s1_rate = (i + 1) / elapsed
            eta_min = (len(s1) - (i + 1)) / max(s1_rate, 0.001) / 60.0
            cur_ram, _peak_ram = get_ram_mb()
            log(
                f"[{i+1:5d}/{len(s1)}] {100*(i+1)/len(s1):5.1f}% | Elapsed: {elapsed/60:4.1f}m | Rate: {s1_rate:4.1f} S1/s | Pairs: {n_pos+n_neg:6d} ({n_pos} pos, {n_neg} neg) | RAM: {cur_ram:.1f} MB | ETA: {eta_min:4.1f}m"
            )

    log("\nAssembling feature array and saving artifacts...")
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)

    assert not np.isnan(X).any(), "NaN found in X!"
    assert not np.isinf(X).any(), "Inf found in X!"

    np.save(EXP_DIR / "train_X.npy", X)
    np.save(EXP_DIR / "train_y.npy", y)

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
        "max_neg_per_s1": MAX_NEG_PER_S1,
        "random_seed": SEED,
        "runtime_seconds": round(total_time, 1),
        "peak_ram_mb": round(final_peak_ram, 1),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }
    with open(EXP_DIR / "training_pairs_report.json", "w", encoding="utf-8") as f:
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
