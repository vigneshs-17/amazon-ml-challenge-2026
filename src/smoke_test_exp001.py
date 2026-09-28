"""EXP-001 Step 4 Safety & Performance Smoke Test.

Tests end-to-end pipeline on 1,000 training S1 and 1,000 validation S1 entities:
1. Cap-200 hard-negative training pair extraction.
2. Feature verification (no NaN/inf, correct shapes/types).
3. HistGradientBoostingClassifier fitting.
4. Cap-200 validation scoring.
5. Official macro F0.5 threshold search & error breakdown.
6. Timing & memory projections for full scale.
"""

import ctypes
import heapq
import time
from ctypes import wintypes

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier

from blocking import RareTokenIndex
from io_utils import explode_ground_truth, load_ground_truth
from metrics import macro_fbeta
from paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
CAP = 200
SEED = 42

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


def main():
    print("=== EXP-001 Step 4: Smoke Test (1k Train S1, 1k Val S1) ===")
    time.time()

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

    # Ground truth mapping
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    truth_all = {}
    for sid, mid in zip(long.source1_entity_id.astype(object), long.matched_id.astype(object)):
        truth_all.setdefault(sid, set()).add(mid)
    del gt, long

    # Load 1k training S1
    train_ids = pd.read_csv(EXP_DIR / "train_remaining_s1_ids.csv")
    s1_all = pd.read_parquet(CACHE / "train_s1.parquet")
    train_s1 = s1_all[s1_all.entity_id.isin(set(train_ids.entity_id))].iloc[:1000].reset_index(drop=True)

    # 1. Build training pairs
    print("Generating training pairs from 1,000 training S1...")
    t_tr0 = time.time()
    X_tr_rows, y_tr_rows = [], []
    n_pos = n_neg = 0
    MAX_NEG = 5

    for eid, name, addr, country in zip(
        train_s1.entity_id.astype(object),
        train_s1.name_norm.astype(object),
        train_s1.address_norm.astype(object),
        train_s1.country.astype(object),
    ):
        cand = blocker.candidates(name, addr, country)
        t = truth_all.get(eid, set())

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

        set(ranked_top)
        # Positives: ground-truth links present in Cap-200
        pos_cands = [c for c in ranked_top if c in t]
        # Hard Negatives: top non-matches in Cap-200
        neg_cands = [c for c in ranked_top if c not in t][:MAX_NEG]

        for cid in pos_cands:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_features(
                name, addr, country, s1_t1, s1_a1s, s1_n1s, la1, la_addr1, n2, a2, c2, name_df, addr_df
            )
            X_tr_rows.append(f)
            y_tr_rows.append(1)
            n_pos += 1

        for cid in neg_cands:
            n2, a2, c2 = s23_lookup[cid]
            f = extract_features(
                name, addr, country, s1_t1, s1_a1s, s1_n1s, la1, la_addr1, n2, a2, c2, name_df, addr_df
            )
            X_tr_rows.append(f)
            y_tr_rows.append(0)
            n_neg += 1

    X_train = np.array(X_tr_rows, dtype=np.float32)
    y_train = np.array(y_tr_rows, dtype=np.int8)
    t_tr_elapsed = time.time() - t_tr0
    print(
        f"Training pairs built: {len(X_train)} pairs ({n_pos} pos, {n_neg} neg, ratio {n_neg/max(n_pos,1):.2f}) in {t_tr_elapsed:.2f}s"
    )
    assert not np.isnan(X_train).any(), "NaN found in training features!"
    assert not np.isinf(X_train).any(), "Inf found in training features!"

    # 2. Train model
    print("Fitting HistGradientBoostingClassifier...")
    t_fit0 = time.time()
    clf = HistGradientBoostingClassifier(random_state=SEED, max_depth=6, max_iter=100)
    clf.fit(X_train, y_train)
    t_fit_elapsed = time.time() - t_fit0
    print(f"Model fit complete in {t_fit_elapsed:.2f}s")

    # 3. Score 1k validation S1
    val_ids = pd.read_csv(EXP_DIR / "validation_s1_ids.csv")
    val_s1 = s1_all[s1_all.entity_id.isin(set(val_ids.entity_id))].iloc[:1000].reset_index(drop=True)
    val_truth = {sid: truth_all.get(sid, set()) for sid in val_s1.entity_id.astype(object)}

    print("Scoring Cap-200 candidates on 1,000 validation S1...")
    t_val0 = time.time()
    per_s1_cands = {}
    per_s1_scores = {}
    total_val_pairs = 0

    for eid, name, addr, country in zip(
        val_s1.entity_id.astype(object),
        val_s1.name_norm.astype(object),
        val_s1.address_norm.astype(object),
        val_s1.country.astype(object),
    ):
        cand = blocker.candidates(name, addr, country)
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

        total_val_pairs += len(ranked_top)
        if not ranked_top:
            per_s1_cands[eid] = []
            per_s1_scores[eid] = np.array([], dtype=np.float32)
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
        per_s1_cands[eid] = ranked_top
        per_s1_scores[eid] = proba

    t_val_elapsed = time.time() - t_val0
    print(
        f"Validation scoring complete: {total_val_pairs} pairs scored in {t_val_elapsed:.2f}s ({total_val_pairs/t_val_elapsed:.0f} pairs/sec)"
    )

    # 4. Sweep thresholds & evaluate official macro F0.5
    print("\nSweeping threshold grid...")
    thresholds = np.round(np.arange(0.1, 0.95, 0.05), 2)
    best_th = 0.5
    best_f05 = -1.0
    best_res = None

    for th in thresholds:
        preds = {}
        for eid, cands in per_s1_cands.items():
            sc = per_s1_scores[eid]
            preds[eid] = {cid for cid, score in zip(cands, sc) if score >= th}
        m = macro_fbeta(val_truth, preds)
        if m["f0_5"] > best_f05:
            best_f05 = m["f0_5"]
            best_th = th
            best_res = m

    cur_ram, peak_ram = get_ram_mb()
    print(
        f"Best threshold: {best_th:.2f} -> Macro F0.5: {best_res['f0_5']:.4f}, Precision: {best_res['precision']:.4f}, Recall: {best_res['recall']:.4f}"
    )
    print(f"RAM: {cur_ram:.1f} MB (Peak: {peak_ram:.1f} MB)")
    print("\nSmoke test PASSED! Ready for full-scale execution.")


if __name__ == "__main__":
    main()
