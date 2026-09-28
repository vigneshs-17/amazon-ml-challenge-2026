"""EXP-001 Step 7: candidate-budget curve (optimized, single-pass).

Uses the selected generator B5 (name+addr rare-1 union, country-soft).
Computes the original deterministic evidence-based ranking score for each
candidate, extracts top-1000 candidates per S1 once, and evaluates all budget
caps [50, 100, 200, 500, 1000] simultaneously in a single pass over the
100,000 validation S1 entities.
"""

import ctypes
import heapq
import json
import time
from ctypes import wintypes

import pandas as pd

from blocking import RareTokenIndex
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
LOG_FILE = PROJECT_ROOT / "logs" / "run_budget_curve.log"
CAPS = [50, 100, 200, 500, 1000]
QTS = [0.5, 0.9, 0.95, 0.99, 1.0]

# Windows ctypes memory tracking
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


def load_val():
    val_ids = pd.read_csv(EXP_DIR / "validation_s1_ids.csv")
    s1 = pd.read_parquet(CACHE / "train_s1.parquet")
    s1 = s1[s1.entity_id.isin(set(val_ids.entity_id))].reset_index(drop=True)
    return s1


def load_s23():
    cols = ["entity_id", "name_norm", "address_norm", "country"]
    s2 = pd.read_parquet(CACHE / "train_s2.parquet", columns=cols)
    s3 = pd.read_parquet(CACHE / "train_s3.parquet", columns=cols)
    return pd.concat([s2, s3], ignore_index=True)


def truth_map(s1):
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    ids = set(s1.entity_id.astype(object))
    long = long[long.source1_entity_id.astype(object).isin(ids)]
    truth = {sid: set() for sid in ids}
    for sid, mid in zip(long.source1_entity_id.astype(object), long.matched_id.astype(object)):
        truth[sid].add(mid)
    return truth


def compute_rank_score(s1_name, s1_addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df):
    """Original deterministic ranking formula (preserves exact Phase-1 precision weights)."""
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
    """Dedicated B5 candidate generator: rare-1 name + rare-1 address with soft country fallback."""

    def __init__(self, s23: pd.DataFrame):
        self.name_idx = RareTokenIndex(s23, "name_norm")
        self.addr_idx = RareTokenIndex(s23, "address_norm")

    def candidates(self, name: str, addr: str, country: str):
        # Name channel
        name_cand = self.name_idx.query_union(name, country, 1, country_aware=True)
        if not name_cand:
            name_cand = self.name_idx.query_union(name, country, 1, country_aware=False)
        # Addr channel
        addr_cand = self.addr_idx.query_union(addr, country, 1, country_aware=True)
        if not addr_cand:
            addr_cand = self.addr_idx.query_union(addr, country, 1, country_aware=False)
        return name_cand | addr_cand


def main():
    # Initialize log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== EXP-001 Step 7: Candidate Budget Curve Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    t_start = time.time()
    log("Loading validation S1 and S2/S3 pools...")
    s1 = load_val()
    s23 = load_s23()
    log(f"Loaded {len(s1)} validation S1 and {len(s23)} S2/S3 rows.")

    log("Building candidate lookup dict and B5 inverted indexes...")
    t_idx0 = time.time()
    s23_lookup = dict(
        zip(s23.entity_id.astype(object), zip(s23.name_norm.astype(object), s23.address_norm.astype(object)))
    )
    blocker = B5Blocker(s23)
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    del s23

    truth = truth_map(s1)
    total_truth_links = sum(len(v) for v in truth.values())
    non_singleton_entities = [eid for eid, t in truth.items() if len(t) > 0]
    non_singleton_truth_links = sum(len(truth[eid]) for eid in non_singleton_entities)
    n_singletons = len(s1) - len(non_singleton_entities)

    setup_time = time.time() - t_idx0
    cur_ram, peak_ram = get_ram_mb()
    log(f"Setup complete in {setup_time:.1f}s. Memory: {cur_ram:.1f} MB (Peak: {peak_ram:.1f} MB)")
    log(
        f"Validation summary: {len(s1)} S1 entities | {n_singletons} singletons | {len(non_singleton_entities)} non-singletons | {total_truth_links} true links"
    )

    # Data structures for tracking metrics per cap
    cap_retained_tp = {cap: 0 for cap in CAPS}
    cap_counts = {cap: [] for cap in CAPS}
    cap_all_true = {cap: 0 for cap in CAPS}
    cap_at_least_one = {cap: 0 for cap in CAPS}

    # Tracking raw B5 metrics
    raw_b5_retained_tp = 0
    raw_b5_counts = []
    raw_b5_all_true = 0
    raw_b5_at_least_one = 0

    total_s1 = len(s1)
    total_raw_cands = 0
    t_eval0 = time.time()
    log_interval = 2500

    log("\nStarting single-pass evaluation across all 100,000 S1 entities...")

    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1.entity_id.astype(object),
            s1.name_norm.astype(object),
            s1.address_norm.astype(object),
            s1.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country)
        t = truth[eid]
        is_multi = len(t) > 0
        n_cand = len(cand)
        total_raw_cands += n_cand

        # Raw B5 metrics
        raw_b5_counts.append(n_cand)
        raw_hits = len(t & cand)
        raw_b5_retained_tp += raw_hits
        if is_multi:
            if t.issubset(cand):
                raw_b5_all_true += 1
            if raw_hits > 0:
                raw_b5_at_least_one += 1

        # Precompute S1 tokens once
        s1_t1 = set(name.split())
        s1_a1s = set(addr.split())
        s1_n1s = {tok for tok in s1_a1s if any(c.isdigit() for c in tok)}

        # Rank candidates deterministically
        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2 = s23_lookup[cid]
            scores[cid] = compute_rank_score(name, addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df)

        # Top-1000 selection
        if len(cand_list) > 1000:
            ranked_top1000 = heapq.nlargest(1000, cand_list, key=scores.__getitem__)
        else:
            ranked_top1000 = sorted(cand_list, key=lambda c: -scores[c])

        # Evaluate all caps simultaneously from the same ranking
        for cap in CAPS:
            cap_cand_list = ranked_top1000[:cap]
            cap_cand_set = set(cap_cand_list)

            cap_counts[cap].append(len(cap_cand_set))
            hits = len(t & cap_cand_set)
            cap_retained_tp[cap] += hits

            if is_multi:
                if t.issubset(cap_cand_set):
                    cap_all_true[cap] += 1
                if hits > 0:
                    cap_at_least_one[cap] += 1

        # Periodic progress logging
        if (i + 1) % log_interval == 0 or (i + 1) == total_s1:
            elapsed = time.time() - t_eval0
            s1_rate = (i + 1) / elapsed
            cand_rate = total_raw_cands / elapsed
            pct = 100.0 * (i + 1) / total_s1
            remaining_s1 = total_s1 - (i + 1)
            eta_sec = remaining_s1 / max(s1_rate, 0.001)
            eta_min = eta_sec / 60.0
            cur_ram_step, peak_ram_step = get_ram_mb()

            log(
                f"[{i+1:6d}/{total_s1}] {pct:5.1f}% | Elapsed: {elapsed/60:4.1f}m | Rate: {s1_rate:4.1f} S1/s ({cand_rate:6.0f} cands/s) | RAM: {cur_ram_step:5.1f} MB (Peak: {peak_ram_step:5.1f} MB) | ETA: {eta_min:4.1f}m"
            )

    eval_total_time = time.time() - t_eval0
    overall_total_time = time.time() - t_start
    _final_ram, final_peak_ram = get_ram_mb()

    log("\n==================== BUDGET CURVE RESULTS ====================")
    log(f"Total Evaluation Time: {eval_total_time:.1f} s ({eval_total_time/60:.2f} min)")
    log(f"Total Job Time: {overall_total_time:.1f} s ({overall_total_time/60:.2f} min)")
    log(f"Final Peak RAM: {final_peak_ram:.1f} MB")

    # Compile results table
    results = []
    # Reference RAW B5
    raw_s = pd.Series(raw_b5_counts)
    raw_rec = round(100.0 * raw_b5_retained_tp / total_truth_links, 3)
    raw_non_single_rec = round(100.0 * raw_b5_retained_tp / non_singleton_truth_links, 3)
    raw_tp_missed = total_truth_links - raw_b5_retained_tp
    raw_all_pct = round(100.0 * raw_b5_all_true / len(non_singleton_entities), 2)
    raw_any_pct = round(100.0 * raw_b5_at_least_one / len(non_singleton_entities), 2)

    results.append(
        {
            "cap": "RAW_B5",
            "blocking_recall_pct": raw_rec,
            "non_singleton_recall_pct": raw_non_single_rec,
            "tp_retained": int(raw_b5_retained_tp),
            "tp_missed": int(raw_tp_missed),
            "total_candidate_pairs": int(raw_s.sum()),
            "avg_candidates_per_s1": round(float(raw_s.mean()), 2),
            "median_candidates_per_s1": float(raw_s.median()),
            "p90_candidates_per_s1": float(raw_s.quantile(0.90)),
            "p95_candidates_per_s1": float(raw_s.quantile(0.95)),
            "p99_candidates_per_s1": float(raw_s.quantile(0.99)),
            "max_candidates_per_s1": float(raw_s.max()),
            "pct_s1_zero_candidates": round(100.0 * float((raw_s == 0).mean()), 3),
            "all_true_entity_pct": raw_all_pct,
            "at_least_one_true_entity_pct": raw_any_pct,
            "runtime_seconds": round(eval_total_time, 1),
        }
    )

    for cap in CAPS:
        s = pd.Series(cap_counts[cap])
        rec = round(100.0 * cap_retained_tp[cap] / total_truth_links, 3)
        non_single_rec = round(100.0 * cap_retained_tp[cap] / non_singleton_truth_links, 3)
        tp_ret = cap_retained_tp[cap]
        tp_mis = total_truth_links - tp_ret
        all_pct = round(100.0 * cap_all_true[cap] / len(non_singleton_entities), 2)
        any_pct = round(100.0 * cap_at_least_one[cap] / len(non_singleton_entities), 2)

        results.append(
            {
                "cap": cap,
                "blocking_recall_pct": rec,
                "non_singleton_recall_pct": non_single_rec,
                "tp_retained": int(tp_ret),
                "tp_missed": int(tp_mis),
                "total_candidate_pairs": int(s.sum()),
                "avg_candidates_per_s1": round(float(s.mean()), 2),
                "median_candidates_per_s1": float(s.median()),
                "p90_candidates_per_s1": float(s.quantile(0.90)),
                "p95_candidates_per_s1": float(s.quantile(0.95)),
                "p99_candidates_per_s1": float(s.quantile(0.99)),
                "max_candidates_per_s1": float(s.max()),
                "pct_s1_zero_candidates": round(100.0 * float((s == 0).mean()), 3),
                "all_true_entity_pct": all_pct,
                "at_least_one_true_entity_pct": any_pct,
                "runtime_seconds": round(eval_total_time, 1),
            }
        )

    # Save artifacts
    results_df = pd.DataFrame(results)
    results_df.to_csv(EXP_DIR / "budget_curve.csv", index=False)
    with open(EXP_DIR / "budget_curve.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    runtime_report = {
        "setup_seconds": round(setup_time, 1),
        "evaluation_seconds": round(eval_total_time, 1),
        "total_seconds": round(overall_total_time, 1),
        "validation_s1_count": total_s1,
        "total_raw_candidates": total_raw_cands,
        "s1_per_sec": round(total_s1 / eval_total_time, 1),
        "candidates_per_sec": round(total_raw_cands / eval_total_time, 0),
        "peak_ram_mb": round(final_peak_ram, 1),
    }
    with open(EXP_DIR / "budget_curve_runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime_report, f, indent=2)

    log("\nResults saved to:")
    log(f"  - {EXP_DIR / 'budget_curve.json'}")
    log(f"  - {EXP_DIR / 'budget_curve.csv'}")
    log(f"  - {EXP_DIR / 'budget_curve_runtime.json'}")

    log("\nSummary Table:")
    log(
        f"{'Cap':>8} | {'Recall':>7} | {'TP Ret':>7} | {'TP Miss':>7} | {'Avg Cand':>8} | {'P95':>6} | {'All-True%':>9} | {'At-Least-1%':>11}"
    )
    log("-" * 80)
    for r in results:
        cap_str = str(r["cap"])
        log(
            f"{cap_str:>8} | {r['blocking_recall_pct']:6.3f}% | {r['tp_retained']:7d} | {r['tp_missed']:7d} | {r['avg_candidates_per_s1']:8.2f} | {r['p95_candidates_per_s1']:6.1f} | {r['all_true_entity_pct']:8.2f}% | {r['at_least_one_true_entity_pct']:10.2f}%"
        )

    log("\nEXP-001 candidate budget curve completed successfully.")


if __name__ == "__main__":
    main()
