"""EXP-001 Step 7 safety benchmark: test candidate-budget curve on 1,000 validation S1 entities.
Verifies determinism, nested caps, valid IDs, metric formulas, and performance projection.
"""

import ctypes
import heapq
import time
from ctypes import wintypes

import pandas as pd

from blocking import Blocker
from io_utils import explode_ground_truth, load_ground_truth
from paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "experiments" / "EXP-001"
CACHE = PROJECT_ROOT / "data" / "interim" / "exp001"
CAPS = [50, 100, 200, 500, 1000]

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


def load_val(n=1000):
    val_ids = pd.read_csv(EXP_DIR / "validation_s1_ids.csv")
    s1 = pd.read_parquet(CACHE / "train_s1.parquet")
    s1 = s1[s1.entity_id.isin(set(val_ids.entity_id))].reset_index(drop=True)
    if n is not None and len(s1) > n:
        s1 = s1.iloc[:n].copy()
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


def main():
    print("--- EXP-001 Step 7: Safety Benchmark (1,000 S1) ---")
    t0_init = time.time()
    s1 = load_val(1000)
    s23 = load_s23()
    s23_lookup = dict(
        zip(s23.entity_id.astype(object), zip(s23.name_norm.astype(object), s23.address_norm.astype(object)))
    )
    blocker = Blocker(s23)
    name_df = dict(blocker.name_idx.df_global)
    addr_df = dict(blocker.addr_idx.df_global)
    truth = truth_map(s1)
    valid_s23_ids = set(s23.entity_id.astype(object))
    del s23

    init_time = time.time() - t0_init
    cur_ram, peak_ram = get_ram_mb()
    print(f"Setup complete in {init_time:.1f}s. RAM: {cur_ram:.1f} MB (Peak: {peak_ram:.1f} MB)")

    total_truth_links = sum(len(v) for v in truth.values())
    non_singleton_entities = [eid for eid, t in truth.items() if len(t) > 0]
    sum(len(truth[eid]) for eid in non_singleton_entities)
    print(
        f"Benchmark subset: {len(s1)} S1 entities ({len(non_singleton_entities)} non-singletons, {total_truth_links} true links)"
    )

    # Test processing loop
    t0 = time.time()
    total_raw_cands = 0
    cap_retained_tp = {cap: 0 for cap in CAPS}
    cap_counts = {cap: [] for cap in CAPS}
    cap_all_true = {cap: 0 for cap in CAPS}
    cap_at_least_one = {cap: 0 for cap in CAPS}

    # Tracking raw B5 reference on this 1,000 subset
    raw_b5_retained_tp = 0
    raw_b5_counts = []
    raw_b5_all_true = 0
    raw_b5_at_least_one = 0

    verification_errors = []

    for i, (eid, name, addr, country) in enumerate(
        zip(
            s1.entity_id.astype(object),
            s1.name_norm.astype(object),
            s1.address_norm.astype(object),
            s1.country.astype(object),
        )
    ):
        cand = blocker.candidates(name, addr, country, 1, 1, ("name", "addr"), country_soft=True)
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

        # Check candidate IDs valid
        if not cand.issubset(valid_s23_ids):
            verification_errors.append(f"{eid}: invalid S2/S3 candidate ID found")

        # Precompute S1 tokens once
        s1_t1 = set(name.split())
        s1_a1s = set(addr.split())
        s1_n1s = {tok for tok in s1_a1s if any(c.isdigit() for c in tok)}

        # Rank candidates
        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2 = s23_lookup[cid]
            scores[cid] = compute_rank_score(name, addr, s1_t1, s1_a1s, s1_n1s, n2, a2, name_df, addr_df)

        # Select top-1000 (or all if <= 1000)
        if len(cand_list) > 1000:
            ranked_top1000 = heapq.nlargest(1000, cand_list, key=scores.__getitem__)
        else:
            ranked_top1000 = sorted(cand_list, key=lambda c: -scores[c])

        # Evaluate nested caps
        ranked_sets = {}
        for cap in CAPS:
            cap_cand_list = ranked_top1000[:cap]
            cap_cand_set = set(cap_cand_list)
            ranked_sets[cap] = cap_cand_set

            cap_counts[cap].append(len(cap_cand_set))
            hits = len(t & cap_cand_set)
            cap_retained_tp[cap] += hits

            if is_multi:
                if t.issubset(cap_cand_set):
                    cap_all_true[cap] += 1
                if hits > 0:
                    cap_at_least_one[cap] += 1

        # Verify nesting property: top50 <= top100 <= top200 <= top500 <= top1000
        if not (ranked_sets[50] <= ranked_sets[100] <= ranked_sets[200] <= ranked_sets[500] <= ranked_sets[1000]):
            verification_errors.append(f"{eid}: nested caps property violated")

        if (i + 1) % 250 == 0:
            sub_elapsed = time.time() - t0
            cur_ram_step, _ = get_ram_mb()
            print(
                f"[{i+1}/{len(s1)}] {i+1} S1 processed in {sub_elapsed:.1f}s ({(i+1)/sub_elapsed:.1f} S1/s, {total_raw_cands/sub_elapsed:.0f} cands/s) | RAM: {cur_ram_step:.1f} MB"
            )

    bench_elapsed = time.time() - t0
    s1_per_sec = len(s1) / bench_elapsed
    cands_per_sec = total_raw_cands / bench_elapsed
    projected_100k_sec = 100_000 / s1_per_sec
    cur_ram, peak_ram = get_ram_mb()

    print("\n--- Benchmark Results ---")
    print(f"Processed: {len(s1)} S1 in {bench_elapsed:.2f} s")
    print(f"Speed: {s1_per_sec:.1f} S1/sec, {cands_per_sec:.0f} candidates/sec")
    print(f"RAM: {cur_ram:.1f} MB (Peak: {peak_ram:.1f} MB)")
    print(f"Projected 100k S1 runtime: {projected_100k_sec:.1f} s ({projected_100k_sec/60:.1f} min)")
    print(f"Verification errors: {len(verification_errors)}")

    print("\n--- 1k Budget Curve Preview ---")
    raw_s = pd.Series(raw_b5_counts)
    print(
        f"RAW B5: recall={100*raw_b5_retained_tp/total_truth_links:.3f}%, avg={raw_s.mean():.1f}, all_true={100*raw_b5_all_true/len(non_singleton_entities):.2f}%, any_true={100*raw_b5_at_least_one/len(non_singleton_entities):.2f}%"
    )
    for cap in CAPS:
        s = pd.Series(cap_counts[cap])
        rec = 100 * cap_retained_tp[cap] / total_truth_links
        all_t = 100 * cap_all_true[cap] / len(non_singleton_entities)
        any_t = 100 * cap_at_least_one[cap] / len(non_singleton_entities)
        print(
            f"Cap {cap:4d}: recall={rec:6.3f}%, TP={cap_retained_tp[cap]}, avg={s.mean():6.2f}, p95={s.quantile(0.95):5.1f}, all_true={all_t:5.2f}%, any_true={any_t:5.2f}%"
        )


if __name__ == "__main__":
    main()
