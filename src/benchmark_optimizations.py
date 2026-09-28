"""Micro-benchmark and Equivalence Test for Candidate Ranking and Feature Extraction."""

import heapq
import sys
import time

sys.path.insert(0, "src")
from ranking_v2 import (
    compute_rank_score_v2,
    precompute_global_idfs,
    precompute_s1_ranking,
)
from run_exp004 import B5Blocker, load_s23, load_val


def compute_rank_score_fast(s1_name, s1_addr, t1, a1, name_idfs, addr_idfs, s1_postals, len_t1, len_a1, n2, a2):
    score = 0.0
    if s1_addr and s1_addr == a2:
        score += 15.0
    if s1_name and s1_name == n2:
        score += 10.0

    t2 = set(n2.split()) if n2 else set()
    a2_set = set(a2.split()) if a2 else set()

    shared_n = t1 & t2
    shared_a = a1 & a2_set

    if shared_n:
        score += sum(name_idfs[t] for t in shared_n)
    if shared_a:
        score += sum(addr_idfs[t] for t in shared_a)

    if s1_postals and (s1_postals & a2_set):
        score += 5.0

    if shared_n and shared_a:
        score += 4.0

    if shared_n:
        u_n_len = len_t1 + len(t2) - len(shared_n)
        if u_n_len > 0:
            score += 3.0 * (len(shared_n) / u_n_len)

    if shared_a:
        u_a_len = len_a1 + len(a2_set) - len(shared_a)
        if u_a_len > 0:
            score += 3.0 * (len(shared_a) / u_a_len)

    return score


def main():
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
    name_idf, addr_idf = precompute_global_idfs(name_df, addr_df)
    del s23

    s1_sample = load_val().iloc[:100]

    def baseline_ranking(eid, name, addr, country):
        cand = blocker.candidates(name, addr, country)
        s1_rank_pre = precompute_s1_ranking(name, addr, name_idf, addr_idf)
        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score_v2(s1_rank_pre, n2, a2)
        if len(cand_list) > 200:
            ranked_top = heapq.nlargest(200, cand_list, key=lambda c: (scores[c], c))
        else:
            ranked_top = sorted(cand_list, key=lambda c: (-scores[c], c))
        return ranked_top, scores

    def opt1_ranking(eid, name, addr, country):
        cand = blocker.candidates(name, addr, country)
        t1 = set(name.split()) if name else set()
        a1 = set(addr.split()) if addr else set()
        num_tokens = {t for t in a1 if any(c.isdigit() for c in t)}
        s1_postals = {t for t in num_tokens if len(t) in (5, 6) and t.isdigit()}
        name_idfs = {t: name_idf.get(t, 0.0) for t in t1}
        addr_idfs = {t: addr_idf.get(t, 0.0) for t in a1}
        len_t1 = len(t1)
        len_a1 = len(a1)

        cand_list = list(cand)
        scores = {}
        for cid in cand_list:
            n2, a2, _ = s23_lookup[cid]
            scores[cid] = compute_rank_score_fast(
                name, addr, t1, a1, name_idfs, addr_idfs, s1_postals, len_t1, len_a1, n2, a2
            )

        if len(cand_list) > 200:
            ranked_top = heapq.nlargest(200, cand_list, key=lambda c: (scores[c], c))
        else:
            ranked_top = sorted(cand_list, key=lambda c: (-scores[c], c))
        return ranked_top, scores

    # Test equivalence & timing on 100 entities
    print("Running equivalence test on 100 S1...", flush=True)
    mismatches = 0
    for eid, name, addr, country in zip(
        s1_sample.entity_id.astype(object),
        s1_sample.name_norm.astype(object),
        s1_sample.address_norm.astype(object),
        s1_sample.country.astype(object),
    ):
        r_base, s_base = baseline_ranking(eid, name, addr, country)
        r_opt1, s_opt1 = opt1_ranking(eid, name, addr, country)
        if r_base != r_opt1:
            mismatches += 1
        for cid in r_base:
            if abs(s_base[cid] - s_opt1[cid]) > 1e-6:
                mismatches += 1

    print(f"Equivalence test: {mismatches} mismatches out of 100 S1 (100% MATCH: {mismatches==0})")

    # Timing comparison
    t0 = time.perf_counter()
    for eid, name, addr, country in zip(
        s1_sample.entity_id, s1_sample.name_norm, s1_sample.address_norm, s1_sample.country
    ):
        baseline_ranking(eid, name, addr, country)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    for eid, name, addr, country in zip(
        s1_sample.entity_id, s1_sample.name_norm, s1_sample.address_norm, s1_sample.country
    ):
        opt1_ranking(eid, name, addr, country)
    t_opt1 = time.perf_counter() - t0

    print(f"Timing (100 S1): Baseline = {t_base:.2f}s | Opt1 = {t_opt1:.2f}s | Speedup = {t_base/t_opt1:.2f}x")


if __name__ == "__main__":
    main()
