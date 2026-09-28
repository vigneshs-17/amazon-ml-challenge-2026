"""EXP-002 Feature Engineering Module.

Contains 42 interpretable features grouped into 4 ablation slices:
- Slice A (indices 0..21, 22 feats): EXP-001 baseline features
- Slice B (indices 0..31, 32 feats): Slice A + 10 token-IDF weighted features
- Slice C (indices 0..35, 36 feats): Slice B + 4 character 3/4-gram Jaccard features
- Slice D (indices 0..41, 42 feats): Slice C + 6 address structure & cross-field features
"""

import math

from rapidfuzz import fuzz

N_CORPUS_RECORDS = 10_320_219

FEATURE_NAMES_A = [
    "name_exact_eq",
    "name_tok_jaccard",
    "name_tok_containment",
    "name_ratio",
    "name_token_set_ratio",
    "name_len_ratio",
    "name_shared_tok_count",
    "name_min_shared_df",
    "addr_exact_eq",
    "addr_tok_jaccard",
    "addr_tok_containment",
    "addr_ratio",
    "addr_token_set_ratio",
    "addr_len_ratio",
    "addr_shared_tok_count",
    "addr_min_shared_df",
    "num_tok_intersection",
    "num_tok_jaccard",
    "num_tok_conflict",
    "country_eq",
    "name_missing",
    "addr_missing",
]

FEATURE_NAMES_B = FEATURE_NAMES_A + [
    "name_weighted_jaccard",
    "name_weighted_containment",
    "name_shared_idf_sum",
    "name_max_shared_idf",
    "name_mean_shared_idf",
    "addr_weighted_jaccard",
    "addr_weighted_containment",
    "addr_shared_idf_sum",
    "addr_max_shared_idf",
    "addr_mean_shared_idf",
]

FEATURE_NAMES_C = FEATURE_NAMES_B + [
    "name_char3_jaccard",
    "name_char4_jaccard",
    "addr_char3_jaccard",
    "addr_char4_jaccard",
]

FEATURE_NAMES_D = FEATURE_NAMES_C + [
    "postal_match",
    "postal_conflict",
    "address_zero_overlap",
    "name_zero_overlap",
    "strong_name_weak_addr",
    "weak_name_strong_addr",
]

ALL_FEATURE_NAMES = FEATURE_NAMES_D


def get_idf(df: int) -> float:
    return math.log(1.0 + N_CORPUS_RECORDS / (1.0 + max(df, 0)))


def precompute_s1(name1: str, addr1: str, country1: str, name_df: dict, addr_df: dict):
    """Precompute all S1-specific tokens, IDFs, and n-grams once per S1."""
    t1 = set(name1.split())
    a1 = set(addr1.split())
    n1 = {t for t in a1 if any(c.isdigit() for c in t)}
    postals1 = {t for t in n1 if len(t) in (5, 6) and t.isdigit()}

    # Char n-grams
    c3_name = {name1[i : i + 3] for i in range(len(name1) - 2)} if len(name1) >= 3 else set()
    c4_name = {name1[i : i + 4] for i in range(len(name1) - 3)} if len(name1) >= 4 else set()
    c3_addr = {addr1[i : i + 3] for i in range(len(addr1) - 2)} if len(addr1) >= 3 else set()
    c4_addr = {addr1[i : i + 4] for i in range(len(addr1) - 3)} if len(addr1) >= 4 else set()

    # IDFs
    name_idfs = {t: get_idf(name_df.get(t, 0)) for t in t1}
    addr_idfs = {t: get_idf(addr_df.get(t, 0)) for t in a1}
    name_idf_sum = sum(name_idfs.values())
    addr_idf_sum = sum(addr_idfs.values())

    return {
        "name1": name1,
        "addr1": addr1,
        "country1": country1,
        "len_name1": len(name1),
        "len_addr1": len(addr1),
        "t1": t1,
        "a1": a1,
        "n1": n1,
        "postals1": postals1,
        "c3_name": c3_name,
        "c4_name": c4_name,
        "c3_addr": c3_addr,
        "c4_addr": c4_addr,
        "name_idfs": name_idfs,
        "addr_idfs": addr_idfs,
        "name_idf_sum": name_idf_sum,
        "addr_idf_sum": addr_idf_sum,
    }


def extract_pair_features_v2(s1_pre: dict, name2: str, addr2: str, country2: str, name_df: dict, addr_df: dict):
    """Compute the full 42-feature vector for (S1, Candidate)."""
    t1 = s1_pre["t1"]
    a1 = s1_pre["a1"]
    n1 = s1_pre["n1"]
    postals1 = s1_pre["postals1"]
    name1 = s1_pre["name1"]
    addr1 = s1_pre["addr1"]
    c1 = s1_pre["country1"]

    t2 = set(name2.split())
    a2 = set(addr2.split())
    shared_name = t1 & t2
    shared_addr = a1 & a2
    n2 = {t for t in a2 if any(c.isdigit() for c in t)}
    postals2 = {t for t in n2 if len(t) in (5, 6) and t.isdigit()}

    # --- Slice A: Baseline features (22) ---
    la1, la2 = s1_pre["len_name1"], len(name2)
    name_len_ratio = min(la1, la2) / max(la1, la2) if max(la1, la2) else 0.0
    la_addr1, la_addr2 = s1_pre["len_addr1"], len(addr2)
    addr_len_ratio = min(la_addr1, la_addr2) / max(la_addr1, la_addr2) if max(la_addr1, la_addr2) else 0.0

    u_n = t1 | t2
    u_a = a1 | a2
    u_num = n1 | n2
    i_num = n1 & n2

    min_df_n = float(min(name_df.get(t, 0) for t in shared_name)) if shared_name else 0.0
    min_df_a = float(min(addr_df.get(t, 0) for t in shared_addr)) if shared_addr else 0.0

    n_ratio = fuzz.ratio(name1, name2) if (name1 or name2) else 0.0
    n_tok_set = fuzz.token_set_ratio(name1, name2) if (name1 or name2) else 0.0
    a_ratio = fuzz.ratio(addr1, addr2) if (addr1 or addr2) else 0.0
    a_tok_set = fuzz.token_set_ratio(addr1, addr2) if (addr1 or addr2) else 0.0

    feats_a = [
        1.0 if (name1 and name1 == name2) else 0.0,
        len(shared_name) / len(u_n) if u_n else 1.0,
        len(shared_name) / min(len(t1), len(t2)) if (t1 and t2) else 0.0,
        n_ratio,
        n_tok_set,
        name_len_ratio,
        float(len(shared_name)),
        min_df_n,
        1.0 if (addr1 and addr1 == addr2) else 0.0,
        len(shared_addr) / len(u_a) if u_a else 1.0,
        len(shared_addr) / min(len(a1), len(a2)) if (a1 and a2) else 0.0,
        a_ratio,
        a_tok_set,
        addr_len_ratio,
        float(len(shared_addr)),
        min_df_a,
        float(len(i_num)),
        len(i_num) / len(u_num) if u_num else 1.0,
        1.0 if (n1 and n2 and not i_num) else 0.0,
        1.0 if c1 == country2 else 0.0,
        1.0 if not name1 else 0.0,
        1.0 if not addr1 else 0.0,
    ]

    # --- Slice B: Token-IDF weighted features (10) ---
    s1_name_idfs = s1_pre["name_idfs"]
    s1_addr_idfs = s1_pre["addr_idfs"]
    sh_n_idfs = [s1_name_idfs[t] for t in shared_name]
    sh_n_sum = sum(sh_n_idfs)
    max_n_idf = max(sh_n_idfs) if sh_n_idfs else 0.0
    mean_n_idf = sh_n_sum / len(shared_name) if shared_name else 0.0

    sh_a_idfs = [s1_addr_idfs[t] for t in shared_addr]
    sh_a_sum = sum(sh_a_idfs)
    max_a_idf = max(sh_a_idfs) if sh_a_idfs else 0.0
    mean_a_idf = sh_a_sum / len(shared_addr) if shared_addr else 0.0

    c2_name_idf_sum = sum(get_idf(name_df.get(t, 0)) for t in t2)
    c2_addr_idf_sum = sum(get_idf(addr_df.get(t, 0)) for t in a2)

    u_n_idf = s1_pre["name_idf_sum"] + c2_name_idf_sum - sh_n_sum
    u_a_idf = s1_pre["addr_idf_sum"] + c2_addr_idf_sum - sh_a_sum

    name_w_jaccard = sh_n_sum / u_n_idf if u_n_idf > 0 else 1.0
    addr_w_jaccard = sh_a_sum / u_a_idf if u_a_idf > 0 else 1.0
    name_w_containment = (
        sh_n_sum / min(s1_pre["name_idf_sum"], c2_name_idf_sum)
        if min(s1_pre["name_idf_sum"], c2_name_idf_sum) > 0
        else 0.0
    )
    addr_w_containment = (
        sh_a_sum / min(s1_pre["addr_idf_sum"], c2_addr_idf_sum)
        if min(s1_pre["addr_idf_sum"], c2_addr_idf_sum) > 0
        else 0.0
    )

    feats_b = [
        name_w_jaccard,
        name_w_containment,
        sh_n_sum,
        max_n_idf,
        mean_n_idf,
        addr_w_jaccard,
        addr_w_containment,
        sh_a_sum,
        max_a_idf,
        mean_a_idf,
    ]

    # --- Slice C: Character 3/4-gram Jaccard (4) ---
    c2_name_c3 = {name2[i : i + 3] for i in range(len(name2) - 2)} if len(name2) >= 3 else set()
    c2_name_c4 = {name2[i : i + 4] for i in range(len(name2) - 3)} if len(name2) >= 4 else set()
    c2_addr_c3 = {addr2[i : i + 3] for i in range(len(addr2) - 2)} if len(addr2) >= 3 else set()
    c2_addr_c4 = {addr2[i : i + 4] for i in range(len(addr2) - 3)} if len(addr2) >= 4 else set()

    s1_nc3, s1_nc4 = s1_pre["c3_name"], s1_pre["c4_name"]
    s1_ac3, s1_ac4 = s1_pre["c3_addr"], s1_pre["c4_addr"]

    name_c3_j = len(s1_nc3 & c2_name_c3) / len(s1_nc3 | c2_name_c3) if (s1_nc3 | c2_name_c3) else 1.0
    name_c4_j = len(s1_nc4 & c2_name_c4) / len(s1_nc4 | c2_name_c4) if (s1_nc4 | c2_name_c4) else 1.0
    addr_c3_j = len(s1_ac3 & c2_addr_c3) / len(s1_ac3 | c2_addr_c3) if (s1_ac3 | c2_addr_c3) else 1.0
    addr_c4_j = len(s1_ac4 & c2_addr_c4) / len(s1_ac4 | c2_addr_c4) if (s1_ac4 | c2_addr_c4) else 1.0

    feats_c = [name_c3_j, name_c4_j, addr_c3_j, addr_c4_j]

    # --- Slice D: Address structure & cross-field signals (6) ---
    postal_match = 1.0 if (postals1 and postals2 and (postals1 & postals2)) else 0.0
    postal_conflict = 1.0 if (postals1 and postals2 and not (postals1 & postals2)) else 0.0
    zero_addr = 1.0 if not shared_addr else 0.0
    zero_name = 1.0 if not shared_name else 0.0
    strong_n_weak_a = 1.0 if (n_tok_set >= 80 and a_tok_set <= 35) else 0.0
    weak_n_strong_a = 1.0 if (n_tok_set <= 50 and a_tok_set >= 80) else 0.0

    feats_d = [postal_match, postal_conflict, zero_addr, zero_name, strong_n_weak_a, weak_n_strong_a]

    return feats_a + feats_b + feats_c + feats_d
