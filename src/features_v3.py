"""EXP-003.6 Feature Engineering Module.

Contains 52 features:
- Slice V2 (indices 0..41, 42 features): EXP-002 / EXP-003 baseline features
- Slice V3a (indices 0..46, 47 features): Slice V2 + 5 tolerant numeric & postal features
- Slice V3b (indices 0..51, 52 features): Slice V3a + 5 continuous cross-field asymmetry features
"""

import math
import re

from rapidfuzz import fuzz

N_CORPUS_RECORDS = 10_320_219
DIGIT_RE = re.compile(r"\d+")

FEATURE_NAMES_V2 = [
    # Slice A (22)
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
    # Slice B (10)
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
    # Slice C (4)
    "name_char3_jaccard",
    "name_char4_jaccard",
    "addr_char3_jaccard",
    "addr_char4_jaccard",
    # Slice D (6)
    "postal_match",
    "postal_conflict",
    "address_zero_overlap",
    "name_zero_overlap",
    "strong_name_weak_addr",
    "weak_name_strong_addr",
]

# 5 Tolerant Numeric & Postal Features
FEATURE_NAMES_NUMERIC = [
    "numeric_base_overlap",  # Base digits match ignoring letter suffix (e.g. 12 vs 12A)
    "numeric_partial_overlap_ratio",  # len(shared_nums) / min(len(nums1), len(nums2))
    "address_alpha_ratio",  # RapidFuzz ratio on address text with digits stripped
    "address_alpha_tok_jaccard",  # Token Jaccard on address text with digits stripped
    "postal_partial_match",  # First 3 digits of 5/6 digit postal codes match (district)
]

# 5 Continuous Cross-Field Asymmetry Features
FEATURE_NAMES_ASYMMETRY = [
    "max_field_ratio",  # max(name_ratio, addr_ratio)
    "min_field_ratio",  # min(name_ratio, addr_ratio)
    "max_field_tok_set",  # max(name_tok_set, addr_tok_set)
    "field_ratio_gap",  # abs(name_ratio - addr_ratio)
    "max_weighted_jaccard",  # max(name_weighted_jaccard, addr_weighted_jaccard)
]

FEATURE_NAMES_V3A = FEATURE_NAMES_V2 + FEATURE_NAMES_NUMERIC
FEATURE_NAMES_V3B = FEATURE_NAMES_V3A + FEATURE_NAMES_ASYMMETRY
ALL_FEATURE_NAMES_V3 = FEATURE_NAMES_V3B


def get_idf(df: int) -> float:
    return math.log(1.0 + N_CORPUS_RECORDS / (1.0 + max(df, 0)))


def get_base_number(tok: str) -> str:
    m = DIGIT_RE.search(tok)
    return m.group(0) if m else ""


def precompute_s1_v3(name1: str, addr1: str, country1: str, name_df: dict, addr_df: dict):
    """Precompute all S1 tokens, IDFs, n-grams, and alpha-stripped strings."""
    t1 = set(name1.split()) if name1 else set()
    a1 = set(addr1.split()) if addr1 else set()
    n1 = {t for t in a1 if any(c.isdigit() for c in t)}
    postals1 = {t for t in n1 if len(t) in (5, 6) and t.isdigit()}
    base_nums1 = {get_base_number(t) for t in n1 if get_base_number(t)}

    # Alpha-only address (digits and extra spaces stripped)
    addr1_alpha_toks = [t for t in a1 if not any(c.isdigit() for c in t)]
    addr1_alpha_str = " ".join(addr1_alpha_toks)
    addr1_alpha_set = set(addr1_alpha_toks)

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
        "base_nums1": base_nums1,
        "addr1_alpha_str": addr1_alpha_str,
        "addr1_alpha_set": addr1_alpha_set,
        "c3_name": c3_name,
        "c4_name": c4_name,
        "c3_addr": c3_addr,
        "c4_addr": c4_addr,
        "name_idfs": name_idfs,
        "addr_idfs": addr_idfs,
        "name_idf_sum": name_idf_sum,
        "addr_idf_sum": addr_idf_sum,
    }


def extract_pair_features_v3(s1_pre: dict, name2: str, addr2: str, country2: str, name_df: dict, addr_df: dict):
    """Compute the full 52-feature vector for (S1, Candidate)."""
    t1 = s1_pre["t1"]
    a1 = s1_pre["a1"]
    n1 = s1_pre["n1"]
    postals1 = s1_pre["postals1"]
    base_nums1 = s1_pre["base_nums1"]
    name1 = s1_pre["name1"]
    addr1 = s1_pre["addr1"]
    c1 = s1_pre["country1"]

    t2 = set(name2.split()) if name2 else set()
    a2 = set(addr2.split()) if addr2 else set()
    shared_name = t1 & t2
    shared_addr = a1 & a2
    n2 = {t for t in a2 if any(c.isdigit() for c in t)}
    postals2 = {t for t in n2 if len(t) in (5, 6) and t.isdigit()}
    base_nums2 = {get_base_number(t) for t in n2 if get_base_number(t)}

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

    name_tok_jacc = len(shared_name) / len(u_n) if u_n else 1.0
    addr_tok_jacc = len(shared_addr) / len(u_a) if u_a else 1.0

    num_conflict_val = 1.0 if (n1 and n2 and not i_num) else 0.0

    feats_a = [
        1.0 if (name1 and name1 == name2) else 0.0,
        name_tok_jacc,
        len(shared_name) / min(len(t1), len(t2)) if (t1 and t2) else 0.0,
        n_ratio,
        n_tok_set,
        name_len_ratio,
        float(len(shared_name)),
        min_df_n,
        1.0 if (addr1 and addr1 == addr2) else 0.0,
        addr_tok_jacc,
        len(shared_addr) / min(len(a1), len(a2)) if (a1 and a2) else 0.0,
        a_ratio,
        a_tok_set,
        addr_len_ratio,
        float(len(shared_addr)),
        min_df_a,
        float(len(i_num)),
        len(i_num) / len(u_num) if u_num else 1.0,
        num_conflict_val,
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

    denom_n = s1_pre["name_idf_sum"] + c2_name_idf_sum - sh_n_sum
    name_w_jacc = sh_n_sum / denom_n if denom_n > 0 else 1.0

    min_n_sum = min(s1_pre["name_idf_sum"], c2_name_idf_sum)
    name_w_cont = sh_n_sum / min_n_sum if min_n_sum > 0 else 0.0

    denom_a = s1_pre["addr_idf_sum"] + c2_addr_idf_sum - sh_a_sum
    addr_w_jacc = sh_a_sum / denom_a if denom_a > 0 else 1.0

    min_a_sum = min(s1_pre["addr_idf_sum"], c2_addr_idf_sum)
    addr_w_cont = sh_a_sum / min_a_sum if min_a_sum > 0 else 0.0

    feats_b = [
        name_w_jacc,
        name_w_cont,
        sh_n_sum,
        max_n_idf,
        mean_n_idf,
        addr_w_jacc,
        addr_w_cont,
        sh_a_sum,
        max_a_idf,
        mean_a_idf,
    ]

    # --- Slice C: Char 3/4-grams (4) ---
    c3_cand_name = {name2[i : i + 3] for i in range(len(name2) - 2)} if len(name2) >= 3 else set()
    c4_cand_name = {name2[i : i + 4] for i in range(len(name2) - 3)} if len(name2) >= 4 else set()
    c3_cand_addr = {addr2[i : i + 3] for i in range(len(addr2) - 2)} if len(addr2) >= 3 else set()
    c4_cand_addr = {addr2[i : i + 4] for i in range(len(addr2) - 3)} if len(addr2) >= 4 else set()

    u_c3_n = s1_pre["c3_name"] | c3_cand_name
    u_c4_n = s1_pre["c4_name"] | c4_cand_name
    u_c3_a = s1_pre["c3_addr"] | c3_cand_addr
    u_c4_a = s1_pre["c4_addr"] | c4_cand_addr

    feats_c = [
        len(s1_pre["c3_name"] & c3_cand_name) / len(u_c3_n) if u_c3_n else 1.0,
        len(s1_pre["c4_name"] & c4_cand_name) / len(u_c4_n) if u_c4_n else 1.0,
        len(s1_pre["c3_addr"] & c3_cand_addr) / len(u_c3_a) if u_c3_a else 1.0,
        len(s1_pre["c4_addr"] & c4_cand_addr) / len(u_c4_a) if u_c4_a else 1.0,
    ]

    # --- Slice D: Address Structure & Cross-Field Flags (6) ---
    has_post_match = 1.0 if (postals1 and postals2 and (postals1 & postals2)) else 0.0
    has_post_conflict = 1.0 if (postals1 and postals2 and not (postals1 & postals2)) else 0.0

    feats_d = [
        has_post_match,
        has_post_conflict,
        1.0 if len(shared_addr) == 0 else 0.0,
        1.0 if len(shared_name) == 0 else 0.0,
        1.0 if (n_ratio >= 80.0 and len(shared_addr) == 0) else 0.0,
        1.0 if (a_ratio >= 80.0 and len(shared_name) == 0) else 0.0,
    ]

    # --- Slice V3a: Tolerant Numeric & Postal Features (5) ---
    # 1. Base number overlap (e.g. 12 vs 12A)
    base_match = 1.0 if (base_nums1 and base_nums2 and (base_nums1 & base_nums2)) else 0.0

    # 2. Numeric partial overlap ratio
    if n1 and n2:
        num_part_overlap = len(i_num) / min(len(n1), len(n2))
    elif not n1 and not n2:
        num_part_overlap = 1.0
    else:
        num_part_overlap = 0.0

    # 3 & 4. Address alpha similarity (digits stripped)
    addr2_alpha_toks = [t for t in a2 if not any(c.isdigit() for c in t)]
    addr2_alpha_str = " ".join(addr2_alpha_toks)
    addr2_alpha_set = set(addr2_alpha_toks)

    addr1_alpha_str = s1_pre["addr1_alpha_str"]
    addr1_alpha_set = s1_pre["addr1_alpha_set"]

    if addr1_alpha_str or addr2_alpha_str:
        alpha_ratio = fuzz.ratio(addr1_alpha_str, addr2_alpha_str)
    else:
        alpha_ratio = 100.0

    u_alpha = addr1_alpha_set | addr2_alpha_set
    alpha_jacc = len(addr1_alpha_set & addr2_alpha_set) / len(u_alpha) if u_alpha else 1.0

    # 5. Postal partial match (first 3 digits match -> same postal district)
    postal_partial = 0.0
    if postals1 and postals2:
        p1_prefixes = {p[:3] for p in postals1 if len(p) >= 3}
        p2_prefixes = {p[:3] for p in postals2 if len(p) >= 3}
        if p1_prefixes & p2_prefixes:
            postal_partial = 1.0

    feats_numeric = [
        base_match,
        num_part_overlap,
        alpha_ratio,
        alpha_jacc,
        postal_partial,
    ]

    # --- Slice V3b: Continuous Cross-Field Asymmetry Features (5) ---
    max_f_ratio = max(n_ratio, a_ratio)
    min_f_ratio = min(n_ratio, a_ratio)
    max_f_tok_set = max(n_tok_set, a_tok_set)
    f_ratio_gap = abs(n_ratio - a_ratio)
    max_w_jacc = max(name_w_jacc, addr_w_jacc)

    feats_asymmetry = [
        max_f_ratio,
        min_f_ratio,
        max_f_tok_set,
        f_ratio_gap,
        max_w_jacc,
    ]

    return feats_a + feats_b + feats_c + feats_d + feats_numeric + feats_asymmetry


def extract_pair_features_v3a(s1_pre: dict, name2: str, addr2: str, country2: str, name_df: dict, addr_df: dict):
    """Compute the frozen 47-feature vector for EXP-004 (Slice V2 + 5 Tolerant Numeric)."""
    return extract_pair_features_v3(s1_pre, name2, addr2, country2, name_df, addr_df)[:47]
