"""EXP-001 pair features. Interpretable, small feature set per Phase-1
evidence: RapidFuzz name/address similarities, exact-equality flags (kept
separate per Phase-1's precision-profile finding), token Jaccard/containment,
numeric-token agreement, country equality, missing-field flags, and
strongest-shared-token rarity.
"""

import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
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


def _nums(s):
    return {t for t in s.split() if any(c.isdigit() for c in t)}


def _containment(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _len_ratio(a, b):
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0


def pair_features(name1, addr1, country1, name2, addr2, country2, name_df=None, addr_df=None):
    """Compute the feature vector for one S1<->candidate pair. `name_df`/
    `addr_df` are optional dict-like (token -> document frequency) used for
    the shared-token-rarity feature; if omitted that feature is 0.
    """
    t1, t2 = set(name1.split()), set(name2.split())
    a1, a2 = set(addr1.split()), set(addr2.split())
    shared_name = t1 & t2
    shared_addr = a1 & a2
    n1, n2 = _nums(addr1), _nums(addr2)

    def min_df(shared, dfm):
        if not shared or dfm is None:
            return 0.0
        return float(min(dfm.get(t, 0) for t in shared))

    feats = [
        1.0 if (name1 and name1 == name2) else 0.0,
        len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 1.0,
        _containment(t1, t2),
        fuzz.ratio(name1, name2) if (name1 or name2) else 0.0,
        fuzz.token_set_ratio(name1, name2) if (name1 or name2) else 0.0,
        _len_ratio(name1, name2),
        float(len(shared_name)),
        min_df(shared_name, name_df),
        1.0 if (addr1 and addr1 == addr2) else 0.0,
        len(a1 & a2) / len(a1 | a2) if (a1 | a2) else 1.0,
        _containment(a1, a2),
        fuzz.ratio(addr1, addr2) if (addr1 or addr2) else 0.0,
        fuzz.token_set_ratio(addr1, addr2) if (addr1 or addr2) else 0.0,
        _len_ratio(addr1, addr2),
        float(len(shared_addr)),
        min_df(shared_addr, addr_df),
        float(len(n1 & n2)),
        len(n1 & n2) / len(n1 | n2) if (n1 | n2) else 1.0,
        1.0 if (n1 and n2 and not (n1 & n2)) else 0.0,
        1.0 if country1 == country2 else 0.0,
        1.0 if not name1 else 0.0,
        1.0 if not addr1 else 0.0,
    ]
    return np.array(feats, dtype=np.float32)
