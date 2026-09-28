"""EXP-001 candidate-generation module.

Efficient inverted-index blocking over normalized name/address tokens.
Deliberately avoids the Phase-1 diagnostic's O(bucket-size) `id in python_list`
pattern: every bucket is a `set`, so membership/union is O(1)/O(len(result)),
not O(bucket size) per check. Document frequencies are learned only from the
provided S2/S3 records (no external data).

Channels:
  A. rare name-token retrieval   (RareTokenIndex over name_norm)
  B. rare address-token retrieval (RareTokenIndex over address_norm)
  C. exact normalized name        (ExactIndex over name_norm)
  D. exact normalized address     (ExactIndex over address_norm)

Country handling is "soft": primary retrieval is same-country; if that
yields zero candidates the same query is re-run unrestricted. Country is
treated as an open-set string throughout -- nothing here special-cases
"US"/"India"/"France".
"""

from collections import defaultdict

import pandas as pd


class ExactIndex:
    """key -> set(entity_id), key = (country, normalized_field) or just the
    normalized field for the country-agnostic fallback index."""

    def __init__(self, df: pd.DataFrame, field: str):
        self.by_country = defaultdict(set)
        self.global_ = defaultdict(set)
        for eid, val, country in zip(df.entity_id.astype(object), df[field].astype(object), df.country.astype(object)):
            if not val:
                continue
            self.by_country[(country, val)].add(eid)
            self.global_[val].add(eid)

    def query(self, value: str, country: str):
        if not value:
            return set()
        hit = self.by_country.get((country, value))
        if hit:
            return hit
        return set()  # no cross-country fallback for exact match by design

    def query_global(self, value: str):
        return self.global_.get(value, set()) if value else set()


class RareTokenIndex:
    """Inverted index token -> set(entity_id), plus document frequency, both
    country-aware and global. Built once over the full S2+S3 pool.
    """

    def __init__(self, df: pd.DataFrame, field: str):
        self.by_country = defaultdict(set)  # (country, token) -> set(eid)
        self.global_ = defaultdict(set)  # token -> set(eid)
        self.df_country = defaultdict(int)  # (country, token) -> count
        self.df_global = defaultdict(int)  # token -> count
        for eid, val, country in zip(df.entity_id.astype(object), df[field].astype(object), df.country.astype(object)):
            if not val:
                continue
            for tok in set(val.split()):
                self.by_country[(country, tok)].add(eid)
                self.global_[tok].add(eid)
                self.df_country[(country, tok)] += 1
                self.df_global[tok] += 1

    def rare_tokens(self, value: str, country: str, k: int, country_aware: bool = True):
        toks = value.split() if value else []
        if not toks:
            return []
        keyf = (
            (lambda t: (self.df_country.get((country, t), 0), t))
            if country_aware
            else (lambda t: (self.df_global.get(t, 0), t))
        )
        return sorted(set(toks), key=keyf)[:k]

    def query_union(self, value: str, country: str, k: int, country_aware: bool = True):
        """Union of the buckets for the k rarest tokens in `value`."""
        out = set()
        for t in self.rare_tokens(value, country, k, country_aware):
            key = (country, t) if country_aware else t
            bucket = (self.by_country if country_aware else self.global_).get(key)
            if bucket:
                out |= bucket
        return out


class Blocker:
    """Bundles all four channels with country-soft fallback."""

    def __init__(self, s23: pd.DataFrame):
        self.name_idx = RareTokenIndex(s23, "name_norm")
        self.addr_idx = RareTokenIndex(s23, "address_norm")
        self.exact_name = ExactIndex(s23, "name_norm")
        self.exact_addr = ExactIndex(s23, "address_norm")

    def _soft(self, primary: set, fallback_fn):
        if primary:
            return primary
        return fallback_fn()

    def name_candidates(self, name: str, country: str, k: int = 1, country_soft: bool = True):
        primary = self.name_idx.query_union(name, country, k, country_aware=True)
        if country_soft:
            return self._soft(primary, lambda: self.name_idx.query_union(name, country, k, country_aware=False))
        return primary

    def addr_candidates(self, addr: str, country: str, k: int = 1, country_soft: bool = True):
        primary = self.addr_idx.query_union(addr, country, k, country_aware=True)
        if country_soft:
            return self._soft(primary, lambda: self.addr_idx.query_union(addr, country, k, country_aware=False))
        return primary

    def exact_name_candidates(self, name: str, country: str, country_soft: bool = True):
        primary = self.exact_name.query(name, country)
        if country_soft:
            return self._soft(primary, lambda: self.exact_name.query_global(name))
        return primary

    def exact_addr_candidates(self, addr: str, country: str, country_soft: bool = True):
        primary = self.exact_addr.query(addr, country)
        if country_soft:
            return self._soft(primary, lambda: self.exact_addr.query_global(addr))
        return primary

    def candidates(
        self,
        name: str,
        addr: str,
        country: str,
        name_k: int = 1,
        addr_k: int = 1,
        channels=("name", "addr", "exact_name", "exact_addr"),
        country_soft: bool = True,
    ):
        out = set()
        if "name" in channels:
            out |= self.name_candidates(name, country, name_k, country_soft)
        if "addr" in channels:
            out |= self.addr_candidates(addr, country, addr_k, country_soft)
        if "exact_name" in channels:
            out |= self.exact_name_candidates(name, country, country_soft)
        if "exact_addr" in channels:
            out |= self.exact_addr_candidates(addr, country, country_soft)
        return out
