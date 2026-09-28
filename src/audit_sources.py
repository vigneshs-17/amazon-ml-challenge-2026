"""EXP-000 step 2: per-file audit (shapes, quality, country, text characteristics).

Writes a normalized parquet cache per file to data/interim/ (derived data only;
raw TSVs are never modified) and a JSON report to reports/audit_sources.json.
"""

import json
import re
import time
from collections import Counter

import pandas as pd

from io_utils import load_source
from paths import INTERIM, REPORTS
from textnorm import SCRIPT_PATTERNS, normalize_basic

ID_RE = r"^S[123]-\d+$"
QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def qdict(s):
    return {str(q): float(v) for q, v in s.quantile(QUANTILES).items()} | {"mean": float(s.mean())}


def field_quality(col):
    obj = col.astype(object)
    return {
        "null": int(col.isna().sum()),
        "empty": int((obj == "").sum()),
        "empty_pct": round(100 * float((obj == "").mean()), 3),
        "whitespace_only": int(((obj != "") & (col.str.strip() == "")).sum()),
        "leading_trailing_ws": int((col != col.str.strip()).sum()),
        "double_space": int(col.str.contains("  ", regex=False).sum()),
    }


def text_profile(col, norm_col, top_k=60):
    lens = col.str.len().astype("float64")
    ntok = norm_col.str.count(" ").astype("float64") + (norm_col.str.len() > 0)
    prof = {"char_len": qdict(lens), "token_count": qdict(ntok)}
    prof["script_pct"] = {
        k: round(100 * float(col.str.contains(p, regex=True).mean()), 3) for k, p in SCRIPT_PATTERNS.items()
    }
    # punctuation character frequency (share of rows containing each char)
    sample = col.sample(min(len(col), 300_000), random_state=0).astype(object)
    rowpunct = Counter()
    for s in sample:
        for ch in set(re.findall(r"[^\w\s]", s)):
            rowpunct[ch] += 1
    prof["punct_row_pct_sample"] = {ch: round(100 * c / len(sample), 3) for ch, c in rowpunct.most_common(25)}
    tok_counter = Counter()
    last_tok = Counter()
    first_tok = Counter()
    for s in norm_col.astype(object):
        if not s:
            continue
        t = s.split()
        tok_counter.update(t)
        last_tok[t[-1]] += 1
        first_tok[t[0]] += 1
    n_types = len(tok_counter)
    singletons = sum(1 for c in tok_counter.values() if c == 1)
    prof["vocab_size"] = n_types
    prof["hapax_tokens"] = singletons
    prof["hapax_pct_of_vocab"] = round(100 * singletons / max(n_types, 1), 2)
    prof["top_tokens"] = tok_counter.most_common(top_k)
    prof["top_last_tokens"] = last_tok.most_common(40)
    prof["top_first_tokens"] = first_tok.most_common(25)
    prof["rare_token_examples"] = [t for t, c in tok_counter.items() if c == 1][:30]
    return prof, tok_counter


def audit_file(split, source):
    t0 = time.time()
    df = load_source(split, source)
    key = f"{split}_s{source}"
    rep = {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "memory_mb_deep": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }
    df["norm_name"] = pd.array([normalize_basic(x) for x in df.business_name.astype(object)], dtype="string[pyarrow]")
    df["norm_addr"] = pd.array(
        [normalize_basic(x) for x in df.business_address.astype(object)], dtype="string[pyarrow]"
    )
    df.to_parquet(INTERIM / f"{key}.parquet", index=False)

    q = {c: field_quality(df[c]) for c in ["entity_id", "business_name", "business_address", "country"]}
    rep["field_quality"] = q
    raw_cols = ["entity_id", "business_name", "business_address", "country"]
    rep["duplicates"] = {
        "exact_duplicate_rows": int(df.duplicated(raw_cols).sum()),
        "duplicate_rows_ignoring_id": int(df.duplicated(raw_cols[1:]).sum()),
        "duplicate_entity_ids": int(df.entity_id.duplicated().sum()),
        "bad_entity_id_format": int((~df.entity_id.str.match(ID_RE)).sum()),
        "duplicate_business_name_raw": int(df.business_name.duplicated().sum()),
        "distinct_business_name_raw": int(df.business_name.nunique()),
        "duplicate_business_name_norm": int(df.norm_name.duplicated().sum()),
        "distinct_business_name_norm": int(df.norm_name.nunique()),
        "duplicate_name_addr_norm": int(df.duplicated(["norm_name", "norm_addr"]).sum()),
        "duplicate_name_addr_country_norm": int(df.duplicated(["norm_name", "norm_addr", "country"]).sum()),
    }
    # suspicious / malformed records
    nn = df.norm_name.astype(object)
    starts_nonalnum = (
        df.business_name.str.slice(0, 1).astype(object).map(lambda c: c != "" and not c.isalnum()).astype(bool)
    )
    rep["suspicious"] = {
        "name_no_alnum": int((nn == "").sum()),
        "name_len_le_2": int((df.business_name.str.len() <= 2).sum()),
        "name_is_url_like": int(df.business_name.str.contains(r"(?i)(www\.|\.com|\.in\b|\.fr\b|http)").sum()),
        "name_starts_nonalnum": int(starts_nonalnum.sum()),
        "name_all_upper": int(
            (df.business_name == df.business_name.str.upper()).sum()
            - (df.business_name.str.upper() == df.business_name.str.lower()).sum()
        ),
        "name_all_lower": int(
            (df.business_name == df.business_name.str.lower()).sum()
            - (df.business_name.str.upper() == df.business_name.str.lower()).sum()
        ),
        "addr_empty": int((df.business_address.astype(object) == "").sum()),
        "addr_no_digit": int((~df.business_address.str.contains(r"[0-9]")).sum()),
        "name_equals_addr": int((df.norm_name == df.norm_addr).sum()),
        "examples_name_starts_nonalnum": df.business_name[starts_nonalnum.values].head(15).tolist(),
        "examples_short_names": df.business_name[df.business_name.str.len() <= 2].head(10).tolist(),
    }
    cc = df.country.value_counts()
    rep["country"] = {k: {"n": int(v), "pct": round(100 * v / len(df), 3)} for k, v in cc.items()}

    rep["per_country"] = {}
    for c in cc.index:
        sub = df[df.country == c]
        pc = {}
        pc["name"], _ = text_profile(sub.business_name, sub.norm_name)
        pc["address"], _ = text_profile(sub.business_address, sub.norm_addr)
        pc["name_empty_pct"] = round(100 * float((sub.business_name.astype(object) == "").mean()), 3)
        pc["addr_empty_pct"] = round(100 * float((sub.business_address.astype(object) == "").mean()), 3)
        pc["examples"] = sub.sample(min(12, len(sub)), random_state=42)[
            ["business_name", "business_address"]
        ].values.tolist()
        rep["per_country"][c] = pc
    rep["seconds"] = round(time.time() - t0, 1)
    print(key, "done", rep["seconds"], "s", flush=True)
    return key, rep


if __name__ == "__main__":
    INTERIM.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    out = {}
    for split in ["train", "test"]:
        for s in [1, 2, 3]:
            k, r = audit_file(split, s)
            out[k] = r
            json.dump(
                out,
                open(REPORTS / "audit_sources.json", "w", encoding="utf-8"),
                indent=1,
                ensure_ascii=False,
                default=str,
            )
