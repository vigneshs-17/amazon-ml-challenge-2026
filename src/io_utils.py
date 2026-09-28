"""Loading helpers for the raw challenge TSVs.

Design decisions (verified in EXP-000 structural audit):
* sep="\\t" always; every line has exactly 4 fields (2 for ground truth).
* Some fields use CSV-style quoting with doubled quotes (e.g. '\"\"\"chrs & cie sasu\"'),
  so we keep pandas' default QUOTE_MINIMAL decoding and assert the row count
  matches the raw line count to guarantee no rows were merged.
* keep_default_na=False: a business literally named "NA"/"None"/"null" must not
  become NaN. Only a truly empty field is treated as missing ("").
* Strings are stored as pyarrow-backed strings to keep memory low (~5M rows/file).
"""

import pandas as pd

from paths import GROUND_TRUTH, SOURCE_FILES

STR = "string[pyarrow]"
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _raw_line_count(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # minus header


def read_tsv(path, check_rows=True):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=STR,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        engine="c",
    )
    if check_rows:
        n = _raw_line_count(path)
        assert len(df) == n, f"{path}: parsed {len(df)} rows but file has {n} data lines"
    return df


def load_source(split, source, check_rows=True):
    df = read_tsv(SOURCE_FILES[(split, source)], check_rows)
    assert list(df.columns) == SOURCE_COLS, df.columns
    return df


def load_ground_truth(check_rows=True):
    df = read_tsv(GROUND_TRUTH, check_rows)
    assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"], df.columns
    return df


def explode_ground_truth(gt):
    """Return long-form (source1_entity_id, matched_id) positive pairs."""
    s = gt.set_index("source1_entity_id")["matched_entity_ids"].astype(object)
    s = s[s.str.len() > 0].str.split(",")
    long = s.explode().rename("matched_id").reset_index()
    long["matched_id"] = long["matched_id"].str.strip()
    return long
