"""Quantify placeholder artifacts (<NULL>, <CITY_NAME>, corrupted variants) and
mojibake-style encoding corruption, per file/field/country. Read-only; nothing
is deleted from raw data. Uses RAW columns (not the normalized cache) so counts
reflect exactly what is in the source TSVs.
"""

import json
import re
from collections import Counter

import pandas as pd

from io_utils import explode_ground_truth, load_ground_truth, load_source
from paths import REPORTS

PLACEHOLDER_RE = re.compile(r"<[A-Za-z_ ]{2,20}>")
CORRUPT_PLACEHOLDER_RE = re.compile(r"<[^>]{2,20}>")
# Mojibake: UTF-8 bytes of a non-ASCII char misread as Latin-1/cp1252, landing
# in the C1 control block (U+0080-U+009F) or the common "Ã?" / "â€™" patterns.
MOJIBAKE_RE = re.compile(r"[\x80-\x9f]|Ã[\x80-\xbf]|â€[\x80-\x9f]")


def scan_field(df, col, field_name, key):
    obj = df[col].astype(object)
    moji = obj.str.contains(MOJIBAKE_RE, regex=True, na=False)
    null_lit = obj.str.strip().str.casefold() == "<null>"
    city_lit = obj.str.contains(r"<CITY_NAME>", regex=False, na=False)
    all_tags = Counter()
    for s in obj[obj.str.contains("<", regex=False, na=False)]:
        for m in CORRUPT_PLACEHOLDER_RE.findall(s):
            all_tags[m] += 1
    return {
        f"{field_name}_null_placeholder": int(null_lit.sum()),
        f"{field_name}_city_name_placeholder": int(city_lit.sum()),
        f"{field_name}_any_angle_bracket_tag": int(obj.str.contains("<", regex=False, na=False).sum()),
        f"{field_name}_distinct_tag_variants": len(all_tags),
        f"{field_name}_top_tag_variants": all_tags.most_common(25),
        f"{field_name}_mojibake_rows": int(moji.sum()),
        f"{field_name}_mojibake_examples": obj[moji].head(8).tolist(),
    }


def main():
    rep = {}
    for split in ("train", "test"):
        for s in (1, 2, 3):
            df = load_source(split, s)
            key = f"{split}_s{s}"
            r = {"rows": len(df)}
            for col, fname in [("business_name", "name"), ("business_address", "address")]:
                r.update(scan_field(df, col, fname, key))
            r["by_country"] = {}
            for c in df.country.unique():
                sub = df[df.country == c]
                nlit = (sub.business_name.astype(object).str.strip().str.casefold() == "<null>").sum() + (
                    sub.business_address.astype(object).str.strip().str.casefold() == "<null>"
                ).sum()
                city = (
                    sub.business_name.astype(object).str.contains("<CITY_NAME>", regex=False).sum()
                    + sub.business_address.astype(object).str.contains("<CITY_NAME>", regex=False).sum()
                )
                moji = (
                    sub.business_name.astype(object).str.contains(MOJIBAKE_RE, regex=True, na=False).sum()
                    + sub.business_address.astype(object).str.contains(MOJIBAKE_RE, regex=True, na=False).sum()
                )
                r["by_country"][c] = {
                    "null_placeholder": int(nlit),
                    "city_name_placeholder": int(city),
                    "mojibake": int(moji),
                }
            rep[key] = r
            print(key, "placeholders/mojibake scanned", flush=True)

    # Do placeholders appear inside TRUE positive pairs (train)?
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    s1 = load_source("train", 1)[["entity_id", "business_name", "business_address"]]
    s23 = pd.concat(
        [
            load_source("train", 2)[["entity_id", "business_name", "business_address"]],
            load_source("train", 3)[["entity_id", "business_name", "business_address"]],
        ],
        ignore_index=True,
    )
    pos = long.merge(s1, left_on="source1_entity_id", right_on="entity_id").merge(
        s23, left_on="matched_id", right_on="entity_id", suffixes=("_1", "_2")
    )
    has_ph_1 = pos.business_name_1.astype(object).str.contains("<", regex=False) | pos.business_address_1.astype(
        object
    ).str.contains("<", regex=False)
    has_ph_2 = pos.business_name_2.astype(object).str.contains("<", regex=False) | pos.business_address_2.astype(
        object
    ).str.contains("<", regex=False)
    has_moji = (
        pos.business_name_1.astype(object).str.contains(MOJIBAKE_RE, regex=True)
        | pos.business_address_1.astype(object).str.contains(MOJIBAKE_RE, regex=True)
        | pos.business_name_2.astype(object).str.contains(MOJIBAKE_RE, regex=True)
        | pos.business_address_2.astype(object).str.contains(MOJIBAKE_RE, regex=True)
    )
    rep["positive_pairs_with_placeholder_on_either_side"] = int((has_ph_1 | has_ph_2).sum())
    rep["positive_pairs_total"] = len(pos)
    rep["positive_pairs_with_mojibake"] = int(has_moji.sum())
    rep["placeholder_pair_examples"] = (
        pos[has_ph_1 | has_ph_2][
            [
                "source1_entity_id",
                "business_name_1",
                "business_address_1",
                "matched_id",
                "business_name_2",
                "business_address_2",
            ]
        ]
        .head(10)
        .astype(str)
        .values.tolist()
    )
    rep["mojibake_pair_examples"] = (
        pos[has_moji][
            [
                "source1_entity_id",
                "business_name_1",
                "business_address_1",
                "matched_id",
                "business_name_2",
                "business_address_2",
            ]
        ]
        .head(10)
        .astype(str)
        .values.tolist()
    )

    json.dump(
        rep, open(REPORTS / "audit_placeholders.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str
    )
    print("done")


if __name__ == "__main__":
    main()
