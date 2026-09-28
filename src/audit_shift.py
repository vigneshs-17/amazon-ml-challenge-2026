"""EXP-000 step 5: train vs test distribution shift (per source, per country)."""

import json
from collections import Counter

import pandas as pd

from paths import INTERIM, REPORTS


def token_counter(s):
    c = Counter()
    for x in s.astype(object):
        c.update(x.split())
    return c


def main():
    rep = {}
    for src in (1, 2, 3):
        tr = pd.read_parquet(INTERIM / f"train_s{src}.parquet", columns=["country", "norm_name", "norm_addr"])
        te = pd.read_parquet(INTERIM / f"test_s{src}.parquet", columns=["country", "norm_name", "norm_addr"])
        r = {}
        for fld in ("norm_name", "norm_addr"):
            ctr_all = token_counter(tr[fld])
            for c in te.country.unique():
                cte = token_counter(te.loc[te.country == c, fld])
                tot = sum(cte.values())
                seen_occ = sum(v for t, v in cte.items() if t in ctr_all)
                seen_types = sum(1 for t in cte if t in ctr_all)
                top_unseen = [(t, v) for t, v in cte.most_common(3000) if t not in ctr_all][:40]
                r[f"{fld}|{c}"] = {
                    "test_token_occurrences_seen_in_train_pct": round(100 * seen_occ / max(tot, 1), 2),
                    "test_token_types_seen_in_train_pct": round(100 * seen_types / max(len(cte), 1), 2),
                    "top_test_tokens_unseen_in_train": top_unseen,
                    "top_tokens": cte.most_common(40),
                }
        rep[f"s{src}"] = r
        print("shift done", src, flush=True)
    json.dump(rep, open(REPORTS / "audit_shift.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
