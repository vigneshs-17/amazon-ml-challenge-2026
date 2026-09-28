"""Cheap, bounded (<2 min) missed-positive proxy analysis.

Rather than rerunning the expensive full blocker-specific miss analysis
(audit_missed.py, which needs a fresh multi-GB inverted-index build), this
reuses the EXACT same fixed-seed 300k positive-pair sample methodology as
audit_pairs.py to answer the single most important open question cheaply:
what fraction of true positives share ZERO normalized tokens with their
match on name / address / both? This is a hard structural ceiling that no
token-overlap blocker (exact-name, rarest-1/2/3, prefix) can ever cross,
independent of which specific blocker or k is chosen.
"""

import json

import pandas as pd

from io_utils import explode_ground_truth, load_ground_truth
from paths import INTERIM, REPORTS

SEED = 42
N_SAMPLE = 300_000


def jacc_tokens(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def main():
    cols = ["entity_id", "business_name", "business_address", "country", "norm_name", "norm_addr"]
    s1 = pd.read_parquet(INTERIM / "train_s1.parquet", columns=cols)
    s23 = pd.concat([pd.read_parquet(INTERIM / f"train_s{i}.parquet", columns=cols) for i in (2, 3)], ignore_index=True)
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    pos = long.sample(min(N_SAMPLE, len(long)), random_state=SEED)
    pos = pos.merge(s1.astype({"entity_id": object}), left_on="source1_entity_id", right_on="entity_id")
    pos = pos.merge(
        s23.astype({"entity_id": object}), left_on="matched_id", right_on="entity_id", suffixes=("_1", "_2")
    )

    name_shared = [
        bool(set(a.split()) & set(b.split()))
        for a, b in zip(pos.norm_name_1.astype(object), pos.norm_name_2.astype(object))
    ]
    addr_shared = [
        bool(set(a.split()) & set(b.split()))
        for a, b in zip(pos.norm_addr_1.astype(object), pos.norm_addr_2.astype(object))
    ]
    pos["name_shared"] = name_shared
    pos["addr_shared"] = addr_shared

    n = len(pos)
    zero_name = int(sum(not x for x in name_shared))
    zero_addr = int(sum(not x for x in addr_shared))
    zero_both = int(sum((not a) and (not b) for a, b in zip(name_shared, addr_shared)))

    rep = {
        "sample_size": n,
        "note": (
            "This is a proxy for 'structurally unreachable by ANY token-overlap "
            "blocker' -- computed on the same fixed-seed 300k positive-pair "
            "sample as audit_pairs.py, NOT blocker-specific full-scale recall. "
            "A pair with zero shared name tokens cannot be retrieved by "
            "exact-name, prefix, or rarest-K-name blocking at any k; likewise "
            "for address. 'zero_both' pairs need a fundamentally different "
            "signal (numeric/exact-field, embedding, phonetic, etc.)."
        ),
        "zero_shared_name_tokens": zero_name,
        "zero_shared_name_tokens_pct": round(100 * zero_name / n, 3),
        "zero_shared_addr_tokens": zero_addr,
        "zero_shared_addr_tokens_pct": round(100 * zero_addr / n, 3),
        "zero_shared_both_pct": round(100 * zero_both / n, 3),
    }

    # representative examples of the "zero shared both" hardest tail
    both = pos[(~pos.name_shared) & (~pos.addr_shared)]
    if len(both):
        ex = both.sample(min(15, len(both)), random_state=SEED)
        rep["examples_zero_shared_both"] = (
            ex[
                [
                    "source1_entity_id",
                    "business_name_1",
                    "business_address_1",
                    "matched_id",
                    "business_name_2",
                    "business_address_2",
                    "country_1",
                ]
            ]
            .astype(str)
            .values.tolist()
        )
    else:
        rep["examples_zero_shared_both"] = []

    # name-only-zero (address still shares something -- these ARE retrievable
    # by address-token blocking even though name-token blocking would miss them)
    name_only = pos[(~pos.name_shared) & (pos.addr_shared)]
    if len(name_only):
        ex2 = name_only.sample(min(10, len(name_only)), random_state=SEED)
        rep["examples_zero_shared_name_but_addr_shares"] = (
            ex2[
                [
                    "source1_entity_id",
                    "business_name_1",
                    "business_address_1",
                    "matched_id",
                    "business_name_2",
                    "business_address_2",
                    "country_1",
                ]
            ]
            .astype(str)
            .values.tolist()
        )
    else:
        rep["examples_zero_shared_name_but_addr_shares"] = []

    json.dump(
        rep, open(REPORTS / "audit_missed_lite.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str
    )
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("examples")}, indent=1))


if __name__ == "__main__":
    main()
