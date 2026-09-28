"""Build the EXP-001 normalized cache (final normalizer, incl. embedded
placeholder fix). Written to data/interim/exp001/ -- separate from the
Phase-1 caches in data/interim/*.parquet, which are left untouched as the
historical record of what Phase-1 numbers were computed against.
"""

import time

import pandas as pd

from io_utils import load_source
from paths import PROJECT_ROOT
from textnorm import normalize_basic

OUT = PROJECT_ROOT / "data" / "interim" / "exp001"
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["entity_id", "name_norm", "address_norm", "country"]


def build(split, source):
    t0 = time.time()
    df = load_source(split, source)
    name_norm = pd.array([normalize_basic(x) for x in df.business_name.astype(object)], dtype="string[pyarrow]")
    addr_norm = pd.array([normalize_basic(x) for x in df.business_address.astype(object)], dtype="string[pyarrow]")
    out = pd.DataFrame(
        {
            "entity_id": df.entity_id,
            "name_norm": name_norm,
            "address_norm": addr_norm,
            "country": df.country,
        }
    )
    path = OUT / f"{split}_s{source}.parquet"
    out.to_parquet(path, index=False)
    print(f"{split}_s{source}: {len(out)} rows, {round(time.time()-t0,1)}s -> {path}")


if __name__ == "__main__":
    for split in ("train", "test"):
        for s in (1, 2, 3):
            build(split, s)
    print("EXP-001 cache build complete (final normalizer, embedded-placeholder fix included).")
