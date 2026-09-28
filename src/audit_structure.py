"""EXP-000 step 1: byte-level structural audit of every raw TSV (no pandas).

Checks field count per line, quote characters, stray CR, invalid UTF-8,
and ID-prefix consistency. Pandas can silently mangle any of these.
"""

import json
from collections import Counter

from paths import GROUND_TRUTH, REPORTS, SOURCE_FILES


def scan(path, expected_prefix=None):
    field_counts = Counter()
    stats = Counter()
    bad_examples = []
    with open(path, "rb") as f:
        header = f.readline().rstrip(b"\n").split(b"\t")
        for n, raw in enumerate(f, start=2):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                stats["invalid_utf8"] += 1
                line = raw.decode("utf-8", "replace")
            line = line.rstrip("\n")
            if "\r" in line:
                stats["contains_CR"] += 1
            if '"' in line:
                stats["contains_double_quote"] += 1
            if line.strip() == "":
                stats["blank_lines"] += 1
                continue
            parts = line.split("\t")
            field_counts[len(parts)] += 1
            if len(parts) != len(header) and len(bad_examples) < 5:
                bad_examples.append((n, line[:200]))
            if expected_prefix and not parts[0].startswith(expected_prefix):
                stats["wrong_id_prefix"] += 1
    return {
        "header": [h.decode() for h in header],
        "field_count_distribution": dict(field_counts),
        **stats,
        "bad_examples": bad_examples,
    }


if __name__ == "__main__":
    out = {}
    for (split, s), p in SOURCE_FILES.items():
        out[f"{split}_s{s}"] = scan(p, f"S{s}-")
        print(split, s, out[f"{split}_s{s}"])
    out["ground_truth"] = scan(GROUND_TRUTH, "S1-")
    print("gt", out["ground_truth"])
    REPORTS.mkdir(exist_ok=True)
    json.dump(out, open(REPORTS / "audit_structure.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
