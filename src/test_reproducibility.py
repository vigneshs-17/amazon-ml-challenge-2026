"""Regression test for deterministic rare-token tie-breaking across different Python hash seeds."""

import os
import subprocess
import sys

TEST_CODE = """
import sys; sys.path.insert(0, 'src')
import pandas as pd
from blocking import RareTokenIndex

df = pd.DataFrame({
    'entity_id': ['S2-1', 'S2-2'],
    'field': ['alpha beta gamma', 'alpha beta delta'],
    'country': ['US', 'US']
})
idx = RareTokenIndex(df, 'field')
# 'gamma' and 'delta' have identical document frequency (1).
# Ties must be broken deterministically by token string alphabetical order.
toks = idx.rare_tokens('gamma delta', 'US', k=1)
print(toks[0])
"""


def main():
    outputs = []
    for seed in ["0", "1", "42", "2026", "random"]:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        res = subprocess.run([sys.executable, "-c", TEST_CODE], env=env, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        outputs.append(res.stdout.strip())

    print(f"Outputs across seeds 0, 1, 42, 2026, random: {outputs}")
    assert len(set(outputs)) == 1, f"Non-deterministic token selection! Outputs: {outputs}"
    assert outputs[0] == "delta", f"Expected 'delta' (alphabetical tie-breaker), got {outputs[0]}"
    print("Regression test PASSED: rare-token tie-breaking is 100% deterministic and process-independent.")


if __name__ == "__main__":
    main()
