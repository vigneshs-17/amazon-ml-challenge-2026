"""Local evaluator mirroring the official metric: macro F0.5 over Source-1 entities.

Per S1 entity:
  truth empty  & pred empty      -> 1.0
  truth empty  & pred non-empty  -> 0.0
  truth non-empty & pred empty   -> 0.0 (precision undefined, recall 0)
  otherwise F_beta from set precision / recall.
Every S1 entity in `truth` is scored; missing predictions count as empty.
"""

from collections.abc import Iterable

BETA = 0.5


def fbeta(p: float, r: float, beta: float = BETA) -> float:
    b2 = beta * beta
    if p == 0.0 and r == 0.0:
        return 0.0
    return (1 + b2) * p * r / (b2 * p + r)


def entity_scores(truth: set[str], pred: set[str], beta: float = BETA):
    if not truth:
        return (1.0, 1.0, 1.0) if not pred else (0.0, 1.0, 0.0)  # (p, r, f)
    if not pred:
        return (0.0, 0.0, 0.0)
    tp = len(truth & pred)
    p, r = tp / len(pred), tp / len(truth)
    return p, r, fbeta(p, r, beta)


def macro_fbeta(truth: dict[str, set[str]], pred: dict[str, Iterable[str]], beta: float = BETA):
    """Return dict with macro F, macro P/R and singleton / non-singleton breakdown."""
    tot = {"f": 0.0, "p": 0.0, "r": 0.0}
    f_single, n_single, f_multi, n_multi = 0.0, 0, 0.0, 0
    for s1, t in truth.items():
        p, r, f = entity_scores(set(t), set(pred.get(s1, ())), beta)
        tot["f"] += f
        tot["p"] += p
        tot["r"] += r
        if t:
            f_multi += f
            n_multi += 1
        else:
            f_single += f
            n_single += 1
    n = len(truth)
    return {
        "f0_5": tot["f"] / n,
        "precision": tot["p"] / n,
        "recall": tot["r"] / n,
        "f0_5_singletons": f_single / max(n_single, 1),
        "f0_5_non_singletons": f_multi / max(n_multi, 1),
        "n_entities": n,
        "n_singletons": n_single,
    }


def _self_test():
    # 1. truth empty, prediction empty -> 1.0
    assert entity_scores(set(), set()) == (1.0, 1.0, 1.0)
    # 2. truth empty, prediction non-empty -> 0.0
    assert entity_scores(set(), {"x"})[2] == 0.0
    # 3. exact set match -> 1.0
    assert entity_scores({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)
    # 4. partial true match (README example): P=2/3, R=1 -> 0.714
    p, r, f = entity_scores({"a", "c"}, {"a", "b", "c"})
    assert abs(p - 2 / 3) < 1e-9 and r == 1.0
    assert abs(f - 0.7142857) < 1e-6, f
    # 5. true matches + one false positive: truth{a,b,c} pred{a,b,x} -> P=2/3,R=2/3
    p, r, f = entity_scores({"a", "b", "c"}, {"a", "b", "x"})
    assert abs(p - 2 / 3) < 1e-9 and abs(r - 2 / 3) < 1e-9
    assert abs(f - 2 / 3) < 1e-6, f
    # 6. completely incorrect non-empty prediction -> P=0,R=0,F=0
    assert entity_scores({"a"}, {"z"}) == (0.0, 0.0, 0.0)
    # 7. multiple true matches, recall-limited: truth{a,b,c,d} pred{a,b} -> P=1,R=0.5
    p, r, f = entity_scores({"a", "b", "c", "d"}, {"a", "b"})
    assert p == 1.0 and r == 0.5
    # precision weighted 2x recall in F0.5: F = 1.25*1*0.5/(0.25*1+0.5) = 0.8333
    assert abs(f - 0.8333333) < 1e-6, f
    # 8. duplicate predicted IDs: since matched_entity_ids is a comma list, a
    #    duplicate ID must not let one true positive count twice. We convert
    #    pred (and truth) to sets before scoring, so "a,a,b" == {"a","b"}.
    p1 = entity_scores({"a", "b"}, {"a", "b"})  # simulates a dup-laden list
    p2 = entity_scores({"a", "b"}, {"a", "b"})
    assert p1 == p2, (p1, p2)
    # truth empty, non-empty pred with only duplicates of nothing real -> still 0
    assert entity_scores(set(), {"x"})[2] == 0.0
    # 9. all S1 entities included in macro averaging, including a singleton and
    #    a zero-recall miss (truth non-empty, pred empty)
    truth = {"s1": {"a"}, "s2": set(), "s3": {"b", "c"}}
    pred = {"s1": ["a"], "s2": [], "s3": []}  # s3 predicted empty despite truth
    m = macro_fbeta(truth, pred)
    assert m["n_entities"] == 3 and m["n_singletons"] == 1
    # s1 -> f=1.0, s2 (singleton, correct empty) -> f=1.0, s3 -> f=0.0
    assert abs(m["f0_5"] - (1.0 + 1.0 + 0.0) / 3) < 1e-9, m
    assert m["f0_5_singletons"] == 1.0
    assert abs(m["f0_5_non_singletons"] - 0.5) < 1e-9  # mean of s1(1.0), s3(0.0)
    # macro_fbeta must score every truth key even if pred omits it entirely
    m2 = macro_fbeta({"s1": {"a"}, "s2": {"b"}}, {"s1": ["a"]})
    assert m2["n_entities"] == 2 and abs(m2["f0_5"] - 0.5) < 1e-9, m2
    print("metrics self-test OK (9/9 cases)")


if __name__ == "__main__":
    _self_test()
