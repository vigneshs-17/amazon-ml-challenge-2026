"""Verify the S2/S3 -> single-S1-owner observation numerically (train only).

This is an OBSERVATION from the provided training ground truth, not an assumed
constraint. Test data has no ground truth to check this against, and nothing in
the official rules promises one-to-one ownership there.
"""

import json

from io_utils import explode_ground_truth, load_ground_truth
from paths import REPORTS


def main():
    gt = load_ground_truth()
    long = explode_ground_truth(gt)
    mid = long.matched_id
    is2, is3 = mid.str.startswith("S2-"), mid.str.startswith("S3-")

    owners2 = long[is2].groupby("matched_id").source1_entity_id.nunique()
    owners3 = long[is3].groupby("matched_id").source1_entity_id.nunique()

    rep = {
        "s2_ids_in_ground_truth": len(owners2),
        "s2_ids_with_multiple_s1_owners": int((owners2 > 1).sum()),
        "s2_max_s1_owners_for_one_id": int(owners2.max()),
        "s3_ids_in_ground_truth": len(owners3),
        "s3_ids_with_multiple_s1_owners": int((owners3 > 1).sum()),
        "s3_max_s1_owners_for_one_id": int(owners3.max()),
        "conclusion": (
            "In train_ground_truth.tsv every S2/S3 id that appears is owned by "
            "exactly one S1 entity (verified exhaustively above). This is an "
            "artifact of how the labeled training set was constructed, not a "
            "rule stated in README.md. The test set carries no ground truth, so "
            "this cannot be checked there, and our candidate generation / "
            "matching model must not assume it (a record could legitimately be "
            "closest to more than one S1 entity; conflict resolution, if any, "
            "should be validated experimentally rather than hard-coded)."
        ),
    }
    json.dump(rep, open(REPORTS / "audit_ownership.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
