"""Tests for 47-feature pairwise extraction and tolerant numeric/postal logic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features_v3 import FEATURE_NAMES_V3A, extract_pair_features_v3a, precompute_s1_v3


def test_47_feature_shape_and_names():
    """Verify that the model feature set contains exactly 47 features."""
    assert len(FEATURE_NAMES_V3A) == 47


def test_feature_extraction_values():
    """Verify that feature extraction produces valid 47 float features without NaNs."""
    name_df = {"amazon": 100, "development": 50, "center": 80}
    addr_df = {"hyderabad": 30, "telangana": 40, "500081": 5}

    s1_name = "amazon development center india"
    s1_addr = "hitech city hyderabad 500081"
    s1_country = "India"

    s1_pre = precompute_s1_v3(s1_name, s1_addr, s1_country, name_df, addr_df)

    cand_name = "amazon dev center"
    cand_addr = "plot 12 hitech city hyderabad 500081"
    cand_country = "India"

    feats = extract_pair_features_v3a(s1_pre, cand_name, cand_addr, cand_country, name_df, addr_df)

    assert isinstance(feats, (list, np.ndarray))
    feats = np.asarray(feats, dtype=np.float32)
    assert feats.shape == (47,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_tolerant_numeric_matching():
    """Verify tolerant numeric feature captures apartment/building numbers with letter suffixes."""
    name_df = {}
    addr_df = {}

    s1_name = "retail shop"
    s1_addr = "shop 12 main market"
    s1_country = "India"

    s1_pre = precompute_s1_v3(s1_name, s1_addr, s1_country, name_df, addr_df)

    # 12A vs 12: base digit overlap should be 1.0
    cand_name = "retail shop"
    cand_addr = "shop 12a main market"
    cand_country = "India"

    feats = np.asarray(extract_pair_features_v3a(s1_pre, cand_name, cand_addr, cand_country, name_df, addr_df))
    # Feature 42 is numeric_base_overlap
    assert feats[42] == 1.0
